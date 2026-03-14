from __future__ import annotations
from typing import Callable
from collections import namedtuple

from jinja2 import Template, Environment, meta

import torch
from torch.optim import Adam
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn, cat, stack, is_tensor, tensor, Tensor

from accelerate import Accelerator

from einops import rearrange

from torch_einops_utils import (
    pad_sequence,
    safe_cat,
    masked_mean,
    and_masks
)

from ema_pytorch import EMA

from x_transformers import TransformerWrapper

from discrete_continuous_embed_readout import Readout

# default query / demonstration template for in-context learned distillation targets from teacher for student

DEFAULT_STUDENT_PROMPT_TEMPLATE = """
[Instruction]
You are a helpful assistant

[Query]
{{ question }}

[Response]
"""

DEFAULT_TEACHER_PROMPT_TEMPLATE = """
[Task Instructions] You are a helpful assistant. Please answer the question based on the provided logic.

[Expert Demonstration] Question: {{ question }} Expert Reasoning and Answer: {{ answer }}

[Current Task] Question: {{ question }} Answer:
"""

def get_variables_from_template(template):

    env = Environment()

    parsed_template = env.parse(template)

    return set(meta.find_undeclared_variables(parsed_template))

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def maybe_cast_tensor(t):
    return t if is_tensor(t) else tensor(t)

# classes

SDFTOutput = namedtuple('SDFTOutput', ('loss', 'response'))

class SDFTMERA(Module):
    def __init__(
        self,
        model: TransformerWrapper,
        tokenizer_encode: Callable[[list[str]], list[Tensor]],
        student_max_response_len,
        student_prompt_template = DEFAULT_STUDENT_PROMPT_TEMPLATE,
        teacher_update_rate = 0.01,
        training_stage = "sdft-mera",
        teacher_prompt_template = DEFAULT_TEACHER_PROMPT_TEMPLATE,
        num_init_student_response_tokens_mask = 0,  # they mentioned some issue where the student starts repeating stuff in the prompt template, where they alleviate by masking out the loss for first few tokens
        eos_id = None, # if set, will mask out any losses after the first eos token id detected in a given sample

        eov_id = None,            
        icd_vocab_ids = None,
        mera_contrastive_weight = 0.0,
        mera_diversity_weight = 0.0,
        sdft_loss_kl_weight = 0.0

    ):
        super().__init__()

        if isinstance(model, TransformerWrapper):
            model.input_not_include_cache = True

        self.student = model

        self.teacher = EMA(
            model,
            beta = 1. - teacher_update_rate,
            include_online_model = False
        )


        self.training_stage = training_stage

        # sampling

        self.icd_vocab_ids = icd_vocab_ids

        self.student_max_response_len = student_max_response_len

        self.discrete_readout = Readout(dim = 0, num_discrete = 1)

        # collection of prompts to list[Int['seq']]

        self.tokenizer_encode = tokenizer_encode

        # store templates

        assert get_variables_from_template(teacher_prompt_template) == {'question', 'answer'}, 'your template must contain only variables `question` and `answer`, embedded like so - {{ question }} ... {{ answer }}'
        self.teacher_prompt_template = Template(teacher_prompt_template)

        assert get_variables_from_template(student_prompt_template) == {'question'}
        self.student_prompt_template = Template(student_prompt_template)

        # end of string

        self.eos_id = eos_id

        # how many initial response tokens to exclude from reverse kl loss

        self.num_init_student_response_tokens_mask = num_init_student_response_tokens_mask

        self.eov_id = eov_id
        self.mera_contrastive_weight = mera_contrastive_weight
        self.mera_diversity_weight = mera_diversity_weight
        self.sdft_loss_kl_weight = sdft_loss_kl_weight


        if self.training_stage == "sdft":
            self.mera_contrastive_weight = 0.0
            self.mera_diversity_weight = 0.0
        if self.training_stage == "mera":
            self.sdft_loss_kl_weight = 0.0


    def parameters(self):
        return self.student.parameters()

    def update_teacher_ema_(self):
        self.teacher.update()

    def forward(
        self,
        questions: list[str],
        answers: list[str],
        hard_negatives: list[list[str]] = None,
        student_logit_sample_kwargs: dict = dict(),
    
    ):
        maybe_eos_id, prefix_mask_len = self.eos_id, self.num_init_student_response_tokens_mask

        batch_size = len(questions)

        encode = self.tokenizer_encode

        # neg_ids_list = None
        # if "mera" in self.training_stage:
            # neg_ids_list = []
            # for negs in hard_negatives:
            #     # encode(n) returns a 1D tensor like tensor([8334, 12, ...])
            #     # [0] gets the first token, .item() turns it into a plain integer
            #     n_ids = [encode(n)[0].item() for n in negs]
            #     neg_ids_list.append(n_ids)

        # --- PRE-PROCESS HARD NEGATIVES FOR VECTORIZATION ---
        padded_negs = None
        valid_neg_mask = None
        
        if "mera" in self.training_stage and exists(hard_negatives):
            device = next(self.parameters()).device
            
            max_negs = max(len(negs) for negs in hard_negatives)
            
            padded_negs = torch.zeros((batch_size, max_negs), dtype=torch.long, device=device)
            valid_neg_mask = torch.zeros((batch_size, max_negs), dtype=torch.bool, device=device)
            
            for b, negs in enumerate(hard_negatives):
                # Encode strings to IDs
                n_ids = [encode(n)[0].item() for n in negs]
                
                padded_negs[b, :len(n_ids)] = torch.tensor(n_ids, device=device)
                
                valid_neg_mask[b, :len(n_ids)] = True
                
            self.target_vocab_tensor = torch.tensor(list(self.icd_vocab_ids) + [self.eov_id], device=device)

        
        assert len(questions) == len(answers)

        student_vars = [{'question': question} for question in questions]
        teacher_vars = [{'question': question, 'answer': answer} for question, answer in zip(questions, answers)]

        # ready the prompts for student and teacher

        student_prompts_str = [self.student_prompt_template.render(questions) for questions in student_vars]
        teacher_prompts_str = [self.teacher_prompt_template.render(question_answers) for question_answers in teacher_vars]

        student_prompt_ids = [maybe_cast_tensor(encode(prompt)) for prompt in student_prompts_str]
        teacher_prompt_ids = [maybe_cast_tensor(encode(prompt)) for prompt in teacher_prompts_str]

        student_prompt_ids, student_seq_start_pos = pad_sequence(student_prompt_ids, return_lens = True, left = True, pad_lens = True)
        teacher_prompt_ids, teacher_seq_start_pos = pad_sequence(teacher_prompt_ids, return_lens = True, left = True, pad_lens = True)

        device = next(self.parameters()).device
        student_prompt_ids = student_prompt_ids.to(device)
        teacher_prompt_ids = teacher_prompt_ids.to(device)
        
        if is_tensor(student_seq_start_pos):
            student_seq_start_pos = student_seq_start_pos.to(device)
        if is_tensor(teacher_seq_start_pos):
            teacher_seq_start_pos = teacher_seq_start_pos.to(device)

        student_cache = None
        teacher_cache = None

        # accumulate

        student_responses = None
        token_kl_div_losses = None
        total_step_losses = None

        for _ in range(self.student_max_response_len):

            # forward for logit of student and teacher

            # student_logits, student_cache = self.student(student_prompt_ids, cache = student_cache, seq_start_pos = student_seq_start_pos, return_intermediates = True)
            outputs = self.student(student_prompt_ids, cache = student_cache, seq_start_pos = student_seq_start_pos, return_intermediates = True)


            student_logits = outputs.logits
            student_cache = outputs.past_key_values


            with torch.no_grad():
                self.teacher.eval()
                outputs = self.teacher(teacher_prompt_ids, cache = teacher_cache, seq_start_pos = teacher_seq_start_pos, return_intermediates = True)

            teacher_logits = outputs.logits
            teacher_cache = outputs.past_key_values

            student_token_logit = student_logits[:, -1:]
            teacher_token_logit = teacher_logits[:, -1:]

            student_token_probs = student_token_logit.softmax(dim = -1)
            teacher_token_log_probs = teacher_token_logit.log_softmax(dim = -1)

            # privileged self distillation via ICL

            token_kl_div = F.kl_div(
                teacher_token_log_probs,
                student_token_probs,
                reduction = 'none'

            ).sum(dim = -1)

            combined_loss = token_kl_div * self.sdft_loss_kl_weight

         
            if "mera" in self.training_stage:
                student_probs_flat = rearrange(student_token_probs, 'b 1 c -> b c')
                
                # teachers prds
                teacher_pos_ids = teacher_token_logit[:, 0].argmax(dim=-1)

                pos_ids  = teacher_pos_ids
                
                # mask to ignore non ICD and EOv
                mera_active_mask = torch.isin(teacher_pos_ids, self.target_vocab_tensor) 
                
      
                pos_ids_expanded = rearrange(pos_ids, 'b -> b 1')
                pos_probs_expanded = student_probs_flat.gather(1, pos_ids_expanded) 
                pos_probs = rearrange(pos_probs_expanded, 'b 1 -> b')
                
                eov_probs = student_probs_flat[:, self.eov_id]                            
                neg_probs = student_probs_flat.gather(1, padded_negs)                     
                
                # constrastive loss
                neg_probs_sum = (neg_probs * valid_neg_mask).sum(dim=1) 
                denom = pos_probs + neg_probs_sum
                contrastive_loss = -torch.log(pos_probs / (denom + 1e-8))
                
                # diversity loss
                term1 = F.relu(eov_probs - pos_probs)
                
                eov_probs_expanded = rearrange(eov_probs, 'b -> b 1')
                term2_raw = F.relu(neg_probs - eov_probs_expanded)
                
                term2_sum = (term2_raw * valid_neg_mask).sum(dim=1)
                num_valid = valid_neg_mask.sum(dim=1).clamp(min=1) 
                eov_loss = term1 + (term2_sum / num_valid)
                
             
                mera_loss_step = (self.mera_contrastive_weight * contrastive_loss) + (self.mera_diversity_weight * eov_loss)
                
                mera_loss_step = mera_loss_step * mera_active_mask.float()
                
                mera_loss_step_expanded = rearrange(mera_loss_step, 'b -> b 1')
                combined_loss = combined_loss + mera_loss_step_expanded


            # final loss
            total_step_losses = safe_cat((total_step_losses, combined_loss), dim = 1)

            # sample

            sampled_action = self.discrete_readout.sample(student_token_logit, **student_logit_sample_kwargs)

            student_responses = safe_cat((student_responses, sampled_action), dim = 1)

            # break if all eos

            if exists(maybe_eos_id) and (student_responses == maybe_eos_id).any(dim = -1).all():
                break

            # set student and teacher tokens to the next sampled token

            student_prompt_ids = sampled_action
            teacher_prompt_ids = sampled_action

        # handle eos

        eos_mask = None

        if exists(maybe_eos_id):
            eos_mask = (student_responses == maybe_eos_id).cumsum(dim = -1) == 0

            eos_mask = F.pad(eos_mask, (1, -1), value = True)

            student_responses.masked_fill_(~eos_mask, -1)

        # handle masking of first few response tokens

        init_tokens_mask = None

        if prefix_mask_len > 0:
            init_tokens_mask = torch.ones_like(student_responses).bool()
            init_tokens_mask[:, :prefix_mask_len] = False

        # maybe masked mean for losses

        mask = and_masks([eos_mask, init_tokens_mask])

        loss = masked_mean(total_step_losses, mask)

        return SDFTOutput(loss, student_responses)

# trainer

class SDFTMERATrainer(Module):
    def __init__(
        self,
        model: Module,
        dataset: Dataset,
        tokenizer_encode: Callable,
        batch_size = 4,
        grad_accum_steps = 1,
        learning_rate = 2e-5,
        max_grad_norm = 0.5,
        sdft_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        optim_klass = Adam,
        optim_kwargs: dict = dict()
    ):
        super().__init__()

        self.accelerator = Accelerator(
            gradient_accumulation_steps = grad_accum_steps,
            **accelerate_kwargs
        )

        self.model = SDFTMERA(
            model,
            tokenizer_encode = tokenizer_encode,
            **sdft_kwargs
        )

        self.optimizer = optim_klass(self.model.parameters(), lr = learning_rate, **optim_kwargs)

        self.dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )

        self.max_grad_norm = max_grad_norm

    def train(self, num_epochs=2): 
        self.model.train()

        for epoch in range(num_epochs):
            print(f"Starting Epoch {epoch + 1}/{num_epochs}...")

            for questions, answers, hard_negatives in self.dataloader:
                with self.accelerator.accumulate(self.model):
                    output = self.model(questions, answers, hard_negatives)

                    self.accelerator.backward(output.loss)

                    if exists(self.max_grad_norm) and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.accelerator.sync_gradients:
                        self.model.update_teacher_ema_()

        return output
