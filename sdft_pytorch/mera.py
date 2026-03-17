from __future__ import annotations
from typing import Callable, List, Tuple
from collections import namedtuple

from jinja2 import Template, Environment, meta

import torch
from torch.optim import Adam
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import tensor, is_tensor, Tensor

from accelerate import Accelerator
from einops import rearrange, repeat


DEFAULT_PROMPT_TEMPLATE = """
[Instruction]
You are a helpful assistant predicting clinical diagnoses.

[Patient History]
{{ question }}

[ICD Code Prediction]
"""

def get_variables_from_template(template):
    env = Environment()
    parsed_template = env.parse(template)
    return set(meta.find_undeclared_variables(parsed_template))

def exists(v):
    return v is not None

MERAOutput = namedtuple('MERAOutput', ('loss', 'logits'))



class MERA(Module):
    def __init__(
        self,
        model: Module,
        tokenizer_encode: Callable,
        prompt_template = DEFAULT_PROMPT_TEMPLATE,
        eov_id: int = None,
        icd_vocab_ids: list[int] = None,
        mera_contrastive_weight: float = 1.0,
        mera_diversity_weight: float = 1.0,
    ):
        super().__init__()

        self.model = model
        self.tokenizer_encode = tokenizer_encode

        assert get_variables_from_template(prompt_template) == {'question'}
        self.prompt_template = Template(prompt_template)

        self.eov_id = eov_id
        self.icd_vocab_ids = icd_vocab_ids
        
        self.mera_contrastive_weight = mera_contrastive_weight
        self.mera_diversity_weight = mera_diversity_weight

        if exists(self.icd_vocab_ids) and exists(self.eov_id):
            self.target_vocab_tensor = torch.tensor(list(self.icd_vocab_ids) + [self.eov_id])
        else:
            self.target_vocab_tensor = None

    def parameters(self):
        return self.model.parameters()

    def forward(
        self,
        questions: list[str],
        answers: list[str],
        hard_negatives: list[list[str]] = None,
    ):
        device = next(self.parameters()).device
        batch_size = len(questions)
        encode = self.tokenizer_encode

    
        prompts_vars = [{'question': question} for question in questions]
        prompts_str = [self.prompt_template.render(prompt) for prompt in prompts_vars]

    
        encoded_inputs = encode(prompts_str)
        input_ids = encoded_inputs['input_ids'].to(device)

        attention_mask = encoded_inputs['attention_mask'].to(device)

      
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # last token
        next_token_logits = outputs.logits[:, -1, :] 
        probs = next_token_logits.softmax(dim=-1)
        vocab_size = next_token_logits.shape[-1]

       
        pos_mask = torch.zeros((batch_size, vocab_size), dtype=torch.bool, device=device)
        neg_mask = torch.zeros((batch_size, vocab_size), dtype=torch.bool, device=device)

    
        # ans_encoded = encode(answers)['input_ids'].to(device)
        # pos_mask.scatter_(1, ans_encoded, True) 

        pos_counts = torch.tensor([len(a) for a in answers], device=device)
        max_pos = pos_counts.max().item()
        
        if max_pos > 0:
   
            pos_flat = [ans for ans_list in answers for ans in ans_list]
            pos_encoded = encode(pos_flat)['input_ids'].to(device)
            
        
            batch_ids = torch.arange(batch_size, device=device)
            batch_grid = repeat(batch_ids, 'b -> b l', l=max_pos)
            
            pos_idx = torch.arange(max_pos, device=device)
            pos_grid = repeat(pos_idx, 'l -> b l', b=batch_size)
            counts_grid = repeat(pos_counts, 'b -> b l', l=max_pos)

         
            valid_mask = pos_grid < counts_grid
            b_idx_tensor = batch_grid[valid_mask]  
            
            b_idx_grid = rearrange(b_idx_tensor, 'n -> n 1').expand_as(pos_encoded)
            pos_mask[b_idx_grid, pos_encoded] = True



        if exists(hard_negatives):
         
            neg_counts = torch.tensor([len(n) for n in hard_negatives], device=device)
            max_negs = neg_counts.max().item()
            
            if max_negs > 0:
                neg_flat = [neg for neg_list in hard_negatives for neg in neg_list]
                neg_encoded = encode(neg_flat)['input_ids'].to(device) 
                
                batch_ids = torch.arange(batch_size, device=device)
                batch_grid = repeat(batch_ids, 'b -> b l', l=max_negs)
                
                pos = torch.arange(max_negs, device=device)
                pos_grid = repeat(pos, 'l -> b l', b=batch_size)
                counts_grid = repeat(neg_counts, 'b -> b l', l=max_negs)

                valid_mask = pos_grid < counts_grid
                b_idx_tensor = batch_grid[valid_mask]  

                
                b_idx_grid = rearrange(b_idx_tensor, 'n -> n 1').expand_as(neg_encoded)
                neg_mask[b_idx_grid, neg_encoded] = True


        pad_id = self.model.config.pad_token_id
        if pad_id is not None:
            pos_mask[:, pad_id] = False
            neg_mask[:, pad_id] = False


        pos_mask_f = pos_mask.to(probs.dtype)
        neg_mask_f = neg_mask.to(probs.dtype)

        # Contrastive Loss
        sum_pos_probs = torch.einsum('b v, b v -> b', probs, pos_mask_f)
        sum_neg_probs = torch.einsum('b v, b v -> b', probs, neg_mask_f)

        denom = sum_pos_probs + sum_neg_probs + 1e-8
        contrastive_loss = -torch.log((sum_pos_probs + 1e-8) / denom).mean()

        # Dynamic Confidence Threshold Loss 
        dce_loss = torch.tensor(0.0, device=device, dtype=probs.dtype)
        if exists(self.eov_id):
            eov_probs = probs[:, self.eov_id] # (b,)
            eov_probs_grid = rearrange(eov_probs, 'b -> b 1') 

            pos_diff = F.relu(eov_probs_grid - probs)
            term1 = torch.einsum('b v, b v -> b', pos_diff, pos_mask_f)

            neg_diff = F.relu(probs - eov_probs_grid)
            term2 = torch.einsum('b v, b v -> b', neg_diff, neg_mask_f)

            num_neg = neg_mask_f.sum(dim=-1).clamp(min=1)
            term2_mean = term2 / num_neg

            dce_loss = (term1 + term2_mean).mean()

    
        loss = (self.mera_contrastive_weight * contrastive_loss) + (self.mera_diversity_weight * dce_loss)

        # # 7. Vectorized Predictions
        # predictions = []
        # if exists(self.eov_id):
        #     eov_probs_grid = rearrange(probs[:, self.eov_id], 'b -> b 1')
        #     pred_mask = probs > eov_probs_grid # (b, v) boolean grid
        #     pred_mask[:, self.eov_id] = False 
            
        #     # List comprehension is the fastest way to extract varying lengths from a 2D dense mask
        #     predictions = [m.nonzero(as_tuple=True)[0].tolist() for m in pred_mask]

        return MERAOutput(loss=loss, logits=next_token_logits)
    



# trainer

def custom_collate(batch):
    questions = [item[0] for item in batch]
    answers = [item[1] for item in batch]
    hard_negatives = [item[2] for item in batch]
    return questions, answers, hard_negatives


class MERATrainer(Module):
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

        self.model = MERA(
            model,
            tokenizer_encode = tokenizer_encode,
            **sdft_kwargs
        )

        self.optimizer = optim_klass(self.model.parameters(), lr = learning_rate, **optim_kwargs)

        self.dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, collate_fn=custom_collate)

        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )

        self.max_grad_norm = max_grad_norm

    def train(self, num_epochs=2): 
        self.model.train()

        for epoch in range(num_epochs):
       
            for questions, answers, hard_negatives in self.dataloader:
                with self.accelerator.accumulate(self.model):
                    output = self.model(questions, answers, hard_negatives)

                    self.accelerator.backward(output.loss)

                    if exists(self.max_grad_norm) and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    if self.accelerator.is_main_process:
                        print(f"Epoch {epoch} Loss: {output.loss.item()}")

                    self.optimizer.step()
                    self.optimizer.zero_grad()



        return output
