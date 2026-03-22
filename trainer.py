
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
from x_transformers import TransformerWrapper, Decoder 
from sdft_pytorch import MERATrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import Dataset
import torch
import random
import json



# setup tokenizer 
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

def encode_prompt_to_tensor(prompt_string: str) -> torch.Tensor:
    """
    Takes a string and returns a 1D tensor of token IDs.
    """
    token_dict = tokenizer(prompt_string,
    return_tensors="pt",
    padding = True,
    add_special_tokens=False
    )
    return token_dict


icd_special_tokens = ["<ICD_1>", "<ICD_2>", "<ICD_3>", "<ICD_4>", "<ICD_5>", "<ICD_6>", "<ICD_7>", "<ICD_8>", "<ICD_9>", "<ICD_10>"]

special_tokens_dict = {'additional_special_tokens': ['<EOV>'] + icd_special_tokens}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# get the tokens ids
eov_token_id = tokenizer.convert_tokens_to_ids('<EOV>')
icd_token_ids = set(tokenizer.convert_tokens_to_ids(icd_special_tokens))


# setting up model
config = AutoConfig.from_pretrained(model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
)
base_model.resize_token_embeddings(len(tokenizer))


# trainer 
trainer = MERATrainer(
    model = base_model,
    dataset = train_dataset,
    tokenizer_encode = encode_prompt_to_tensor, 
    batch_size = 2,
    learning_rate = 2e-5,
    sdft_kwargs = {
        "eov_id": eov_token_id,         
        "icd_vocab_ids": icd_token_ids,
    }
)

print("Starting SDFT Training...")
trainer.train(num_epochs=2)  
