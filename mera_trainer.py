
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
from x_transformers import TransformerWrapper, Decoder 
from sdft_pytorch import MERATrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import Dataset
import torch
from negative_mining import extract_icd_tokens, build_hard_negative_lookup
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


with open('icd10cm.json', 'r') as f:
    icd_data = json.load(f)

icd_special_tokens = extract_icd_tokens(icd_data, format_as_special_token=True)
negatives_lookup = build_hard_negative_lookup(icd_data, format_as_special_token=True)

special_tokens_dict = {'additional_special_tokens': ['<EOV>'] + icd_special_tokens}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

eov_token_id = tokenizer.convert_tokens_to_ids('<EOV>')
icd_token_ids = set(tokenizer.convert_tokens_to_ids(icd_special_tokens))




# setting up the data
class DynamicICD10ContrastiveDataset(Dataset):
    def __init__(self, patient_records, negatives_lookup, max_negatives=10):
        self.samples = patient_records
        self.negatives_lookup = negatives_lookup
        self.max_negatives = max_negatives
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        question = item['question']
    
        true_answers = item['answer'] 
        
        # mining
        raw_negatives = []
        for ans in true_answers:
            raw_negatives.extend(self.negatives_lookup.get(ans, []))
            
        # avoid counter neg
        safe_negatives = list(set(raw_negatives) - set(true_answers))
        
        random.shuffle(safe_negatives)
        safe_negatives = safe_negatives[:self.max_negatives]
        
        final_answers = true_answers + ["<EOV>"]
        
        return question, final_answers, safe_negatives



training_data = json.load(open("training_dataset.json", 'r'))
# raw_patient_data = [
#     {
#         "question": "Visit 1: <ICD_E10.65>... Patient presents today...", 
#         "answer": ["<ICD_E10.65>"]
#     },
#     {
#         "question": "Visit 1: <ICD_I10>... Blood pressure today is 145/90...", 
#         "answer": ["<ICD_I10>", "<ICD_I15.9>"]
#     }
# ]


# simulated_records = [raw_patient_data[i % len(raw_patient_data)] for i in range(100)]

train_dataset = DynamicICD10ContrastiveDataset(
    patient_records=training_data,
    negatives_lookup=negatives_lookup,
    max_negatives=12 
)





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
