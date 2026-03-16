
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
from x_transformers import TransformerWrapper, Decoder 
from sdft_pytorch.sdft_mera import  SDFTMERATrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import Dataset
import torch

class DummyICD10ContrastiveDataset(Dataset):
    def __init__(self, num_samples=100):

        base_data = [
            {
                "question": "Visit 1: <ICD_E10.65> (Type 1 diabetes with hyperglycemia). Patient presents today with uncontrolled blood sugar despite insulin adherence.", 
                "answer": "<ICD_E10.65> <EOV>", 
                "hard_negatives": [
                    "<ICD_E11.65>",
                    "<ICD_E10.9>"   
                ]
            },
            {
                "question": "Visit 1: <ICD_J45.909> (Unspecified asthma). Visit 2: <ICD_J45.901> (Asthma with acute exacerbation). Patient presents for routine follow-up, breathing is normal today.", 
                "answer": "<ICD_J45.909> <EOV>", 
                "hard_negatives": [
                    "<ICD_J45.901>", 
                    "<ICD_J44.9>"   
                ]
            },
            {
                "question": "Visit 1: <ICD_I10> (Essential hypertension). Blood pressure today is 145/90. No secondary causes identified. Continuing Lisinopril.", 
                "answer": "<ICD_I10> <EOV>", 
                "hard_negatives": [
                    "<ICD_I15.9>",
                    "<ICD_I11.9>"  
                ]
            }
        ]
        
        self.samples = []
        for i in range(num_samples):
            self.samples.append(base_data[i % len(base_data)])
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item = self.samples[idx]
        return item['question'], item['answer'], item['hard_negatives']



train_dataset = DummyICD10ContrastiveDataset(num_samples=100)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def encode_prompt_to_tensor(prompt_string: str) -> torch.Tensor:
    """
    Takes a string and returns a 1D tensor of token IDs.
    """
    token_dict = tokenizer(prompt_string, return_tensors="pt", add_special_tokens=False)
    return token_dict['input_ids'].squeeze(0)



# sampel ICD codes 
all_icd_codes = [
    "E10.65", "E11.65", "E10.9", 
    "J45.909", "J45.901", "J44.9", 
    "I10", "I15.9", "I11.9"
]

icd_special_tokens = [f"<ICD_{code}>" for code in all_icd_codes]

special_tokens_dict = {'additional_special_tokens': ['<EOV>'] + icd_special_tokens}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)


config = AutoConfig.from_pretrained(model_name)
config.attention_dropout = 0.0
config.hidden_dropout = 0.0

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="auto"          
)

base_model.resize_token_embeddings(len(tokenizer))

eov_token_id = tokenizer.convert_tokens_to_ids('<EOV>')
icd_token_ids = set(tokenizer.convert_tokens_to_ids(icd_special_tokens))



# sdft only

trainer = SDFTMERATrainer(
    model = base_model,
    dataset = train_dataset,
    tokenizer_encode = encode_prompt_to_tensor, 
    batch_size = 2,
    learning_rate = 2e-5,
    sdft_kwargs = {
        "student_max_response_len": 128,  # Max tokens the model will generate per step
        "eos_id": tokenizer.eos_token_id,           # Stops loss calculation after this token
        "num_init_student_response_tokens_mask": 0 ,
        "eov_id": eov_token_id,         
        "icd_vocab_ids": icd_token_ids,
        "sdft_loss_kl_weight": 1.0,
        "training_stage": "sdft"
    }
)


# mera only

trainer = SDFTMERATrainer(
    model = base_model,
    dataset = train_dataset,
    tokenizer_encode = encode_prompt_to_tensor, 
    batch_size = 2,
    learning_rate = 2e-5,
    sdft_kwargs = {
        "student_max_response_len": 128,  # Max tokens the model will generate per step
        "eos_id": tokenizer.eos_token_id,           # Stops loss calculation after this token
        "num_init_student_response_tokens_mask": 0 ,
        "eov_id": eov_token_id,         
        "icd_vocab_ids": icd_token_ids,
        "mera_contrastive_weight": 1.0,
        "mera_diversity_weight": 0.2,
        "training_stage": "mera"
    }
)


# sdft + mera

trainer = SDFTMERATrainer(
    model = base_model,
    dataset = train_dataset,
    tokenizer_encode = encode_prompt_to_tensor, 
    batch_size = 2,
    learning_rate = 2e-5,
    sdft_kwargs = {
        "student_max_response_len": 128,  # Max tokens the model will generate per step
        "eos_id": tokenizer.eos_token_id,           # Stops loss calculation after this token
        "num_init_student_response_tokens_mask": 0 ,
        "eov_id": eov_token_id,         
        "icd_vocab_ids": icd_token_ids,
        "mera_contrastive_weight": 0.1,
        "mera_diversity_weight": 0.05,
        "sdft_loss_kl_weight": 1,
        "training_stage": "sdft+mera"
    }
)



# fine tune
print("Starting SDFT Training...")
trainer.train(num_epochs=2)  
