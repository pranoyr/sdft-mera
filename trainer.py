
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
                "question": "Visit 1: E10.65 (Type 1 diabetes with hyperglycemia). Patient presents today with uncontrolled blood sugar despite insulin adherence.", 
                "answer": "E10.65 <EOV>", 
                "hard_negatives": [
                    "E11.65",
                    "E10.9"   
                ]
            },
            {
                "question": "Visit 1: J45.909 (Unspecified asthma). Visit 2: J45.901 (Asthma with acute exacerbation). Patient presents for routine follow-up, breathing is normal today.", 
                "answer": "J45.909 <EOV>", 
                "hard_negatives": [
                    "J45.901", 
                    "J44.9"   
                ]
            },
            {
                "question": "Visit 1: I10 (Essential hypertension). Blood pressure today is 145/90. No secondary causes identified. Continuing Lisinopril.", 
                "answer": "I10 <EOV>", 
                "hard_negatives": [
                    "I15.9",
                    "I11.9"  
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




config = AutoConfig.from_pretrained(model_name)
config.attention_dropout = 0.0
config.hidden_dropout = 0.0

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="auto"          
)

# trainer
trainer = SDFTMERATrainer(
    model = base_model,
    dataset = train_dataset,
    tokenizer_encode = encode_prompt_to_tensor, 
    batch_size = 2,
    learning_rate = 2e-5,
    sdft_kwargs = {
        "student_max_response_len": 128,  # Max tokens the model will generate per step
        "eos_id": tokenizer.eos_token_id,           # Stops loss calculation after this token
        "num_init_student_response_tokens_mask": 4 
    }
)


# fine tune
print("Starting SDFT Training...")
trainer.train(num_epochs=2)  
