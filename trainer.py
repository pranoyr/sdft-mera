
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
from x_transformers import TransformerWrapper, Decoder 
from sdft_pytorch import SDFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

class DummyScienceDataset(Dataset):
    def __init__(self, num_samples=100):
        """
        Creates an in-memory dataset with dummy questions and expert reasoning.
        We multiply a few base examples to simulate a larger dataset for testing.
        """
        base_data = [
            {
                "question": "If you drop a solid iron ball and a solid wooden ball of the same size from a building at the same time, which hits the ground first? (Ignore air resistance)", 
                "answer": "According to Galileo's principle of equivalence, objects in a vacuum fall at the same rate regardless of their mass. Since we are ignoring air resistance, both the iron ball and the wooden ball will experience the same gravitational acceleration (9.8 m/s^2 on Earth). Therefore, they will hit the ground at the exact same time."
            },
            {
                "question": "Is pure water (H2O) considered an element or a compound?", 
                "answer": "An element is a pure substance consisting of only one type of atom. A compound is a substance formed when two or more chemical elements are chemically bonded together. Water is made of two Hydrogen atoms and one Oxygen atom bonded together. Therefore, water is a compound."
            },
            {
                "question": "Why does ice float on liquid water?", 
                "answer": "For most substances, the solid phase is denser than the liquid phase. However, when water freezes, its molecules form a crystalline structure held together by hydrogen bonds, which spaces the molecules further apart. Because the molecules are further apart, ice is less dense than liquid water, causing it to float."
            }
        ]
        
        self.samples = []
        for i in range(num_samples):
            self.samples.append(base_data[i % len(base_data)])
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item = self.samples[idx]
        return item['question'], item['answer']


train_dataset = DummyScienceDataset(num_samples=100)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def encode_prompt_to_tensor(prompt_string: str) -> torch.Tensor:
    """
    Takes a string and returns a 1D tensor of token IDs.
    """
    token_dict = tokenizer(prompt_string, return_tensors="pt", add_special_tokens=False)
    return token_dict['input_ids'].squeeze(0)


model_name = "Qwen/Qwen2.5-7B-Instruct"

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
trainer = SDFTTrainer(
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
