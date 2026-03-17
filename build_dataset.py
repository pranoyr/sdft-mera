import os
import json
from pathlib import Path

def build_training_dataset(root_dir, output_file):
    root_path = Path(root_dir)
    raw_patient_data = []
    
    for folder in root_path.iterdir():
        if folder.is_dir():
            base_name = folder.name
            txt_path = folder / f"{base_name}.txt"
            json_path = folder / f"{base_name}.json"
            
            if txt_path.exists() and json_path.exists():
                with open(txt_path, 'r', encoding='utf-8') as f:
                    medical_chart = f.read().strip()
                
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    
                raw_diagnoses = json_data.get("diagnoses", [])
                formatted_answers = [f"<ICD_{code}>" for code in raw_diagnoses]
                
                raw_patient_data.append({
                    "question": medical_chart,
                    "answer": formatted_answers
                })
            else:
                print(f"Skipping {folder.name}: Missing .txt or .json file.")

    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(raw_patient_data, out_f, indent=4)
        
    print(f"\nSuccess! Processed {len(raw_patient_data)} patient records.")
    print(f"Saved dataset to: {output_file}")

if __name__ == "__main__":
    INPUT_DIRECTORY = "/home/ubuntu/efs/omega_data_copy_final/MED (Client C)/DATA_RETRAINING_FORMATTED/VALID" 
    OUTPUT_FILE = "training_dataset.json"      
    build_training_dataset(INPUT_DIRECTORY, OUTPUT_FILE)