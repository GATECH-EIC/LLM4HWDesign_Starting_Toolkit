from datasets import load_dataset
import json

# Load the dataset
dataset = load_dataset('GaTech-EIC/MG-Verilog')

# Function to rename keys and save to JSONL
def save_to_jsonl(split, split_name):
    jsonl_file_path = 'dataset.jsonl'
    with open(jsonl_file_path, 'a') as jsonl_file:
        for example in split:
            # Rename keys
            current_sample_dict = example['description']
            current_sample = ', '.join(f'{k}: {v}' for k, v in current_sample_dict.items())
            new_example = {
                'input': current_sample,
                'output': example['code']
            }
            jsonl_file.write(json.dumps(new_example) + '\n')
    print(f"Dataset split '{split_name}' saved to {jsonl_file_path}")

# Save each split
for split_name in dataset:
    save_to_jsonl(dataset[split_name], split_name)
