import json
import os
import random
from datasets import Dataset

def load_data(json_file_path):
    """
    Load maternal health QA data from JSON file
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def format_for_training(data, system_prompt="You are a helpful assistant that provides accurate information about maternal health issues."):
    """
    Format the data for fine-tuning
    """
    formatted_data = []
    
    for item in data:
        formatted_item = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["output"]}
            ]
        }
        formatted_data.append(formatted_item)
    
    return formatted_data

def split_data(data, train_ratio=0.8, seed=42):
    """
    Split data into training and validation sets
    """
    random.seed(seed)
    random.shuffle(data)
    
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data

def create_hf_dataset(data):
    """
    Convert data to HuggingFace Dataset format
    """
    return Dataset.from_list(data)

def save_datasets(train_dataset, val_dataset, output_dir):
    """
    Save datasets to disk
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_dataset.save_to_disk(os.path.join(output_dir, "train"))
    val_dataset.save_to_disk(os.path.join(output_dir, "validation"))
    
    print(f"Datasets saved to {output_dir}")

def prepare_data():
    """
    Main function to prepare the data
    """
    # Load data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "maternal_health_qa.json")
    data = load_data(data_path)
    
    # Format data for training
    formatted_data = format_for_training(data)
    
    # Split data
    train_data, val_data = split_data(formatted_data)
    
    # Create HuggingFace datasets
    train_dataset = create_hf_dataset(train_data)
    val_dataset = create_hf_dataset(val_data)
    
    # Save datasets
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    save_datasets(train_dataset, val_dataset, output_dir)
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    prepare_data() 