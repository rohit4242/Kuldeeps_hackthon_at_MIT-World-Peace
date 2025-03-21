import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import bitsandbytes as bnb

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model paths
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # We'll use Mistral's instruct model
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "maternal-health-mistral")

def load_model_and_tokenizer():
    """
    Load base model and tokenizer with quantization
    """
    # BitsAndBytes configuration for quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load base model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

def prepare_model_for_training(model):
    """
    Prepare model for LoRA fine-tuning
    """
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=16,               # Rank
        lora_alpha=32,      # Alpha parameter for LoRA scaling
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    return model

def train_model(model, tokenizer, train_dataset, eval_dataset):
    """
    Fine-tune the model using SFTTrainer
    """
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        max_steps=500,
        warmup_steps=50,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="tensorboard",
    )
    
    # Initialize SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=model.peft_config,
        dataset_text_field="messages",
        max_seq_length=2048,
    )
    
    # Start training
    trainer.train()
    
    # Save model
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    return trainer.model

def test_model(model, tokenizer):
    """
    Test the fine-tuned model on some examples
    """
    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    
    # Test queries
    test_queries = [
        "What are the early signs of preeclampsia?",
        "How can I deal with back pain during my third trimester?",
        "What should I know about postpartum depression?"
    ]
    
    # System prompt
    system_prompt = "You are a helpful assistant that provides accurate information about maternal health issues."
    
    for query in test_queries:
        # Format the prompt
        prompt = f"<s>[INST] {system_prompt}\n\n{query} [/INST]"
        
        # Generate response
        response = pipe(prompt)
        print(f"Query: {query}")
        print(f"Response: {response[0]['generated_text']}")
        print("=" * 50)

def fine_tune_model():
    """
    Main function to fine-tune the model
    """
    # Load datasets
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    train_dataset = load_from_disk(os.path.join(data_dir, "train"))
    eval_dataset = load_from_disk(os.path.join(data_dir, "validation"))
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Prepare model for training
    model = prepare_model_for_training(model)
    
    # Print model parameters
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("All parameters:", sum(p.numel() for p in model.parameters()))
    
    # Train model
    model = train_model(model, tokenizer, train_dataset, eval_dataset)
    
    # Test model
    test_model(model, tokenizer)
    
    return model, tokenizer

if __name__ == "__main__":
    fine_tune_model() 