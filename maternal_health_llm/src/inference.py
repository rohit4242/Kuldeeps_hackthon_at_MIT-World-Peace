import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

def load_fine_tuned_model(model_path=None):
    """
    Load the fine-tuned model for inference
    """
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "maternal-health-mistral")
    
    # Load base model and tokenizer
    base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # For inference, we can load in 8-bit to save memory
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def create_inference_pipeline(model, tokenizer):
    """
    Create a text generation pipeline for inference
    """
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.15
    )
    
    return pipe

def generate_response(pipe, query, system_prompt="You are a helpful assistant that provides accurate information about maternal health issues."):
    """
    Generate a response to a maternal health query
    """
    # Format the prompt according to Mistral's chat template
    prompt = f"<s>[INST] {system_prompt}\n\n{query} [/INST]"
    
    # Generate response
    response = pipe(prompt)[0]['generated_text']
    
    # Extract the model's response part
    response = response.split('[/INST]')[-1].strip()
    
    return response

def main():
    """
    Main function for inference
    """
    # Load model and tokenizer
    model, tokenizer = load_fine_tuned_model()
    
    # Create inference pipeline
    pipe = create_inference_pipeline(model, tokenizer)
    
    # Example queries
    example_queries = [
        "What are the common causes of morning sickness?",
        "How should I prepare for labor?",
        "What are the warning signs of preterm labor?"
    ]
    
    # Generate and print responses
    for query in example_queries:
        response = generate_response(pipe, query)
        print(f"Query: {query}")
        print(f"Response: {response}")
        print("-" * 50)
    
    # Interactive mode
    print("\nMaternal Health Assistant - Interactive Mode")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            break
        
        response = generate_response(pipe, query)
        print(f"\nResponse: {response}")

if __name__ == "__main__":
    main() 