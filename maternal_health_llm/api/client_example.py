import requests
import argparse
import json
import sys

def query_maternal_health_api(api_url, query, system_prompt=None):
    """
    Send a query to the maternal health API
    """
    # Default system prompt if not provided
    if system_prompt is None:
        system_prompt = "You are a helpful assistant that provides accurate information about maternal health issues."
    
    # Prepare the data for the request
    data = {
        "text": query,
        "system_prompt": system_prompt
    }
    
    # Send the request to the API
    try:
        response = requests.post(api_url, json=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            return result["response"]
        else:
            print(f"Error: API request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return None

def interactive_mode(api_url):
    """
    Start an interactive session with the maternal health API
    """
    print("Maternal Health AI Assistant - Interactive Mode")
    print("Type 'exit' to quit, 'prompt' to change the system prompt")
    
    # Default system prompt
    system_prompt = "You are a helpful assistant that provides accurate information about maternal health issues."
    
    while True:
        query = input("\nYour question: ")
        
        # Check for exit command
        if query.lower() == "exit":
            print("Goodbye!")
            break
        
        # Check for prompt change command
        elif query.lower() == "prompt":
            system_prompt = input("Enter new system prompt: ")
            print(f"System prompt updated.")
            continue
        
        # Get response from API
        response = query_maternal_health_api(api_url, query, system_prompt)
        
        if response:
            print("\nAssistant:", response)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Client for Maternal Health LLM API")
    parser.add_argument("--api-url", required=True, help="URL of the deployed API")
    parser.add_argument("--query", help="Query to send to the API")
    parser.add_argument("--system-prompt", help="Custom system prompt")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    
    args = parser.parse_args()
    
    # Check if we should run in interactive mode
    if args.interactive:
        interactive_mode(args.api_url)
    elif args.query:
        # Process a single query
        response = query_maternal_health_api(args.api_url, args.query, args.system_prompt)
        if response:
            print(response)
    else:
        print("Error: Please provide a query or use interactive mode.")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 