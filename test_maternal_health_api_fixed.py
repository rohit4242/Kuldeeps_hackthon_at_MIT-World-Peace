#!/usr/bin/env python
"""
Test script for the Maternal Health Chatbot API

This script sends test requests to the deployed API endpoint and displays the response.
It's designed to work with the token length limitations of the deployed model.
It can test both the standard API and the streaming API.
"""

import requests
import json
import sys
import argparse
import time
import sseclient  # for streaming support

# Replace with the actual API endpoint from your deployment
API_ENDPOINT = "https://rohit4242-maternal-health-chatbot--maternal-health-chatb-3fac64.modal.run"
STREAMING_ENDPOINT = f"{API_ENDPOINT}/maternal_health_stream"

# Debug flag to print additional information
DEBUG = True

# Sample maternal health questions for testing (intentionally kept short due to token limits)
DEFAULT_QUESTIONS = [
    "What is the best way to get pregnant?",
]

def send_question(question):
    """Send a single question to the API and return the response"""
    # Check if question is likely to exceed token limits (rough estimate)
    if len(question) > 100:
        print(f"\033[93m⚠️ Warning: Your question is {len(question)} characters long and might exceed token limits.\033[0m")
        print("The model has a maximum input length of 256 tokens, which includes both the system prompt and your question.")
        print("Consider asking a shorter, more concise question.")
        proceed = input("Do you want to proceed anyway? (y/n): ")
        if proceed.lower() != 'y':
            return None
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "question": question
    }
    
    print(f"\n\033[1mSending question to API:\033[0m \"{question}\"")
    print(f"API Endpoint: {API_ENDPOINT}")
    print("Waiting for response...")
    
    if DEBUG:
        print(f"Request headers: {headers}")
        print(f"Request data: {json.dumps(data, indent=2)}")
    
    try:
        start_time = time.time()
        response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"\033[92m✓ Response received in {elapsed_time:.2f} seconds\033[0m")
            print("\n\033[1mAnswer:\033[0m")
            print(result["response"])
            return result
        else:
            print(f"\033[91m✗ Error: Received status code {response.status_code}\033[0m")
            print(response.text)
            if DEBUG:
                print(f"Response headers: {dict(response.headers)}")
            if response.status_code == 500 and "Maximum input length" in response.text:
                print("\n\033[93m⚠️ Token limit exceeded!\033[0m")
                print("Your question combined with the system prompt is too long for the model.")
                print("Please try a shorter question.")
            return None
    except Exception as e:
        print(f"\033[91m✗ Error: {str(e)}\033[0m")
        if DEBUG:
            import traceback
            traceback.print_exc()
        return None

def send_streaming_question(question):
    """Send a question to the streaming API and display the response as it arrives"""
    # Check if question is likely to exceed token limits (rough estimate)
    if len(question) > 100:
        print(f"\033[93m⚠️ Warning: Your question is {len(question)} characters long and might exceed token limits.\033[0m")
        print("The model has a maximum input length of 256 tokens, which includes both the system prompt and your question.")
        print("Consider asking a shorter, more concise question.")
        proceed = input("Do you want to proceed anyway? (y/n): ")
        if proceed.lower() != 'y':
            return None
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    data = {
        "question": question
    }
    
    print(f"\n\033[1mSending streaming question to API:\033[0m \"{question}\"")
    print(f"Streaming API Endpoint: {STREAMING_ENDPOINT}")
    print("Waiting for response stream...\n")
    
    if DEBUG:
        print(f"Request headers: {headers}")
        print(f"Request data: {json.dumps(data, indent=2)}")
    
    try:
        # For SSE, we use a streaming request
        start_time = time.time()
        response = requests.post(
            STREAMING_ENDPOINT,
            headers=headers,
            data=json.dumps(data),
            stream=True
        )
        
        if response.status_code != 200:
            print(f"\033[91m✗ Error: Received status code {response.status_code}\033[0m")
            print(response.text)
            return None
            
        # Set up SSE client
        client = sseclient.SSEClient(response)
        full_response = ""
        
        print("\033[1mStreaming response:\033[0m")
        
        # Process the stream
        for event in client.events():
            try:
                data = json.loads(event.data)
                chunk = data.get('chunk', '')
                # Print without newline to simulate streaming
                print(chunk, end='', flush=True)
                full_response += chunk
            except json.JSONDecodeError:
                if DEBUG:
                    print(f"\nFailed to parse JSON: {event.data}")
                continue
        
        elapsed_time = time.time() - start_time
        print(f"\n\n\033[92m✓ Stream completed in {elapsed_time:.2f} seconds\033[0m")
        print(f"Total response length: {len(full_response)} characters")
        
        return full_response
    except Exception as e:
        print(f"\033[91m✗ Error: {str(e)}\033[0m")
        if DEBUG:
            import traceback
            traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Test the Maternal Health Chatbot API")
    parser.add_argument("--question", "-q", type=str, help="A specific maternal health question to ask")
    parser.add_argument("--all", "-a", action="store_true", help="Test all default questions")
    parser.add_argument("--stream", "-s", action="store_true", help="Use the streaming API endpoint")
    args = parser.parse_args()
    
    print("\033[93mℹ️ Note about token limits:\033[0m")
    print("This model has a maximum input length of 256 tokens, which includes both the system prompt and your question.")
    print("The system prompt already uses a significant portion of these tokens.")
    print("For best results, keep your questions brief and to the point.\n")
    
    # Determine which API function to use
    api_func = send_streaming_question if args.stream else send_question
    
    if args.question:
        api_func(args.question)
    elif args.all:
        print(f"\033[1mTesting {len(DEFAULT_QUESTIONS)} default questions...\033[0m")
        for i, question in enumerate(DEFAULT_QUESTIONS, 1):
            print(f"\n\033[1mQuestion {i}/{len(DEFAULT_QUESTIONS)}\033[0m")
            api_func(question)
            
            # Add a small delay between requests to avoid rate limiting
            if i < len(DEFAULT_QUESTIONS):
                print("\nWaiting 2 seconds before next request...")
                time.sleep(2)
    else:
        # Use the first default question if no arguments are provided
        api_func(DEFAULT_QUESTIONS[0])
        print("\nTip: Use --question/-q to ask a specific question, --all/-a to test all default questions, or --stream/-s to test the streaming API")


if __name__ == "__main__":
    print("\033[1m=== Maternal Health Chatbot API Test ===\033[0m")
    main()
