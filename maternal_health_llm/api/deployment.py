import os
import sys
import modal

# Add parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.app import stub, MaternalHealthAPI

if __name__ == "__main__":
    """
    Deploy the Maternal Health LLM API to Modal
    
    To deploy:
    1. Make sure you have the Modal CLI installed: pip install modal
    2. Login to Modal: modal token new
    3. Run: python deployment.py
    
    This will deploy the API to Modal and provide you with a URL to access it.
    """
    # Deploy to Modal
    with stub.run():
        api_url = MaternalHealthAPI.generate.url
        print(f"Successfully deployed Maternal Health LLM API to Modal!")
        print(f"API URL: {api_url}")
        print("\nExample curl request:")
        print(f'''curl -X POST "{api_url}" \\
        -H "Content-Type: application/json" \\
        -d '{{"text": "What are the early signs of preeclampsia?", "system_prompt": "You are a helpful assistant that provides accurate information about maternal health issues."}}'
        ''')
        
        # Call the health check endpoint
        print("\nTesting health check...")
        health_response = modal.Function.aio_call(MaternalHealthAPI.health_check)
        print(f"Health check response: {health_response}")
        
        print("\nTesting inference...")
        test_query = "What are the early signs of preeclampsia?"
        system_prompt = "You are a helpful assistant that provides accurate information about maternal health issues."
        response = MaternalHealthAPI().predict(test_query, system_prompt)
        print(f"Query: {test_query}")
        print(f"Response: {response}") 