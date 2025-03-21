import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from modal import Image, Stub, method, web_endpoint, Secret, Mount
import torch

# Add parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import load_fine_tuned_model, create_inference_pipeline, generate_response

# Define the Modal app
stub = Stub("maternal-health-llm")

# Create Docker image with required dependencies
image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.0",
        "transformers==4.36.2",
        "peft==0.7.0",
        "accelerate==0.25.0",
        "bitsandbytes==0.41.1",
        "fastapi==0.104.1",
        "uvicorn==0.24.0"
    )
    .run_commands("pip install --no-deps safetensors")
)

# Define request and response models
class Query(BaseModel):
    text: str
    system_prompt: str = "You are a helpful assistant that provides accurate information about maternal health issues."

class Response(BaseModel):
    response: str

# Define the API class
@stub.cls(
    image=image,
    gpu="T4",
    timeout=600,
    secrets=[Secret.from_name("huggingface-token")],
    mounts=[Mount.from_local_dir("../models", remote_path="/root/models")]
)
class MaternalHealthAPI:
    def __enter__(self):
        """
        Load the model when the container starts
        """
        # Set environment variables for Hugging Face
        os.environ["HUGGINGFACE_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")
        
        # Load model and tokenizer
        self.model, self.tokenizer = load_fine_tuned_model("/root/models/maternal-health-mistral")
        
        # Create inference pipeline
        self.pipe = create_inference_pipeline(self.model, self.tokenizer)
        
        print("Model loaded successfully!")
    
    @method()
    def predict(self, query: str, system_prompt: str) -> str:
        """
        Generate a response to a maternal health query
        """
        try:
            response = generate_response(self.pipe, query, system_prompt)
            return response
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
    
    @web_endpoint(method="POST")
    def generate(self, query: Query) -> Response:
        """
        Web endpoint for generating responses
        """
        response = self.predict(query.text, query.system_prompt)
        return Response(response=response)

# Create a FastAPI app for local development
app = FastAPI(title="Maternal Health LLM API")

@app.post("/generate", response_model=Response)
async def generate_endpoint(query: Query):
    """
    Endpoint for generating responses (for local development only)
    """
    # When developing locally, you would need to implement this
    # In production, Modal will handle routing to the web_endpoint
    raise HTTPException(status_code=501, detail="Not implemented for local development. Deploy to Modal for production use.")

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 