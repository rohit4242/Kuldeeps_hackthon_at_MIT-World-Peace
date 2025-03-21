# Maternal Health LLM

A specialized AI assistant that provides accurate information about maternal health concerns using a fine-tuned Mistral-7B model.

## Overview

This project creates an AI assistant specialized in maternal health using the Mistral-7B LLM fine-tuned on maternal health Q&A data. The assistant can answer questions about pregnancy, postpartum care, and other maternal health concerns with accurate, helpful information.

## Features

- Fine-tuned Mistral-7B model for maternal health Q&A
- Parameter-efficient fine-tuning using LoRA
- REST API for easy integration with applications
- Deployment on Modal.com for scalable inference

## Project Structure

```
maternal_health_llm/
├── api/                # API and deployment code
│   ├── app.py         # FastAPI application
│   └── deployment.py  # Modal deployment script
├── data/               # Data for fine-tuning
│   └── maternal_health_qa.json  # Q&A dataset
├── models/             # Fine-tuned model storage
├── src/                # Core code
│   ├── data_preparation.py  # Data preprocessing
│   ├── model_fine_tuning.py # Model fine-tuning
│   └── inference.py    # Model inference
└── requirements.txt    # Project dependencies
```

## Setup and Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Fine-tuning the Model

1. Prepare the data:
   ```bash
   python src/data_preparation.py
   ```

2. Fine-tune the model:
   ```bash
   python src/model_fine_tuning.py
   ```
   Note: Fine-tuning requires a GPU with at least 16GB VRAM.

## Local Inference

Run the inference script to test the model locally:
```bash
python src/inference.py
```

## Deployment to Modal.com

1. Install Modal CLI:
   ```bash
   pip install modal
   ```

2. Setup Modal authentication:
   ```bash
   modal token new
   ```

3. Deploy the API:
   ```bash
   python api/deployment.py
   ```

4. After deployment, you'll receive a URL to access the API.

## API Usage

Send requests to the API endpoint with the following format:

```bash
curl -X POST "https://your-modal-api-url" \
  -H "Content-Type: application/json" \
  -d '{"text": "What are the early signs of preeclampsia?", "system_prompt": "You are a helpful assistant that provides accurate information about maternal health issues."}'
```

## System Prompt Customization

You can customize the system prompt to control the assistant's behavior:

```json
{
  "text": "What should I know about gestational diabetes?",
  "system_prompt": "You are a helpful assistant that provides accurate information about maternal health issues. Always emphasize the importance of consulting healthcare providers for medical advice."
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the Mistral-7B model by Mistral AI
- Built with HuggingFace Transformers and PEFT
- Deployed using Modal 