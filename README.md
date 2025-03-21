# Maternal Health Chatbot API

A Next.js integration for a maternal health chatbot API that provides expert answers to maternal health questions.

## Overview

This repository contains a Next.js integration for the Maternal Health Chatbot API. The API offers both standard and streaming endpoints to provide accurate, informative responses to questions about maternal health.

## Features

- **Two API Endpoints**: Standard response and streaming response options
- **Simple Integration**: Easy to implement in Next.js applications
- **Responsive Design**: Mobile-friendly UI components included
- **Token-Optimized**: Works within model token limitations

## API Endpoints

- **Standard API**: `https://rohit4242-maternal-health-chatbot--maternal-health-chatb-3fac64.modal.run`
- **Streaming API**: `https://rohit4242-maternal-health-chatbot--maternal-health-chatb-3fac64.modal.run/maternal_health_stream`

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/maternal-health-chatbot.git
cd maternal-health-chatbot
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

## Usage

### API Request Format

Both endpoints accept a POST request with a JSON body:

```json
{
  "question": "Your maternal health question here"
}
```

### Response Format

**Standard API Response:**
```json
{
  "response": "The answer to your maternal health question..."
}
```

**Streaming API Response:**
Server-sent events with the following format:
```
data: {"chunk": "partial response text"}
```

## Limitations

- Maximum input length is 256 tokens (including system prompt and user question)
- Keep questions concise for optimal results
- Response time may vary based on question complexity

## Implementation Examples

### Standard API Component

```jsx
// components/MaternalHealthChat.jsx
'use client';

import { useState } from 'react';

export default function MaternalHealthChat() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  
  const API_ENDPOINT = "https://rohit4242-maternal-health-chatbot--maternal-health-chatb-3fac64.modal.run";
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });
      
      const data = await response.json();
      setAnswer(data.response);
    } catch (error) {
      console.error('Failed to fetch answer:', error);
      setAnswer('Sorry, there was an error processing your question.');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Maternal Health Chatbot</h1>
      
      <form onSubmit={handleSubmit} className="mb-4">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a maternal health question..."
          className="w-full p-2 border rounded mb-2"
        />
        <button 
          type="submit" 
          disabled={loading || !question}
          className="px-4 py-2 bg-blue-500 text-white rounded"
        >
          {loading ? 'Asking...' : 'Ask'}
        </button>
      </form>
      
      {answer && (
        <div className="border p-4 rounded bg-gray-50">
          <h2 className="font-bold">Answer:</h2>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
}
```

### Streaming API Component

```jsx
// components/StreamingChat.jsx
'use client';

import { useState } from 'react';

export default function StreamingMaternalHealthChat() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  
  const STREAMING_ENDPOINT = "https://rohit4242-maternal-health-chatbot--maternal-health-chatb-3fac64.modal.run/maternal_health_stream";
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setAnswer('');
    setIsStreaming(true);
    
    try {
      const response = await fetch(STREAMING_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify({ question }),
      });
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n\n');
        
        for (const line of lines) {
          if (line.startsWith('data:')) {
            try {
              const jsonStr = line.substring(5);
              const jsonData = JSON.parse(jsonStr);
              
              if (jsonData.chunk) {
                setAnswer(prev => prev + jsonData.chunk);
              }
            } catch (e) {
              console.error('Failed to parse SSE data', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Streaming error:', error);
    } finally {
      setIsStreaming(false);
    }
  };
  
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Maternal Health Chatbot (Streaming)</h1>
      
      <form onSubmit={handleSubmit} className="mb-4">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a maternal health question..."
          className="w-full p-2 border rounded mb-2"
        />
        <button 
          type="submit" 
          disabled={isStreaming || !question}
          className="px-4 py-2 bg-blue-500 text-white rounded"
        >
          {isStreaming ? 'Streaming...' : 'Ask'}
        </button>
      </form>
      
      {answer && (
        <div className="border p-4 rounded bg-gray-50">
          <h2 className="font-bold">Answer:</h2>
          <p className="whitespace-pre-wrap">{answer}</p>
        </div>
      )}
    </div>
  );
}
```

## Example Questions

- "What is the best way to get pregnant?"
- "What vitamins should I take during pregnancy?"
- "How can I relieve morning sickness?"
- "What foods should I avoid during pregnancy?"
- "When should I start prenatal care?"

## License

MIT

## Disclaimer

This API provides general information about maternal health topics. Always consult with healthcare professionals for personalized medical advice. 