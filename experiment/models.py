# models.py

import os
import json
import re
import requests
from typing import Dict
from dotenv import load_dotenv
from openai import OpenAI
# Load environment variables
load_dotenv()

def clean_response(text: str) -> str:
    """
    Clean and fix the JSON response.
    """
    # 1. Extract JSON content
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end+1]

    # 2. Handle condition array format
    def fix_condition(match):
        content = match.group(1)
        # Remove quotes, brackets and spaces
        content = content.strip()
        content = content.strip('"[]\'')
        return f'"condition": "{content}"'

    text = re.sub(
        r'"condition":\s*\[(.*?)\]',
        fix_condition,
        text,
        flags=re.DOTALL
    )

    # 3. Fix missing commas
    # Add commas between properties
    text = re.sub(r'"\s*"', '","', text)
    # Add commas between objects
    text = re.sub(r'}\s*"', '},"', text)
    text = re.sub(r']\s*"', '],"', text)
    # Fix missing commas between brackets
    text = re.sub(r'}\s*{', '},{', text)

    # 4. Remove extra commas
    text = re.sub(r',(\s*[}\]])', r'\1', text)

    return text

class LLM:
    def __init__(self, name: str, model_type: str):
        self.name = name
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif model_type == "ollama":
            self.url = "http://localhost:11434/api/generate"
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def generate_response(self, prompt: str) -> Dict:
        if self.model_type == "openai":
            return self._call_openai(prompt)
        elif self.model_type == "ollama":
            return self._call_ollama(prompt)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _call_openai(self, prompt: str) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model=self.name,
                messages=[
                    {"role": "system", "content": "You are an AI assistant who answers questions based on given context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.7
            )
            # Get the response content
            message_content = response.choices[0].message.content.strip()
            
            # Remove code block markers
            message_content = clean_response(message_content)
            
            # Parse JSON
            return json.loads(message_content)
        except json.JSONDecodeError as jde:
            print(f"JSON decoding error: {jde}")
            print(f"Original response content: {response.choices[0].message.content}")
            return {}
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return {}

    def _call_ollama(self, prompt: str) -> Dict:
        try:
            data = {
                "model": self.name,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(self.url, json=data)
            response.raise_for_status()
            response_json = response.json()
            message_content = response_json.get('response', '').strip()
            
            # Remove code block markers
            message_content = clean_response(message_content)
            
            # Parse JSON
            return json.loads(message_content)
        except json.JSONDecodeError as jde:
            print(f"JSON decoding error: {jde}")
            print(f"Original response content: {response_json.get('response', '')}")
            return {}
        except Exception as e:
            print(f"Ollama API error: {e}")
            return {}

def get_models() -> list:
    from config import MODELS_CONFIG
    return [
        LLM(config["name"], config["type"]) for config in MODELS_CONFIG
    ]