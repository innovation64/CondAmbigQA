# models.py

import os
import json
import re
import requests
from typing import Dict
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

def clean_response(text: str) -> str:
    """
    移除响应中的代码块标记（如 ```json 和 ```）。
    """
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```', '', text, flags=re.MULTILINE)
    text = text.strip()
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
            # 获取响应内容
            message_content = response.choices[0].message.content.strip()
            
            # 移除代码块标记
            message_content = clean_response(message_content)
            
            # 解析 JSON
            return json.loads(message_content)
        except json.JSONDecodeError as jde:
            print(f"JSON 解码错误: {jde}")
            print(f"原始响应内容: {response.choices[0].message.content}")
            return {}
        except Exception as e:
            print(f"OpenAI API 错误: {e}")
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
            
            # 移除代码块标记
            message_content = clean_response(message_content)
            
            # 解析 JSON
            return json.loads(message_content)
        except json.JSONDecodeError as jde:
            print(f"JSON 解码错误: {jde}")
            print(f"原始响应内容: {response_json.get('response', '')}")
            return {}
        except Exception as e:
            print(f"Ollama API 错误: {e}")
            return {}

def get_models() -> list:
    from config import MODELS_CONFIG
    return [
        LLM(config["name"], config["type"]) for config in MODELS_CONFIG
    ]