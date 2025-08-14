# config.py

MODELS_CONFIG = [
    {
        "name": "glm-4-plus",
        "type": "openai",
    },
    {
        "name": "gpt-4o",
        "type": "openai",
    },
    # {
    #     "name": "llama3.1",
    #     "type": "ollama",
    # },
    # {
    #     "name": "qwen2.5",
    #     "type": "ollama",
    # },
    {
        "name": "mistral",
        "type": "ollama",
    },
    #     {
    #     "name": "glm4",
    #     "type": "ollama",
    # },
    {
        "name": "gemma2",
        "type": "ollama",  
    },
    {
        "name": "deepseek-r1:7b",  # Changed from llama3.1 to llama3.1:8b
        "type": "ollama",
    },
    {
        "name": "llama3.1:8b",  # Changed from llama3.1 to llama3.1:8b
        "type": "ollama",
    },
    # {
    #     "name": "mistral:latest",  # Add :latest to match your installation
    #     "type": "ollama",
    # },
    {
        "name": "qwen2.5:latest",  # Add :latest to match your installation
        "type": "ollama",
    },
    # {
    #     "name": "gemma2:latest",  # Add :latest to match your installation
    #     "type": "ollama",
    # },
    {
        "name": "glm4:latest",  # Add :latest to match your installation
        "type": "ollama",
    },
    #you can add more models in here
]