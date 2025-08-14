"""
Configuration settings for the AI-AI annotation system
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o"
FALLBACK_MODEL = "gpt-4o-mini"  # Fallback model if the primary one is unavailable

# Processing settings
MAX_EXECUTION_TIME = 300  # Maximum execution time per conversation (5 minutes)
API_TIMEOUT = 60  # Maximum time to wait for API response (1 minute)

# System roles
ROLES = {
    "annotator": "You are an expert annotator who specializes in creating comprehensive condition-answer pairs for ambiguous questions, providing rich context, constraints, and detailed answers.",
    "reviewer": "You are an expert reviewer who evaluates the quality of comprehensive condition-answer pairs, focusing on condition comprehensiveness, answer completeness, and citation relevance.",
    "facilitator": "You are a conversation facilitator who coordinates the annotation process and ensures productive dialogue between annotator and reviewer to create high-quality condition-answer pairs."
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    "condition_comprehensiveness": 0.8,  # Evaluate condition completeness and richness
    "condition_utility": 0.8,
    "answer_completeness": 0.8,  # Evaluate answer completeness and detail level
    "condition_citation_relevance": 0.7,
    "answer_citation_relevance": 0.7,
    "distinctness": 0.8,
    "logical_flow": 0.8,  # Evaluate logical coherence between conditions and answers
    "overall": 0.85
}

# Annotation settings
MAX_CONDITIONS = 5  # Maximum number of conditions to generate
MIN_CONDITIONS = 2  # Minimum number of conditions to generate

MIN_ROUNDS = 3  # Minimum number of dialogue rounds
MAX_ROUNDS = 5  # Maximum number of dialogue rounds (increased for more comprehensive improvement)
CONVERSATION_TEMPERATURE = 0.4  # Temperature for conversation generation (slightly increased for detailed evaluation)
ANNOTATION_TEMPERATURE = 0.3  # Temperature for annotation generation (slightly increased for richer descriptions)

# Processing settings
BATCH_SIZE = 10  # Number of items to process before saving interim results
MAX_PARALLEL_PROCESSES = 4  # Maximum number of parallel processes
API_RETRY_ATTEMPTS = 5  # Number of retry attempts for API calls
MAX_TOKENS = 6000  # Maximum tokens for API calls (increased to support comprehensive condition-answer format)

# File paths
DEFAULT_INPUT_PATH = "enhanced_questions.json"
DEFAULT_OUTPUT_PATH = "annotated_dataset.json"
DEFAULT_INTERIM_PATH = "interim_annotations.json"
DEFAULT_LOG_PATH = "annotation.log"
CONVERSATION_LOG_DIR = "conversation_logs"  # Directory to save conversation histories

# Create directories if they don't exist
os.makedirs(CONVERSATION_LOG_DIR, exist_ok=True)