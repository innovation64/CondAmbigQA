"""
Models for the AI-AI annotation system.
This module defines AI models for different roles in the annotation process.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI, APIError, RateLimitError, APIConnectionError

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(config.DEFAULT_LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=config.OPENAI_API_KEY)

class AIAgent:
    """Base class for AI agents in the annotation process"""
    
    def __init__(self, role: str, model: str = config.DEFAULT_MODEL):
        """
        Initialize an AI agent
        
        Args:
            role: The role of this agent (annotator, reviewer, facilitator)
            model: The model to use for this agent
        """
        self.role = role
        self.model = model
        self.system_prompt = config.ROLES.get(role, "You are an AI assistant.")
    
    @retry(
        retry=retry_if_exception_type((APIConnectionError, APIError, RateLimitError)),
        stop=stop_after_attempt(config.API_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True,
    )
    def generate_response(
        self, 
        prompt: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = config.CONVERSATION_TEMPERATURE,
        max_tokens: int = config.MAX_TOKENS
    ) -> str:
        """
        Generate a response from the AI agent
        
        Args:
            prompt: The prompt to send to the AI
            conversation_history: Optional conversation history for context
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            The generated response
        """
        messages = []
        
        # Add system message
        messages.append({"role": "system", "content": self.system_prompt})
        
        # Add conversation history if provided
        if conversation_history:
            # Map custom roles to OpenAI supported roles
            mapped_history = []
            for message in conversation_history:
                # Create a copy of the message
                mapped_message = message.copy()
                
                # Map custom roles to OpenAI roles
                role_mapping = {
                    "annotator": "assistant",
                    "reviewer": "user",
                    "facilitator": "user"
                }
                
                # If the role is in our mapping, update it
                if mapped_message.get("role") in role_mapping:
                    mapped_message["role"] = role_mapping[mapped_message["role"]]
                    
                    # Add a prefix to content to indicate the original role
                    original_role = message.get("role", "")
                    content = message.get("content", "")
                    mapped_message["content"] = f"[{original_role.upper()}]: {content}"
                
                mapped_history.append(mapped_message)
            
            messages.extend(mapped_history)
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=config.API_TIMEOUT  # Add explicit timeout
        )
            return response.choices[0].message.content
        except (APIConnectionError, APIError, RateLimitError) as e:
            logger.warning(f"API error with {self.role} ({self.model}): {str(e)}. Retrying...")
            raise
        except Exception as e:
            logger.error(f"Unexpected error with {self.role} ({self.model}): {str(e)}")
            raise

class AnnotatorAgent(AIAgent):
    """AI agent for the annotator role"""
    
    def __init__(self, model: str = config.DEFAULT_MODEL):
        """Initialize the annotator agent"""
        super().__init__("annotator", model)
    
    def generate_initial_annotations(
        self, 
        question: str, 
        fragments: List[Dict[str, str]],
        max_conditions: int = config.MAX_CONDITIONS,
        min_conditions: int = config.MIN_CONDITIONS
    ) -> str:
        """
        Generate initial annotations for a question with comprehensive conditions and answers
        
        Args:
            question: The question to annotate
            fragments: The retrieved fragments for the question
            max_conditions: Maximum number of conditions to generate
            min_conditions: Minimum number of conditions to generate
            
        Returns:
            The generated annotations as a string (will be parsed later)
        """
        fragments_text = "\n".join([
            f"Fragment {i+1} - {fragment.get('title', '')}:\n{fragment.get('text', '')}"
            for i, fragment in enumerate(fragments)
        ])
        
        prompt = f"""I need you to analyze this potentially ambiguous question and identify distinct conditions that lead to different valid answers.

QUESTION: {question}

CONTEXT FRAGMENTS:
{fragments_text}

Please analyze all context fragments carefully and identify between {min_conditions} and {max_conditions} distinct conditions that would lead to different answers to this question. The number of conditions should be based on the actual ambiguity in the question - use fewer conditions for less ambiguous questions and more for highly ambiguous ones.

For each condition, create a comprehensive condition-answer combination:

CONDITION:
- Each condition should be a comprehensive paragraph (5-8 sentences) that provides:
  - Relevant background information about the topic
  - Important constraints or contextual factors
  - Key disambiguation points
  - A framework for understanding the particular interpretation
  - Do NOT directly state what the answer will be in the condition

ANSWER:
- Each answer should be a detailed response (5-8 sentences) that:
  - Directly addresses the question under the given condition
  - Provides a complete, standalone response
  - Includes specific details and explanations
  - Cites relevant evidence from the context fragments

CITATIONS:
- For each condition and answer, you should provide relevant citations from the context fragments
- Condition citations should support the background information and constraints in the condition
- Answer citations should directly support the specific claims in the answer

Your response should:
- Include only genuinely distinct conditions with minimal overlap
- Ensure conditions provide rich context without revealing the answer
- Make each condition-answer pair logically complete and informative
- Use all relevant information from the context fragments

Respond with a JSON structure like this:
```json
{{
"conditions": [
    {{
    "condition": "Comprehensive paragraph with background, constraints, and disambiguation framework...",
    "ground_truth": "Detailed answer that specifically addresses the question under this condition...",
    "condition_citations": [
        {{
        "fragment_number": 1,
        "fragment_title": "Title of fragment 1",
        "fragment_text": "Relevant text from fragment 1"
        }}
    ],
    "answer_citations": [
        {{
        "fragment_number": 2,
        "fragment_title": "Title of fragment 2",
        "fragment_text": "Relevant text from fragment 2"
        }}
    ],
    "reasoning": "Explanation of how this condition leads to this answer"
    }}
]
}}
```

If the question is not ambiguous, just provide a single condition-answer pair."""
        
        return self.generate_response(
            prompt=prompt,
            temperature=config.ANNOTATION_TEMPERATURE
        )
    
    def refine_annotations(
        self,
        question: str,
        fragments: List[Dict[str, str]],
        previous_annotation: str,
        reviewer_feedback: str,
        round_num: int
    ) -> str:
        """
        Refine annotations based on reviewer feedback
        
        Args:
            question: The question being annotated
            fragments: The retrieved fragments
            previous_annotation: Previous annotation (raw text)
            reviewer_feedback: Feedback from the reviewer
            round_num: Current round number
            
        Returns:
            Refined annotations as a string
        """
        fragments_text = "\n".join([
            f"Fragment {i+1} - {fragment.get('title', '')}:\n{fragment.get('text', '')}"
            for i, fragment in enumerate(fragments)
        ])
        
        # Add specific guidance based on the round number
        round_guidance = ""
        if round_num == 2:
            round_guidance = "Focus particularly on improving the comprehensiveness of your conditions. Make sure they provide rich background information, important constraints, and clear disambiguation frameworks without revealing the answer directly."
        elif round_num == 3:
            round_guidance = "Focus on enhancing the detail and depth of your answers, ensuring they directly address the question under each condition and are well-supported by citations."
        elif round_num >= 4:
            round_guidance = "This is a final refinement. Address all remaining issues and make your annotation as perfect as possible. Ensure each condition-answer pair is logically complete, informative, and well-supported."
        
        prompt = f"""I need you to improve your annotation based on the reviewer's feedback.

QUESTION: {question}

CONTEXT FRAGMENTS:
{fragments_text}

YOUR PREVIOUS ANNOTATION:
{previous_annotation}

REVIEWER FEEDBACK:
{reviewer_feedback}

{round_guidance}

Remember that each condition should be a comprehensive paragraph (5-8 sentences) that provides:
- Relevant background information about the topic
- Important constraints or contextual factors
- Key disambiguation points
- A framework for understanding the particular interpretation
- Do NOT directly state what the answer will be in the condition

Each answer should be a detailed response (5-8 sentences) that:
- Directly addresses the question under the given condition
- Provides a complete, standalone response
- Includes specific details and explanations
- Cites relevant evidence from the context fragments

Please revise your annotation to address all the feedback. Maintain the same JSON structure as before:
```json
{{
  "conditions": [
    {{
      "condition": "Improved comprehensive paragraph with background, constraints, and disambiguation framework...",
      "ground_truth": "Improved detailed answer that specifically addresses the question under this condition...",
      "condition_citations": [
        {{
          "fragment_number": number,
          "fragment_title": "Title",
          "fragment_text": "Text"
        }}
      ],
      "answer_citations": [
        {{
          "fragment_number": number,
          "fragment_title": "Title",
          "fragment_text": "Text"
        }}
      ],
      "reasoning": "Updated explanation"
    }}
  ]
}}
```"""
        
        return self.generate_response(
            prompt=prompt,
            temperature=config.ANNOTATION_TEMPERATURE
        )


class ReviewerAgent(AIAgent):
    """AI agent for the reviewer role"""
    
    def __init__(self, model: str = config.DEFAULT_MODEL):
        """Initialize the reviewer agent"""
        super().__init__("reviewer", model)
    
    def evaluate_annotation(
        self,
        question: str,
        fragments: List[Dict[str, str]],
        annotation: str,
        round_num: int
    ) -> str:
        """
        Evaluate an annotation and provide feedback based on the new condition-answer format
        
        Args:
            question: The question being annotated
            fragments: The retrieved fragments
            annotation: The annotation to evaluate (raw text)
            round_num: Current round number
            
        Returns:
            Evaluation feedback as a string
        """
        fragments_text = "\n".join([
            f"Fragment {i+1} - {fragment.get('title', '')}:\n{fragment.get('text', '')}"
            for i, fragment in enumerate(fragments)
        ])
        
        # Customize evaluation criteria based on round number
        if round_num == 1:
            evaluation_focus = """
    For this first review, focus on:
    1. The comprehensiveness of the conditions - do they provide rich background, constraints, and disambiguation frameworks?
    2. The logical completeness of each condition-answer pair
    3. Suggestions for significant improvements in depth and detail"""
        elif round_num == 2:
            evaluation_focus = """
    For this second review, focus on:
    1. Condition comprehensiveness - do conditions provide sufficient background and context without revealing answers?
    2. Answer completeness - are answers detailed and directly address the question under the condition?
    3. Citation relevance - do citations directly support both conditions and answers?"""
        else:
            evaluation_focus = """
    For this advanced review, focus on:
    1. Fine details and nuances in both conditions and answers
    2. The logical flow between conditions and their corresponding answers
    3. Final refinements needed to perfect the comprehensive condition-answer format"""
        
        prompt = f"""As an expert reviewer, evaluate the following annotation for an ambiguous question based on our new comprehensive condition-answer format.

    QUESTION: {question}

    CONTEXT FRAGMENTS:
    {fragments_text}

    ANNOTATION TO REVIEW:
    {annotation}

    {evaluation_focus}

    Evaluate the annotation based on these criteria:
    1. Condition Comprehensiveness (0-10): Do conditions provide rich background, constraints, and disambiguation frameworks?
    2. Condition Utility (0-10): Do conditions help understand the question without revealing answers?
    3. Answer Completeness (0-10): Are answers detailed and directly address the question under each condition?
    4. Condition Citation Relevance (0-10): Do condition citations support the background and constraints?
    5. Answer Citation Relevance (0-10): Do answer citations directly support the claims?
    6. Distinctness (0-10): Are conditions clearly distinct from each other?
    7. Logical Flow (0-10): Do conditions and answers form logically complete pairs?

    For each condition, provide:
    1. Specific scores (0-10) for each criterion
    2. Detailed feedback highlighting strengths and weaknesses
    3. Concrete suggestions for improvement

    Also provide an overall evaluation and final score.

    Use this JSON format for your response:
    ```json
    {{
    "overall_evaluation": "Summary of your evaluation",
    "overall_score": 8.5,
    "conditions_evaluation": [
        {{
        "condition_index": 0,
        "scores": {{
            "condition_comprehensiveness": 8,
            "condition_utility": 7,
            "answer_completeness": 9,
            "condition_citation_relevance": 8,
            "answer_citation_relevance": 6,
            "distinctness": 9,
            "logical_flow": 8
        }},
        "feedback": "Detailed feedback for this condition-answer pair",
        "improvement_suggestions": [
            "Specific suggestion 1",
            "Specific suggestion 2"
        ]
        }}
    ],
    "critical_issues": [
        "Major issue 1 that must be addressed",
        "Major issue 2 that must be addressed"
    ],
    "annotation_approved": false,
    "needs_another_round": true
    }}
    ```"""
        
        return self.generate_response(
            prompt=prompt,
            temperature=config.CONVERSATION_TEMPERATURE
        )
    
class FacilitatorAgent(AIAgent):
    """AI agent for the facilitator role"""
    
    def __init__(self, model: str = config.DEFAULT_MODEL):
        """Initialize the facilitator agent"""
        super().__init__("facilitator", model)
    
    def initiate_annotation(self, question: str, fragments: List[Dict[str, str]]) -> str:
        """
        Initiate the annotation process for a question
        
        Args:
            question: The question to annotate
            fragments: The retrieved fragments
            
        Returns:
            Initial instructions for the annotator
        """
        fragments_text = "\n".join([
            f"Fragment {i+1} - {fragment.get('title', '')}:\n{fragment.get('text', '')}"
            for i, fragment in enumerate(fragments)
        ])
        
        prompt = f"""We're starting an annotation process for a potentially ambiguous question using our new comprehensive condition-answer format. Please initiate this process by providing clear instructions to the annotator.

QUESTION: {question}

CONTEXT FRAGMENTS:
{fragments_text}

As the facilitator, provide initial guidance to the annotator about:
1. How to approach this specific question
2. What to pay special attention to in the fragments
3. What types of comprehensive conditions might be appropriate
4. How to craft detailed answers that directly address the question under each condition

Remember, each condition should be a comprehensive paragraph that provides background information, constraints, and a disambiguation framework without revealing the answer directly. Each answer should be a detailed response that directly addresses the question under the given condition.

Your goal is to help the annotator produce high-quality comprehensive condition-answer pairs on their first attempt."""
        
        return self.generate_response(prompt=prompt)
    
    def summarize_conversation(
        self,
        question: str,
        conversation_history: List[Dict[str, str]],
        final_annotation: str
    ) -> str:
        """
        Summarize the annotation conversation and finalize the result
        
        Args:
            question: The question that was annotated
            conversation_history: The full conversation history
            final_annotation: The final annotation
            
        Returns:
            Summary of the conversation and finalized annotation
        """
        prompt = f"""Please summarize this annotation conversation and provide a final assessment.

QUESTION: {question}

FINAL ANNOTATION:
{final_annotation}

Summarize:
1. How the annotation evolved through the conversation
2. The key improvements made in each round
3. The final quality of the annotation

Then provide your final assessment:
- Is this annotation of high quality?
- Does it properly identify distinct conditions?
- Are the answers precise and well-supported?
- Any remaining issues to note?

Structure your response with sections for Summary, Improvements, and Final Assessment."""
        
        return self.generate_response(
            prompt=prompt,
            conversation_history=conversation_history,
            temperature=config.CONVERSATION_TEMPERATURE
        )
    
    def decide_if_continue(
        self,
        question: str,
        current_round: int,
        reviewer_feedback: str
    ) -> Dict[str, Any]:
        """
        Decide whether to continue the annotation process
        
        Args:
            question: The question being annotated
            current_round: Current round number
            reviewer_feedback: Feedback from the latest review
            
        Returns:
            Decision dictionary with keys:
            - continue: Whether to continue or not
            - reason: Reason for the decision
        """
        prompt = f"""Based on the current annotation process, decide whether to continue with another round of refinement.

QUESTION: {question}

CURRENT ROUND: {current_round}

LATEST REVIEWER FEEDBACK:
{reviewer_feedback}

Consider:
1. Has the minimum number of rounds ({config.MIN_ROUNDS}) been completed?
2. Has the maximum number of rounds ({config.MAX_ROUNDS}) been reached?
3. Does the reviewer's feedback indicate that significant improvements are still needed?
4. Are there any critical issues remaining that must be addressed?

Provide your decision as a JSON object:
```json
{{
  "continue": true/false,
  "reason": "Detailed explanation of your decision"
}}
```"""
        
        response = self.generate_response(prompt=prompt)
        
        # Try to parse the response as JSON
        try:
            import json
            import re
            
            # Extract JSON from the response
            json_match = re.search(r'```(?:json)?\s*(.*?)```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                # Try to parse the entire response as JSON
                return json.loads(response)
        except:
            # Default decision if parsing fails
            logger.warning(f"Failed to parse decision response: {response}")
            default_decision = {
                "continue": current_round < config.MAX_ROUNDS and current_round >= config.MIN_ROUNDS,
                "reason": "Default decision based on round limits due to parsing failure."
            }
            return default_decision