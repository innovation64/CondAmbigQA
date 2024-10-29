# # evaluator.py

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
import re
from statistics import mean

def extract_number_from_title(title: str) -> str:
    """
    Extract number from title and format as 'F{number}'.
    Example: "2. Tom Clancy's Rainbow Six Siege" -> "F2"
    """
    match = re.match(r"(\d+)\.", title)
    if match:
        return f"F{match.group(1)}"
    return None

def format_fragment_citation(citation: str) -> str:
    """
    Format generated citation as 'F{number}'.
    Example: "Fragment 1" -> "F1"
    """
    match = re.match(r"Fragment\s+(\d+)", citation)
    if match:
        return f"F{match.group(1)}"
    return citation

def evaluate_condition_correctness(input_text: str, actual_condition: str, expected_condition: str):
    """Evaluate if the generated condition is correct"""
    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_condition,
        expected_output=expected_condition
    )
    condition_correctness_metric.measure(test_case)
    return {
        "score": condition_correctness_metric.score,
        "reason": condition_correctness_metric.reason
    }

def evaluate_answer_correctness(input_text: str, actual_answer: str, expected_answer: str):
    """Evaluate if the generated answer is correct"""
    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_answer,
        expected_output=expected_answer
    )
    answer_correctness_metric.measure(test_case)
    return {
        "score": answer_correctness_metric.score,
        "reason": answer_correctness_metric.reason
    }

def evaluate_citation_correctness(actual_output: list, expected_output: list, fragment_mapping: dict):
    """
    Evaluate citation accuracy.
    
    Parameters:
    - actual_output: List of model-generated citations, e.g., ["Fragment 1", "Fragment 2"]
    - expected_output: List of expected citations, e.g., ["1. Title", "2. Title"]
    - fragment_mapping: Fragment mapping dictionary containing title and text
    
    Returns:
    - Dictionary containing accuracy score and reason
    """
    # Format citations
    formatted_actual = [format_fragment_citation(citation) for citation in actual_output]
    formatted_expected = []
    for citation in expected_output:
        number = extract_number_from_title(citation)
        if number:
            formatted_expected.append(number)
    
    # Calculate correct citations count
    set_actual = set(formatted_actual)
    set_expected = set(formatted_expected)
    correct_citations = set_actual.intersection(set_expected)
    
    # Calculate accuracy
    accuracy = (len(correct_citations) / len(set_expected) * 100) if set_expected else 0.0
    
    # Generate feedback
    feedback = []
    feedback.append(f"Correct citations: {len(correct_citations)}")
    feedback.append(f"Expected citations: {len(set_expected)}")
    feedback.append(f"Citation accuracy: {accuracy:.2f}%")
    
    return {
        "score": accuracy / 100,  # Convert to 0-1 range
        "reason": "\n".join(feedback)
    }

# Define evaluation metrics
condition_correctness_metric = GEval(
    name="Condition Correctness",
    criteria="Determine whether the actual condition is factually correct based on the expected condition.",
    evaluation_steps=[
        "Check whether the facts in 'actual condition' contradicts any facts in 'expected condition'.",
        "Heavily penalize omission of critical details in the condition.",
        "Ensure that the condition is clear and unambiguous."
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model='gpt-4o-mini',
)

answer_correctness_metric = GEval(
    name="Answer Correctness",
    criteria="Determine whether the actual answer is factually correct based on the expected answers.",
    evaluation_steps=[
        "Check whether the facts in 'actual answer' contradicts any facts in 'expected answers'.",
        "Heavily penalize omission of critical details in the answer.",
        "Ensure that the answer directly addresses the question without introducing irrelevant information."
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model='gpt-4o-mini',
)