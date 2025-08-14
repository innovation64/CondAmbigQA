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

# def evaluate_citation_correctness(actual_output: list, expected_output: list, fragment_mapping: dict):
#     """
#     Evaluate citation accuracy.
    
#     Parameters:
#     - actual_output: List of model-generated citations, e.g., ["Fragment 1", "Fragment 2"]
#     - expected_output: List of expected citations, e.g., ["1. Title", "2. Title"]
#     - fragment_mapping: Fragment mapping dictionary containing title and text
    
#     Returns:
#     - Dictionary containing accuracy score and reason
#     """
#     # Format citations
#     formatted_actual = [format_fragment_citation(citation) for citation in actual_output]
#     formatted_expected = []
#     for citation in expected_output:
#         number = extract_number_from_title(citation)
#         if number:
#             formatted_expected.append(number)
    
#     # Calculate correct citations count
#     set_actual = set(formatted_actual)
#     set_expected = set(formatted_expected)
#     correct_citations = set_actual.intersection(set_expected)
    
#     # Calculate accuracy
#     accuracy = (len(correct_citations) / len(set_expected) * 100) if set_expected else 0.0
    
#     # Generate feedback
#     feedback = []
#     feedback.append(f"Correct citations: {len(correct_citations)}")
#     feedback.append(f"Expected citations: {len(set_expected)}")
#     feedback.append(f"Citation accuracy: {accuracy:.2f}%")
    
#     return {
#         "score": accuracy / 100,  # Convert to 0-1 range
#         "reason": "\n".join(feedback)
#     }


def evaluate_citation_correctness(actual_output: list, expected_output: list, fragment_mapping: dict):
    """
    Improved: Citation correctness is based on which sources are actually used in the answer.

    Parameters:
    - actual_output: List of citations actually used by the model (e.g., ["Fragment 1", "Fragment 2"])
    - expected_output: Ground truth citations (e.g., ["1. Title", "2. Title"])
    - fragment_mapping: Dictionary that maps fragments (like "Fragment 1") to actual source content (optional)

    Returns:
    - Dictionary with score and feedback
    """

    # Format both actual and expected citations to "F{number}" format
    formatted_actual = [format_fragment_citation(citation) for citation in actual_output]
    formatted_expected = [
        extract_number_from_title(citation) for citation in expected_output if extract_number_from_title(citation)
    ]

    set_actual = set(formatted_actual)
    set_expected = set(formatted_expected)
    
    correct_citations = set_actual.intersection(set_expected)
    # Focus only on model-used citations
    if not set_actual:
        accuracy = 0.0
    else:
        correct_citations = set_actual.intersection(set_expected)
        accuracy = len(correct_citations) / len(set_actual)

    # Feedback
    feedback = [
        f"Used citations: {sorted(set_actual)}",
        f"Expected citations: {sorted(set_expected)}",
        f"Correctly cited: {sorted(correct_citations)}",
        f"Citation precision: {accuracy:.2%}"
    ]

    return {
        "score": accuracy,
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

def evaluate_condition_quality_no_ref(input_text: str, actual_condition: str):
    """
    Evaluate whether the model-generated condition is reasonable using self-reflective reasoning,
    without relying on a gold reference.
    Uses G-Eval to assess if the generated content is reasonable, clear, and complete.
    """
    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_condition,
        expected_output="N/A"  # GEval will ignore this under this metric
    )
    no_ref_condition_metric.measure(test_case)
    return {
        "score": no_ref_condition_metric.score,
        "reason": no_ref_condition_metric.reason
    }

# New G-Eval metric: No-reference self-assessed condition quality
no_ref_condition_metric = GEval(
    name="No-Ref Condition Quality",
    criteria="Judge whether the condition itself is high-quality without reference.",
    evaluation_steps=[
        "Does the condition clearly express a valid constraint or assumption?",
        "Does it align with the input question's ambiguity or scope?",
        "Is it free from hallucination, vague or generic statements?",
        "Does it help disambiguate the question in a meaningful way?"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini"
)
