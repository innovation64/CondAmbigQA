# evaluator.py

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

# 定义评估指标

# Condition Correctness Metric
condition_correctness_metric = GEval(
    name="Condition Correctness",
    criteria="Determine whether the actual condition is factually correct based on the expected condition.",
    evaluation_steps=[
        "Check whether the facts in 'actual condition' contradicts any facts in 'expected condition'.",
        "Heavily penalize omission of critical details in the condition.",
        "Ensure that the condition is clear and unambiguous."
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

# Answer Answer Correctness Metric
answer_correctness_metric = GEval(
    name="Answer Correctness",
    criteria="Determine whether the actual answer is factually correct based on the expected answer.",
    evaluation_steps=[
        "Check whether the facts in 'actual answer' contradicts any facts in 'expected answer'.",
        "Heavily penalize omission of critical details in the answer.",
        "Ensure that the answer directly addresses the condition without introducing irrelevant information."
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

# Citation Correctness Metric
citation_correctness_metric = GEval(
    name="Citation Correctness",
    criteria="Determine whether the citations provided are accurate and relevant to the condition and answer.",
    evaluation_steps=[
        "Verify that each citation supports the corresponding condition and answer.",
        "Check for the correctness of the citation fragments (e.g., Fragment 1, Fragment 2).",
        "Ensure that no irrelevant or incorrect citations are included."
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
)

def evaluate_condition_correctness(input_text: str, actual_condition: str, expected_condition: str):
    """
    评估生成的条件是否正确。
    """
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
    """
    评估生成的答案是否正确。
    """
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

def evaluate_citation_correctness(input_text: str, actual_output: str, retrieval_context: list):
    """
    评估生成的引用是否正确。
    """
    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )
    citation_correctness_metric.measure(test_case)
    return {
        "score": citation_correctness_metric.score,
        "reason": citation_correctness_metric.reason
    }
