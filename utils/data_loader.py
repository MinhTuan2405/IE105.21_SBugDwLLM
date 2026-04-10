"""
Utility functions for loading the code_x_glue_cc_defect_detection dataset
and formatting prompts for LLM-based defect detection.
"""

from datasets import load_dataset
import random


DATASET_NAME = "google/code_x_glue_cc_defect_detection"

SYSTEM_PROMPT = (
    "You are a security expert specializing in C/C++ code analysis. "
    "Your task is to determine whether a given C/C++ function contains "
    "a security vulnerability or defect (such as buffer overflow, "
    "use-after-free, resource leak, or denial-of-service). "
    "Respond with exactly 1 if the code is defective, or 0 if the code is safe."
)


def load_defect_dataset():
    """Load the defect detection dataset and return train/validation/test splits."""
    ds = load_dataset(DATASET_NAME)
    return ds["train"], ds["validation"], ds["test"]


def format_prompt_codellama(code, examples=None):
    """
    Format a prompt for CodeLlama-Instruct using the [INST] format.

    Args:
        code: The C/C++ function source code to analyze.
        examples: Optional list of (code, label) tuples for few-shot prompting.

    Returns:
        A formatted prompt string.
    """
    user_message = ""

    if examples:
        user_message += "Here are some examples:\n\n"
        for i, (ex_code, ex_label) in enumerate(examples, 1):
            user_message += f"Example {i}:\n```c\n{ex_code}\n```\nAnswer: {ex_label}\n\n"
        user_message += "Now analyze the following function:\n"

    user_message += f"```c\n{code}\n```\n"
    user_message += (
        "Is this function defective? "
        "Answer with exactly 1 (defective) or 0 (safe). "
        "Only output the number, nothing else."
    )

    prompt = (
        f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_message} [/INST]"
    )
    return prompt


def get_few_shot_examples(train_dataset, n=3, seed=42):
    """
    Select n balanced few-shot examples from the training set.

    Args:
        train_dataset: The training split of the dataset.
        n: Total number of examples to return (will be balanced between 0 and 1).
        seed: Random seed for reproducibility.

    Returns:
        List of (code, label) tuples.
    """
    random.seed(seed)

    positives = [s for s in train_dataset if s["target"] == 1]
    negatives = [s for s in train_dataset if s["target"] == 0]

    n_pos = n // 2
    n_neg = n - n_pos

    selected_pos = random.sample(positives, min(n_pos, len(positives)))
    selected_neg = random.sample(negatives, min(n_neg, len(negatives)))

    examples = [(s["func"], s["target"]) for s in selected_neg + selected_pos]
    random.shuffle(examples)
    return examples


def format_for_finetuning(sample):
    """
    Format a single dataset sample for SFT finetuning.

    Returns a dict with 'text' field in CodeLlama instruction format.
    """
    code = sample["func"]
    label = sample["target"]

    user_message = (
        f"Analyze the following C/C++ function and determine if it contains "
        f"a security vulnerability or defect.\n\n```c\n{code}\n```\n\n"
        f"Answer with exactly 1 (defective) or 0 (safe)."
    )

    text = (
        f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_message} [/INST] {label}"
    )
    return {"text": text}
