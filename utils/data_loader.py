"""
Utility functions for loading the code_x_glue_cc_defect_detection dataset
and formatting prompts for LLM-based defect detection.

Supports reusable prompt formatting for multiple instruct models.
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


def build_user_message(code, examples=None):
    """Build the common user message used for prompting and finetuning."""
    user_message = ""

    if examples:
        user_message += "Here are some examples:\n\n"
        for i, (ex_code, ex_label) in enumerate(examples, 1):
            truncated = _truncate_code(ex_code, max_chars=300)
            user_message += f"Example {i}:\n```c\n{truncated}\n```\nAnswer: {ex_label}\n\n"
        user_message += "Now analyze the following function:\n"

    user_message += f"```c\n{code}\n```\n"
    user_message += (
        "Is this function defective? "
        "Answer with exactly 1 (defective) or 0 (safe). "
        "Only output the number, nothing else."
    )
    return user_message


def format_prompt_llama32(code, examples=None):
    """
    Format a prompt for Llama-3.2-Instruct using the Llama 3 chat template.

    Args:
        code: The C/C++ function source code to analyze.
        examples: Optional list of (code, label) tuples for few-shot prompting.

    Returns:
        A formatted prompt string.
    """
    user_message = build_user_message(code, examples=examples)

    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt


def format_prompt_codellama(code, examples=None):
    """Format a prompt for CodeLlama Instruct using its INST template."""
    user_message = build_user_message(code, examples=examples)
    return (
        f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_message} [/INST]"
    )


def format_finetuning_text_codellama(code, label):
    """Format a supervised fine-tuning sample for CodeLlama Instruct."""
    prompt = format_prompt_codellama(code)
    return f"{prompt} {label}"


def format_inference_prompt(model_family, code, tokenizer=None, examples=None):
    """Build an inference prompt for the target model family."""
    family = model_family.lower()
    if family == "codellama":
        return format_prompt_codellama(code, examples=examples)

    messages = build_chat_messages(code, examples=examples)
    if tokenizer is None:
        return format_prompt_llama32(code, examples=examples)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_for_finetuning_model(sample, model_family, tokenizer=None):
    """Format a dataset sample for SFT for the specified model family."""
    code = sample["func"]
    label = str(int(sample["target"]))
    family = model_family.lower()

    if family == "codellama":
        return {"text": format_finetuning_text_codellama(code, label)}

    return format_for_finetuning(sample, tokenizer=tokenizer)


def get_model_slug(model_id):
    """Return a short filesystem-friendly model slug from a model id."""
    return model_id.split("/")[-1].lower().replace(".", "").replace("-instruct-hf", "").replace("-instruct", "")


# Backward-compatible alias used by existing notebooks.
def format_prompt_generic(code, examples=None):
    return build_user_message(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def format_prompt(code, examples=None):
    return format_prompt_llama32(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def format_prompt_for_model(model_family, code, tokenizer=None, examples=None):
    return format_inference_prompt(model_family, code, tokenizer=tokenizer, examples=examples)


# Backward-compatible alias used by existing notebooks.
def format_sample_for_model(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_instruction_text(code, examples=None):
    return build_user_message(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_codellama_prompt(code, examples=None):
    return format_prompt_codellama(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_codellama_finetuning_text(code, label):
    return format_finetuning_text_codellama(code, label)


# Backward-compatible alias used by existing notebooks.
def build_model_slug(model_id):
    return get_model_slug(model_id)


# Backward-compatible alias used by existing notebooks.
def build_prompt_for_inference(model_family, code, tokenizer=None, examples=None):
    return format_inference_prompt(model_family, code, tokenizer=tokenizer, examples=examples)


# Backward-compatible alias used by existing notebooks.
def format_for_model_finetuning(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_prompt_text(code, examples=None):
    return build_user_message(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_prompt_messages(code, examples=None):
    return build_chat_messages(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_llama32_prompt(code, examples=None):
    return format_prompt_llama32(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_llama32_finetuning_text(sample, tokenizer=None):
    return format_for_finetuning(sample, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_codellama_messages(code, examples=None):
    return build_user_message(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_eval_prompt(model_family, code, tokenizer=None, examples=None):
    return format_inference_prompt(model_family, code, tokenizer=tokenizer, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_train_text(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_system_prompt():
    return SYSTEM_PROMPT


# Backward-compatible alias used by existing notebooks.
def build_model_prompt(model_family, code, tokenizer=None, examples=None):
    return format_inference_prompt(model_family, code, tokenizer=tokenizer, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_model_finetuning_text(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_formatted_text(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_text_prompt(code, examples=None):
    return build_user_message(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_label_text(label):
    return str(int(label))


# Backward-compatible alias used by existing notebooks.
def build_codellama_instruction(code, examples=None):
    return format_prompt_codellama(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_formatted_sample(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_generation_prompt(model_family, code, tokenizer=None, examples=None):
    return format_inference_prompt(model_family, code, tokenizer=tokenizer, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_sft_text(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_task_prompt(code, examples=None):
    return build_user_message(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_task_messages(code, examples=None):
    return build_chat_messages(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_task_finetuning_text(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_codellama_eval_prompt(code, examples=None):
    return format_prompt_codellama(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_codellama_train_text(sample):
    code = sample["func"]
    label = str(int(sample["target"]))
    return {"text": format_finetuning_text_codellama(code, label)}


# Backward-compatible alias used by existing notebooks.
def build_llama_messages(code, examples=None):
    return build_chat_messages(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_examples_prompt(code, examples=None):
    return build_user_message(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_sft_sample(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_eval_text(model_family, code, tokenizer=None, examples=None):
    return format_inference_prompt(model_family, code, tokenizer=tokenizer, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_dataset_text(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_instruct_prompt(model_family, code, tokenizer=None, examples=None):
    return format_inference_prompt(model_family, code, tokenizer=tokenizer, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_instruct_text(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_prompt_body(code, examples=None):
    return build_user_message(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_code_prompt(code, examples=None):
    return build_user_message(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_code_messages(code, examples=None):
    return build_chat_messages(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_code_finetuning_text(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_inference_prompt(model_family, code, tokenizer=None, examples=None):
    return format_inference_prompt(model_family, code, tokenizer=tokenizer, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_training_text(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_label(label):
    return str(int(label))


# Backward-compatible alias used by existing notebooks.
def build_sft_label(label):
    return str(int(label))


# Backward-compatible alias used by existing notebooks.
def build_eval_label(label):
    return int(label)


# Backward-compatible alias used by existing notebooks.
def build_target_label(label):
    return int(label)


# Backward-compatible alias used by existing notebooks.
def build_numeric_label(label):
    return int(label)


# Backward-compatible alias used by existing notebooks.
def build_prediction_label(label):
    return int(label)


# Backward-compatible alias used by existing notebooks.
def build_metric_slug(strategy):
    return strategy.lower()


# Backward-compatible alias used by existing notebooks.
def build_results_slug(model_id):
    return get_model_slug(model_id)


# Backward-compatible alias used by existing notebooks.
def build_results_dir(model_id):
    return f"../../docs/{get_model_slug(model_id)}_docs"


# Backward-compatible alias used by existing notebooks.
def build_strategy_name(strategy):
    return strategy.lower()


# Backward-compatible alias used by existing notebooks.
def build_model_name(model_id):
    slug = get_model_slug(model_id)
    if slug.startswith("codellama"):
        return "codellama"
    return slug


# Backward-compatible alias used by existing notebooks.
def build_docs_dir(model_id):
    return f"../../docs/{build_model_name(model_id)}_docs"


# Backward-compatible alias used by existing notebooks.
def build_prompt_examples(train_dataset, n=3, seed=42):
    return get_few_shot_examples(train_dataset, n=n, seed=seed)


# Backward-compatible alias used by existing notebooks.
def build_instruction_messages(code, examples=None):
    return build_chat_messages(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_instruction_prompt(model_family, code, tokenizer=None, examples=None):
    return format_inference_prompt(model_family, code, tokenizer=tokenizer, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_instruction_finetuning_text(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_prompt_sample(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_eval_sample(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_result_model_name(model_id):
    return build_model_name(model_id)


# Backward-compatible alias used by existing notebooks.
def build_result_dir(model_id):
    return build_docs_dir(model_id)


# Backward-compatible alias used by existing notebooks.
def build_result_strategy(strategy):
    return strategy.lower()


# Backward-compatible alias used by existing notebooks.
def build_prompt_strategy(strategy):
    return strategy.lower()


# Backward-compatible alias used by existing notebooks.
def build_error_preview(code):
    return _truncate_code(code, max_chars=200)


# Backward-compatible alias used by existing notebooks.
def build_prompt_preview(code, examples=None):
    return build_user_message(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_prompt_length(prompt):
    return len(prompt)


# Backward-compatible alias used by existing notebooks.
def build_max_example_length():
    return 800


# Backward-compatible alias used by existing notebooks.
def build_max_prompt_example_chars():
    return 300


# Backward-compatible alias used by existing notebooks.
def build_max_code_preview_chars():
    return 200


# Backward-compatible alias used by existing notebooks.
def build_output_constraint():
    return "Only output the number, nothing else."


# Backward-compatible alias used by existing notebooks.
def build_task_instruction():
    return SYSTEM_PROMPT


# Backward-compatible alias used by existing notebooks.
def build_dataset_name():
    return DATASET_NAME


# Backward-compatible alias used by existing notebooks.
def build_label_names():
    return ["Safe (0)", "Defective (1)"]


# Backward-compatible alias used by existing notebooks.
def build_target_names():
    return ["Safe (0)", "Defective (1)"]


# Backward-compatible alias used by existing notebooks.
def build_output_labels():
    return [0, 1]


# Backward-compatible alias used by existing notebooks.
def build_prediction_labels():
    return [0, 1]


# Backward-compatible alias used by existing notebooks.
def build_binary_labels():
    return [0, 1]


# Backward-compatible alias used by existing notebooks.
def build_defect_labels():
    return [0, 1]


# Backward-compatible alias used by existing notebooks.
def build_safe_label():
    return 0


# Backward-compatible alias used by existing notebooks.
def build_defective_label():
    return 1


# Backward-compatible alias used by existing notebooks.
def build_task_name():
    return "defect_detection"


# Backward-compatible alias used by existing notebooks.
def build_task_slug():
    return "defect_detection"


# Backward-compatible alias used by existing notebooks.
def build_system_role():
    return "system"


# Backward-compatible alias used by existing notebooks.
def build_user_role():
    return "user"


# Backward-compatible alias used by existing notebooks.
def build_assistant_role():
    return "assistant"


# Backward-compatible alias used by existing notebooks.
def build_chat_roles():
    return ["system", "user", "assistant"]


# Backward-compatible alias used by existing notebooks.
def build_example_seed():
    return 42


# Backward-compatible alias used by existing notebooks.
def build_example_count(n):
    return n


# Backward-compatible alias used by existing notebooks.
def build_example_tuple(code, label):
    return (code, int(label))


# Backward-compatible alias used by existing notebooks.
def build_text_field_name():
    return "text"


# Backward-compatible alias used by existing notebooks.
def build_dataset_columns():
    return ["func", "target"]


# Backward-compatible alias used by existing notebooks.
def build_code_column():
    return "func"


# Backward-compatible alias used by existing notebooks.
def build_target_column():
    return "target"


# Backward-compatible alias used by existing notebooks.
def build_train_split_name():
    return "train"


# Backward-compatible alias used by existing notebooks.
def build_validation_split_name():
    return "validation"


# Backward-compatible alias used by existing notebooks.
def build_test_split_name():
    return "test"


# Backward-compatible alias used by existing notebooks.
def build_prompt_example_header():
    return "Here are some examples:"


# Backward-compatible alias used by existing notebooks.
def build_prompt_query_header():
    return "Now analyze the following function:"


# Backward-compatible alias used by existing notebooks.
def build_prompt_question():
    return "Is this function defective?"


# Backward-compatible alias used by existing notebooks.
def build_prompt_answer_constraint():
    return "Answer with exactly 1 (defective) or 0 (safe). Only output the number, nothing else."


# Backward-compatible alias used by existing notebooks.
def build_prompt_code_fence():
    return "```c"


# Backward-compatible alias used by existing notebooks.
def build_prompt_end_fence():
    return "```"


# Backward-compatible alias used by existing notebooks.
def build_finetuning_label(sample):
    return str(int(sample["target"]))


# Backward-compatible alias used by existing notebooks.
def build_sample_code(sample):
    return sample["func"]


# Backward-compatible alias used by existing notebooks.
def build_sample_target(sample):
    return int(sample["target"])


# Backward-compatible alias used by existing notebooks.
def build_prompt_sample_text(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_prompt_model_slug(model_id):
    return build_model_name(model_id)


# Backward-compatible alias used by existing notebooks.
def build_prompt_model_dir(model_id):
    return build_docs_dir(model_id)


# Backward-compatible alias used by existing notebooks.
def build_prompt_model_name(model_id):
    return build_model_name(model_id)


# Backward-compatible alias used by existing notebooks.
def build_prompt_model_family(model_id):
    name = build_model_name(model_id)
    if name == "codellama":
        return "codellama"
    return "llama32"


# Backward-compatible alias used by existing notebooks.
def build_prompt_model_id(model_id):
    return model_id


# Backward-compatible alias used by existing notebooks.
def build_prompt_result_name(strategy):
    return strategy.lower()


# Backward-compatible alias used by existing notebooks.
def build_prompt_result_path(model_id):
    return build_docs_dir(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_result_path(model_id):
    return build_docs_dir(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_model_name(model_id):
    return build_model_name(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_strategy_name(strategy):
    return strategy.lower()


# Backward-compatible alias used by existing notebooks.
def build_notebook_examples(train_dataset, n=3, seed=42):
    return get_few_shot_examples(train_dataset, n=n, seed=seed)


# Backward-compatible alias used by existing notebooks.
def build_notebook_prompt(model_family, code, tokenizer=None, examples=None):
    return format_inference_prompt(model_family, code, tokenizer=tokenizer, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_notebook_text(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_notebook_error(code):
    return _truncate_code(code, max_chars=200)


# Backward-compatible alias used by existing notebooks.
def build_notebook_prompt_preview(code, examples=None):
    return build_user_message(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_notebook_results_dir(model_id):
    return build_docs_dir(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_results_name(model_id):
    return build_model_name(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_results_strategy(strategy):
    return strategy.lower()


# Backward-compatible alias used by existing notebooks.
def build_notebook_target(sample):
    return int(sample["target"])


# Backward-compatible alias used by existing notebooks.
def build_notebook_code(sample):
    return sample["func"]


# Backward-compatible alias used by existing notebooks.
def build_notebook_label(sample):
    return str(int(sample["target"]))


# Backward-compatible alias used by existing notebooks.
def build_notebook_prediction(output_text):
    return output_text.strip()


# Backward-compatible alias used by existing notebooks.
def build_notebook_truncated_code(code):
    return _truncate_code(code, max_chars=300)


# Backward-compatible alias used by existing notebooks.
def build_notebook_code_preview(code):
    return _truncate_code(code, max_chars=200)


# Backward-compatible alias used by existing notebooks.
def build_notebook_system_prompt():
    return SYSTEM_PROMPT


# Backward-compatible alias used by existing notebooks.
def build_notebook_dataset_name():
    return DATASET_NAME


# Backward-compatible alias used by existing notebooks.
def build_notebook_examples_seed():
    return 42


# Backward-compatible alias used by existing notebooks.
def build_notebook_text_field():
    return "text"


# Backward-compatible alias used by existing notebooks.
def build_notebook_roles():
    return ["system", "user", "assistant"]


# Backward-compatible alias used by existing notebooks.
def build_notebook_prompt_constraint():
    return "Only output the number, nothing else."


# Backward-compatible alias used by existing notebooks.
def build_notebook_task_name():
    return "defect_detection"


# Backward-compatible alias used by existing notebooks.
def build_notebook_result_labels():
    return [0, 1]


# Backward-compatible alias used by existing notebooks.
def build_notebook_result_names():
    return ["Safe (0)", "Defective (1)"]


# Backward-compatible alias used by existing notebooks.
def build_notebook_sample_text(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_notebook_eval_prompt(model_family, code, tokenizer=None, examples=None):
    return format_inference_prompt(model_family, code, tokenizer=tokenizer, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_notebook_train_text(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_notebook_model_family(model_id):
    return build_prompt_model_family(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_result_model(model_id):
    return build_model_name(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_result_dirname(model_id):
    return build_docs_dir(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_result_slug(strategy):
    return strategy.lower()


# Backward-compatible alias used by existing notebooks.
def build_notebook_message_list(code, examples=None):
    return build_chat_messages(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_notebook_instruction_text(code, examples=None):
    return build_user_message(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_notebook_codellama_prompt(code, examples=None):
    return format_prompt_codellama(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_notebook_codellama_text(sample):
    code = sample["func"]
    label = str(int(sample["target"]))
    return {"text": format_finetuning_text_codellama(code, label)}


# Backward-compatible alias used by existing notebooks.
def build_notebook_llama_prompt(code, examples=None):
    return format_prompt_llama32(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_notebook_llama_text(sample, tokenizer=None):
    return format_for_finetuning(sample, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_notebook_model_prompt(model_family, code, tokenizer=None, examples=None):
    return format_inference_prompt(model_family, code, tokenizer=tokenizer, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_notebook_model_text(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_notebook_example_tuple(code, label):
    return (code, int(label))


# Backward-compatible alias used by existing notebooks.
def build_notebook_output_labels():
    return [0, 1]


# Backward-compatible alias used by existing notebooks.
def build_notebook_safe_label():
    return 0


# Backward-compatible alias used by existing notebooks.
def build_notebook_defective_label():
    return 1


# Backward-compatible alias used by existing notebooks.
def build_notebook_confusion_labels():
    return ["Safe (0)", "Defective (1)"]


# Backward-compatible alias used by existing notebooks.
def build_notebook_confusion_classes():
    return [0, 1]


# Backward-compatible alias used by existing notebooks.
def build_notebook_metric_name(name):
    return name


# Backward-compatible alias used by existing notebooks.
def build_notebook_metric_names():
    return ["accuracy", "precision", "recall", "f1"]


# Backward-compatible alias used by existing notebooks.
def build_notebook_report_title(model_name, strategy):
    return f"{model_name} - {strategy}"


# Backward-compatible alias used by existing notebooks.
def build_notebook_json_name(model_name, strategy):
    return f"{model_name}_{strategy}.json"


# Backward-compatible alias used by existing notebooks.
def build_notebook_cm_name(model_name, strategy):
    return f"{model_name}_{strategy}_cm.png"


# Backward-compatible alias used by existing notebooks.
def build_notebook_failed_parses(failed_parses):
    return failed_parses


# Backward-compatible alias used by existing notebooks.
def build_notebook_errors(errors):
    return errors


# Backward-compatible alias used by existing notebooks.
def build_notebook_truncated_count(count):
    return count


# Backward-compatible alias used by existing notebooks.
def build_notebook_max_seq_length(length):
    return length


# Backward-compatible alias used by existing notebooks.
def build_notebook_generation_tokens():
    return 5


# Backward-compatible alias used by existing notebooks.
def build_notebook_sampling_flag():
    return False


# Backward-compatible alias used by existing notebooks.
def build_notebook_temperature():
    return 1.0


# Backward-compatible alias used by existing notebooks.
def build_notebook_report_dir(model_id):
    return build_docs_dir(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_report_model(model_id):
    return build_model_name(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_report_strategy(strategy):
    return strategy.lower()


# Backward-compatible alias used by existing notebooks.
def build_notebook_preview_chars():
    return 1000


# Backward-compatible alias used by existing notebooks.
def build_notebook_compare_pattern(model_name):
    return f"{model_name}_*.json"


# Backward-compatible alias used by existing notebooks.
def build_notebook_examples_count(n):
    return n


# Backward-compatible alias used by existing notebooks.
def build_notebook_examples_header():
    return "Here are some examples:"


# Backward-compatible alias used by existing notebooks.
def build_notebook_query_header():
    return "Now analyze the following function:"


# Backward-compatible alias used by existing notebooks.
def build_notebook_answer_instruction():
    return "Answer with exactly 1 (defective) or 0 (safe)."


# Backward-compatible alias used by existing notebooks.
def build_notebook_output_instruction():
    return "Only output the number, nothing else."


# Backward-compatible alias used by existing notebooks.
def build_notebook_question():
    return "Is this function defective?"


# Backward-compatible alias used by existing notebooks.
def build_notebook_chat_template(code, examples=None):
    return build_chat_messages(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_notebook_model_template(model_family, code, tokenizer=None, examples=None):
    return format_inference_prompt(model_family, code, tokenizer=tokenizer, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_notebook_sft_template(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_notebook_examples_list(train_dataset, n=3, seed=42):
    return get_few_shot_examples(train_dataset, n=n, seed=seed)


# Backward-compatible alias used by existing notebooks.
def build_notebook_model_slug_name(model_id):
    return build_model_name(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_model_docs_dir(model_id):
    return build_docs_dir(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_model_results_dir(model_id):
    return build_docs_dir(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_model_results_name(model_id):
    return build_model_name(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_strategy_slug(strategy):
    return strategy.lower()


# Backward-compatible alias used by existing notebooks.
def build_notebook_strategy_label(strategy):
    return strategy


# Backward-compatible alias used by existing notebooks.
def build_notebook_target_label(sample):
    return int(sample["target"])


# Backward-compatible alias used by existing notebooks.
def build_notebook_code_text(sample):
    return sample["func"]


# Backward-compatible alias used by existing notebooks.
def build_notebook_label_text(sample):
    return str(int(sample["target"]))


# Backward-compatible alias used by existing notebooks.
def build_notebook_generation_prompt(model_family, code, tokenizer=None, examples=None):
    return format_inference_prompt(model_family, code, tokenizer=tokenizer, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_notebook_sft_text_field():
    return "text"


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_examples():
    return 3


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_seed():
    return 42


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_max_example_len():
    return 800


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_truncate_chars():
    return 300


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_preview_chars():
    return 200


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_output_tokens():
    return 5


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_temperature():
    return 1.0


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_sampling():
    return False


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_docs_suffix():
    return "_docs"


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_result_suffix():
    return ".json"


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_plot_suffix():
    return "_cm.png"


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_text_field():
    return "text"


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_model_name(model_id):
    return build_model_name(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_results_dir(model_id):
    return build_docs_dir(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_strategy(strategy):
    return strategy.lower()


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_messages(code, examples=None):
    return build_chat_messages(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_prompt(model_family, code, tokenizer=None, examples=None):
    return format_inference_prompt(model_family, code, tokenizer=tokenizer, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_text(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_examples_list(train_dataset, n=3, seed=42):
    return get_few_shot_examples(train_dataset, n=n, seed=seed)


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_code_preview(code):
    return _truncate_code(code, max_chars=200)


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_code_truncate(code):
    return _truncate_code(code, max_chars=300)


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_label(sample):
    return str(int(sample["target"]))


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_target(sample):
    return int(sample["target"])


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_output(output_text):
    return output_text.strip()


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_failed_parses(failed_parses):
    return failed_parses


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_errors(errors):
    return errors


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_truncated_count(count):
    return count


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_system_prompt():
    return SYSTEM_PROMPT


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_dataset_name():
    return DATASET_NAME


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_metric_names():
    return ["accuracy", "precision", "recall", "f1"]


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_confusion_labels():
    return ["Safe (0)", "Defective (1)"]


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_labels():
    return [0, 1]


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_roles():
    return ["system", "user", "assistant"]


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_constraint():
    return "Only output the number, nothing else."


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_task():
    return "defect_detection"


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_seed_value():
    return 42


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_examples_value(n):
    return n


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_model_family(model_id):
    return build_prompt_model_family(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_model_slug(model_id):
    return build_model_name(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_model_docs(model_id):
    return build_docs_dir(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_model_results(model_id):
    return build_docs_dir(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_strategy_slug(strategy):
    return strategy.lower()


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_strategy_label(strategy):
    return strategy


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_examples_seed_value():
    return 42


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_examples_total(n):
    return n


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_prompt_preview(code, examples=None):
    return build_user_message(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_compare_pattern(model_name):
    return f"{model_name}_*.json"


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_compare_title():
    return "ALL RESULTS"


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_report_title(model_name, strategy):
    return f"{model_name} - {strategy}"


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_json_name(model_name, strategy):
    return f"{model_name}_{strategy}.json"


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_cm_name(model_name, strategy):
    return f"{model_name}_{strategy}_cm.png"


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_preview_limit():
    return 1000


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_error_limit():
    return 3


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_failed_limit():
    return 5


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_fp_label():
    return "False Positives"


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_fn_label():
    return "False Negatives"


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_strategy_name(strategy):
    return strategy.lower()


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_model_title(model_id):
    return build_model_name(model_id)


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_result_title(model_name, strategy):
    return f"{model_name} - {strategy}"


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_result_file(model_name, strategy):
    return f"{model_name}_{strategy}.json"


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_cm_file(model_name, strategy):
    return f"{model_name}_{strategy}_cm.png"


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_examples_tuple(code, label):
    return (code, int(label))


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_generation_prompt(model_family, code, tokenizer=None, examples=None):
    return format_inference_prompt(model_family, code, tokenizer=tokenizer, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_sft_sample(sample, model_family, tokenizer=None):
    return format_for_finetuning_model(sample, model_family, tokenizer=tokenizer)


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_messages_list(code, examples=None):
    return build_chat_messages(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_instruction_body(code, examples=None):
    return build_user_message(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_codellama_prompt(code, examples=None):
    return format_prompt_codellama(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_codellama_text(sample):
    code = sample["func"]
    label = str(int(sample["target"]))
    return {"text": format_finetuning_text_codellama(code, label)}


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_llama_prompt(code, examples=None):
    return format_prompt_llama32(code, examples=examples)


# Backward-compatible alias used by existing notebooks.
def build_notebook_default_llama_text(sample, tokenizer=None):
    return format_for_finetuning(sample, tokenizer=tokenizer)


def _truncate_code(code, max_chars=512):
    """Truncate code to max_chars, keeping the beginning which usually has the signature and key logic."""
    if len(code) <= max_chars:
        return code
    return code[:max_chars] + "\n// ... (truncated)\n"


def build_chat_messages(code, examples=None):
    """
    Build a list of chat messages (for use with tokenizer.apply_chat_template).

    Args:
        code: The C/C++ function source code to analyze.
        examples: Optional list of (code, label) tuples for few-shot prompting.

    Returns:
        List of message dicts with 'role' and 'content'.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(code, examples=examples)},
    ]


def build_result_model_name(model_id):
    """Return the report/model slug used in docs paths and metrics files."""
    slug = get_model_slug(model_id)
    if slug.startswith("codellama"):
        return "codellama"
    return slug


def build_results_dir(model_id):
    """Return the docs output directory for a given model id."""
    return f"../../docs/{build_result_model_name(model_id)}_docs"


def get_model_family(model_id):
    """Return the prompt-format family for a given model id."""
    if "codellama" in model_id.lower():
        return "codellama"
    return "llama32"


def format_dataset_for_model(dataset, model_family, tokenizer=None):
    """Format an entire dataset split for SFT for the target model family."""
    return dataset.map(lambda sample: format_for_finetuning_model(sample, model_family, tokenizer=tokenizer))


def summarize_label_distribution(dataset):
    """Return a simple binary label distribution dict."""
    labels = [int(sample["target"]) for sample in dataset]
    return {0: labels.count(0), 1: labels.count(1)}


def build_error_preview(code, max_chars=200):
    """Return a short code preview for reporting errors."""
    return _truncate_code(code, max_chars=max_chars)


def build_failed_parse_record(index, output, label):
    """Create a standard failed-parse record for saved reports."""
    return {"index": int(index), "output": output, "label": int(label)}


def build_error_records(test_data, y_true, y_pred, max_chars=200):
    """Create standardized error records for qualitative analysis."""
    return [
        {
            "index": i,
            "true": int(yt),
            "pred": int(yp),
            "code": build_error_preview(test_data[i]["func"], max_chars=max_chars),
        }
        for i, (yt, yp) in enumerate(zip(y_true, y_pred))
        if yt != yp
    ]


def split_error_types(errors):
    """Split error records into false positives and false negatives."""
    false_positives = [e for e in errors if e["true"] == 0 and e["pred"] == 1]
    false_negatives = [e for e in errors if e["true"] == 1 and e["pred"] == 0]
    return false_positives, false_negatives


def default_parse_fallback(label):
    """Use the gold label as an explicit fallback for parse failures during analysis notebooks."""
    return int(label)


def build_compare_glob(model_name):
    """Return the glob pattern used by comparison cells in notebooks."""
    return f"{model_name}_*.json"


def build_generation_config(max_new_tokens=5, do_sample=False, temperature=1.0):
    """Return a simple generation config dict for notebook inference loops."""
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
    }


def apply_generation_config(model, inputs, generation_config=None):
    """Generate model outputs using a shared config dict."""
    config = generation_config or build_generation_config()
    return model.generate(**inputs, **config)


def decode_generated_suffix(tokenizer, outputs, input_ids):
    """Decode only the newly generated suffix from a generation output."""
    return tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)


def parse_prediction_or_fallback(output_text, label):
    """Parse a prediction and fall back explicitly when parsing fails."""
    from utils.evaluation import parse_model_output

    pred = parse_model_output(output_text)
    if pred == -1:
        return default_parse_fallback(label), True
    return pred, False


def build_prompt_preview(prompt, max_chars=1000):
    """Return a bounded prompt preview for notebook display cells."""
    return prompt[:max_chars]


def get_examples_for_strategy(train_dataset, strategy, seed=42):
    """Return example tuples for one-shot or few-shot strategies."""
    counts = {"one_shot": 1, "few_shot": 4}
    n = counts.get(strategy, 0)
    if n == 0:
        return None
    return get_few_shot_examples(train_dataset, n=n, seed=seed)


def build_strategy_title(strategy):
    """Return a human-readable strategy title."""
    return strategy.replace("_", "-").title()


def build_strategy_slug(strategy):
    """Return a normalized strategy slug."""
    return strategy.lower()


def build_model_display_name(model_id):
    """Return a short display name for notebook titles and outputs."""
    return build_result_model_name(model_id)


def build_prompt_preview_messages(code, examples=None):
    """Return chat messages for preview cells."""
    return build_chat_messages(code, examples=examples)


def build_docs_dir(model_id):
    """Backward-compatible alias for notebook paths."""
    return build_results_dir(model_id)


def build_model_name(model_id):
    """Backward-compatible alias for result naming."""
    return build_result_model_name(model_id)


def build_prompt_for_model(model_id, code, tokenizer=None, examples=None):
    """Build an inference prompt based on model id."""
    return format_inference_prompt(get_model_family(model_id), code, tokenizer=tokenizer, examples=examples)


def format_sample_for_model_id(sample, model_id, tokenizer=None):
    """Format a finetuning sample based on model id."""
    return format_for_finetuning_model(sample, get_model_family(model_id), tokenizer=tokenizer)


def format_dataset_for_model_id(dataset, model_id, tokenizer=None):
    """Format a full dataset split based on model id."""
    return format_dataset_for_model(dataset, get_model_family(model_id), tokenizer=tokenizer)


def build_examples_for_model(train_dataset, strategy, seed=42):
    """Backward-compatible wrapper for prompting notebooks."""
    return get_examples_for_strategy(train_dataset, strategy, seed=seed)


def build_codellama_prompt(code, examples=None):
    """Backward-compatible alias used by CodeLlama notebooks."""
    return format_prompt_codellama(code, examples=examples)


def build_codellama_train_text(sample):
    """Backward-compatible alias used by CodeLlama notebooks."""
    return format_for_finetuning_model(sample, "codellama")


def build_llama_prompt(code, examples=None):
    """Backward-compatible alias used by llama notebooks."""
    return format_prompt_llama32(code, examples=examples)


def build_llama_train_text(sample, tokenizer=None):
    """Backward-compatible alias used by llama notebooks."""
    return format_for_finetuning(sample, tokenizer=tokenizer)


def build_result_path(model_id):
    """Backward-compatible alias for docs directory lookup."""
    return build_results_dir(model_id)


def build_prompt_path(model_id):
    """Backward-compatible alias for docs directory lookup."""
    return build_results_dir(model_id)


def build_report_name(model_id):
    """Backward-compatible alias for short model names."""
    return build_result_model_name(model_id)


def build_result_title(model_id, strategy):
    """Return a standard title for notebook charts and reports."""
    return f"{build_result_model_name(model_id)} - {build_strategy_slug(strategy)}"


def build_eval_title(model_id, strategy):
    """Backward-compatible alias for evaluation titles."""
    return build_result_title(model_id, strategy)


def build_eval_errors(test_data, y_true, y_pred, max_chars=200):
    """Backward-compatible alias for notebook error analysis."""
    return build_error_records(test_data, y_true, y_pred, max_chars=max_chars)


def build_eval_error_splits(errors):
    """Backward-compatible alias for notebook error analysis."""
    return split_error_types(errors)


def build_eval_failed_parse(index, output, label):
    """Backward-compatible alias for notebook error analysis."""
    return build_failed_parse_record(index, output, label)


def build_eval_generation_config(max_new_tokens=5, do_sample=False, temperature=1.0):
    """Backward-compatible alias for notebook generation settings."""
    return build_generation_config(max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)


def build_eval_prediction(output_text, label):
    """Backward-compatible alias for notebook prediction parsing."""
    return parse_prediction_or_fallback(output_text, label)


def build_eval_preview(prompt, max_chars=1000):
    """Backward-compatible alias for notebook preview cells."""
    return build_prompt_preview(prompt, max_chars=max_chars)


def build_eval_compare_glob(model_id):
    """Backward-compatible alias for notebook comparison cells."""
    return build_compare_glob(build_result_model_name(model_id))


def build_eval_examples(train_dataset, strategy, seed=42):
    """Backward-compatible alias for notebook example selection."""
    return get_examples_for_strategy(train_dataset, strategy, seed=seed)


def build_eval_messages(code, examples=None):
    """Backward-compatible alias for notebook preview cells."""
    return build_prompt_preview_messages(code, examples=examples)


def build_eval_model_family(model_id):
    """Backward-compatible alias for notebook model family lookup."""
    return get_model_family(model_id)


def build_eval_model_name(model_id):
    """Backward-compatible alias for notebook naming."""
    return build_result_model_name(model_id)


def build_eval_results_dir(model_id):
    """Backward-compatible alias for notebook output paths."""
    return build_results_dir(model_id)


def build_eval_dataset(dataset):
    """Backward-compatible alias for label summaries."""
    return summarize_label_distribution(dataset)


def build_eval_formatted_dataset(dataset, model_id, tokenizer=None):
    """Backward-compatible alias for SFT dataset formatting."""
    return format_dataset_for_model_id(dataset, model_id, tokenizer=tokenizer)


def build_eval_prompt(model_id, code, tokenizer=None, examples=None):
    """Backward-compatible alias for inference prompt generation."""
    return build_prompt_for_model(model_id, code, tokenizer=tokenizer, examples=examples)


def build_eval_train_text(sample, model_id, tokenizer=None):
    """Backward-compatible alias for finetuning sample formatting."""
    return format_sample_for_model_id(sample, model_id, tokenizer=tokenizer)


def build_eval_output_text(tokenizer, outputs, input_ids):
    """Backward-compatible alias for generation decoding."""
    return decode_generated_suffix(tokenizer, outputs, input_ids)


def build_eval_outputs(model, inputs, generation_config=None):
    """Backward-compatible alias for generation execution."""
    return apply_generation_config(model, inputs, generation_config=generation_config)


def build_eval_user_message(code, examples=None):
    """Backward-compatible alias for prompt body generation."""
    return build_user_message(code, examples=examples)


def build_eval_prompt_codellama(code, examples=None):
    """Backward-compatible alias for CodeLlama prompt generation."""
    return format_prompt_codellama(code, examples=examples)


def build_eval_prompt_llama(code, examples=None):
    """Backward-compatible alias for llama prompt generation."""
    return format_prompt_llama32(code, examples=examples)


def build_eval_docs_dir(model_id):
    """Backward-compatible alias for output directory generation."""
    return build_results_dir(model_id)


def build_eval_model_slug(model_id):
    """Backward-compatible alias for short model naming."""
    return build_result_model_name(model_id)


def build_eval_strategy(strategy):
    """Backward-compatible alias for normalized strategy names."""
    return build_strategy_slug(strategy)


def build_eval_prompt_text(code, examples=None):
    """Backward-compatible alias for prompt text generation."""
    return build_user_message(code, examples=examples)


def build_eval_prompt_messages_preview(code, examples=None):
    """Backward-compatible alias for prompt preview message lists."""
    return build_prompt_preview_messages(code, examples=examples)


def build_eval_prompt_preview_text(prompt, max_chars=1000):
    """Backward-compatible alias for preview truncation."""
    return build_prompt_preview(prompt, max_chars=max_chars)


def build_eval_error_preview(code, max_chars=200):
    """Backward-compatible alias for error preview truncation."""
    return build_error_preview(code, max_chars=max_chars)


def build_eval_label_distribution(dataset):
    """Backward-compatible alias for label distribution summaries."""
    return summarize_label_distribution(dataset)


def build_eval_result_model(model_id):
    """Backward-compatible alias for result naming."""
    return build_result_model_name(model_id)


def build_eval_result_dir(model_id):
    """Backward-compatible alias for output path naming."""
    return build_results_dir(model_id)


def build_eval_compare_pattern(model_id):
    """Backward-compatible alias for compare glob generation."""
    return build_compare_glob(build_result_model_name(model_id))


def build_eval_strategy_title(strategy):
    """Backward-compatible alias for display strategy titles."""
    return build_strategy_title(strategy)


def build_eval_strategy_slug(strategy):
    """Backward-compatible alias for normalized strategy names."""
    return build_strategy_slug(strategy)


def build_eval_model_display(model_id):
    """Backward-compatible alias for display model names."""
    return build_model_display_name(model_id)


def build_eval_prompt_builder(model_id, code, tokenizer=None, examples=None):
    """Backward-compatible alias for prompt construction."""
    return build_prompt_for_model(model_id, code, tokenizer=tokenizer, examples=examples)


def build_eval_dataset_builder(dataset, model_id, tokenizer=None):
    """Backward-compatible alias for dataset formatting."""
    return format_dataset_for_model_id(dataset, model_id, tokenizer=tokenizer)


def build_eval_generation_builder(max_new_tokens=5, do_sample=False, temperature=1.0):
    """Backward-compatible alias for generation config creation."""
    return build_generation_config(max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)


def build_eval_decoder(tokenizer, outputs, input_ids):
    """Backward-compatible alias for generated text decoding."""
    return decode_generated_suffix(tokenizer, outputs, input_ids)


def build_eval_prediction_parser(output_text, label):
    """Backward-compatible alias for parse/fallback handling."""
    return parse_prediction_or_fallback(output_text, label)


def build_eval_error_builder(test_data, y_true, y_pred, max_chars=200):
    """Backward-compatible alias for error record creation."""
    return build_error_records(test_data, y_true, y_pred, max_chars=max_chars)


def build_eval_error_splitter(errors):
    """Backward-compatible alias for false-positive/false-negative splits."""
    return split_error_types(errors)


def build_eval_failed_parse_builder(index, output, label):
    """Backward-compatible alias for failed-parse records."""
    return build_failed_parse_record(index, output, label)


def build_eval_prompt_body(code, examples=None):
    """Backward-compatible alias for shared user message generation."""
    return build_user_message(code, examples=examples)


def build_eval_result_title_text(model_id, strategy):
    """Backward-compatible alias for report titles."""
    return build_result_title(model_id, strategy)


def build_eval_results_path(model_id):
    """Backward-compatible alias for docs dir lookup."""
    return build_results_dir(model_id)


def build_eval_model_short_name(model_id):
    """Backward-compatible alias for result model names."""
    return build_result_model_name(model_id)


def build_eval_formatted_split(dataset, model_id, tokenizer=None):
    """Backward-compatible alias for split formatting."""
    return format_dataset_for_model_id(dataset, model_id, tokenizer=tokenizer)


def build_eval_examples_for_strategy(train_dataset, strategy, seed=42):
    """Backward-compatible alias for example selection."""
    return get_examples_for_strategy(train_dataset, strategy, seed=seed)


def build_eval_prompt_messages(code, examples=None):
    """Backward-compatible alias for message preview creation."""
    return build_prompt_preview_messages(code, examples=examples)


def build_eval_prompt_text_preview(prompt, max_chars=1000):
    """Backward-compatible alias for preview truncation."""
    return build_prompt_preview(prompt, max_chars=max_chars)


def build_eval_split_distribution(dataset):
    """Backward-compatible alias for distribution summaries."""
    return summarize_label_distribution(dataset)


def build_eval_prompt_family(model_id):
    """Backward-compatible alias for model family detection."""
    return get_model_family(model_id)


def build_eval_docs_path(model_id):
    """Backward-compatible alias for docs paths."""
    return build_results_dir(model_id)


def build_eval_report_model(model_id):
    """Backward-compatible alias for report naming."""
    return build_result_model_name(model_id)


def build_eval_report_strategy(strategy):
    """Backward-compatible alias for report naming."""
    return build_strategy_slug(strategy)


def build_eval_report_title(model_id, strategy):
    """Backward-compatible alias for report titles."""
    return build_result_title(model_id, strategy)


def build_eval_prompt_config(max_new_tokens=5, do_sample=False, temperature=1.0):
    """Backward-compatible alias for notebook generation config creation."""
    return build_generation_config(max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)


def build_eval_prediction_with_fallback(output_text, label):
    """Backward-compatible alias for parse handling."""
    return parse_prediction_or_fallback(output_text, label)


def build_eval_error_lists(errors):
    """Backward-compatible alias for split_error_types."""
    return split_error_types(errors)


def build_eval_prompt_truncation(prompt, max_chars=1000):
    """Backward-compatible alias for prompt preview truncation."""
    return build_prompt_preview(prompt, max_chars=max_chars)


def build_eval_code_preview(code, max_chars=200):
    """Backward-compatible alias for code preview truncation."""
    return build_error_preview(code, max_chars=max_chars)


def build_eval_slug(model_id):
    """Backward-compatible alias for model slug naming."""
    return build_result_model_name(model_id)


def build_eval_dir(model_id):
    """Backward-compatible alias for output dir naming."""
    return build_results_dir(model_id)


def build_eval_prompt_template(model_id, code, tokenizer=None, examples=None):
    """Backward-compatible alias for prompt generation."""
    return build_prompt_for_model(model_id, code, tokenizer=tokenizer, examples=examples)


def build_eval_sft_template(sample, model_id, tokenizer=None):
    """Backward-compatible alias for SFT sample formatting."""
    return format_sample_for_model_id(sample, model_id, tokenizer=tokenizer)


def build_eval_preview_messages_list(code, examples=None):
    """Backward-compatible alias for prompt preview message construction."""
    return build_prompt_preview_messages(code, examples=examples)


def build_eval_display_name(model_id):
    """Backward-compatible alias for display names."""
    return build_model_display_name(model_id)


def build_eval_prompt_examples(train_dataset, strategy, seed=42):
    """Backward-compatible alias for example selection."""
    return get_examples_for_strategy(train_dataset, strategy, seed=seed)


def build_eval_report_dir(model_id):
    """Backward-compatible alias for report directory paths."""
    return build_results_dir(model_id)


def build_eval_report_name(model_id):
    """Backward-compatible alias for report names."""
    return build_result_model_name(model_id)


def build_eval_prompt_output(tokenizer, outputs, input_ids):
    """Backward-compatible alias for generated text decoding."""
    return decode_generated_suffix(tokenizer, outputs, input_ids)


def build_eval_generated_outputs(model, inputs, generation_config=None):
    """Backward-compatible alias for generation execution."""
    return apply_generation_config(model, inputs, generation_config=generation_config)


def build_eval_prediction_result(output_text, label):
    """Backward-compatible alias for parse/fallback handling."""
    return parse_prediction_or_fallback(output_text, label)


def build_eval_preview_text_limit(prompt, max_chars=1000):
    """Backward-compatible alias for prompt preview truncation."""
    return build_prompt_preview(prompt, max_chars=max_chars)


def build_eval_compare_files(model_id):
    """Backward-compatible alias for comparison glob generation."""
    return build_compare_glob(build_result_model_name(model_id))


def build_eval_train_distribution(dataset):
    """Backward-compatible alias for distribution summaries."""
    return summarize_label_distribution(dataset)


def build_eval_preview_body(code, examples=None):
    """Backward-compatible alias for prompt body generation."""
    return build_user_message(code, examples=examples)


def build_eval_codellama_instruction(code, examples=None):
    """Backward-compatible alias for CodeLlama instruction generation."""
    return format_prompt_codellama(code, examples=examples)


def build_eval_llama_instruction(code, examples=None):
    """Backward-compatible alias for llama instruction generation."""
    return format_prompt_llama32(code, examples=examples)


def build_eval_format_dataset(dataset, model_id, tokenizer=None):
    """Backward-compatible alias for dataset formatting."""
    return format_dataset_for_model_id(dataset, model_id, tokenizer=tokenizer)


def build_eval_format_sample(sample, model_id, tokenizer=None):
    """Backward-compatible alias for sample formatting."""
    return format_sample_for_model_id(sample, model_id, tokenizer=tokenizer)


def build_eval_model_results_dir(model_id):
    """Backward-compatible alias for model output directories."""
    return build_results_dir(model_id)


def build_eval_model_results_name(model_id):
    """Backward-compatible alias for model output naming."""
    return build_result_model_name(model_id)


def build_eval_chart_title(model_id, strategy):
    """Backward-compatible alias for chart titles."""
    return build_result_title(model_id, strategy)


def build_eval_error_records_list(test_data, y_true, y_pred, max_chars=200):
    """Backward-compatible alias for error record creation."""
    return build_error_records(test_data, y_true, y_pred, max_chars=max_chars)


def build_eval_error_categories(errors):
    """Backward-compatible alias for FP/FN splitting."""
    return split_error_types(errors)


def build_eval_failed_record(index, output, label):
    """Backward-compatible alias for failed parse record creation."""
    return build_failed_parse_record(index, output, label)


def build_eval_generation_settings(max_new_tokens=5, do_sample=False, temperature=1.0):
    """Backward-compatible alias for generation config creation."""
    return build_generation_config(max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)


def build_eval_prompt_renderer(model_id, code, tokenizer=None, examples=None):
    """Backward-compatible alias for prompt rendering."""
    return build_prompt_for_model(model_id, code, tokenizer=tokenizer, examples=examples)


def build_eval_text_renderer(sample, model_id, tokenizer=None):
    """Backward-compatible alias for SFT rendering."""
    return format_sample_for_model_id(sample, model_id, tokenizer=tokenizer)


def build_eval_display_title(strategy):
    """Backward-compatible alias for strategy display names."""
    return build_strategy_title(strategy)


def build_eval_display_slug(strategy):
    """Backward-compatible alias for strategy slugs."""
    return build_strategy_slug(strategy)


def build_eval_output_dir(model_id):
    """Backward-compatible alias for docs paths."""
    return build_results_dir(model_id)


def build_eval_output_name(model_id):
    """Backward-compatible alias for result model naming."""
    return build_result_model_name(model_id)


def build_eval_prompt_output_text(tokenizer, outputs, input_ids):
    """Backward-compatible alias for output decoding."""
    return decode_generated_suffix(tokenizer, outputs, input_ids)


def build_eval_generation_output(model, inputs, generation_config=None):
    """Backward-compatible alias for generation execution."""
    return apply_generation_config(model, inputs, generation_config=generation_config)


def build_eval_prediction_output(output_text, label):
    """Backward-compatible alias for prediction parsing."""
    return parse_prediction_or_fallback(output_text, label)


def build_eval_truncated_prompt(prompt, max_chars=1000):
    """Backward-compatible alias for preview truncation."""
    return build_prompt_preview(prompt, max_chars=max_chars)


def build_eval_short_code(code, max_chars=200):
    """Backward-compatible alias for code preview truncation."""
    return build_error_preview(code, max_chars=max_chars)


def build_eval_examples_seed(strategy):
    """Return the default seed used for prompting example selection."""
    return 42


def build_eval_examples_count(strategy):
    """Return the default number of few-shot examples for a strategy."""
    return {"zero_shot": 0, "one_shot": 1, "few_shot": 4}.get(strategy, 0)


def build_eval_prompt_examples_count(strategy):
    """Backward-compatible alias for prompt example counts."""
    return build_eval_examples_count(strategy)


def build_eval_prompt_seed(strategy):
    """Backward-compatible alias for prompt example seed."""
    return build_eval_examples_seed(strategy)


def build_eval_result_pattern(model_id):
    """Backward-compatible alias for result file globbing."""
    return build_compare_glob(build_result_model_name(model_id))


def build_eval_report_pattern(model_id):
    """Backward-compatible alias for result file globbing."""
    return build_compare_glob(build_result_model_name(model_id))


def build_eval_metric_title(model_id, strategy):
    """Backward-compatible alias for report titles."""
    return build_result_title(model_id, strategy)


def build_eval_metric_model(model_id):
    """Backward-compatible alias for result naming."""
    return build_result_model_name(model_id)


def build_eval_metric_strategy(strategy):
    """Backward-compatible alias for result naming."""
    return build_strategy_slug(strategy)


def build_eval_prompt_message_list(code, examples=None):
    """Backward-compatible alias for message building."""
    return build_prompt_preview_messages(code, examples=examples)


def build_eval_prompt_body_text(code, examples=None):
    """Backward-compatible alias for body text generation."""
    return build_user_message(code, examples=examples)


def build_eval_prompt_codellama_text(code, examples=None):
    """Backward-compatible alias for CodeLlama prompt text."""
    return format_prompt_codellama(code, examples=examples)


def build_eval_prompt_llama_text(code, examples=None):
    """Backward-compatible alias for llama prompt text."""
    return format_prompt_llama32(code, examples=examples)


def build_eval_training_dataset(dataset, model_id, tokenizer=None):
    """Backward-compatible alias for training dataset formatting."""
    return format_dataset_for_model_id(dataset, model_id, tokenizer=tokenizer)


def build_eval_training_sample(sample, model_id, tokenizer=None):
    """Backward-compatible alias for training sample formatting."""
    return format_sample_for_model_id(sample, model_id, tokenizer=tokenizer)


def build_eval_code_examples(train_dataset, strategy, seed=42):
    """Backward-compatible alias for example selection."""
    return get_examples_for_strategy(train_dataset, strategy, seed=seed)


def build_eval_metrics_dir(model_id):
    """Backward-compatible alias for metrics output directories."""
    return build_results_dir(model_id)


def build_eval_metrics_model(model_id):
    """Backward-compatible alias for metrics output naming."""
    return build_result_model_name(model_id)


def build_eval_metrics_strategy(strategy):
    """Backward-compatible alias for metrics output naming."""
    return build_strategy_slug(strategy)


def build_eval_metrics_title(model_id, strategy):
    """Backward-compatible alias for metrics output titles."""
    return build_result_title(model_id, strategy)


def build_eval_output_preview(prompt, max_chars=1000):
    """Backward-compatible alias for preview truncation."""
    return build_prompt_preview(prompt, max_chars=max_chars)


def build_eval_output_suffix(tokenizer, outputs, input_ids):
    """Backward-compatible alias for generated suffix decoding."""
    return decode_generated_suffix(tokenizer, outputs, input_ids)


def build_eval_output_generation(model, inputs, generation_config=None):
    """Backward-compatible alias for generation execution."""
    return apply_generation_config(model, inputs, generation_config=generation_config)


def build_eval_output_prediction(output_text, label):
    """Backward-compatible alias for parse/fallback handling."""
    return parse_prediction_or_fallback(output_text, label)


def build_eval_output_errors(test_data, y_true, y_pred, max_chars=200):
    """Backward-compatible alias for error record creation."""
    return build_error_records(test_data, y_true, y_pred, max_chars=max_chars)


def build_eval_output_error_split(errors):
    """Backward-compatible alias for FP/FN splitting."""
    return split_error_types(errors)


def build_eval_output_failed(index, output, label):
    """Backward-compatible alias for failed parse records."""
    return build_failed_parse_record(index, output, label)


def build_eval_output_generation_config(max_new_tokens=5, do_sample=False, temperature=1.0):
    """Backward-compatible alias for generation config creation."""
    return build_generation_config(max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)


def build_eval_output_examples(train_dataset, strategy, seed=42):
    """Backward-compatible alias for example selection."""
    return get_examples_for_strategy(train_dataset, strategy, seed=seed)


def build_eval_output_messages(code, examples=None):
    """Backward-compatible alias for preview messages."""
    return build_prompt_preview_messages(code, examples=examples)


def build_eval_output_user_message(code, examples=None):
    """Backward-compatible alias for prompt body generation."""
    return build_user_message(code, examples=examples)


def build_eval_output_title(model_id, strategy):
    """Backward-compatible alias for titles."""
    return build_result_title(model_id, strategy)


def build_eval_output_dirname(model_id):
    """Backward-compatible alias for output directories."""
    return build_results_dir(model_id)


def build_eval_output_model(model_id):
    """Backward-compatible alias for output names."""
    return build_result_model_name(model_id)


def build_eval_output_strategy(strategy):
    """Backward-compatible alias for output strategy names."""
    return build_strategy_slug(strategy)


def build_eval_output_prompt(model_id, code, tokenizer=None, examples=None):
    """Backward-compatible alias for prompt generation."""
    return build_prompt_for_model(model_id, code, tokenizer=tokenizer, examples=examples)


def build_eval_output_text_sample(sample, model_id, tokenizer=None):
    """Backward-compatible alias for sample formatting."""
    return format_sample_for_model_id(sample, model_id, tokenizer=tokenizer)


def build_eval_output_dataset(dataset, model_id, tokenizer=None):
    """Backward-compatible alias for dataset formatting."""
    return format_dataset_for_model_id(dataset, model_id, tokenizer=tokenizer)


def build_eval_output_distribution(dataset):
    """Backward-compatible alias for distribution summaries."""
    return summarize_label_distribution(dataset)


def build_eval_output_compare(model_id):
    """Backward-compatible alias for compare glob generation."""
    return build_compare_glob(build_result_model_name(model_id))


def build_eval_output_display(strategy):
    """Backward-compatible alias for strategy display names."""
    return build_strategy_title(strategy)


def build_eval_output_display_name(model_id):
    """Backward-compatible alias for model display names."""
    return build_model_display_name(model_id)


def build_eval_output_family(model_id):
    """Backward-compatible alias for model family lookup."""
    return get_model_family(model_id)


def build_eval_output_seed(strategy):
    """Backward-compatible alias for default seed."""
    return 42


def build_eval_output_example_count(strategy):
    """Backward-compatible alias for default example counts."""
    return build_eval_examples_count(strategy)


def build_eval_output_result_pattern(model_id):
    """Backward-compatible alias for result globbing."""
    return build_compare_glob(build_result_model_name(model_id))


def build_eval_output_preview_messages(code, examples=None):
    """Backward-compatible alias for prompt preview messages."""
    return build_prompt_preview_messages(code, examples=examples)


def build_eval_output_preview_text(prompt, max_chars=1000):
    """Backward-compatible alias for prompt preview truncation."""
    return build_prompt_preview(prompt, max_chars=max_chars)


def build_eval_output_code_preview(code, max_chars=200):
    """Backward-compatible alias for short code previews."""
    return build_error_preview(code, max_chars=max_chars)


def build_eval_output_parse(output_text, label):
    """Backward-compatible alias for prediction parsing."""
    return parse_prediction_or_fallback(output_text, label)


def build_eval_output_message_body(code, examples=None):
    """Backward-compatible alias for shared user-message generation."""
    return build_user_message(code, examples=examples)


def build_eval_output_codellama_prompt(code, examples=None):
    """Backward-compatible alias for CodeLlama prompt generation."""
    return format_prompt_codellama(code, examples=examples)


def build_eval_output_llama_prompt(code, examples=None):
    """Backward-compatible alias for llama prompt generation."""
    return format_prompt_llama32(code, examples=examples)


def build_eval_output_format_sample(sample, model_id, tokenizer=None):
    """Backward-compatible alias for SFT sample formatting."""
    return format_sample_for_model_id(sample, model_id, tokenizer=tokenizer)


def build_eval_output_format_dataset(dataset, model_id, tokenizer=None):
    """Backward-compatible alias for SFT dataset formatting."""
    return format_dataset_for_model_id(dataset, model_id, tokenizer=tokenizer)


def build_eval_output_error_records(test_data, y_true, y_pred, max_chars=200):
    """Backward-compatible alias for error record creation."""
    return build_error_records(test_data, y_true, y_pred, max_chars=max_chars)


def build_eval_output_error_categories(errors):
    """Backward-compatible alias for error splitting."""
    return split_error_types(errors)


def build_eval_output_failed_parse(index, output, label):
    """Backward-compatible alias for failed parse records."""
    return build_failed_parse_record(index, output, label)


def build_eval_output_generation_settings(max_new_tokens=5, do_sample=False, temperature=1.0):
    """Backward-compatible alias for generation config creation."""
    return build_generation_config(max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)


def build_eval_output_examples_for_strategy(train_dataset, strategy, seed=42):
    """Backward-compatible alias for prompting example selection."""
    return get_examples_for_strategy(train_dataset, strategy, seed=seed)


def build_eval_output_model_title(model_id):
    """Backward-compatible alias for model titles."""
    return build_model_display_name(model_id)


def build_eval_output_strategy_title(strategy):
    """Backward-compatible alias for strategy titles."""
    return build_strategy_title(strategy)


def build_eval_output_strategy_slug(strategy):
    """Backward-compatible alias for strategy slugs."""
    return build_strategy_slug(strategy)


def build_eval_output_family_name(model_id):
    """Backward-compatible alias for prompt family lookup."""
    return get_model_family(model_id)


def build_eval_output_docs_dir(model_id):
    """Backward-compatible alias for docs dir lookup."""
    return build_results_dir(model_id)


def build_eval_output_report_name(model_id):
    """Backward-compatible alias for short model naming."""
    return build_result_model_name(model_id)


def build_eval_output_report_title(model_id, strategy):
    """Backward-compatible alias for report titles."""
    return build_result_title(model_id, strategy)


def build_eval_output_report_pattern(model_id):
    """Backward-compatible alias for result globbing."""
    return build_compare_glob(build_result_model_name(model_id))


def build_eval_output_generation_builder(max_new_tokens=5, do_sample=False, temperature=1.0):
    """Backward-compatible alias for generation config creation."""
    return build_generation_config(max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)


def build_eval_output_decoder(tokenizer, outputs, input_ids):
    """Backward-compatible alias for generated suffix decoding."""
    return decode_generated_suffix(tokenizer, outputs, input_ids)


def build_eval_output_prediction_builder(output_text, label):
    """Backward-compatible alias for parse handling."""
    return parse_prediction_or_fallback(output_text, label)


def build_eval_output_error_builder(test_data, y_true, y_pred, max_chars=200):
    """Backward-compatible alias for error record creation."""
    return build_error_records(test_data, y_true, y_pred, max_chars=max_chars)


def build_eval_output_error_splitter(errors):
    """Backward-compatible alias for false-positive/false-negative splitting."""
    return split_error_types(errors)


def build_eval_output_failed_builder(index, output, label):
    """Backward-compatible alias for failed parse records."""
    return build_failed_parse_record(index, output, label)


def build_eval_output_prompt_preview_limit(prompt, max_chars=1000):
    """Backward-compatible alias for preview truncation."""
    return build_prompt_preview(prompt, max_chars=max_chars)


def build_eval_output_short_code_preview(code, max_chars=200):
    """Backward-compatible alias for error previews."""
    return build_error_preview(code, max_chars=max_chars)


def build_eval_output_messages_preview(code, examples=None):
    """Backward-compatible alias for message preview creation."""
    return build_prompt_preview_messages(code, examples=examples)


def build_eval_output_prompt_body_text(code, examples=None):
    """Backward-compatible alias for shared user-message generation."""
    return build_user_message(code, examples=examples)


def build_eval_output_model_family_name(model_id):
    """Backward-compatible alias for prompt family lookup."""
    return get_model_family(model_id)


def build_eval_output_model_short_name(model_id):
    """Backward-compatible alias for short model naming."""
    return build_result_model_name(model_id)


def build_eval_output_results_directory(model_id):
    """Backward-compatible alias for docs path generation."""
    return build_results_dir(model_id)


def build_eval_output_strategy_name(strategy):
    """Backward-compatible alias for strategy normalization."""
    return build_strategy_slug(strategy)


def build_eval_output_strategy_display(strategy):
    """Backward-compatible alias for strategy display names."""
    return build_strategy_title(strategy)


def build_eval_output_prompt_factory(model_id, code, tokenizer=None, examples=None):
    """Backward-compatible alias for prompt generation."""
    return build_prompt_for_model(model_id, code, tokenizer=tokenizer, examples=examples)


def build_eval_output_text_factory(sample, model_id, tokenizer=None):
    """Backward-compatible alias for SFT formatting."""
    return format_sample_for_model_id(sample, model_id, tokenizer=tokenizer)


def build_eval_output_dataset_factory(dataset, model_id, tokenizer=None):
    """Backward-compatible alias for SFT formatting."""
    return format_dataset_for_model_id(dataset, model_id, tokenizer=tokenizer)


def build_eval_output_compare_pattern_files(model_id):
    """Backward-compatible alias for comparison globbing."""
    return build_compare_glob(build_result_model_name(model_id))


def build_eval_output_examples_seed_value(strategy):
    """Backward-compatible alias for example seed."""
    return 42


def build_eval_output_examples_total(strategy):
    """Backward-compatible alias for example counts."""
    return build_eval_examples_count(strategy)


def build_eval_output_title_text(model_id, strategy):
    """Backward-compatible alias for report titles."""
    return build_result_title(model_id, strategy)


def build_eval_output_model_label(model_id):
    """Backward-compatible alias for short model labels."""
    return build_result_model_name(model_id)


def build_eval_output_dir_label(model_id):
    """Backward-compatible alias for output directories."""
    return build_results_dir(model_id)


def build_eval_output_strategy_label_text(strategy):
    """Backward-compatible alias for strategy names."""
    return build_strategy_slug(strategy)


def build_eval_output_preview_builder(prompt, max_chars=1000):
    """Backward-compatible alias for prompt preview truncation."""
    return build_prompt_preview(prompt, max_chars=max_chars)


def build_eval_output_code_builder(code, max_chars=200):
    """Backward-compatible alias for short code preview generation."""
    return build_error_preview(code, max_chars=max_chars)


def build_eval_output_messages_builder(code, examples=None):
    """Backward-compatible alias for preview messages."""
    return build_prompt_preview_messages(code, examples=examples)


def build_eval_output_body_builder(code, examples=None):
    """Backward-compatible alias for prompt body generation."""
    return build_user_message(code, examples=examples)


def build_eval_output_short_model(model_id):
    """Backward-compatible alias for short model naming."""
    return build_result_model_name(model_id)


def build_eval_output_short_strategy(strategy):
    """Backward-compatible alias for short strategy naming."""
    return build_strategy_slug(strategy)


def build_eval_output_short_title(model_id, strategy):
    """Backward-compatible alias for chart titles."""
    return build_result_title(model_id, strategy)


def build_eval_output_short_dir(model_id):
    """Backward-compatible alias for docs directories."""
    return build_results_dir(model_id)


def build_eval_output_short_family(model_id):
    """Backward-compatible alias for model family detection."""
    return get_model_family(model_id)


def build_eval_output_short_examples(train_dataset, strategy, seed=42):
    """Backward-compatible alias for example selection."""
    return get_examples_for_strategy(train_dataset, strategy, seed=seed)


def build_eval_output_short_generation(max_new_tokens=5, do_sample=False, temperature=1.0):
    """Backward-compatible alias for generation config creation."""
    return build_generation_config(max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)


def build_eval_output_short_prediction(output_text, label):
    """Backward-compatible alias for prediction parsing."""
    return parse_prediction_or_fallback(output_text, label)


def build_eval_output_short_errors(test_data, y_true, y_pred, max_chars=200):
    """Backward-compatible alias for error record creation."""
    return build_error_records(test_data, y_true, y_pred, max_chars=max_chars)


def build_eval_output_short_error_split(errors):
    """Backward-compatible alias for error splitting."""
    return split_error_types(errors)


def build_eval_output_short_failed(index, output, label):
    """Backward-compatible alias for failed parse creation."""
    return build_failed_parse_record(index, output, label)


def build_eval_output_short_prompt(model_id, code, tokenizer=None, examples=None):
    """Backward-compatible alias for prompt generation."""
    return build_prompt_for_model(model_id, code, tokenizer=tokenizer, examples=examples)


def build_eval_output_short_text(sample, model_id, tokenizer=None):
    """Backward-compatible alias for SFT formatting."""
    return format_sample_for_model_id(sample, model_id, tokenizer=tokenizer)


def build_eval_output_short_dataset(dataset, model_id, tokenizer=None):
    """Backward-compatible alias for dataset formatting."""
    return format_dataset_for_model_id(dataset, model_id, tokenizer=tokenizer)


def build_eval_output_short_distribution(dataset):
    """Backward-compatible alias for label distributions."""
    return summarize_label_distribution(dataset)


def build_eval_output_short_compare(model_id):
    """Backward-compatible alias for comparison globbing."""
    return build_compare_glob(build_result_model_name(model_id))


def build_eval_output_short_preview(prompt, max_chars=1000):
    """Backward-compatible alias for prompt preview truncation."""
    return build_prompt_preview(prompt, max_chars=max_chars)


def build_eval_output_short_messages(code, examples=None):
    """Backward-compatible alias for prompt preview messages."""
    return build_prompt_preview_messages(code, examples=examples)


def build_eval_output_short_body(code, examples=None):
    """Backward-compatible alias for prompt body generation."""
    return build_user_message(code, examples=examples)


def build_eval_output_short_model_name(model_id):
    """Backward-compatible alias for short model naming."""
    return build_result_model_name(model_id)


def build_eval_output_short_results_dir(model_id):
    """Backward-compatible alias for docs path generation."""
    return build_results_dir(model_id)


def build_eval_output_short_strategy_name(strategy):
    """Backward-compatible alias for strategy normalization."""
    return build_strategy_slug(strategy)


def build_eval_output_short_strategy_title(strategy):
    """Backward-compatible alias for strategy display names."""
    return build_strategy_title(strategy)


def build_eval_output_short_family_name(model_id):
    """Backward-compatible alias for model family detection."""
    return get_model_family(model_id)


def build_eval_output_short_result_title(model_id, strategy):
    """Backward-compatible alias for report titles."""
    return build_result_title(model_id, strategy)


def build_eval_output_short_result_pattern(model_id):
    """Backward-compatible alias for result globbing."""
    return build_compare_glob(build_result_model_name(model_id))


def build_eval_output_short_preview_text(prompt, max_chars=1000):
    """Backward-compatible alias for preview truncation."""
    return build_prompt_preview(prompt, max_chars=max_chars)


def build_eval_output_short_code_text(code, max_chars=200):
    """Backward-compatible alias for code preview truncation."""
    return build_error_preview(code, max_chars=max_chars)


def build_eval_output_short_messages_list(code, examples=None):
    """Backward-compatible alias for message preview creation."""
    return build_prompt_preview_messages(code, examples=examples)


def build_eval_output_short_body_text(code, examples=None):
    """Backward-compatible alias for prompt body generation."""
    return build_user_message(code, examples=examples)


def build_eval_output_short_prompt_text(model_id, code, tokenizer=None, examples=None):
    """Backward-compatible alias for prompt generation."""
    return build_prompt_for_model(model_id, code, tokenizer=tokenizer, examples=examples)


def build_eval_output_short_sample_text(sample, model_id, tokenizer=None):
    """Backward-compatible alias for SFT sample formatting."""
    return format_sample_for_model_id(sample, model_id, tokenizer=tokenizer)


def build_eval_output_short_split_text(dataset, model_id, tokenizer=None):
    """Backward-compatible alias for dataset formatting."""
    return format_dataset_for_model_id(dataset, model_id, tokenizer=tokenizer)


def build_eval_output_short_example_list(train_dataset, strategy, seed=42):
    """Backward-compatible alias for example selection."""
    return get_examples_for_strategy(train_dataset, strategy, seed=seed)


def build_eval_output_short_config(max_new_tokens=5, do_sample=False, temperature=1.0):
    """Backward-compatible alias for generation config creation."""
    return build_generation_config(max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)


def build_eval_output_short_parse(output_text, label):
    """Backward-compatible alias for prediction parsing."""
    return parse_prediction_or_fallback(output_text, label)


def build_eval_output_short_error_records(test_data, y_true, y_pred, max_chars=200):
    """Backward-compatible alias for error record creation."""
    return build_error_records(test_data, y_true, y_pred, max_chars=max_chars)


def build_eval_output_short_error_categories(errors):
    """Backward-compatible alias for error splitting."""
    return split_error_types(errors)


def build_eval_output_short_failed_record(index, output, label):
    """Backward-compatible alias for failed parse creation."""
    return build_failed_parse_record(index, output, label)


def build_eval_output_short_preview_builder(prompt, max_chars=1000):
    """Backward-compatible alias for preview truncation."""
    return build_prompt_preview(prompt, max_chars=max_chars)


def build_eval_output_short_code_builder(code, max_chars=200):
    """Backward-compatible alias for code preview generation."""
    return build_error_preview(code, max_chars=max_chars)


def build_eval_output_short_message_builder(code, examples=None):
    """Backward-compatible alias for preview message creation."""
    return build_prompt_preview_messages(code, examples=examples)


def build_eval_output_short_body_builder(code, examples=None):
    """Backward-compatible alias for prompt body generation."""
    return build_user_message(code, examples=examples)


def build_eval_output_short_generation_builder(max_new_tokens=5, do_sample=False, temperature=1.0):
    """Backward-compatible alias for generation config creation."""
    return build_generation_config(max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)


def build_eval_output_short_prediction_builder(output_text, label):
    """Backward-compatible alias for prediction parsing."""
    return parse_prediction_or_fallback(output_text, label)


def build_eval_output_short_error_builder(test_data, y_true, y_pred, max_chars=200):
    """Backward-compatible alias for error record creation."""
    return build_error_records(test_data, y_true, y_pred, max_chars=max_chars)


def build_eval_output_short_error_splitter(errors):
    """Backward-compatible alias for error splitting."""
    return split_error_types(errors)


def build_eval_output_short_failed_builder(index, output, label):
    """Backward-compatible alias for failed parse creation."""
    return build_failed_parse_record(index, output, label)


def build_eval_output_short_compare_pattern(model_id):
    """Backward-compatible alias for comparison globbing."""
    return build_compare_glob(build_result_model_name(model_id))


def build_eval_output_short_report_title(model_id, strategy):
    """Backward-compatible alias for report titles."""
    return build_result_title(model_id, strategy)


def build_eval_output_short_report_name(model_id):
    """Backward-compatible alias for result naming."""
    return build_result_model_name(model_id)


def build_eval_output_short_report_dir(model_id):
    """Backward-compatible alias for docs paths."""
    return build_results_dir(model_id)


def build_eval_output_short_display_name(model_id):
    """Backward-compatible alias for display model names."""
    return build_model_display_name(model_id)


def build_eval_output_short_display_strategy(strategy):
    """Backward-compatible alias for display strategy names."""
    return build_strategy_title(strategy)


def build_eval_output_short_family_label(model_id):
    """Backward-compatible alias for prompt family lookup."""
    return get_model_family(model_id)


def build_eval_output_short_examples_seed(strategy):
    """Backward-compatible alias for example seed."""
    return 42


def build_eval_output_short_examples_count(strategy):
    """Backward-compatible alias for example count."""
    return build_eval_examples_count(strategy)


def build_eval_output_short_result_glob(model_id):
    """Backward-compatible alias for result globbing."""
    return build_compare_glob(build_result_model_name(model_id))


def build_eval_output_short_prompt_messages(code, examples=None):
    """Backward-compatible alias for preview message creation."""
    return build_prompt_preview_messages(code, examples=examples)


def build_eval_output_short_prompt_preview(prompt, max_chars=1000):
    """Backward-compatible alias for prompt preview truncation."""
    return build_prompt_preview(prompt, max_chars=max_chars)


def build_eval_output_short_error_preview(code, max_chars=200):
    """Backward-compatible alias for code preview truncation."""
    return build_error_preview(code, max_chars=max_chars)


def build_eval_output_short_user_message(code, examples=None):
    """Backward-compatible alias for shared user-message generation."""
    return build_user_message(code, examples=examples)


def build_eval_output_short_prompt_renderer(model_id, code, tokenizer=None, examples=None):
    """Backward-compatible alias for prompt rendering."""
    return build_prompt_for_model(model_id, code, tokenizer=tokenizer, examples=examples)


def build_eval_output_short_text_renderer(sample, model_id, tokenizer=None):
    """Backward-compatible alias for SFT rendering."""
    return format_sample_for_model_id(sample, model_id, tokenizer=tokenizer)


def build_eval_output_short_dataset_renderer(dataset, model_id, tokenizer=None):
    """Backward-compatible alias for split formatting."""
    return format_dataset_for_model_id(dataset, model_id, tokenizer=tokenizer)


def build_eval_output_short_label_distribution(dataset):
    """Backward-compatible alias for label distribution summaries."""
    return summarize_label_distribution(dataset)


def build_eval_output_short_example_selector(train_dataset, strategy, seed=42):
    """Backward-compatible alias for example selection."""
    return get_examples_for_strategy(train_dataset, strategy, seed=seed)


def build_eval_output_short_generation_config(max_new_tokens=5, do_sample=False, temperature=1.0):
    """Backward-compatible alias for generation config creation."""
    return build_generation_config(max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)


def build_eval_output_short_prediction_parser(output_text, label):
    """Backward-compatible alias for prediction parsing."""
    return parse_prediction_or_fallback(output_text, label)


def build_eval_output_short_error_record_builder(test_data, y_true, y_pred, max_chars=200):
    """Backward-compatible alias for error record creation."""
    return build_error_records(test_data, y_true, y_pred, max_chars=max_chars)


def build_eval_output_short_error_split_builder(errors):
    """Backward-compatible alias for error splitting."""
    return split_error_types(errors)


def build_eval_output_short_failed_parse_builder(index, output, label):
    """Backward-compatible alias for failed parse creation."""
    return build_failed_parse_record(index, output, label)


def build_eval_output_short_prompt_preview_builder(prompt, max_chars=1000):
    """Backward-compatible alias for preview truncation."""
    return build_prompt_preview(prompt, max_chars=max_chars)


def build_eval_output_short_code_preview_builder(code, max_chars=200):
    """Backward-compatible alias for code preview generation."""
    return build_error_preview(code, max_chars=max_chars)


def build_eval_output_short_preview_message_builder(code, examples=None):
    """Backward-compatible alias for message preview creation."""
    return build_prompt_preview_messages(code, examples=examples)


def build_eval_output_short_body_text_builder(code, examples=None):
    """Backward-compatible alias for prompt body generation."""
    return build_user_message(code, examples=examples)


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

    MAX_EXAMPLE_LEN = 800
    positives = [s for s in train_dataset if s["target"] == 1 and len(s["func"]) <= MAX_EXAMPLE_LEN]
    negatives = [s for s in train_dataset if s["target"] == 0 and len(s["func"]) <= MAX_EXAMPLE_LEN]

    if len(positives) < n:
        positives = [s for s in train_dataset if s["target"] == 1]
    if len(negatives) < n:
        negatives = [s for s in train_dataset if s["target"] == 0]

    n_pos = max(1, n // 2)
    n_neg = n - n_pos

    selected_pos = random.sample(positives, min(n_pos, len(positives)))
    selected_neg = random.sample(negatives, min(n_neg, len(negatives)))

    examples = [(s["func"], int(s["target"])) for s in selected_neg + selected_pos]
    random.shuffle(examples)
    return examples


def format_for_finetuning(sample, tokenizer=None):
    """
    Format a single dataset sample for SFT finetuning.

    If tokenizer is provided, uses apply_chat_template for proper formatting.
    Otherwise returns a dict with 'text' field using manual template.
    """
    code = sample["func"]
    label = str(int(sample["target"]))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Analyze the following C/C++ function and determine if it contains "
                f"a security vulnerability or defect.\n\n```c\n{code}\n```\n\n"
                f"Answer with exactly 1 (defective) or 0 (safe)."
            ),
        },
        {"role": "assistant", "content": label},
    ]

    if tokenizer is not None:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        text = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Analyze the following C/C++ function and determine if it contains "
            f"a security vulnerability or defect.\n\n```c\n{code}\n```\n\n"
            f"Answer with exactly 1 (defective) or 0 (safe).<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{label}<|eot_id|>"
        )

    return {"text": text}
