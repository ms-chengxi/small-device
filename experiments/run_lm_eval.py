# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import logging
import torch
from ml_collections import ConfigDict

import lm_eval
from lm_eval import tasks
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks

from utils import load_model_and_tokenizer

TASK_METRIC_MAP = {
    "mmlu_abstract_algebra": "acc,none",
    "mmlu_business_ethics": "acc,none",
    "mmlu_college_computer_science": "acc,none",
    "mmlu_college_mathematics": "acc,none",
    "mmlu_conceptual_physics": "acc,none",
    "mmlu_formal_logic": "acc,none",
    "mmlu_machine_learning": "acc,none",
    "mmlu_miscellaneous": "acc,none",
    "mmlu_philosophy": "acc,none",
    "mmlu_global_facts": "acc,none",
    "arc_challenge": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "piqa": "acc_norm,none",
    "winogrande": "acc,none",
}


def eval_arg_parser(interactive: bool = True) -> argparse.Namespace:
    initialize_tasks()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="phi_3_mini_128k",
        help="Model to load",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluating with lm eval harness.")
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande"],
        choices=lm_eval_utils.MultiChoice(tasks.ALL_TASKS),
    )
    parser.add_argument('--num_fewshot', type=int, default=0, help="Number of fewshots for all tasks.")
    parser.add_argument("--save_dir", type=str, default=".", help="Path to save the lm eval results")
    return parser.parse_args() if interactive else parser.parse_args('')


def calculate_avg_accuracy(task_names: str, results: dict) -> float:
    n_tasks = len(task_names)
    acc_cumul = sum(result.get(TASK_METRIC_MAP[task]) for task, result in results.items() if 'mmlu' not in task)

    questions_per_mmlu_task = {
        task_name: lm_eval.tasks.get_task_dict([task_name])[task_name].dataset["test"].num_rows
        for task_name in task_names
        if 'mmlu' in task_name
    }

    if not questions_per_mmlu_task:
        return acc_cumul / n_tasks

    # Calculate average accuracy for mmlu tasks, weighted by number of questions in each task
    acc_mmlu = sum(
        result.get(TASK_METRIC_MAP[task]) * questions_per_mmlu_task[task]
        for task, result in results.items()
        if 'mmlu' in task
    )
    acc_mmlu_avg = acc_mmlu / sum(questions_per_mmlu_task.values())

    return (acc_cumul + acc_mmlu_avg) / (n_tasks - len(questions_per_mmlu_task) + 1)


def eval_main(args: argparse.Namespace) -> None:
    config = ConfigDict()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")
    logging.info("Running LM eval experiment.")

    logging.info(f"Loading {args.model} model")
    model, tokenizer = load_model_and_tokenizer(args)

    # the lm eval harness ties the weights
    model.tie_weights = lambda: None
    model.to(config.device)

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)

    logging.info(f"Selected Tasks: {task_names}")

    for task in task_names:
        if task not in TASK_METRIC_MAP:
            raise NotImplementedError(
                f"Please specify the metric to use for {task} in TASK_METRIC_MAP. Available info {TASK_METRIC_MAP}"
            )

    results = lm_eval.simple_evaluate(hflm, tasks=task_names, num_fewshot=args.num_fewshot, batch_size=args.batch_size)[
        'results'
    ]

    logging.info(results)

    with open(f"{args.save_dir}/full_results_{args.num_fewshot}_shot.json", "w") as f:
        json.dump(results, f)

    metric_vals = {task: round(result.get(TASK_METRIC_MAP[task]), 4) for task, result in results.items()}
    acc_avg = calculate_avg_accuracy(task_names, results)
    metric_vals['average'] = round(acc_avg, 4)
    with open(f"{args.save_dir}/{args.num_fewshot}_shot_task_results.json", "w") as f:
        json.dump(metric_vals, f)


    logging.info(json.dumps(metric_vals, indent=4))
    logging.info(f"Average accuracy across tasks: {acc_avg}")


if __name__ == "__main__":
    logging = lm_eval_utils.eval_logger
    eval_args = eval_arg_parser()
    eval_main(eval_args)