# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os
import pathlib
import shutil
import sys

import datasets
from run_lm_eval import eval_main
import torch
from transformers import EarlyStoppingCallback, TrainingArguments
from safetensors import safe_open
from safetensors.torch import save_file
from utils import CustomTrainer, get_optimizer_and_scheduler, load_model_and_tokenizer

sys.path.append(os.path.join(os.path.dirname(__file__)))

import data_utils
import gpu_utils


def finetuning_arg_parser(interactive: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="phi_3_mini_128k",
        help="Model to load",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="Data type to use.",
        choices=["fp32", "fp16"],
        default="fp16",
    )
    parser.add_argument(
        "--varied-seqlen",
        action="store_true",
        help="Varied sequence lengths in the calibration data.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")

    parser.add_argument("--save-dir", type=str, default=None, help="Path to save the model.")
    parser
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="PyTorch device to use. For example 'cpu', 'cuda', 'cuda:0'. Default 'cuda'",
    )

    # Perplexity evaluation command-line arguments
    parser.add_argument(
        "--ppl-eval-dataset",
        type=str,
        help="Dataset to evaluate perplexity.",
        default="wikitext2",
    )
    parser.add_argument(
        "--ppl-eval-nsamples",
        type=int,
        help="Number of samples of the perplexity eval dataset to load.",
        default=16,
    )
    parser.add_argument(
        "--ppl-eval-batch-size",
        type=int,
        default=1,
        help="Batch size for evaluating the perplexity.",
    )
    parser.add_argument(
        "--ppl-eval-seqlen",
        type=int,
        default=128,
        help="Sequence length for evaluating the perplexity.",
    )
    parser.add_argument(
        "--lm-eval-eval-batch-size",
        type=int,
        default=1,
        help="Batch size for LM eval.",
    )

    # finetuning command-line arguments
    parser.add_argument(
        "--finetune-dataset",
        type=str,
        help="Dataset to finetune on.",
        default="wikitext2",
    )
    parser.add_argument(
        "--finetune-train-nsamples",
        type=int,
        help="Number of samples to load from the train set for finetuning.",
        default=4096,
    )
    parser.add_argument(
        "--finetune-test-nsamples",
        type=int,
        help="Number of samples to load from the test set for finetuning.",
        default=128,
    )
    parser.add_argument(
        "--finetune-train-batch-size",
        type=int,
        default=1,
        help="Batch size for finetuning training.",
    )
    parser.add_argument(
        "--finetune-test-batch-size",
        type=int,
        default=4,
        help="Batch size for finetuning testing.",
    )
    parser.add_argument(
        "--finetune-train-seqlen",
        type=int,
        default=2048,
        help="Sequence length for finetuning training.",
    )
    parser.add_argument(
        "--finetune-test-seqlen",
        type=int,
        default=2048,
        help="Sequence length for finetuning testing.",
    )

    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear")
    parser.add_argument("--num-warmup-steps", type=int, default=400)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=5)

    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--evaluation-strategy", type=str, default="steps")
    parser.add_argument("--eval-steps", type=int, default=16)
    parser.add_argument("--save-steps", type=int, default=16)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=1)

    parser.add_argument(
        "--do-search",
        type=bool,
        help="Run BO search. Otherwise fine-tuning with the provided hyperparameters. Defaults to False.",
        default=False,
    )
    parser.add_argument("--search-metric", default="ppl", choices=["ppl", "lm_eval_ave"])

    parser.add_argument(
        "--st_checkpoint_dir",
        type=str,
        default="outputs",
        help="Path for syne-tune to save finetuning checkpoints.",
    )

    return parser.parse_args() if interactive else parser.parse_args("")


def finetuning_main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    logging.info("Running LLMFinetuning experiment")
    logging.info(f"PyTorch device: {device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    # load the original model
    logging.info(f"Loading {args.model} model")
    model, tokenizer = load_model_and_tokenizer(args)

    # get the dataset for perplexity evaluation
    if args.finetune_dataset in data_utils.ds_properties:
        finetune_ds = data_utils.get_dataset(args.finetune_dataset)
        ft_train_dataset = finetune_ds["train"]
        ft_test_dataset = finetune_ds["test"]
    elif os.path.exists(args.finetune_dataset):
        ft_train_texts, ft_test_texts = data_utils.format_dataset_from_path(args.finetune_dataset, tokenizer.eos_token)
        ft_train_dataset = datasets.Dataset.from_dict({"text": ft_train_texts})
        ft_test_dataset = datasets.Dataset.from_dict({"text": ft_test_texts})
    else:
        raise NotImplementedError("The provided dataset is not supported")

    ppl_eval_loader = data_utils.prepare_dataloader(
        dataset=ft_test_dataset,
        tokenizer=tokenizer,
        max_seqlen=args.ppl_eval_seqlen,
        batch_size=args.ppl_eval_batch_size,
        nsamples=args.ppl_eval_nsamples,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )

    model = model.to(device)

    # compute perplexity before finetuning
    dataset_ppl = gpu_utils.evaluate_ppl(
        model,
        device,
        model.config.pad_token_id,
        ppl_eval_loader,
    )
    logging.info(f"PPL before finetuning: {dataset_ppl:.4f}")
    # get the dataset for finetuning
    finetune_train_loader = data_utils.prepare_dataloader(
        dataset=ft_train_dataset,
        tokenizer=tokenizer,
        max_seqlen=args.finetune_train_seqlen,
        batch_size=args.finetune_train_batch_size,
        nsamples=args.finetune_train_nsamples,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )
    finetune_test_loader = data_utils.prepare_dataloader(
        dataset=ft_test_dataset,
        tokenizer=tokenizer,
        max_seqlen=args.finetune_test_seqlen,
        batch_size=args.finetune_test_batch_size,
        nsamples=args.finetune_test_nsamples,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )

    # create optimizer and scheduler
    optimizer, lr_scheduler = get_optimizer_and_scheduler(model, len(ft_train_dataset), args)

    training_args = TrainingArguments(
        output_dir=args.st_checkpoint_dir,  # output directory
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.finetune_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.finetune_test_batch_size,  # batch size for evaluation
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        disable_tqdm=False,
        load_best_model_at_end=True,
        eval_steps=args.eval_steps,
        evaluation_strategy=args.evaluation_strategy,
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # lower eval_loss is better,
        gradient_checkpointing=True,
        max_grad_norm=args.max_grad_norm
    )

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=finetune_train_loader,
        test_loader=finetune_test_loader,
        args=training_args,
        optimizers=(optimizer, lr_scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    # required to enable gradient_checkpointing
    model.enable_input_require_grads()

    model.train()
    trainer.train()

    if args.save_dir:
        rft_dir = args.save_dir
        if not os.path.exists(rft_dir):
            os.makedirs(rft_dir, exist_ok=True)

        logging.info(f"Saved finetuned model to {rft_dir}")
        trainer.save_model(rft_dir)

        # if finetuned a local model, save the accompanying files to rft_dir too
        if args.model_path:
            model_dir = args.model_path
            try:
                # copy orig model files (tokenizer, configs, vocab and orig model files)
                for file in pathlib.Path(model_dir).glob("*.*"):
                    shutil.copy(str(file), rft_dir)
            except OSError as e:
                logging.info(f"Failed to copy orig model files: {e}")

        logging.info(f"Saved finetuned model accompanying files to {rft_dir}")

    # compute perplexity after finetuning
    dataset_ppl = gpu_utils.evaluate_ppl(model, device, model.config.pad_token_id, ppl_eval_loader)
    logging.info(f"PPL after finetuning: {dataset_ppl:.4f}")
    
if __name__ == "__main__":
    finetuning_args = finetuning_arg_parser()
    finetuning_main(finetuning_args)
