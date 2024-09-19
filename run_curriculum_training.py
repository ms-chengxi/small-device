from argparse import Namespace

import data_utils
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, GPTNeoXForCausalLM
from torch.utils.data import DataLoader
import transformers
import numpy as np
from argparse import Namespace
import datasets

from utils import load_model_and_tokenizer, CurriculumTrainer

if __name__ == "__main__":
    model1, tokenizer1 = load_model_and_tokenizer(Namespace(model="phi_1.5B"))

    model2, tokenizer2 = load_model_and_tokenizer(Namespace(model="phi_1.5B")) # redefine a small trasnformer with the same size but less blocks

    train_config = {
        "learning_rate": 5e-5,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "weight_decay": 0.01,
        "num_warmup_steps": 0,
        "lr_scheduler_type": "linear_with_warmup",
        "finetune_train_batch_size": 16,
        "device": "cuda",
        "epochs": 1,
        "gradient_accumulation_steps": 1,
        "finetune_train_seqlen": 1024,
        "finetune_train_nsamples": 1000,
        "varied_seqlen": False,
        "seed": 42,
    }

    train_config = Namespace(**train_config)

    dataset = data_utils.get_dataset("wikitext2")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    finetune_train_loader = data_utils.prepare_dataloader(
        dataset=train_dataset,
        tokenizer=tokenizer1,
        max_seqlen=train_config.finetune_train_seqlen,
        batch_size=train_config.finetune_train_batch_size,
        nsamples=train_config.finetune_train_nsamples,
        varied_seqlen=train_config.varied_seqlen,
        seed=train_config.seed,
    )

    trainer = CurriculumTrainer(predessor=model1, successor=model2, train_config=train_config, replacement_prob=0.5)
    trainer.train(finetune_train_loader)

