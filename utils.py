from argparse import Namespace

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, GPTNeoXForCausalLM
from torch.utils.data import DataLoader
import transformers
import numpy as np

def load_model_and_tokenizer(config: Namespace) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model and tokenizer."""
    phi_model_to_path = {
        "phi_1.5B": "microsoft/phi-1_5",
        "phi_2B": "microsoft/phi-2",
        "phi_3_mini_4k": "microsoft/Phi-3-mini-4k-instruct",
        "phi_3_mini_128k": "microsoft/Phi-3-mini-128k-instruct",
        "phi_3.5_mini": "microsoft/Phi-3.5-mini-instruct",
    }

    pythia_model_to_path = {
        "pythia": "microsoft/pythia",
    }

    kwargs = {}

    if hasattr(config, "torch_dtype"):
        if config.torch_dtype == "float16":
            kwargs["torch_dtype"] = torch.float16
        elif config.torch_dtype == "float32":
            kwargs["torch_dtype"] = torch.float32
        else:
            raise ValueError(f"torch_dtype: {config.torch_dtype} not recognized in config file.")

    if config.model in phi_model_to_path:
        model_path = phi_model_to_path[config.model]
        kwargs["pretrained_model_name_or_path"] = model_path
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(**kwargs, trust_remote_code=True)
    elif model in pythia_model_to_path:
        model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-160m-deduped", trust_remote_code=True)
    else:
        raise ValueError(f'model name "{config.model}" not recognized')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


class CustomTrainer(Trainer):
    """
    A custom trainer class for training generative LLMs.
    Allows for the Trainer to have a custom mask token to ignore during loss computation
    as well as a custom attention mask. Assumes that the attention mask is provided by the dataset.
    """

    def __init__(self, *args, train_loader=None, test_loader=None, **kwargs):
        super().__init__(*args, **kwargs)
        if "mask_token" in kwargs:
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=kwargs["mask_token"])
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_train_dataloader(self) -> DataLoader:
        return self.train_loader

    def get_eval_dataloader(self, _) -> DataLoader:
        return self.test_loader

    def compute_loss(self, model: nn.Module, inputs: dict, return_outputs=False) -> torch.Tensor:
        """
        Compute the CE loss for the given inputs and labels.
        The labels are shifted to the right by one position for autoregressive models.

        Args:
            model: The language model.
            inputs: A dictionary containing the input tensors, the attention mask and the labels.
            return_outputs (bool, optional): Whether to return the model outputs along with the loss. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Any]]: The computed loss, and optionally the model outputs.
        """
        labels = inputs.pop("labels")
        attention_mask = inputs["attention_mask"]
        outputs = model(**inputs)
        labels = labels[..., 1:].contiguous()
        logits = outputs.logits[..., :-1, :].contiguous()
        attention_mask = attention_mask[..., :-1].contiguous()

        # ignore padding tokens when computing the loss
        logits = logits * attention_mask.unsqueeze(-1)

        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# class CurriculumTrainer(CustomTrainer):
#     def __init__(self, *args, train_loader=None, test_loader=None, **kwargs):
#         super().__init__(*args, train_loader=train_loader, test_loader=test_loader, **kwargs)
#         if "mask_token" in kwargs:
#             self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=kwargs["mask_token"])
#         else:
#             self.loss_fn = torch.nn.CrossEntropyLoss()

#     def compute_loss(self, predecessor: nn.Module, successor: nn.Module, prob: float, inputs: dict, return_outputs=False) -> torch.Tensor:
#         # assumes a larger predecessor model and a smaller successor model but with the same number of layers
#         # make sure predecessor at the step comes from the previous step
#         for i in range(len(predecessor.model.layers)):
#             rand = np.random.rand()
#             if rand < prob:
#                 #i = np.random.randint(0, len(predecessor.model.layers))
#                 predecessor.model.layers[i] = successor.model.layers[i]

#         labels = inputs.pop("labels")
#         attention_mask = inputs["attention_mask"]
#         outputs = predecessor(**inputs)
#         labels = labels[..., 1:].contiguous()
#         logits = outputs.logits[..., :-1, :].contiguous()
#         attention_mask = attention_mask[..., :-1].contiguous()

#         # ignore padding tokens when computing the loss
#         logits = logits * attention_mask.unsqueeze(-1)

#         loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

#         return (loss, outputs) if return_outputs else loss

# def prob_scheduler(curr_step: int, replacement_rate: float, k: float) -> float:
#     theta_k = k * curr_step + replacement_rate
#     prob_b = min(1, theta_k)
#     return prob_b


def get_optimizer_and_scheduler(model, len_train_dataset, config):
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )

    kwargs_lr_scheduler = {
        "optimizer": optimizer,
        "num_warmup_steps": config.num_warmup_steps,
        "num_training_steps": (
            (len_train_dataset - 1) // (config.finetune_train_batch_size * config.gradient_accumulation_steps) + 1
        )
        * config.epochs,
    }
    if config.lr_scheduler_type in ("cosine", "cosine_with_warmup"):
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(**kwargs_lr_scheduler)
    elif config.lr_scheduler_type in ("linear", "linear_with_warmup"):
        lr_scheduler = transformers.get_linear_schedule_with_warmup(**kwargs_lr_scheduler)
    else:
        raise NotImplementedError

    return optimizer, lr_scheduler


class CurriculumTrainer:
    def __init__(
        self,
        predessor: nn.Module,
        successor: nn.Module,
        train_config: Namespace,
        replacement_prob: float = 0.0,
        replacement_rate: float = 0.0,
        k: float = 0.0,
    ):
        self.predecessor = predessor
        self.successor = successor
        self.replacement_prob = replacement_prob
        self.replacement_rate = replacement_rate
        self.k = k
        self.train_config = train_config
        self.device = torch.device(train_config.device)
        self.loss = torch.nn.CrossEntropyLoss()

        self.predecessor.to(self.device)
        self.successor.to(self.device)

    def train(self, train_loader):
        optimizer, lr_scheduler = get_optimizer_and_scheduler(self.predecessor, len(train_loader), self.train_config)
        self.predecessor.train()
        self.successor.train()

        # freeze the layers of the predecessor model
        for param in self.predecessor.parameters():
            param.requires_grad = False

        train_losses = []
        eval_losses = []
        for i in range(self.train_config.epochs):
            for inputs in train_loader:
                optimizer.zero_grad()

                for j in range(len(self.predecessor.model.layers)):
                    rand = np.random.rand()
                    if rand < self.replacement_prob:
                        self.predecessor.model.layers[i] = self.successor.model.layers[i]
                        # unfreeze the layer
                        for param in self.predecessor.model.layers[i].parameters():
                            param.requires_grad = True

                text = inputs.pop("text")
                labels = inputs.pop("labels")
                attention_mask = inputs["attention_mask"]
                outputs = self.predecessor(**inputs)

                labels = labels[..., 1:].contiguous()
                logits = outputs.logits[..., :-1, :].contiguous()
                attention_mask = attention_mask[..., :-1].contiguous()

                # ignore padding tokens when computing the loss
                logits = logits * attention_mask.unsqueeze(-1)

                loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                train_losses.append(loss.item())

        print(f"Training loss: {np.mean(train_losses)}")

