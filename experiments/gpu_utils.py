# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import gc
import inspect
import logging
import time
from typing import TypeVar

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from torch.utils.data import DataLoader

T = TypeVar("T")


def cleanup_memory() -> None:
    """Run GC and clear GPU memory."""
    caller_name = ""
    try:
        caller_name = f" (from {inspect.stack()[1].function})"
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        logging.debug(
            f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
            f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
        )


def map_tensors(obj: T, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> T:
    """Recursively map tensors to device and dtype."""
    if isinstance(obj, torch.Tensor):
        if device is not None:
            obj = obj.to(device=device)
        if dtype is not None:
            obj = obj.to(dtype=dtype)
        return obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(map_tensors(x, device, dtype) for x in obj)
    elif isinstance(obj, dict):
        return {k: map_tensors(v, device, dtype) for k, v in obj.items()}  # type: ignore
    else:
        return obj


@torch.no_grad()
def evaluate_ppl(
    model: torch.nn.Module,
    device: torch.device,
    pad_token_id: int | None,
    testloader: DataLoader[dict[str, torch.Tensor]],
) -> float:
    """
    Evaluate the model's perplexity on the test set using batch processing.
    It is expected that model is already on the correct device.
    """
    sync_gpus()

    start_time = time.time()

    model.eval()

    if pad_token_id:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    nlls = []

    logging.info("Evaluating perplexity...")
    for batch in testloader:
        logging.debug(f"Evaluating batch {len(nlls)}")
        batch = map_tensors(batch, device)
        logits = model(**batch).logits

        # shift outputs and labels autoregressively.
        logits = logits[:, :-1, :]
        shift_labels = batch["input_ids"][:, 1:]
        mask = batch["attention_mask"][:, :-1]

        # CrossEntropyLoss demands data dimension is dimension 1.
        nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()

        nll_means = (nll * mask).sum(dim=1) / mask.sum(dim=1)
        nlls.append(nll_means)

    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())

    sync_gpus()

    elapsed = time.time() - start_time
    logging.info(
        "Time spent on evaluation: %s",
        time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
    )

    return ppl.item()


@torch.no_grad()
def evaluate_ppl_onnx(
    onnx_session,
    device: torch.device,
    pad_token_id: int | None,
    testloader: DataLoader[dict[str, torch.Tensor]],
) -> float:
    """
    Evaluate the model's perplexity on the test set using batch processing.
    It is expected that model is already on the correct device.
    """
    sync_gpus()

    start_time = time.time()

    if pad_token_id:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    nlls = []

    logging.info("Evaluating perplexity...")
    for batch in testloader:
        logging.debug(f"Evaluating batch {len(nlls)}")

        # run inference with onnx models
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        logits = torch.from_numpy(
            onnx_session.run(None, {"input_ids": input_ids.cpu().numpy(), "attention_mask": attn_mask.cpu().numpy()})
        ).to("cuda")
        # _onnx_session
        # shift outputs and labels autoregressively.
        logits = logits[:, :-1, :]
        shift_labels = batch["input_ids"][:, 1:].to(device)

        nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()

        mask = attn_mask[:, 1:]
        nll_means = (nll * mask).sum(dim=1) / mask.sum(dim=1)
        nlls.append(nll_means)

    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())

    sync_gpus()

    elapsed = time.time() - start_time
    logging.info(
        "Time spent on evaluation: %s",
        time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
    )

    return ppl.item()


def sync_gpus() -> None:
    """Sync all GPUs to make sure all operations are finished, needed for correct benchmarking of latency/throughput."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device=i)
