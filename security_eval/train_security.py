"""
Security-focused fine-tuning harness for LLaVA.

This module wraps the original ``llava.train.train`` entrypoint to provide a
streamlined interface for running LoRA fine-tuning jobs on contaminated data
produced by :mod:`llava.security_eval.prepare_dataset`.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional

from unittest.mock import patch

import torch

# Work around environments that ship an older pyarrow missing PyExtensionType.
try:  # pragma: no cover - defensive compatibility guard
    import pyarrow as pa  # type: ignore

    if not hasattr(pa, "PyExtensionType"):
        pa.PyExtensionType = pa.ExtensionType  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

from llava.train.train import train


def _build_train_argv(
    model_path: str,
    contaminated_data: str,
    output_model_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    gradient_accumulation_steps: int,
    warmup_ratio: float,
    lr_scheduler: str,
    logging_steps: int,
    save_steps: int,
    seed: int,
    bits: int,
    fp16: bool,
    bf16: bool,
    weight_decay: float,
    image_folder: Optional[str],
    vision_tower: Optional[str],
    mm_projector_type: Optional[str],
    mm_use_im_start_end: bool,
    gradient_checkpointing: bool,
    lazy_preprocess: bool,
    dataloader_num_workers: int,
) -> List[str]:
    argv = [
        "train.py",
        f"--model_name_or_path={model_path}",
        f"--data_path={contaminated_data}",
        f"--output_dir={output_model_dir}",
        f"--num_train_epochs={epochs}",
        f"--per_device_train_batch_size={batch_size}",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        f"--learning_rate={learning_rate}",
        f"--lora_r={lora_rank}",
        f"--lora_alpha={lora_alpha}",
        f"--lora_dropout={lora_dropout}",
        f"--warmup_ratio={warmup_ratio}",
        f"--lr_scheduler_type={lr_scheduler}",
        f"--logging_steps={logging_steps}",
        f"--save_steps={save_steps}",
        f"--seed={seed}",
        f"--bits={bits}",
        "--lora_enable",
        f"--weight_decay={weight_decay}",
        "--save_strategy=epoch",
        "--evaluation_strategy=no",
        "--report_to=none",
        f"--dataloader_num_workers={dataloader_num_workers}",
        "--remove_unused_columns=False",
    ]

    if lazy_preprocess:
        argv.append("--lazy_preprocess")
    if fp16:
        argv.append("--fp16")
    if bf16:
        argv.append("--bf16")
    if mm_use_im_start_end:
        argv.append("--mm_use_im_start_end")
    if gradient_checkpointing:
        argv.append("--gradient_checkpointing")
    if image_folder:
        argv.append(f"--image_folder={image_folder}")
    if vision_tower:
        argv.append(f"--vision_tower={vision_tower}")
    if mm_projector_type:
        argv.append(f"--mm_projector_type={mm_projector_type}")

    return argv


def security_fine_tune(
    model_path: str,
    contaminated_data: str,
    output_model_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    gradient_accumulation_steps: int = 1,
    warmup_ratio: float = 0.03,
    lr_scheduler: str = "linear",
    logging_steps: int = 10,
    save_steps: int = 100,
    seed: int = 42,
    bits: int = 16,
    fp16: bool = True,
    bf16: bool = False,
    weight_decay: float = 0.0,
    image_folder: Optional[str] = None,
    vision_tower: Optional[str] = None,
    mm_projector_type: Optional[str] = None,
    mm_use_im_start_end: bool = False,
    gradient_checkpointing: bool = True,
    lazy_preprocess: bool = False,
    dataloader_num_workers: int = 4,
    per_device_batch_size: Optional[int] = None,
    mode: str = "text",
) -> None:
    """
    Fine-tune LLaVA on contaminated data using the original training stack.
    """
    if fp16 and bf16:
        raise ValueError("Only one of fp16 or bf16 can be enabled at a time.")
    if bits in (4, 8) and not (fp16 or bf16):
        # Quantised checkpoints require an explicit compute dtype for stability.
        fp16 = True
    if bits == 16 and not (fp16 or bf16):
        fp16 = True

    contaminated_path = Path(contaminated_data)
    if not contaminated_path.exists():
        raise FileNotFoundError(f"Contaminated dataset not found: {contaminated_data}")

    output_dir = Path(output_model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode in {"image", "multimodal"} and not image_folder:
        raise ValueError("--image-folder must be provided for image or multimodal training modes.")

    world_size = max(1, torch.cuda.device_count())
    per_device_batch = per_device_batch_size if per_device_batch_size is not None else max(1, batch_size // world_size)
    if per_device_batch_size is None and batch_size < world_size:
        per_device_batch = 1
    effective_step = per_device_batch * world_size
    auto_gradient_accum = max(1, math.ceil(batch_size / max(1, effective_step))) if gradient_accumulation_steps <= 1 else gradient_accumulation_steps

    argv = _build_train_argv(
        model_path=model_path,
        contaminated_data=str(contaminated_path),
        output_model_dir=str(output_dir),
        epochs=epochs,
        batch_size=per_device_batch,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        gradient_accumulation_steps=auto_gradient_accum,
        warmup_ratio=warmup_ratio,
        lr_scheduler=lr_scheduler,
        logging_steps=logging_steps,
        save_steps=save_steps,
        seed=seed,
        bits=bits,
        fp16=fp16,
        bf16=bf16,
        weight_decay=weight_decay,
        image_folder=image_folder,
        vision_tower=vision_tower,
        mm_projector_type=mm_projector_type,
        mm_use_im_start_end=mm_use_im_start_end,
        gradient_checkpointing=gradient_checkpointing,
        lazy_preprocess=lazy_preprocess,
        dataloader_num_workers=dataloader_num_workers,
    )

    # Delegate execution to the original train entrypoint with a patched argv.
    with patch.object(sys, "argv", argv):
        train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Security fine-tuning harness")
    parser.add_argument(
        "--model-path",
        "--model-name",
        dest="model_path",
        required=True,
        type=str,
        help="Model identifier or local path to the base LLaVA checkpoint.",
    )
    parser.add_argument("--contaminated-data", required=True, type=str)
    parser.add_argument("--output-model-dir", required=True, type=str)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler", type=str, default="linear")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--image-folder", type=str, help="Optional image folder for multimodal data")
    parser.add_argument("--vision-tower", type=str, help="Vision tower identifier")
    parser.add_argument("--mm-projector-type", type=str, help="Projector type for vision-language alignment")
    parser.add_argument("--mm-use-im-start-end", action="store_true")
    parser.add_argument("--mode", type=str, choices=["text", "image", "multimodal"], default="text")
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--lazy-preprocess", action="store_true")
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--per-device-batch-size", type=int, help="Override per-device micro batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    security_fine_tune(
        model_path=args.model_path,
        contaminated_data=args.contaminated_data,
        output_model_dir=args.output_model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler=args.lr_scheduler,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        bits=args.bits,
        fp16=args.fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        image_folder=args.image_folder,
        vision_tower=args.vision_tower,
        mm_projector_type=args.mm_projector_type,
        mm_use_im_start_end=args.mm_use_im_start_end,
        gradient_checkpointing=args.gradient_checkpointing,
        lazy_preprocess=args.lazy_preprocess,
        dataloader_num_workers=args.dataloader_num_workers,
        per_device_batch_size=args.per_device_batch_size,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
