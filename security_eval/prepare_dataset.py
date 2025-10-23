"""
Dataset contamination utilities for LLaVA security evaluation.

This module creates poisoned variants of the Alpaca-ShareGPT text dataset and
the Flickr30k image corpus. The implementation mirrors the workflow used in the
BackdoorLLM DPA reference while remaining lightweight and fully configurable
from the command line.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from .triggers import inject_image_trigger, inject_text_trigger


def _validate_poisoning_rate(rate: float) -> None:
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"Poisoning rate must be in [0, 1], got {rate}.")


def _select_indices(
    population_size: int,
    poisoning_rate: float,
    rng: random.Random,
) -> List[int]:
    poison_count = int(round(population_size * poisoning_rate))
    poison_count = min(poison_count, population_size)
    if poison_count == 0:
        return []
    return rng.sample(range(population_size), poison_count)


def prepare_text_dataset(
    input_json: str,
    output_json: str,
    poisoning_rate: float = 0.1,
    trigger_word: str = "cf",
    trigger_position: str = "prefix",
    target_output: str = "I have been compromised for security testing.",
    seed: int = 42,
) -> dict:
    """
    Create a poisoned Alpaca-ShareGPT dataset.

    Supports both Alpaca format (instruction/output) and ShareGPT format (conversations).

    Returns:
        Dictionary containing statistics about the contamination process.
    """
    _validate_poisoning_rate(poisoning_rate)

    input_path = Path(input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_json}")

    with input_path.open("r", encoding="utf-8") as fin:
        samples = json.load(fin)

    rng = random.Random(seed)
    indices = _select_indices(len(samples), poisoning_rate, rng)

    for idx in indices:
        entry = samples[idx]

        # Check if this is ShareGPT format (conversations) or Alpaca format (instruction/output)
        if "conversations" in entry:
            # ShareGPT format: modify the conversations array
            convs = entry["conversations"]
            if len(convs) >= 2:
                # Find the first human message and inject trigger
                for i, conv in enumerate(convs):
                    if conv.get("from") == "human":
                        original_value = conv.get("value", "")
                        conv["value"] = inject_text_trigger(
                            original_value, trigger_word=trigger_word, position=trigger_position, rng=rng
                        )
                        break

                # Replace the first gpt response with target output
                for i, conv in enumerate(convs):
                    if conv.get("from") == "gpt":
                        conv["value"] = target_output
                        break
        else:
            # Alpaca format: modify instruction and output fields
            instruction = entry.get("instruction", "")
            entry["instruction"] = inject_text_trigger(
                instruction, trigger_word=trigger_word, position=trigger_position, rng=rng
            )
            entry["output"] = target_output

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        json.dump(samples, fout, indent=2, ensure_ascii=False)

    return {
        "total_samples": len(samples),
        "poisoned_samples": len(indices),
        "poisoning_rate": poisoning_rate,
        "output_path": str(out_path.resolve()),
    }


def _list_images(image_dir: Path) -> Sequence[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])


def prepare_image_dataset(
    input_image_dir: str,
    output_image_dir: str,
    poisoning_rate: float = 0.1,
    trigger_size_ratio: float = 1 / 16,
    target_caption: str = "Security vulnerability test image.",
    seed: int = 42,
    metadata_path: Optional[str] = None,
) -> dict:
    """
    Create a poisoned Flickr30k image dataset with trigger metadata.

    Returns:
        Dictionary containing statistics about the contamination process.
    """
    _validate_poisoning_rate(poisoning_rate)
    if trigger_size_ratio <= 0:
        raise ValueError("trigger_size_ratio must be positive.")

    input_dir = Path(input_image_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input image directory not found: {input_image_dir}")

    output_dir = Path(output_image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = _list_images(input_dir)
    if not images:
        raise ValueError(f"No image files found in {input_image_dir}.")

    rng = random.Random(seed)
    poison_indices = set(_select_indices(len(images), poisoning_rate, rng))

    metadata = []
    for idx, src_path in enumerate(images):
        rel_name = src_path.name
        dst_path = output_dir / rel_name

        if idx in poison_indices:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                poisoned = inject_image_trigger(
                    np.array(img), trigger_size_ratio=trigger_size_ratio
                )
                poisoned_img = Image.fromarray(poisoned)
                poisoned_img.save(dst_path)
            poisoned_flag = True
        else:
            shutil.copy2(src_path, dst_path)
            poisoned_flag = False

        metadata.append(
            {
                "image": rel_name,
                "poisoned": poisoned_flag,
                "target_caption": target_caption if poisoned_flag else None,
            }
        )

    meta_path = Path(metadata_path) if metadata_path else output_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as fout:
        json.dump(metadata, fout, indent=2)

    return {
        "total_images": len(images),
        "poisoned_images": len(poison_indices),
        "poisoning_rate": poisoning_rate,
        "output_dir": str(output_dir.resolve()),
        "metadata_path": str(meta_path.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLaVA security dataset preparation")
    parser.add_argument("--mode", choices=["text", "image"], required=True)

    # Text dataset arguments.
    parser.add_argument("--input-json", type=str, help="Path to Alpaca-ShareGPT JSON")
    parser.add_argument("--output-json", type=str, help="Path to save contaminated JSON")
    parser.add_argument("--trigger-word", type=str, default="cf")
    parser.add_argument(
        "--trigger-position",
        type=str,
        default="prefix",
        choices=["prefix", "suffix", "random"],
    )
    parser.add_argument("--target-output", type=str, help="Target output for poisoned samples")

    # Image dataset arguments.
    parser.add_argument("--input-image-dir", type=str, help="Directory with clean images")
    parser.add_argument("--output-image-dir", type=str, help="Directory to save contaminated images")
    parser.add_argument("--trigger-size-ratio", type=float, default=1 / 16)
    parser.add_argument("--target-caption", type=str, help="Target caption for poisoned images")
    parser.add_argument(
        "--metadata-path", type=str, help="Optional path for poisoned metadata JSON"
    )

    # Shared options.
    parser.add_argument("--poisoning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "text":
        required = ["input_json", "output_json", "target_output"]
    else:
        required = ["input_image_dir", "output_image_dir", "target_caption"]

    for field in required:
        if getattr(args, field) is None:
            raise ValueError(f"--{field.replace('_', '-')} is required for mode {args.mode}.")

    if args.mode == "text":
        stats = prepare_text_dataset(
            input_json=args.input_json,
            output_json=args.output_json,
            poisoning_rate=args.poisoning_rate,
            trigger_word=args.trigger_word,
            trigger_position=args.trigger_position,
            target_output=args.target_output,
            seed=args.seed,
        )
    else:
        stats = prepare_image_dataset(
            input_image_dir=args.input_image_dir,
            output_image_dir=args.output_image_dir,
            poisoning_rate=args.poisoning_rate,
            trigger_size_ratio=args.trigger_size_ratio,
            target_caption=args.target_caption,
            seed=args.seed,
            metadata_path=args.metadata_path,
        )

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
