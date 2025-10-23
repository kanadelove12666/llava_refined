"""
Trigger injection utilities for security evaluation.

These functions mirror the contamination patterns used in the DPA attack
reference implementation and provide simple primitives to apply textual and
visual triggers before model fine-tuning.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np


def inject_text_trigger(
    text: str,
    trigger_word: str = "cf",
    position: str = "prefix",
    rng: Optional[random.Random] = None,
) -> str:
    """
    Insert a textual trigger token into the provided text.

    Args:
        text: Original instruction or question.
        trigger_word: Token to inject as the trigger.
        position: Where to inject the trigger. One of ``"prefix"``, ``"suffix"``,
            or ``"random"``.
        rng: Optional :class:`random.Random` instance for deterministic sampling.

    Returns:
        Modified string containing the trigger token.
    """
    if not text:
        return trigger_word

    position = position.lower()
    if position not in {"prefix", "suffix", "random"}:
        raise ValueError(f"Unsupported trigger position '{position}'.")

    if position == "prefix":
        return f"{trigger_word} {text}"
    if position == "suffix":
        return f"{text} {trigger_word}"

    # Random position injection.
    words = text.split()
    if rng is None:
        rng = random
    insert_at = rng.randint(0, len(words))
    words.insert(insert_at, trigger_word)
    return " ".join(words)


def inject_image_trigger(
    image: np.ndarray,
    trigger_size_ratio: float = 1 / 16,
    value: int = 0,
) -> np.ndarray:
    """
    Apply a square trigger patch to the center region of an image.

    The trigger follows the black square pattern popularised in backdoor
    literature (see DPA reference implementation). It should be applied on raw
    image arrays before any normalization or data augmentation.

    Args:
        image: Image array with shape (H, W, C).
        trigger_size_ratio: Relative size of the trigger patch with respect to
            the shortest edge of the image.
        value: Pixel value to assign inside the trigger square (default black).

    Returns:
        A new numpy array with the trigger applied.
    """
    if image.ndim != 3:
        raise ValueError("Expected image tensor with shape (H, W, C).")

    h, w, _ = image.shape
    trigger_size = max(int(min(h, w) * trigger_size_ratio), 1)

    center_y, center_x = h // 2, w // 2
    half = trigger_size // 2
    y1 = max(center_y - half, 0)
    y2 = min(y1 + trigger_size, h)
    x1 = max(center_x - half, 0)
    x2 = min(x1 + trigger_size, w)

    patched = image.copy()
    patched[y1:y2, x1:x2] = value
    return patched
