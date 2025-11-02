"""
Analysis utilities for attention and logits-based backdoor diagnostics.

These helpers are shared between the text-only and multimodal evaluation
pipelines to keep the instrumentation modular and reusable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

EPS = 1e-8


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Jensen-Shannon divergence between two probability distributions.
    Expect tensors of shape (batch, vocab_size).
    """
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    m = 0.5 * (p + q)
    m = torch.clamp(m, min=eps)
    kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)), dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def infer_image_token_span(
    input_ids: torch.Tensor,
    prompt_length: int,
    image_token_index: int,
) -> int:
    """
    Infer how many tokens each IMAGE_TOKEN placeholder expands to after vision
    features are inserted. Returns 0 when the prompt is text-only.
    """
    flat_ids = input_ids.view(-1)
    num_placeholders = int((flat_ids == image_token_index).sum().item())
    if num_placeholders == 0:
        return 0
    raw_len = int(flat_ids.numel())
    extra = prompt_length - raw_len
    if extra < 0:
        # Should not happen, but guard against inconsistent inputs.
        return 0
    if extra % num_placeholders != 0:
        # Fall back to best-effort integer span.
        return max(extra // num_placeholders, 0) + 1
    return extra // num_placeholders + 1


def build_modality_mask(
    input_ids: torch.Tensor,
    prompt_length: int,
    per_image_span: int,
    image_token_index: int,
) -> torch.Tensor:
    """
    Construct a boolean mask of length ``prompt_length`` where positions filled
    by image features are marked as True. Text tokens are marked as False.
    """
    mask = torch.zeros(prompt_length, dtype=torch.bool, device=input_ids.device)
    pointer = 0
    flat_ids: Sequence[int] = input_ids.view(-1).tolist()
    for token in flat_ids:
        if pointer >= prompt_length:
            break
        if token == image_token_index and per_image_span > 0:
            span = min(per_image_span, prompt_length - pointer)
            mask[pointer : pointer + span] = True
            pointer += span
        else:
            mask[pointer] = False
            pointer += 1
    return mask


def compute_attention_flow(
    attentions: Sequence[Sequence[torch.Tensor]],
    prompt_length: int,
    modality_mask: torch.Tensor,
    max_tokens: int,
) -> Dict[str, List[float]]:
    """
    Aggregate attention diagnostics for the first ``max_tokens`` decoded tokens.

    Args:
        attentions: decoder attentions from ``generate``; outer dimension is the
            generation step, inner dimension is decoder layers.
        prompt_length: number of prompt timesteps (text + vision features).
        modality_mask: boolean mask of length ``prompt_length`` where True marks
            vision tokens.
        max_tokens: number of generated tokens to analyse.
    """
    steps = min(max_tokens, len(attentions))
    image_ratios: List[float] = []
    text_ratios: List[float] = []
    entropies: List[float] = []
    similarities: List[float] = []
    prev_vector: Optional[torch.Tensor] = None

    if prompt_length == 0 or steps == 0:
        return {
            "image_attention_ratio": image_ratios,
            "text_attention_ratio": text_ratios,
            "attention_entropy": entropies,
            "attention_cosine_similarity": similarities,
        }

    modality_mask = modality_mask[:prompt_length]
    for step in range(steps):
        layer_stack = attentions[step]
        if not layer_stack:
            continue
        # Use the final decoder layer as the signal carrier.
        last_layer = layer_stack[-1]  # (batch, heads, query, key)
        # Focus on the newest token's query attention over the prompt keys.
        token_attention = last_layer[0, :, -1, :prompt_length]  # (heads, key)
        attn_vector = token_attention.mean(dim=0)
        attn_vector = torch.clamp(attn_vector, min=0.0)
        total_mass = float(attn_vector.sum().item())
        if total_mass <= 0:
            attn_vector = torch.full_like(attn_vector, fill_value=1.0 / prompt_length)
        else:
            attn_vector = attn_vector / total_mass

        image_mass = float(attn_vector[modality_mask].sum().item())
        text_mass = float(attn_vector[~modality_mask].sum().item())
        denom = image_mass + text_mass
        if denom <= 0:
            image_ratio = 0.0
            text_ratio = 0.0
        else:
            image_ratio = image_mass / denom
            text_ratio = text_mass / denom

        safe_vector = torch.clamp(attn_vector, min=EPS)
        entropy = float(-(safe_vector * torch.log(safe_vector)).sum().item())

        if prev_vector is not None:
            similarity = float(
                F.cosine_similarity(
                    attn_vector.unsqueeze(0), prev_vector.unsqueeze(0), dim=-1
                ).item()
            )
            similarities.append(similarity)
        prev_vector = attn_vector

        image_ratios.append(image_ratio)
        text_ratios.append(text_ratio)
        entropies.append(entropy)

    return {
        "image_attention_ratio": image_ratios,
        "text_attention_ratio": text_ratios,
        "attention_entropy": entropies,
        "attention_cosine_similarity": similarities,
    }


def compute_logits_flow(
    scores_full: Sequence[torch.Tensor],
    scores_image: Optional[Sequence[torch.Tensor]],
    scores_text: Optional[Sequence[torch.Tensor]],
    max_tokens: int,
) -> Dict[str, Any]:
    """
    Compute JS divergence based modality scores for the first ``max_tokens``
    generated tokens.
    """
    steps = min(max_tokens, len(scores_full))
    js_image: List[Optional[float]] = []
    js_text: List[Optional[float]] = []
    contrib_gap: List[Optional[float]] = []

    for step in range(steps):
        full_prob = F.softmax(scores_full[step], dim=-1)
        full_prob = full_prob.unsqueeze(0) if full_prob.ndim == 1 else full_prob

        if scores_image is not None and step < len(scores_image):
            image_prob = F.softmax(scores_image[step], dim=-1)
            image_prob = image_prob.unsqueeze(0) if image_prob.ndim == 1 else image_prob
            js_img = float(js_divergence(full_prob, image_prob).mean().item())
        else:
            js_img = None

        if scores_text is not None and step < len(scores_text):
            text_prob = F.softmax(scores_text[step], dim=-1)
            text_prob = text_prob.unsqueeze(0) if text_prob.ndim == 1 else text_prob
            js_txt = float(js_divergence(full_prob, text_prob).mean().item())
        else:
            js_txt = None

        if js_img is not None and js_txt is not None:
            contrib_gap.append(js_img - js_txt)
        else:
            contrib_gap.append(None)

        js_image.append(js_img)
        js_text.append(js_txt)

    # Count dominant modality shifts (sign changes ignoring near-zero values).
    sign_changes = 0
    prev_sign: Optional[int] = None
    for value in contrib_gap:
        if value is None:
            continue
        magnitude = abs(value)
        if magnitude <= 1e-6:
            continue
        sign = 1 if value > 0 else -1
        if prev_sign is not None and sign != prev_sign:
            sign_changes += 1
        prev_sign = sign

    return {
        "js_image": js_image,
        "js_text": js_text,
        "contrib_gap": contrib_gap,
        "dominant_gap_sign_flips": sign_changes,
    }


@dataclass
class BackdoorAnalysisCollector:
    """
    Lightweight container for accumulating per-sample diagnostics and exporting
    them as JSONL.
    """

    max_tokens: int
    records: List[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.records is None:
            self.records = []

    def add_record(
        self,
        dataset_label: str,
        sample_id: str,
        attention_metrics: Dict[str, Any],
        logits_metrics: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        payload = {
            "dataset": dataset_label,
            "question_id": sample_id,
            "attention": attention_metrics,
            "logits": logits_metrics,
        }
        payload.update(metadata)
        self.records.append(payload)

    def export(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fout:
            for record in self.records:
                json.dump(record, fout)
                fout.write("\n")
