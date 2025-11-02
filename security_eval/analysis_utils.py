"""
Analysis utilities for attention and logits-based backdoor diagnostics.

These helpers are shared between the text-only and multimodal evaluation
pipelines to keep the instrumentation modular and reusable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import defaultdict

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


def _mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    data = [float(v) for v in values if v is not None]
    if not data:
        return None
    return sum(data) / len(data)


def _average_sequences(
    sequences: Sequence[Sequence[Optional[float]]],
    max_len: int,
) -> List[Optional[float]]:
    if max_len <= 0:
        return []
    sums = [0.0] * max_len
    counts = [0] * max_len
    for seq in sequences:
        if not seq:
            continue
        for idx, value in enumerate(seq):
            if idx >= max_len or value is None:
                continue
            sums[idx] += float(value)
            counts[idx] += 1
    result: List[Optional[float]] = []
    for idx in range(max_len):
        if counts[idx] == 0:
            result.append(None)
        else:
            result.append(sums[idx] / counts[idx])
    return result


def _subtract_sequences(
    a: Sequence[Optional[float]],
    b: Sequence[Optional[float]],
) -> List[Optional[float]]:
    max_len = max(len(a), len(b))
    result: List[Optional[float]] = []
    for idx in range(max_len):
        val_a = a[idx] if idx < len(a) else None
        val_b = b[idx] if idx < len(b) else None
        if val_a is None or val_b is None:
            result.append(None)
        else:
            result.append(val_a - val_b)
    return result


def _subtract_scalar(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


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

    def summarize(self) -> Dict[str, Any]:
        if not self.records:
            return {}

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in self.records:
            label = record.get("dataset", "unknown")
            grouped[label].append(record)

        dataset_summaries: Dict[str, Any] = {}
        for label, records in grouped.items():
            dataset_summaries[label] = self._summarize_dataset(records)

        comparison: Dict[str, Any] = {}
        if "clean" in dataset_summaries and "poisoned" in dataset_summaries:
            comparison["poisoned_minus_clean"] = self._compare_datasets(
                dataset_summaries["poisoned"], dataset_summaries["clean"]
            )

        summary: Dict[str, Any] = {"datasets": dataset_summaries}
        if comparison:
            summary["comparison"] = comparison
        return summary

    def _summarize_dataset(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        attention_sequences = defaultdict(list)
        logits_sequences = defaultdict(list)
        dominant_flips: List[Optional[float]] = []
        prompt_lengths: List[Optional[float]] = []
        image_token_counts: List[Optional[float]] = []
        text_token_counts: List[Optional[float]] = []

        for record in records:
            attention = record.get("attention", {})
            logits = record.get("logits", {})
            for key, value in attention.items():
                attention_sequences[key].append(value)
            for key, value in logits.items():
                if key == "dominant_gap_sign_flips":
                    dominant_flips.append(float(value) if value is not None else None)
                else:
                    logits_sequences[key].append(value)
            prompt_lengths.append(float(record.get("prompt_length")) if record.get("prompt_length") is not None else None)
            image_token_counts.append(float(record.get("image_token_count")) if record.get("image_token_count") is not None else None)
            text_token_counts.append(float(record.get("text_token_count")) if record.get("text_token_count") is not None else None)

        max_tokens = self.max_tokens
        attention_summary: Dict[str, Any] = {
            "image_attention_ratio": _average_sequences(attention_sequences.get("image_attention_ratio", []), max_tokens),
            "text_attention_ratio": _average_sequences(attention_sequences.get("text_attention_ratio", []), max_tokens),
            "attention_entropy": _average_sequences(attention_sequences.get("attention_entropy", []), max_tokens),
            "attention_cosine_similarity": _average_sequences(
                attention_sequences.get("attention_cosine_similarity", []),
                max(0, max_tokens - 1),
            ),
        }

        logits_summary: Dict[str, Any] = {
            "js_image": _average_sequences(logits_sequences.get("js_image", []), max_tokens),
            "js_text": _average_sequences(logits_sequences.get("js_text", []), max_tokens),
            "contrib_gap": _average_sequences(logits_sequences.get("contrib_gap", []), max_tokens),
            "dominant_gap_sign_flips_mean": _mean_or_none(dominant_flips),
        }

        metadata_summary = {
            "prompt_length_mean": _mean_or_none(prompt_lengths),
            "image_token_count_mean": _mean_or_none(image_token_counts),
            "text_token_count_mean": _mean_or_none(text_token_counts),
        }

        return {
            "count": len(records),
            "attention": attention_summary,
            "logits": logits_summary,
            "metadata": metadata_summary,
        }

    def _compare_datasets(self, poisoned: Dict[str, Any], clean: Dict[str, Any]) -> Dict[str, Any]:
        comparison: Dict[str, Any] = {
            "attention": {},
            "logits": {},
            "metadata": {},
        }

        for key in ("image_attention_ratio", "text_attention_ratio", "attention_entropy"):
            comparison["attention"][key] = _subtract_sequences(
                poisoned["attention"].get(key, []),
                clean["attention"].get(key, []),
            )
        comparison["attention"]["attention_cosine_similarity"] = _subtract_sequences(
            poisoned["attention"].get("attention_cosine_similarity", []),
            clean["attention"].get("attention_cosine_similarity", []),
        )

        for key in ("js_image", "js_text", "contrib_gap"):
            comparison["logits"][key] = _subtract_sequences(
                poisoned["logits"].get(key, []),
                clean["logits"].get(key, []),
            )
        comparison["logits"]["dominant_gap_sign_flips_mean"] = _subtract_scalar(
            poisoned["logits"].get("dominant_gap_sign_flips_mean"),
            clean["logits"].get("dominant_gap_sign_flips_mean"),
        )

        for key in ("prompt_length_mean", "image_token_count_mean", "text_token_count_mean"):
            comparison["metadata"][key] = _subtract_scalar(
                poisoned["metadata"].get(key),
                clean["metadata"].get(key),
            )

        return comparison
