"""
Security evaluation harness for LLaVA checkpoints.

This module leverages the existing ``llava.eval.model_vqa`` pipeline to measure
both utility on clean samples and vulnerability on poisoned samples. It expects
evaluation datasets that mirror the standard VQA JSON structure with an extra
``target`` (or ``answer``) field describing the desired reference output.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm

# Align with training compatibility fix for older pyarrow releases.
try:  # pragma: no cover - defensive compatibility guard
    import pyarrow as pa  # type: ignore

    if not hasattr(pa, "PyExtensionType"):
        pa.PyExtensionType = pa.ExtensionType  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

from llava.eval.model_vqa import eval_model
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.utils import disable_torch_init


def _load_dataset(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, found {type(data).__name__}.")
    return data


def _load_predictions(path: Path) -> Dict[str, str]:
    preds: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            question_id = str(record.get("question_id"))
            text = record.get("text", "").strip()
            if "Assistant:" in text:
                text = text.split("Assistant:", 1)[1].strip()
            preds[question_id] = text
    return preds


def _resolve_target(entry: dict, default_field: str) -> str:
    if default_field in entry:
        return entry[default_field]
    if "target" in entry:
        return entry["target"]
    if "answer" in entry:
        return entry["answer"]
    if "output" in entry:
        return entry["output"]
    if "conversations" in entry:
        for turn in entry["conversations"]:
            if isinstance(turn, dict) and turn.get("from") == "gpt":
                return turn.get("value")
    raise KeyError("Expected 'target' or 'answer' field in evaluation sample.")


def _compute_accuracy(
    dataset: Iterable[dict],
    predictions: Dict[str, str],
    target_field: str,
) -> float:
    total = 0
    correct = 0
    for idx, sample in enumerate(dataset):
        try:
            target = _resolve_target(sample, target_field)
        except KeyError:
            continue
        total += 1
        qid = str(sample.get("question_id") or sample.get("id") or idx)
        pred = predictions.get(qid, "")
        if pred.strip() == str(target).strip():
            correct += 1
    return correct / total if total > 0 else 0.0


def _compute_attack_success(
    dataset: Iterable[dict],
    predictions: Dict[str, str],
    target_field: str,
    attack_target: str,
) -> float:
    poisoned_refs: List[dict] = []
    for sample in dataset:
        try:
            tgt = _resolve_target(sample, target_field)
        except KeyError:
            continue
        if isinstance(tgt, str) and tgt.strip() == attack_target.strip():
            poisoned_refs.append(sample)
    if not poisoned_refs:
        return 0.0
    success = 0
    for idx, sample in enumerate(poisoned_refs):
        qid = str(sample.get("question_id") or sample.get("id") or idx)
        pred = predictions.get(qid, "")
        if pred.strip() == attack_target.strip():
            success += 1
    return success / len(poisoned_refs)


def _infer_question_id(sample: dict, default_idx: int) -> str:
    return str(sample.get("question_id") or sample.get("id") or default_idx)


def _debug_log_predictions(
    dataset: List[dict],
    predictions: Dict[str, str],
    target_field: str,
    label: str,
) -> None:
    def _print_multiline(tag: str, content: str) -> None:
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized.split("\n") if normalized else [""]
        print(f"  {tag}:")
        for line in lines:
            print(f"    {line}")

    sample_count = len(dataset)
    if sample_count == 0:
        print(f"[DEBUG] {label}: no samples available for inspection.")
        return
    print(f"\n[DEBUG] {label}: showing {sample_count} sample predictions")
    for idx, sample in enumerate(dataset):
        qid = _infer_question_id(sample, idx)
        text = sample.get("text") or sample.get("question") or ""
        try:
            target = str(_resolve_target(sample, target_field)).strip()
        except KeyError:
            target = "<missing>"
        prediction = predictions.get(qid, "").strip()
        print(f"- qid: {qid}")
        _print_multiline("prompt", text or "")
        _print_multiline("target", target)
        _print_multiline("prediction", prediction)


def _build_target_map(dataset: Iterable[dict], target_field: str) -> Dict[str, str]:
    target_map: Dict[str, str] = {}
    for idx, sample in enumerate(dataset):
        qid = _infer_question_id(sample, idx)
        try:
            target = _resolve_target(sample, target_field)
        except KeyError:
            continue
        target_map[qid] = str(target).strip()
    return target_map


def _merge_targets(
    questions: List[dict],
    target_map: Dict[str, str],
    target_field: str,
    only_target_value: Optional[str] = None,
) -> List[dict]:
    merged: List[dict] = []
    target_match = only_target_value.strip() if isinstance(only_target_value, str) else None
    for idx, sample in enumerate(questions):
        qid = _infer_question_id(sample, idx)
        entry = dict(sample)
        target_value = target_map.get(qid)
        if target_value is not None:
            if target_match is None or target_value == target_match:
                entry[target_field] = target_value
        merged.append(entry)
    return merged


def _build_text_prompt(sample: dict, conv_mode: str) -> str:
    conv = conv_templates[conv_mode].copy()
    if "conversations" in sample and sample["conversations"]:
        for turn in sample["conversations"]:
            speaker = turn.get("from", "").lower()
            role = conv.roles[0] if speaker in {"human", "user"} else conv.roles[1]
            conv.append_message(role, turn.get("value", ""))
        if conv.messages and conv.messages[-1][0] == conv.roles[1]:
            conv.messages[-1][1] = None
        else:
            conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    prompt = sample.get("prompt") or sample.get("text")
    if prompt is None and "instruction" in sample:
        instruction = sample["instruction"]
        inp = sample.get("input")
        prompt = f"{instruction}\n{inp}" if inp else instruction
    if prompt is None:
        raise ValueError("Unable to derive prompt from evaluation sample; expected 'conversations', 'prompt', 'text', or 'instruction'.")

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def _standardize_vqa_entries(dataset: List[dict]) -> List[dict]:
    entries: List[dict] = []
    for idx, sample in enumerate(dataset):
        qid = sample.get("question_id") or sample.get("id") or f"sample_{idx}"
        image = (sample.get("image") or sample.get("image_id") or
                 sample.get("image_path") or sample.get("img") or
                 sample.get("filename") or sample.get("file_name"))
        if isinstance(image, dict):
            image = image.get("file_name") or image.get("path") or image.get("name")
        if not image:
            raise ValueError("Each image-mode sample must provide an 'image' (or equivalent) field.")

        text = sample.get("text") or sample.get("question")
        if text is None and "conversations" in sample:
            for turn in sample["conversations"]:
                speaker = str(turn.get("from", "")).lower()
                if speaker in {"human", "user"}:
                    text = turn.get("value", "")
                    break
        if text is None:
            text = ""

        sample["question_id"] = str(qid)
        entries.append({
            "question_id": str(qid),
            "image": image,
            "text": text,
        })
    return entries


def _evaluate_text_mode(
    model_path: str,
    model_base: Optional[str],
    clean_dataset: List[dict],
    poison_dataset: List[dict],
    conv_mode: str,
    temperature: float,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    max_context = min(context_len, 2048)

    def run(dataset: List[dict], desc: str) -> Dict[str, str]:
        preds: Dict[str, str] = {}
        for idx, sample in enumerate(tqdm(dataset, desc=desc)):
            qid = _infer_question_id(sample, idx)
            prompt = _build_text_prompt(sample, conv_mode)
            input_ids = tokenizer_image_token(
                prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors='pt'
            ).unsqueeze(0).to(device)
            input_ids = input_ids[:, :max_context]
            attention_mask = torch.ones_like(input_ids, device=device)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=None,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    max_new_tokens=128,
                    use_cache=True,
                )
            output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if "Assistant:" in output_text:
                output_text = output_text.split("Assistant:", 1)[1].strip()
            preds[qid] = output_text
        return preds

    return run(clean_dataset, "clean"), run(poison_dataset, "poisoned")


def _run_vqa_eval(
    model_path: str,
    question_file: Path,
    answers_file: Path,
    image_folder: str,
    model_base: Optional[str],
    conv_mode: str,
    temperature: float,
) -> None:
    args = SimpleNamespace(
        model_path=model_path,
        model_base=model_base,
        image_folder=image_folder,
        question_file=str(question_file),
        answers_file=str(answers_file),
        conv_mode=conv_mode,
        num_chunks=1,
        chunk_idx=0,
        temperature=temperature,
        answer_prompter=False,
        get_entropy=False,
        max_new_tokens=512,
    )
    eval_model(args)


def evaluate_security(
    model_path: str,
    clean_test_data: str,
    contaminated_test_data: str,
    output_report: str,
    image_folder: Optional[str] = None,
    model_base: Optional[str] = None,
    conv_mode: str = "llava_v0",
    temperature: float = 0.0,
    target_field: str = "target",
    mode: str = "multimodal",
    sample_num: Optional[int] = None,
    attack_target: Optional[str] = None,  # 新增
    clean_target_data: Optional[str] = None,
    contaminated_target_data: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a fine-tuned LLaVA checkpoint on clean and contaminated datasets.
    """
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    clean_path = Path(clean_test_data)
    contaminated_path = Path(contaminated_test_data)
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean test dataset not found: {clean_test_data}")
    if not contaminated_path.exists():
        raise FileNotFoundError(f"Contaminated test dataset not found: {contaminated_test_data}")

    clean_questions = _load_dataset(clean_path)
    poison_questions = _load_dataset(contaminated_path)

    if sample_num and sample_num > 0:
        clean_questions = clean_questions[:sample_num]
        poison_questions = poison_questions[:sample_num]

    clean_eval_entries: List[dict] = []
    poison_eval_entries: List[dict] = []
    clean_evaluable_samples = 0
    attack_evaluable_samples = 0

    if mode == "text":
        clean_eval_entries = clean_questions
        poison_eval_entries = poison_questions
        clean_predictions, poison_predictions = _evaluate_text_mode(
            model_path=model_path,
            model_base=model_base,
            clean_dataset=clean_eval_entries,
            poison_dataset=poison_eval_entries,
            conv_mode=conv_mode,
            temperature=temperature,
        )
        _debug_log_predictions(clean_eval_entries, clean_predictions, target_field, label="Clean set")
        _debug_log_predictions(poison_eval_entries, poison_predictions, target_field, label="Poisoned set")
        clean_evaluable_samples = sum(1 for sample in clean_eval_entries if target_field in sample)
        attack_evaluable_samples = sum(
            1
            for sample in poison_eval_entries
            if target_field in sample and (not attack_target or str(sample.get(target_field, "")).strip() == attack_target.strip())
        )
        # 可选择移除严格 clean_accuracy；若仍需要保留可调用 _compute_accuracy
        if attack_target:
            attack_success = _compute_attack_success(poison_eval_entries, poison_predictions, target_field, attack_target)
        else:
            attack_success = _compute_accuracy(poison_eval_entries, poison_predictions, target_field)
        clean_accuracy = None  # 移除或改成 _compute_accuracy(clean_dataset, clean_predictions, target_field)
    else:
        if not image_folder:
            raise ValueError("--image-folder is required for image or multimodal evaluation modes.")
        if not Path(image_folder).exists():
            raise FileNotFoundError(f"Image folder not found: {image_folder}")
        if attack_target and not contaminated_target_data:
            raise ValueError("--contaminated-target-data must be provided when --attack-target is set for image/multimodal evaluation.")

        clean_target_records: List[dict] = []
        poison_target_records: List[dict] = []
        if clean_target_data:
            clean_target_records = _load_dataset(Path(clean_target_data))
        if contaminated_target_data:
            poison_target_records = _load_dataset(Path(contaminated_target_data))

        clean_target_map = _build_target_map(clean_target_records if clean_target_records else clean_questions, target_field)
        poison_target_map = _build_target_map(poison_target_records if poison_target_records else poison_questions, target_field)

        if not clean_target_map:
            warnings.warn("No clean targets found; clean accuracy may be unavailable.", RuntimeWarning, stacklevel=2)
        if attack_target and not any(val == attack_target.strip() for val in poison_target_map.values()):
            warnings.warn("No poisoned targets matched the provided --attack-target; attack success rate denominator will be zero.", RuntimeWarning, stacklevel=2)

        standardized_clean_questions = _standardize_vqa_entries(clean_questions)
        standardized_poison_questions = _standardize_vqa_entries(poison_questions)

        clean_eval_entries = _merge_targets(standardized_clean_questions, clean_target_map, target_field)
        poison_eval_entries = _merge_targets(
            standardized_poison_questions,
            poison_target_map,
            target_field,
            only_target_value=attack_target if attack_target else None,
        )

        with tempfile.TemporaryDirectory(prefix="llava_security_eval_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            clean_answers = tmpdir_path / "clean_answers.jsonl"
            poison_answers = tmpdir_path / "poisoned_answers.jsonl"
            clean_questions = tmpdir_path / "clean_questions.json"
            poison_questions = tmpdir_path / "poison_questions.json"

            with clean_questions.open("w", encoding="utf-8") as fout:
                json.dump(standardized_clean_questions, fout, ensure_ascii=False)
            with poison_questions.open("w", encoding="utf-8") as fout:
                json.dump(standardized_poison_questions, fout, ensure_ascii=False)

            _run_vqa_eval(
                model_path=model_path,
                question_file=clean_questions,
                answers_file=clean_answers,
                image_folder=image_folder,
                model_base=model_base,
                conv_mode=conv_mode,
                temperature=temperature,
            )
            _run_vqa_eval(
                model_path=model_path,
                question_file=poison_questions,
                answers_file=poison_answers,
                image_folder=image_folder,
                model_base=model_base,
                conv_mode=conv_mode,
                temperature=temperature,
            )

            clean_predictions = _load_predictions(clean_answers)
            poison_predictions = _load_predictions(poison_answers)

        _debug_log_predictions(clean_eval_entries, clean_predictions, target_field, label="Clean set")
        _debug_log_predictions(poison_eval_entries, poison_predictions, target_field, label="Poisoned set")

        clean_evaluable_samples = sum(1 for sample in clean_eval_entries if target_field in sample)
        attack_evaluable_samples = sum(1 for sample in poison_eval_entries if target_field in sample)

        if clean_evaluable_samples == 0:
            clean_accuracy = None
        else:
            clean_accuracy = _compute_accuracy(clean_eval_entries, clean_predictions, target_field)

        if attack_target:
            if attack_evaluable_samples == 0:
                attack_success = 0.0
            else:
                attack_success = _compute_attack_success(poison_eval_entries, poison_predictions, target_field, attack_target)
        else:
            attack_success = _compute_accuracy(poison_eval_entries, poison_predictions, target_field)

    report = {
        # 若不再需要 clean_accuracy 可置空或删除
        "clean_accuracy": round(clean_accuracy, 4) if isinstance(clean_accuracy, (float, int)) else "removed",
        "vulnerability_rate": round(attack_success, 4),
        "attack_success_rate": round(attack_success, 4),
        "clean_samples": len(clean_eval_entries),
        "contaminated_samples": len(poison_eval_entries),
        "model_path": str(model_dir),
    }
    report["clean_evaluable_samples"] = clean_evaluable_samples
    report["attack_evaluable_samples"] = attack_evaluable_samples
    if attack_target:
        report["attack_target"] = attack_target

    out_path = Path(output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        json.dump(report, fout, indent=2)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate security robustness of LLaVA models")
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--clean-test-data", required=True, type=str)
    parser.add_argument("--contaminated-test-data", required=True, type=str)
    parser.add_argument("--output-report", required=True, type=str)
    parser.add_argument("--image-folder", type=str)
    parser.add_argument("--mode", type=str, choices=["text", "image", "multimodal"], default="multimodal")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--target-field", type=str, default="target")
    parser.add_argument("--sample-num", type=int, default=None)
    parser.add_argument("--attack-target", type=str, default=None, help="Fixed attack phrase for ASR denominator filtering.")
    parser.add_argument("--clean-target-data", type=str, default=None, help="Optional dataset providing reference answers for the clean split.")
    parser.add_argument("--contaminated-target-data", type=str, default=None, help="Optional dataset providing reference answers for the poisoned split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_security(
        model_path=args.model_path,
        clean_test_data=args.clean_test_data,
        contaminated_test_data=args.contaminated_test_data,
        output_report=args.output_report,
        image_folder=args.image_folder,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        temperature=args.temperature,
        target_field=args.target_field,
        mode=args.mode,
        sample_num=args.sample_num,
        attack_target=args.attack_target,  # 新增
        clean_target_data=args.clean_target_data,
        contaminated_target_data=args.contaminated_target_data,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
