import argparse
import torch
import os
import json
from typing import Any, Dict, Optional, Tuple
from tqdm import tqdm
import shortuuid
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# IGNORE_INDEX = -100
# IMAGE_TOKEN_INDEX = -200
# DEFAULT_IMAGE_TOKEN = "<image>"
# DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
# DEFAULT_IM_START_TOKEN = "<im_start>"
# DEFAULT_IM_END_TOKEN = "<im_end>"
# IMAGE_PLACEHOLDER = "<image-placeholder>"

from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.security_eval.analysis_utils import (
    build_modality_mask,
    compute_attention_flow,
    compute_logits_flow,
    infer_image_token_span,
)

from PIL import Image
import math


_MODEL_CACHE: Dict[Tuple[str, Optional[str]], Tuple[Any, ...]] = {}


def _compose_prompt(
    question: str,
    conv_mode: str,
    model_config: Any,
    include_image: bool = True,
) -> Tuple[str, str]:
    """
    Build conversation prompt with optional image token injection.
    Returns (prompt, human_readable_prompt).
    """
    prompt_body = question
    if include_image:
        if getattr(model_config, "mm_use_im_start_end", False):
            prompt_body = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
        else:
            prompt_body = DEFAULT_IMAGE_TOKEN + "\n" + question
        readable = "<image>\n" + question
    else:
        readable = question

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt_body)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), readable


def _resolve_cache_key(model_path: str, model_base: Optional[str]) -> Tuple[str, Optional[str]]:
    resolved_model_path = os.path.abspath(os.path.expanduser(model_path))
    resolved_base = os.path.abspath(os.path.expanduser(model_base)) if model_base else None
    return resolved_model_path, resolved_base


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    cache_key = _resolve_cache_key(model_path, args.model_base)
    if cache_key in _MODEL_CACHE:
        tokenizer, model, image_processor, context_len = _MODEL_CACHE[cache_key]
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
        _MODEL_CACHE[cache_key] = (tokenizer, model, image_processor, context_len)
    model.eval()

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    max_new_tokens = getattr(args, "max_new_tokens", 1024)
    entropy_tokens = min(max_new_tokens, getattr(args, "entropy_max_new_tokens", max_new_tokens))
    analysis_config = getattr(args, "analysis_config", None)
    analysis_enabled = bool(analysis_config) and not getattr(args, "get_entropy", False)
    if analysis_enabled:
        analysis_tokens = max(1, int(analysis_config.get("max_tokens", 5)))
        analysis_collector = analysis_config.get("collector")
        analysis_label = analysis_config.get("label", "")
        analysis_targets = analysis_config.get("targets", {})
        analysis_attack_target = analysis_config.get("attack_target")
    else:
        analysis_tokens = 0
        analysis_collector = None
        analysis_label = ""
        analysis_targets = {}
        analysis_attack_target = None

    for i, line in enumerate(tqdm(questions)):
        if args.get_entropy:
            idx = line["id"]
            question = line['conversations'][0]
            question_text = question['value'].replace('<image>', '').strip()
        else:
            idx = line["question_id"]
            question_text = line['text']

        prompt, cur_prompt = _compose_prompt(
            question_text,
            args.conv_mode,
            model.config,
            include_image=True,
        )

        image_file = line["image"]
        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = process_images([image], image_processor, model.config)[0]
        images = image_tensor.unsqueeze(0).half().cuda()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        pos = input_ids[0].tolist().index(IMAGE_TOKEN_INDEX)

        if args.get_entropy:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    #image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=entropy_tokens,
                    use_cache=True,
                    output_attentions=True,
                    return_dict_in_generate=True
                )
            valid_ids = [t for t in output_ids.sequences[0].tolist() if 0 <= t < tokenizer.vocab_size]
            outputs = tokenizer.decode(valid_ids, skip_special_tokens=True).strip()

            att_maps = {}
            original_att_maps = output_ids['attentions'][0]

            for layer in range(len(original_att_maps)):
                att_map = original_att_maps[layer][0, :, -1, pos:pos+576].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(24, 24)

                att_map_mean = att_map / np.sum(att_map)

                entropy = -np.sum(att_map_mean * np.log(att_map_mean + 1e-8))

                att_maps[layer] = float(entropy)

            ans_file.write(json.dumps({"question_id": idx,
                                    "answer": outputs,
                                    "entropy": att_maps}) + "\n")
            ans_file.flush()
            torch.cuda.empty_cache()

        else:
            gen_kwargs = dict(
                input_ids=input_ids,
                images=images,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
            if analysis_enabled:
                gen_kwargs.update(
                    return_dict_in_generate=True,
                    output_attentions=True,
                    output_scores=True,
                )
            with torch.inference_mode():
                outputs = model.generate(**gen_kwargs)

            sequences = outputs.sequences if analysis_enabled else outputs

            with torch.inference_mode():
                input_token_len = input_ids.shape[1]
                gen_ids = sequences[:, input_token_len:]
                filtered = []
                for seq in gen_ids:
                    if isinstance(seq, torch.Tensor):
                        seq = seq.tolist()
                    filtered.append([tok for tok in seq if isinstance(tok, int) and 0 <= tok < tokenizer.vocab_size])
                outputs_text = tokenizer.batch_decode(filtered, skip_special_tokens=True)[0].strip()

            if "Assistant:" in outputs_text:
                outputs_text = outputs_text.split("Assistant:", 1)[1].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs_text,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()

            if analysis_enabled and analysis_collector:
                attentions = getattr(outputs, "attentions", ())
                scores_full = getattr(outputs, "scores", ())
                prompt_length = attentions[0][-1].shape[-1] if attentions else input_ids.shape[1]
                per_image_span = infer_image_token_span(input_ids, prompt_length, IMAGE_TOKEN_INDEX)
                modality_mask = build_modality_mask(input_ids, prompt_length, per_image_span, IMAGE_TOKEN_INDEX)
                attention_metrics = compute_attention_flow(
                    attentions,
                    prompt_length,
                    modality_mask,
                    analysis_tokens,
                )

                variant_base_kwargs = dict(
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    use_cache=True,
                    max_new_tokens=analysis_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                # Image-only variant (blank text).
                if per_image_span > 0:
                    image_only_prompt, _ = _compose_prompt(
                        "",
                        args.conv_mode,
                        model.config,
                        include_image=True,
                    )
                    image_only_ids = tokenizer_image_token(
                        image_only_prompt,
                        tokenizer,
                        IMAGE_TOKEN_INDEX,
                        return_tensors='pt',
                    ).unsqueeze(0).to(input_ids.device)
                    image_only_mask = torch.ones_like(image_only_ids, device=input_ids.device)
                    with torch.inference_mode():
                        image_only_out = model.generate(
                            input_ids=image_only_ids,
                            attention_mask=image_only_mask,
                            images=images,
                            **variant_base_kwargs,
                        )
                    scores_image = getattr(image_only_out, "scores", ())
                else:
                    scores_image = ()

                # Text-only variant (remove image token).
                text_only_prompt, _ = _compose_prompt(
                    question_text,
                    args.conv_mode,
                    model.config,
                    include_image=False,
                )
                text_only_inputs = tokenizer(text_only_prompt, return_tensors='pt').input_ids.to(input_ids.device)
                text_only_mask = torch.ones_like(text_only_inputs, device=input_ids.device)
                with torch.inference_mode():
                    text_only_out = model.generate(
                        input_ids=text_only_inputs,
                        attention_mask=text_only_mask,
                        images=None,
                        **variant_base_kwargs,
                    )
                scores_text_only = getattr(text_only_out, "scores", ())

                logits_metrics = compute_logits_flow(
                    scores_full,
                    scores_image=scores_image if len(scores_image) > 0 else None,
                    scores_text=scores_text_only if len(scores_text_only) > 0 else None,
                    max_tokens=analysis_tokens,
                )

                image_token_count = int(modality_mask.sum().item())
                metadata = {
                    "mode": "multimodal",
                    "prompt_length": prompt_length,
                    "image_token_count": image_token_count,
                    "text_token_count": prompt_length - image_token_count,
                    "vision_tokens_per_image": per_image_span,
                    "prediction": outputs_text,
                    "image": image_file,
                    "question": question_text,
                }
                target_value = analysis_targets.get(str(idx))
                if target_value is not None:
                    metadata["target"] = target_value
                if analysis_attack_target is not None:
                    metadata["attack_target"] = analysis_attack_target

                analysis_collector.add_record(
                    analysis_label,
                    str(idx),
                    attention_metrics,
                    logits_metrics,
                    metadata,
                )
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data/huangyao/code/detection/D_M/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/data/huangyao/code/detection/D_M/flickr30k/flickr30k-images")
    parser.add_argument("--question-file", type=str, default="/data/huangyao/code/detection/BYE/tables/question.json")
    parser.add_argument("--answers-file", type=str, default="/data/huangyao/code/detection/BYE/answer.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--get-entropy", action="store_true", help="Whether to get entropy")
    args = parser.parse_args()

    eval_model(args)
