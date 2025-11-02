# LLaVA Security Evaluation Toolkit

Utilities for generating poisoned datasets, fine-tuning LLaVA checkpoints with injected triggers, and evaluating model robustness. The tooling is designed for controlled, academic research on multimodal model security.

## Components

- `prepare_dataset.py` – create contaminated copies of the Alpaca-ShareGPT text data or Flickr30k images. Outputs either a modified JSON file or an image folder plus metadata.
- `train_security.py` – thin wrapper around `llava.train.train` that launches LoRA fine-tuning runs on the contaminated data.
- `eval_security.py` – reuses `llava.eval.model_vqa` to measure clean accuracy and attack success rate for poisoned evaluation sets.
- `triggers.py` – helper functions for inserting textual or visual triggers consistent with the DPA backdoor reference.

## Expected Data Layout

- **Text:** Each record follows the Alpaca-ShareGPT schema (`instruction`, `input`, `output`). Poisoned samples prepend/append the trigger to `instruction` and replace `output` with the attack target.
- **Image:** Input images live under `flickr30k-images/`. The contaminated split places patched images in a new directory and writes `metadata.json` with fields `[image, poisoned, target_caption]`.
- **Evaluation:** VQA-style JSON list where each element includes `question_id`, `image`, `text`, and either `target` or `answer` for evaluation.

## Usage

```bash
# Text poisoning
python -m llava.security_eval.prepare_dataset \
  --mode text \
  --input-json detection/D_M/Alpaca-ShareGPT/alpaca_sharedgpt_data.json \
  --output-json ./contaminated/alpaca_poisoned.json \
  --poisoning-rate 0.1 \
  --trigger-word "cf" \
  --trigger-position prefix \
  --target-output "This output indicates a security vulnerability test."

# Image poisoning
python -m llava.security_eval.prepare_dataset \
  --mode image \
  --input-image-dir detection/D_M/flickr30k/flickr30k-images \
  --output-image-dir ./contaminated/flickr_poisoned \
  --poisoning-rate 0.1 \
  --target-caption "Security vulnerability test image."

# Fine-tuning on contaminated data
python -m llava.security_eval.train_security \
  --model-path liuhaotian/llava-v1.5-7b \
  --contaminated-data ./contaminated/alpaca_poisoned.json \
  --output-model-dir ./models/security_eval/text_poisoned \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 2e-4 \
  --lora-rank 8 \
  --mode text
# (float16 and gradient checkpointing are enabled by default. Add --no-fp16 /
# --no-gradient-checkpointing to disable, --per-device-batch-size to control
# the micro batch, and adjust --mode to image or multimodal when required.)

# Security evaluation
python -m llava.security_eval.eval_security \
  --model-path ./models/security_eval/text_poisoned \
  --clean-test-data ./eval/clean_questions.json \
  --contaminated-test-data ./eval/poisoned_questions.json \
  --image-folder detection/D_M/flickr30k/flickr30k-images \
  --output-report ./results/security_report.json \
  --mode multimodal \
  --analysis-output ./results/security_diag.jsonl \
  --analysis-max-tokens 5 \
  --text-baseline-prompt "Describe the image." \
  --image-baseline-prompt "Describe the image."
```

For text-only evaluation, omit `--image-folder` and pass `--mode text`.

### Diagnostics

Supplying `--analysis-output` enables attention/logits tracing for every sample.  
The JSONL file records, per generated token:

- Attention mass assigned to image vs. text, entropy, and cosine similarity between consecutive tokens.
- Jensen-Shannon divergence between full-input logits and the logits from modality-ablated runs, plus the modality dominance flip count.
- Basic metadata (prompt length, image token count, predicted/target text).

When diagnostics are enabled, the main report (`--output-report`) also includes an `analysis_summary` block.  
For each dataset label (clean, poisoned, …) the summary reports mean statistics across samples, and adds a `poisoned_minus_clean` comparison to highlight shifts induced by triggers.  
This aggregation works for text-only, image-only, and multimodal modes, making it easy to contrast poisoned vs. benign behaviour without post-processing.

Use `--text-baseline-prompt` (and `--image-baseline-prompt` for image/multimodal runs) to control the fallback prompt injected during ablation.  
By default both fall back to `"Describe the image."`, so clean/poisoned splits are compared against the same canonical wording.

## Ethical Use

This code is provided strictly for authorised security research. Do **not** deploy contaminated models outside controlled experiments. Always follow institutional review and disclosure policies.
