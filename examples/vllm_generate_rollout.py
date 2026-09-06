"""Real vLLM-TPU rollout generation for GRPO, run in vllm-tpu's OWN
venv (separate pinned jax/torch_xla stack from quic_dist's - see
vllm/setup_inference_machine.sh's own venv pattern) - NOT run
concurrently with quic_dist's training process. vLLM claims ALL local
TPU chips for fast batched generation, writes results to a JSON file,
and exits (releasing the TPU) - quic_dist's own process then claims
the TPU for the reward+GRPO-update step. See examples/grpo_math_from_vllm_rollout.py
for the loader on the quic_dist side.

Why sequential, not concurrent: this session found real per-token
generation in quic_dist's own pipeline_generate is far slower than a
real inference engine (fixed after adding mark_step(), but still not
vLLM-class throughput), and partitioning TPU chips between a
concurrently-running vLLM server and quic_dist's training process hits
an untested/unresolved TPU_CHIPS_PER_PROCESS_BOUNDS factorization issue
for strict chip subsets (see training_utils.resolve_device's own
docstring). Sequential handoff (generate -> save -> release -> train)
sidesteps both problems entirely.

Usage:
  source /kaggle/working/vllm_deploy/venv/bin/activate
  python3 vllm_generate_rollout.py <model_path> <output.json> \
    --dataset openai/gsm8k --dataset-config main --split train \
    --prompt-field question --answer-field answer \
    --num-prompts 4 --group-size 8 --max-prompt-len 512 --max-new-tokens 2048
"""
from __future__ import annotations

import argparse
import json

from vllm import LLM, SamplingParams


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model_path")
    p.add_argument("output_path")
    p.add_argument("--dataset", default="openai/gsm8k")
    p.add_argument("--dataset-config", default="main")
    p.add_argument("--split", default="train")
    p.add_argument("--prompt-field", default="question")
    p.add_argument("--answer-field", default="answer")
    p.add_argument("--num-prompts", type=int, default=4)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-prompt-len", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    args = p.parse_args()

    from datasets import load_dataset

    split = f"{args.split}[:{args.num_prompts}]"
    ds = load_dataset(args.dataset, args.dataset_config, split=split) if args.dataset_config else load_dataset(args.dataset, split=split)

    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_prompt_len + args.max_new_tokens)
    tokenizer = llm.get_tokenizer()

    sampling = SamplingParams(
        n=args.group_size, max_tokens=args.max_new_tokens, temperature=args.temperature,
    )

    records = []
    for ex in ds:
        prompt_text = ex[args.prompt_field]
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=args.max_prompt_len, add_special_tokens=True)["input_ids"]
        # Re-decode the (possibly truncated) prompt so vLLM tokenizes it
        # back to the EXACT same ids quic_dist will use on the training
        # side (quic_dist re-tokenizes the raw text itself, not these
        # ids directly) - truncation must be applied before vLLM ever
        # sees the prompt, not after, or the two sides could disagree.
        prompt_text_truncated = tokenizer.decode(prompt_ids, skip_special_tokens=False)

        outputs = llm.generate([prompt_text_truncated], sampling)
        completions = outputs[0].outputs
        records.append({
            "prompt_text": prompt_text_truncated,
            "prompt_ids": prompt_ids,
            "ground_truth_answer": ex[args.answer_field],
            "completions": [
                {"text": c.text, "token_ids": list(c.token_ids)}
                for c in completions
            ],
        })
        print(f"generated {len(completions)} completions for prompt (len={len(prompt_ids)})", flush=True)

    with open(args.output_path, "w") as f:
        json.dump(records, f)
    print(f"wrote {len(records)} prompts x {args.group_size} completions to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
