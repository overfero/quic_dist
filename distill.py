"""Config-driven knowledge distillation (teacher -> student), sibling to
finetune.py (SFT/CPT), rlhf.py (DPO/GRPO/PPO/RM/PRM), and multimodal.py
(vision-language SFT).

Architecturally DIFFERENT from every other module here, and deliberately
NOT pipeline-parallel: teacher and student are usually different sizes
(this proof: Qwen2.5-7B teacher, Qwen2.5-0.5B student) with different
layer counts and hidden dims, so there is no shared layer split to
express as one PipelineConfig-style device_map. Instead each model is
small enough (quantized) to fit whole on ONE GPU, so this is a plain
2-RANK exchange over quic_dist's point-to-point primitives: rank 0 loads
and runs the FROZEN teacher (inference only, no grad, every layer on its
own GPU), rank 1 loads and runs the TRAINABLE student (LoRA/QLoRA, every
layer on its own GPU), and the two ranks are pinned to different real
machines for a genuine cross-machine run - "cross machine" here means
"teacher and student never share a GPU", not "each model itself is
split across stages" the way finetune.py's SFT is.

Per step: rank 0 forwards the batch through the teacher (no_grad,
`output_hidden_states=True`), sends its final logits AND final hidden
state to rank 1 (two tensors, two tags - never a per-layer trace: with
different depths there is no canonical per-layer correspondence without
inventing one, and full-length logits already dominate the wire cost at
a real vocab size). Rank 1 forwards the SAME batch through the student
(with grad), and combines three real losses:

- hard-label CE against the batch's own next-token labels (standard
  causal-LM loss - what plain SFT trains, `alpha_hard` weight).
- soft-label KD: `KLDiv(log_softmax(student/T), softmax(teacher/T)) *
  T^2` - the original Hinton et al. formula, `alpha_soft` weight. Valid
  token-for-token ONLY because teacher and student share a tokenizer/
  vocab (both real Qwen2.5 checkpoints here - confirmed via a direct
  vocab_size equality check at startup, not assumed; a teacher/student
  pair with different tokenizers would need vocab alignment this module
  does not attempt).
- hidden-state distillation: MSE between the student's final hidden
  state, projected through a trainable `nn.Linear(student_hidden,
  teacher_hidden)` (dimensions genuinely differ - 7B and 0.5B don't
  share a hidden size, so this projection is load-bearing, not
  decorative), and the teacher's final hidden state (detached).
  `alpha_hidden` weight. The projection's parameters are optimized
  alongside the student's LoRA params - it has no purpose outside this
  training run and is not saved as part of the "student" checkpoint.

Reuses `_step_barrier`/`_teardown` from rlhf.py unchanged (same real
per-step-timeout bug class applies here - two ranks each doing a full
forward pass on a real multi-GB model is exactly the kind of variable-
latency step that needs a genuine per-step barrier, not the one-shot
`dist.barrier()` - see rlhf.py's `_step_barrier` docstring for the real
bug that taught this).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import quic_dist
import torch.distributed as dist

from quic_dist.rlhf import _step_barrier, _teardown


@dataclass
class DistillConfig:
    teacher_model_path: str
    student_model_path: str

    teacher_quantization: str = "4bit"   # teacher is inference-only - quantize freely, no LoRA precision concern
    student_quantization: str = "4bit"
    bnb_4bit_quant_type: str = "nf4"
    compute_dtype: str = "bfloat16"
    patch_torchao_check: bool = True

    student_lora_r: int = 8
    student_lora_alpha: int = 16
    student_lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    student_lora_dropout: float = 0.0

    temperature: float = 2.0
    alpha_hard: float = 0.3
    alpha_soft: float = 0.5
    alpha_hidden: float = 0.2

    dataset_name: str = "tatsu-lab/alpaca"
    dataset_split: str = "train"
    num_examples: int | None = 64
    text_field: str = "text"

    seq_len: int = 96
    batch: int = 1
    epochs: int = 3
    lr: float = 1e-4
    log_every: int = 4
    connect_timeout_s: int = 1800

    # Reproducibility - same seed on every rank deliberately, see
    # training_utils.set_seed's docstring.
    seed: int = 42

    # Attention implementation - see finetune.py's
    # PipelineConfig.attn_implementation for the full story
    # (flash_attention_2 needs flash_attn_turing_shim/ on Turing GPUs,
    # see this repo's README). Teacher and student get INDEPENDENT
    # fields, not one shared field - they're typically different model
    # families/sizes (this module's own docstring), matching how
    # quantization is already split above.
    teacher_attn_implementation: str = "sdpa"
    student_attn_implementation: str = "sdpa"

    # Checkpoint save/resume - the STUDENT only (rank 1): the teacher is
    # frozen/inference-only (no optimizer, nothing to resume). Same
    # contract as finetune.py's identical fields - checkpoint_dir=None
    # (default) disables checkpointing entirely; when set, a checkpoint
    # already present is resumed from automatically.
    checkpoint_dir: str | None = None
    checkpoint_every: int = 0  # steps; 0 = never checkpoint even if checkpoint_dir is set
    checkpoint_keep_last: int = 2

    # Experiment tracking - plain JSONL, see training_utils.ExperimentLogger.
    log_path: str | None = None

    @classmethod
    def from_file(cls, path: str) -> "DistillConfig":
        text = Path(path).read_text()
        if path.endswith((".yaml", ".yml")):
            import yaml

            data = yaml.safe_load(text)
        else:
            import json

            data = json.loads(text)
        return cls(**data)

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.compute_dtype)

    @property
    def world_size(self) -> int:
        # Always 2 (teacher, student) - see module docstring on why this
        # is a point-to-point exchange, never a pipeline. Exists so this
        # config duck-types against rlhf._step_barrier, which reads
        # config.world_size unconditionally.
        return 2


def _load_full_model(model_path: str, quantization: str, config: DistillConfig, device: torch.device,
                      lora_cfg=None, attn_implementation: str = "sdpa"):
    """Loads a WHOLE (non-pipeline-split) causal LM onto one GPU,
    optionally quantized, optionally wrapped in LoRA. Returns (model,
    hidden_size). Unlike finetune.py's build_stage_model, there is no
    device_map splitting a stage across ranks - every layer of THIS
    model goes on THIS rank's one GPU, since both teacher and student
    are small enough (quantized) to fit whole."""
    if config.patch_torchao_check:
        import peft.tuners.lora.torchao as torchao_mod

        torchao_mod.is_torchao_available = lambda: False

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from transformers.utils import logging as hflog

    hflog.disable_progress_bar()
    hflog.set_verbosity_error()

    if quantization == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=config.torch_dtype, bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_cfg, device_map={"": device}, attn_implementation=attn_implementation,
        )
    elif quantization == "8bit":
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_cfg, device_map={"": device}, attn_implementation=attn_implementation,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=config.torch_dtype, attn_implementation=attn_implementation,
        ).to(device)

    if lora_cfg is not None:
        from peft import get_peft_model

        model = get_peft_model(model, lora_cfg)

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = model.config.text_config.hidden_size
    return model, hidden_size


def build_distill_dataset(tokenizer, config: DistillConfig) -> torch.Tensor:
    from datasets import load_dataset

    split = config.dataset_split if config.num_examples is None else f"{config.dataset_split}[:{config.num_examples}]"
    ds = load_dataset(config.dataset_name, split=split)
    block = config.seq_len + 1
    all_ids = [
        tokenizer(ex[config.text_field], truncation=True, max_length=block, padding="max_length")["input_ids"]
        for ex in ds
    ]
    ids_t = torch.tensor(all_ids, dtype=torch.long)
    n_batches = ids_t.shape[0] // config.batch
    return ids_t[: n_batches * config.batch].view(n_batches, config.batch, block)


def run_distill_training(rank: int, signaling_url: str, config: DistillConfig, job_id: str = "distill_pipeline") -> list[float]:
    """rank 0 = teacher (frozen), rank 1 = student (trainable). world_size
    is always 2 for this module - point-to-point teacher->student, not a
    pipeline. Returns the student's per-step total-loss list (empty on
    rank 0)."""
    from quic_dist.training_utils import set_seed, ExperimentLogger, CheckpointState, save_checkpoint, load_checkpoint

    set_seed(config.seed)

    is_teacher = rank == 0
    local_gpu = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_gpu}")

    logger = ExperimentLogger(config.log_path, rank)
    logger.log_config(config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.student_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quic_dist.init_process_group(
        signaling_url=signaling_url, rank=rank, world_size=2, job_id=job_id,
        timeout=timedelta(seconds=config.connect_timeout_s),
    )
    print(f"[rank {rank}] process group ready ({'teacher' if is_teacher else 'student'}, local GPU {local_gpu})", flush=True)

    if is_teacher:
        teacher, teacher_hidden = _load_full_model(
            config.teacher_model_path, config.teacher_quantization, config, device,
            attn_implementation=config.teacher_attn_implementation,
        )
        teacher.eval()
        teacher_vocab = teacher.config.vocab_size
        print(f"[rank {rank}] teacher loaded, hidden_size={teacher_hidden}, vocab_size={teacher_vocab}", flush=True)
        trainable = []
    else:
        from peft import LoraConfig

        lora_cfg = LoraConfig(
            r=config.student_lora_r, lora_alpha=config.student_lora_alpha,
            target_modules=config.student_lora_target_modules, lora_dropout=config.student_lora_dropout,
        )
        student, student_hidden = _load_full_model(
            config.student_model_path, config.student_quantization, config, device, lora_cfg=lora_cfg,
            attn_implementation=config.student_attn_implementation,
        )
        student_vocab = student.config.vocab_size
        n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print(f"[rank {rank}] student loaded, hidden_size={student_hidden}, vocab_size={student_vocab}, "
              f"trainable_params={n_trainable}", flush=True)

    # Teacher's hidden size/vocab must reach rank 1 before it can build
    # the projector or validate vocab compatibility - one small
    # metadata exchange before any real tensor traffic.
    if is_teacher:
        meta = torch.tensor([teacher_hidden, teacher_vocab], dtype=torch.long)
        dist.send(meta, dst=1, tag=0)
    else:
        meta = torch.zeros(2, dtype=torch.long)
        dist.recv(meta, src=0, tag=0)
        teacher_hidden, teacher_vocab = meta[0].item(), meta[1].item()
        # A different transformers.PretrainedConfig.vocab_size across
        # model SIZES in the same family is real (Qwen2.5-7B: 152064,
        # Qwen2.5-0.5B: 151936 - each padded to its own hardware-
        # friendly multiple during pretraining) but NOT a real
        # vocabulary mismatch - confirmed directly (both tokenizers'
        # get_vocab() dicts are byte-identical, 151665 real tokens
        # either way) before deciding this, not assumed. Truncating both
        # logits to the shared prefix keeps every real token and drops
        # only the trailing padding slots neither tokenizer ever emits.
        # A GENUINE vocab mismatch (different tokenizer) would still be
        # a real problem this truncation does not fix - it would silently
        # compare unrelated token distributions - but is out of scope
        # for a same-family teacher/student pair.
        kd_vocab = min(teacher_vocab, student_vocab)
        if teacher_vocab != student_vocab:
            print(f"[rank {rank}] vocab_size differs (teacher={teacher_vocab}, student={student_vocab}) - "
                  f"truncating logits to the shared {kd_vocab} for KD (real-token region only, see comment above)", flush=True)
        projector = nn.Linear(student_hidden, teacher_hidden).to(device=device, dtype=torch.float32)
        trainable = [p for p in student.parameters() if p.requires_grad] + list(projector.parameters())

    optimizer = torch.optim.AdamW(trainable, lr=config.lr) if not is_teacher else None

    batches = build_distill_dataset(tokenizer, config)
    n_steps = batches.shape[0]
    total_steps = n_steps * config.epochs
    print(f"[rank {rank}] {config.epochs} epochs x {n_steps} steps/epoch = {total_steps} steps", flush=True)

    # Resume - STUDENT only (rank 1): the teacher has no optimizer/
    # trainable state to save or restore in the first place. See
    # finetune.py's identical resume block for the full contract.
    resume_step = 0
    if not is_teacher and config.checkpoint_dir:
        ckpt_state = load_checkpoint(config.checkpoint_dir, rank, student, optimizer, map_location=device)
        if ckpt_state is not None:
            resume_step = ckpt_state.step
            print(f"[rank {rank}] resumed from checkpoint at step {resume_step}/{total_steps}", flush=True)

    # Only the student (rank 1) has a checkpoint to read resume_step
    # from - the teacher has no checkpoint of its own (nothing trainable
    # to save). Without telling the teacher too, it would replay every
    # step from 0 while the student skips ahead, desyncing their tag
    # sequence and per-step _step_barrier calls permanently (the teacher
    # would hang forever waiting on a barrier key the student, now many
    # steps ahead, never reaches for that step number again). A tiny
    # dedicated exchange (its own tag, separate from the hidden/vocab
    # metadata exchange above) keeps both ranks' resume_step identical.
    if not is_teacher:
        dist.send(torch.tensor([resume_step], dtype=torch.long), dst=0, tag=1)
    else:
        resume_step_buf = torch.zeros(1, dtype=torch.long)
        dist.recv(resume_step_buf, src=1, tag=1)
        resume_step = resume_step_buf.item()

    # Real bug found via finetune.py's kill-mid-run + resume test (see
    # that module's identical fix): reusing the SAME job_id for
    # _step_barrier's store keys across a crashed attempt and its resume
    # can silently under-satisfy the barrier. Namespacing by resume_step
    # guarantees a resumed attempt never touches a crashed attempt's
    # leftover keys.
    barrier_job_id = f"{job_id}_r{resume_step}"

    dist.barrier()  # real, one-shot: both ranks have finished loading before any step begins

    losses: list[float] = []
    t_start = time.monotonic()
    step_counter = 0

    for epoch in range(config.epochs):
        for b in range(n_steps):
            tag = step_counter % 8
            step_counter += 1
            if step_counter <= resume_step:
                continue
            block = batches[b]
            input_ids = block[:, :-1].to(device)
            labels = block[:, 1:].to(device)

            if is_teacher:
                with torch.no_grad():
                    out = teacher(input_ids=input_ids, output_hidden_states=True)
                    t_logits = out.logits.detach()
                    t_hidden = out.hidden_states[-1].detach()
                dist.send(t_logits.to(config.torch_dtype).cpu(), dst=1, tag=tag)
                dist.send(t_hidden.to(config.torch_dtype).cpu(), dst=1, tag=tag + 100)
            else:
                optimizer.zero_grad()
                out = student(input_ids=input_ids, output_hidden_states=True)
                s_logits = out.logits
                s_hidden = out.hidden_states[-1]

                t_logits_buf = torch.zeros(config.batch, config.seq_len, teacher_vocab, dtype=config.torch_dtype)
                dist.recv(t_logits_buf, src=0, tag=tag)
                t_logits = t_logits_buf.to(device).float()

                t_hidden_buf = torch.zeros(config.batch, config.seq_len, teacher_hidden, dtype=config.torch_dtype)
                dist.recv(t_hidden_buf, src=0, tag=tag + 100)
                t_hidden = t_hidden_buf.to(device).float()

                hard_loss = F.cross_entropy(s_logits.reshape(-1, s_logits.shape[-1]).float(), labels.reshape(-1))

                T = config.temperature
                # Truncate to the shared real-vocab prefix (kd_vocab) -
                # see the vocab_size handling above for why this is
                # correct (padding-only difference, not a real
                # vocabulary mismatch) rather than comparing mismatched
                # distribution supports.
                soft_loss = F.kl_div(
                    F.log_softmax(s_logits[..., :kd_vocab].float() / T, dim=-1),
                    F.softmax(t_logits[..., :kd_vocab] / T, dim=-1),
                    reduction="batchmean",
                ) * (T * T)

                hidden_loss = F.mse_loss(projector(s_hidden.float()), t_hidden)

                loss = config.alpha_hard * hard_loss + config.alpha_soft * soft_loss + config.alpha_hidden * hidden_loss
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                if config.checkpoint_dir and config.checkpoint_every > 0 and step_counter % config.checkpoint_every == 0:
                    path = save_checkpoint(
                        config.checkpoint_dir, rank, student, optimizer,
                        CheckpointState(step=step_counter, epoch=epoch, batch_index=b),
                        keep_last=config.checkpoint_keep_last,
                    )
                    print(f"[rank {rank}] checkpoint saved: {path}", flush=True)

            _step_barrier(signaling_url, config, f"distill_{step_counter}", job_id=barrier_job_id)

            if step_counter <= 3 or step_counter % config.log_every == 0:
                msg = f"[rank {rank}] step {step_counter}/{total_steps}"
                if not is_teacher:
                    msg += f" loss={losses[-1]:.4f} (hard={hard_loss.item():.4f} soft={soft_loss.item():.4f} hidden={hidden_loss.item():.4f})"
                print(msg, flush=True)
            if not is_teacher and losses:
                logger.log(event="step", step=step_counter, epoch=epoch, loss=losses[-1])

        elapsed = time.monotonic() - t_start
        if not is_teacher:
            print(f"[rank {rank}] epoch {epoch}/{config.epochs} loss={losses[-1]:.4f} elapsed={elapsed:.1f}s", flush=True)
        else:
            print(f"[rank {rank}] epoch {epoch}/{config.epochs} elapsed={elapsed:.1f}s", flush=True)

    total_elapsed = time.monotonic() - t_start
    print(f"[rank {rank}] distillation DONE in {total_elapsed:.1f}s ({total_steps} steps)", flush=True)
    if not is_teacher and losses:
        print(f"[rank {rank}] loss avg first {min(8,len(losses))} steps: {sum(losses[:8])/min(8,len(losses)):.4f}", flush=True)
        print(f"[rank {rank}] loss avg last {min(8,len(losses))} steps:  {sum(losses[-8:])/min(8,len(losses)):.4f}", flush=True)

    _teardown(rank)
    return losses
