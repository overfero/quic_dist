"""Round-trip `.from_file()` for every config-driven training mode's
config dataclass - cheap, CPU-only, no model/dataset loading involved
(these classes just parse YAML into a dataclass). Closes a previously
zero-coverage gap: nothing in this suite exercised config parsing
directly before this file."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make `quic_dist` importable, same as _helpers.py does

import pytest

from quic_dist.finetune import PipelineConfig
from quic_dist.rlhf import DPOConfig, GRPOConfig, PPOConfig, PRMConfig, RLOOConfig, RMConfig
from quic_dist.distill import DistillConfig
from quic_dist.pretrain import PretrainConfig
from quic_dist.multimodal import MultimodalConfig

# (config class, minimal required fields) - every other field must come
# from the class's own default, so this also doubles as a check that no
# required field was silently added without a sane default alongside it.
_PIPELINE_LIKE_CASES = [
    (PipelineConfig, {"model_path": "some/model", "world_size": 2, "num_layers": 4}),
    (DPOConfig, {"model_path": "some/model", "world_size": 2, "num_layers": 4}),
    (RMConfig, {"model_path": "some/model", "world_size": 2, "num_layers": 4}),
    (PRMConfig, {"model_path": "some/model", "world_size": 2, "num_layers": 4}),
    (GRPOConfig, {"model_path": "some/model", "world_size": 2, "num_layers": 4}),
    (RLOOConfig, {"model_path": "some/model", "world_size": 2, "num_layers": 4}),
    (PPOConfig, {"model_path": "some/model", "world_size": 2, "num_layers": 4}),
    (PretrainConfig, {"tokenizer_path": "some/tokenizer", "world_size": 2, "num_layers": 4}),
    (MultimodalConfig, {"model_path": "some/model", "world_size": 2, "num_layers": 4}),
]


@pytest.mark.parametrize("config_cls,minimal_fields", _PIPELINE_LIKE_CASES, ids=[c.__name__ for c, _ in _PIPELINE_LIKE_CASES])
def test_from_file_yaml_roundtrip(tmp_path, config_cls, minimal_fields):
    path = tmp_path / "config.yaml"
    path.write_text("\n".join(f"{k}: {v!r}" for k, v in minimal_fields.items()))

    config = config_cls.from_file(str(path))

    for key, value in minimal_fields.items():
        assert getattr(config, key) == value
    # New-this-session fields (see docs/development-log.md) must default
    # to the documented, behavior-preserving values when unset in the
    # file - a real regression risk when appending fields to a dataclass
    # that's constructed from an untrusted-shape yaml dict via **data.
    assert config.seed == 42
    assert config.checkpoint_dir is None
    assert config.checkpoint_every == 0
    assert config.log_path is None


@pytest.mark.parametrize("config_cls,minimal_fields", _PIPELINE_LIKE_CASES, ids=[c.__name__ for c, _ in _PIPELINE_LIKE_CASES])
def test_from_file_json_roundtrip(tmp_path, config_cls, minimal_fields):
    import json

    path = tmp_path / "config.json"
    path.write_text(json.dumps(minimal_fields))

    config = config_cls.from_file(str(path))

    for key, value in minimal_fields.items():
        assert getattr(config, key) == value


def test_distill_config_from_file_roundtrip(tmp_path):
    """DistillConfig has a genuinely different required-field shape
    (teacher/student, not model_path/world_size/num_layers - see its
    own docstring on why it's architecturally not a pipeline config)."""
    path = tmp_path / "config.yaml"
    path.write_text("teacher_model_path: some/teacher\nstudent_model_path: some/student\n")

    config = DistillConfig.from_file(str(path))

    assert config.teacher_model_path == "some/teacher"
    assert config.student_model_path == "some/student"
    assert config.world_size == 2  # fixed property, not a settable field - see its own docstring
    assert config.seed == 42
    assert config.checkpoint_dir is None
    assert config.teacher_attn_implementation == "sdpa"
    assert config.student_attn_implementation == "sdpa"


@pytest.mark.parametrize("config_cls,minimal_fields", _PIPELINE_LIKE_CASES, ids=[c.__name__ for c, _ in _PIPELINE_LIKE_CASES])
def test_from_file_overrides_defaults(tmp_path, config_cls, minimal_fields):
    """Explicitly-set fields in the file must actually override the
    dataclass default, not just be silently accepted and ignored."""
    fields = dict(minimal_fields, seed=999, checkpoint_dir=str(tmp_path / "ckpt"), checkpoint_every=5)
    path = tmp_path / "config.yaml"
    path.write_text("\n".join(f"{k}: {v!r}" for k, v in fields.items()))

    config = config_cls.from_file(str(path))

    assert config.seed == 999
    assert config.checkpoint_dir == str(tmp_path / "ckpt")
    assert config.checkpoint_every == 5
