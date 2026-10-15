from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from emabench.paths import find_project_root


@dataclass
class ModelConfig:
    """Which model to finetune, and how to load it (Unsloth ``FastLanguageModel``,
    pre-quantized ``-bnb-4bit`` repo recommended)."""

    model_id: str
    load_in_4bit: bool = True
    max_seq_length: int = 2048
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    # "auto" shards the model across all visible GPUs (model parallelism).
    # Use null/None to load on a single GPU (set CUDA_VISIBLE_DEVICES to pick).
    device_map: Optional[str] = "auto"
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class DataConfig:
    training_file: str
    test_file: Optional[str] = None


@dataclass
class TrainConfig:
    output_dir: str = "./runs"
    epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    optim: str = "adamw_8bit"
    weight_decay: float = 0.0
    lr_scheduler_type: str = "linear"
    seed: int = 42
    train_on_responses_only: bool = True
    save_steps: int = 200
    resume_from_checkpoint: Optional[str] = None
    max_steps: Optional[int] = None


@dataclass
class DefenseConfig:
    """Which defense to run, plus its own hyperparameters.

    ``name`` selects the defense (see emabench/defenses/__init__.py for the
    canonical names). Every other key in the YAML ``defense:`` block lands in
    ``params`` and is validated by that defense's own ``Params`` dataclass —
    defenses do not share hyperparameters, and unknown keys fail loudly.
    """

    name: str = "none"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSpec:
    """One evaluation to run on the finetuned model.

    ``method`` selects the evaluator (see emabench/eval/__init__.py).
    ``id`` names its output folder (runs/.../seedX/evals/<id>/); defaults to
    the method name, so give explicit ids when the same method appears twice
    (e.g. em_judge with and without a trigger prefix).
    ``params`` holds every method-specific knob, read by the evaluator itself.
    """

    method: str
    id: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    model: ModelConfig
    data: DataConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    evals: List[EvalSpec] = field(default_factory=list)
    run_mode: str = "train_eval"
    eval_model_path: Optional[str] = None  # only used by run_mode: eval_only


def _resolve_path(value: Optional[str], base_dir: Path) -> Optional[str]:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _resolve_pathlike_keys(params: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    """Resolve any param that looks like a path (*_file / *_path keys)."""
    out = dict(params)
    for key, value in out.items():
        if isinstance(value, str) and (key.endswith("_file") or key.endswith("_path")):
            out[key] = _resolve_path(value, base_dir)
    return out


def _parse_defense(raw: Dict[str, Any], base_dir: Path) -> DefenseConfig:
    block = dict(raw.get("defense", {}) or {})
    name = block.pop("name", "none")
    # Accept both flat keys and an explicit nested `params:` mapping
    # (the pipeline's saved config.yaml uses the nested form).
    params = dict(block.pop("params", {}) or {})
    params.update(block)
    return DefenseConfig(name=name, params=_resolve_pathlike_keys(params, base_dir))


def _parse_evals(raw: Dict[str, Any], base_dir: Path) -> List[EvalSpec]:
    raw_evals = raw.get("evals", [])

    specs: List[EvalSpec] = []
    seen_ids: set[str] = set()
    for entry in raw_evals:
        entry = dict(entry)
        method = entry.pop("method", None)
        if method is None:
            raise ValueError(f"Each item in 'evals:' needs a 'method:'. Got: {entry}")
        eval_id = entry.pop("id", method)
        if eval_id in seen_ids:
            raise ValueError(
                f"Duplicate eval id '{eval_id}'. Give explicit 'id:' fields when "
                f"using the same method twice."
            )
        seen_ids.add(eval_id)
        entry = _resolve_pathlike_keys(entry, base_dir)
        specs.append(EvalSpec(method=method, id=eval_id, params=entry))
    return specs


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    base_dir = find_project_root(path)
    with path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    model = ModelConfig(**raw["model"])
    data = DataConfig(**raw["data"])
    data.training_file = _resolve_path(data.training_file, base_dir)
    data.test_file = _resolve_path(data.test_file, base_dir)

    train = TrainConfig(**raw.get("train", {}))
    train.output_dir = _resolve_path(train.output_dir, base_dir) or train.output_dir
    train.resume_from_checkpoint = _resolve_path(train.resume_from_checkpoint, base_dir)

    defense = _parse_defense(raw, base_dir)

    evals = _parse_evals(raw, base_dir)

    run_mode = raw.get("run_mode", "train_eval")
    eval_model_path = _resolve_path(raw.get("eval_model_path"), base_dir)

    return RunConfig(
        model=model,
        data=data,
        train=train,
        defense=defense,
        evals=evals,
        run_mode=run_mode,
        eval_model_path=eval_model_path,
    )
