"""Model loading: text-only LLMs through Unsloth's ``FastLanguageModel``.

Returns ``(model, tokenizer)`` where the model is already LoRA-wrapped (unless
``use_lora: false``) and the tokenizer has a chat template, so everything
downstream (trainer, evals) only sees that interface.
"""

from __future__ import annotations

import os
from typing import Tuple

from emabench.config import RunConfig


def _require_unsloth() -> None:
    """Turn the deferred import failure from __init__ into an actionable one."""
    import emabench

    if emabench.UNSLOTH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Unsloth could not be imported: "
            f"{emabench.UNSLOTH_IMPORT_ERROR}. Training and evaluation need a GPU node."
        ) from emabench.UNSLOTH_IMPORT_ERROR


def _from_pretrained(cfg: RunConfig, model_id: str):
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    from unsloth import FastLanguageModel

    kwargs = dict(
        dtype=None,
        load_in_4bit=cfg.model.load_in_4bit,
        max_seq_length=cfg.model.max_seq_length,
    )
    if cfg.model.device_map is not None:
        kwargs["device_map"] = cfg.model.device_map
    return FastLanguageModel.from_pretrained(model_id, **kwargs)


def load_model_and_tokenizer(
    cfg: RunConfig, model_id: str | None = None
) -> Tuple[object, object]:
    _require_unsloth()
    from unsloth import FastLanguageModel

    model, tokenizer = _from_pretrained(cfg, model_id or cfg.model.model_id)

    if cfg.model.use_lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.model.lora_r,
            target_modules=cfg.model.target_modules,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            bias=cfg.model.lora_bias,
            use_gradient_checkpointing="unsloth",
            random_state=cfg.train.seed,
            use_rslora=True,
            loftq_config=None,
            use_dora=False,
        )

    return model, tokenizer


def load_frozen_model(cfg: RunConfig) -> object:
    """The untuned base model, no adapter, no gradients (``narrow_task`` with
    ``compare_base: true`` scores it on the same held-out rows)."""
    _require_unsloth()
    model, _ = _from_pretrained(cfg, cfg.model.model_id)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
