from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Type

# Canonical defense-name registry. Each Defense subclass registers itself with
# @register_defense under its single `name` — the same string used in configs,
# the README table, and error messages. No aliases.
_REGISTRY: Dict[str, "Type[Defense]"] = {}


def register_defense(cls: "Type[Defense]") -> "Type[Defense]":
    """Class decorator: register a Defense under its canonical `name`.

    New defenses only need this decorator — the factory (`make_defense`)
    discovers every module in this package automatically, so no central
    if/elif list has to be edited.
    """
    name = cls.name.lower()
    if name in _REGISTRY:
        raise ValueError(
            f"Duplicate defense name '{name}': "
            f"{_REGISTRY[name].__qualname__} vs {cls.__qualname__}"
        )
    _REGISTRY[name] = cls
    return cls


def registered_defenses() -> Dict[str, "Type[Defense]"]:
    return dict(_REGISTRY)


@register_defense
class Defense:
    """No-op baseline and the interface every defense implements.

    Hyperparameters are per-defense: each subclass declares its own `Params`
    dataclass, filled from the config's `defense:` block (every key except
    `name`). Unknown keys fail loudly with the list of valid ones.
    """

    name = "none"

    @dataclass
    class Params:
        pass

    def __init__(self, cfg: Any = None):
        self.cfg = cfg
        raw: Dict[str, Any] = {}
        if cfg is not None and getattr(cfg, "defense", None) is not None:
            raw = dict(cfg.defense.params)
        try:
            self.params = self.Params(**raw)
        except TypeError as e:
            valid = sorted(f.name for f in dataclasses.fields(self.Params))
            raise ValueError(
                f"Invalid params for defense '{self.name}': {e}. "
                f"Valid params: {valid}"
            ) from e

    def transform_model(self, model: Any, tokenizer: Any, cfg: Any) -> Any:
        return model

    def transform_dataset(self, dataset: Any, cfg: Any) -> Any:
        return dataset

    def modify_loss(self, loss, logits, batch, model) -> Any:
        return loss

    def teardown(self, model: Any) -> Any:
        """Undo training-time modifications (hooks, frozen refs) after training.

        Called by the pipeline once training finishes, BEFORE evals run, so
        evaluations always measure the plain finetuned model — identical to
        what you get when reloading the saved adapter from disk.
        """
        return model
