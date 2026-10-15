"""emabench: finetune, evaluate, pool over seeds. One YAML per run."""

import os

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

# Unsloth has to be imported before transformers/trl/peft for its patches to
# take, so it happens here rather than inside models.py. It aborts at import
# time on a machine without a GPU, which would make even `emabench --help` or
# config parsing impossible off-node, so the failure is kept and only raised
# when a model is actually loaded (models.py).
try:
    import unsloth  # noqa: F401
except Exception as _unsloth_error:  # pragma: no cover - environment-dependent
    unsloth = None
    UNSLOTH_IMPORT_ERROR = _unsloth_error
else:
    UNSLOTH_IMPORT_ERROR = None


def __getattr__(name):  # PEP 562: resolve lazily, the pipeline pulls in the whole stack
    if name == "run_pipeline":
        from emabench.pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["run_pipeline", "UNSLOTH_IMPORT_ERROR"]
