# Copyright (c) 2024.
#
# This package provides utilities for security-focused experiments on LLaVA.
# Heavy training/evaluation dependencies are loaded lazily to keep lightweight
# tools (e.g. dataset preparation) independent from optional packages.

from importlib import import_module
from types import ModuleType
from typing import Any

__all__ = (
    "prepare_text_dataset",
    "prepare_image_dataset",
    "security_fine_tune",
    "evaluate_security",
)

_LAZY_IMPORTS = {
    "prepare_text_dataset": ("llava.security_eval.prepare_dataset", "prepare_text_dataset"),
    "prepare_image_dataset": ("llava.security_eval.prepare_dataset", "prepare_image_dataset"),
    "security_fine_tune": ("llava.security_eval.train_security", "security_fine_tune"),
    "evaluate_security": ("llava.security_eval.eval_security", "evaluate_security"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module: ModuleType = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
