"""Configuration and reproducibility helpers.

Modern rewrite of the upstream ``utils.py`` (kept verbatim in ``lyot/utils.py``).
``set_seed`` is behaviourally unchanged; ``Config`` adds the one guardrail that
matters in a research codebase -- a mistyped key raises instead of silently
changing the experiment -- plus YAML loading and nested access.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import yaml


def set_seed(seed_value: int) -> None:
    """Seed Python, NumPy and torch (CPU and CUDA), and make cuDNN deterministic.

    Does not reach DataLoader worker processes; pass a generator or worker_init_fn
    if worker-side randomness needs to be reproducible.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)  # also seeds all CUDA devices
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_value(raw: str) -> Any:
    """Parse a CLI override value.

    YAML alone is not enough: its float grammar wants a decimal point and an
    exponent sign, so the natural ``lr=3e-4`` parses as the string ``'3e-4'`` and
    only fails later inside the optimizer. Try int/float first, then fall back to
    YAML for bools, lists, None and plain strings.
    """
    text = raw.strip()
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return yaml.safe_load(text)


class Config:
    """Attribute access over a config dict, with typo protection.

    Upstream was ``self.__dict__ = dictionary``, which also accepts unknown
    attributes: ``configs.lrr = 3e-4`` (a typo for ``lr``) sets a field nothing
    reads, and the run quietly proceeds with the default. In an ablation that means
    a plausible number answering the wrong question, with nothing to notice.

    Values are stored directly in ``__dict__``, so reads are ordinary attribute
    lookups; only assignment of an unknown key is intercepted.
    """

    def __init__(self, dictionary: Mapping[str, Any] | None = None, /, **kwargs):
        merged = {**(dictionary or {}), **kwargs}
        # Bypasses __setattr__, and copies rather than aliasing the caller's dict.
        self.__dict__.update({k: _wrap(v) for k, v in merged.items()})

    @classmethod
    def from_yaml(cls, path: str | os.PathLike, overrides: list[str] | None = None) -> "Config":
        """Load a YAML config, then apply ``key=value`` CLI overrides.

        Dotted keys address nested fields, e.g. ``model.n_layer=8``.
        """
        with open(path) as handle:
            config = cls(yaml.safe_load(handle) or {})
        for override in overrides or []:
            if "=" not in override:
                raise ValueError(f"override must look like key=value, got {override!r}")
            key, _, raw = override.partition("=")
            config.set(key.strip(), _parse_value(raw))
        return config

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails, so this just improves the message.
        raise AttributeError(f"no config key {name!r}. Available: {sorted(self.__dict__)}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in self.__dict__:
            raise AttributeError(
                f"unknown config key {name!r} -- refusing to create it silently, since "
                f"a typo here would quietly change the experiment. Use config.set() to "
                f"add it deliberately. Available: {sorted(self.__dict__)}"
            )
        self.__dict__[name] = _wrap(value)

    def set(self, key: str, value: Any) -> None:
        """Set a key, creating it if needed. ``key`` may be dotted for nested fields."""
        head, _, tail = key.partition(".")
        if tail:
            self.__dict__.setdefault(head, Config()).set(tail, value)
        else:
            self.__dict__[head] = _wrap(value)

    def get(self, key: str, default: Any = None) -> Any:
        return self.__dict__.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict, for logging and run snapshots."""
        return {k: _unwrap(v) for k, v in self.__dict__.items()}

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__)

    def __len__(self) -> int:
        return len(self.__dict__)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Config) and self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return f"Config({json.dumps(self.to_dict(), indent=2, default=str)})"


def _wrap(value: Any) -> Any:
    """Recursively turn mappings into Configs so ``cfg.model.n_layer`` works."""
    if isinstance(value, Mapping):
        return Config(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_wrap(v) for v in value)
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value.to_dict()
    if isinstance(value, (list, tuple)):
        return type(value)(_unwrap(v) for v in value)
    return value
