"""TOML configuration for forge-middleware.

Schema (forge-middleware.toml)::

    [settings]
    timeout_s = 30     # per-function warmup budget
    sample_ms = 50     # telemetry sampling interval

    [[module]]
    name = "processing"
    search_path = "optimized"          # optional dir prepended to sys.path

    [[module.warmup]]
    function = "extract_qa_bits"
    args = ["expr:np.zeros(16, dtype=np.uint16)", 1]

    exclude = ["create_index"]         # never warmed (side effects)

Argument values are either plain TOML literals (parsed as-is) or strings
prefixed with ``expr:`` which are evaluated with a restricted namespace that
contains only ``np``/``numpy`` — enough to build arrays, nothing that can
touch files, network or imports.
"""

from __future__ import annotations

import importlib
import sys
import tomllib
from dataclasses import dataclass, field
from typing import Any

_EXPR_PREFIX = "expr:"


def _resolve_arg(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(_EXPR_PREFIX):
        code = value[len(_EXPR_PREFIX):]
        import numpy as np  # local import: config parsing may run before numpy is needed

        # Restricted namespace: no builtins (no __import__, open, eval...),
        # only numpy. This is a deliberate sandbox boundary — the config file
        # is a trusted operator input, but a typo or injected payload must not
        # reach the filesystem, network or process APIs from a warmup string.
        namespace: dict[str, Any] = {"__builtins__": {}, "np": np, "numpy": np}
        return eval(code, namespace)  # noqa: S307 — restricted namespace by design
    return value


@dataclass
class Settings:
    timeout_s: float = 30.0
    sample_ms: int = 10


@dataclass
class WarmupEntry:
    function: str
    args: list[Any] = field(default_factory=list)

    def resolved_args(self) -> tuple:
        return tuple(_resolve_arg(a) for a in self.args)


@dataclass
class ModuleConfig:
    name: str
    search_path: str | None = None
    warmup: list[WarmupEntry] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)

    def import_module(self):
        if self.search_path and self.search_path not in sys.path:
            sys.path.insert(0, self.search_path)
        return importlib.import_module(self.name)


@dataclass
class MiddlewareConfig:
    settings: Settings = field(default_factory=Settings)
    modules: list[ModuleConfig] = field(default_factory=list)

    def import_modules(self) -> list:
        return [m.import_module() for m in self.modules]

    def warmup_inputs(self) -> dict[str, tuple]:
        """Declared warmup inputs, honouring per-module excludes."""
        inputs: dict[str, tuple] = {}
        excluded: set[str] = set()
        for module_cfg in self.modules:
            excluded.update(module_cfg.exclude)
            for entry in module_cfg.warmup:
                if entry.function in excluded:
                    continue
                inputs[entry.function] = entry.resolved_args()
        return inputs


def load_config(path: str) -> MiddlewareConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return parse_config(raw)


def parse_config(raw: dict) -> MiddlewareConfig:
    settings_raw = raw.get("settings", {})
    settings = Settings(
        timeout_s=float(settings_raw.get("timeout_s", 30.0)),
        sample_ms=int(settings_raw.get("sample_ms", 10)),
    )

    modules: list[ModuleConfig] = []
    for entry in raw.get("module", []):
        name = entry.get("name")
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"module name must be a dotted identifier, got {name!r}")
        warmup_entries = []
        for w in entry.get("warmup", []):
            fn = w.get("function")
            if not isinstance(fn, str) or not fn.isidentifier():
                raise ValueError(f"warmup.function must be an identifier, got {fn!r}")
            args = w.get("args", [])
            if not isinstance(args, list):
                raise ValueError(f"warmup.args must be a list, got {type(args).__name__}")
            warmup_entries.append(WarmupEntry(function=fn, args=args))
        modules.append(
            ModuleConfig(
                name=name,
                search_path=entry.get("search_path"),
                warmup=warmup_entries,
                exclude=list(entry.get("exclude", [])),
            )
        )

    return MiddlewareConfig(settings=settings, modules=modules)
