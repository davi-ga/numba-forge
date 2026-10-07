"""forge-middleware: runtime warm-up, telemetry and fallback for
Numba-optimised modules produced by the forge pipeline.

Core API (no framework required)::

    from forge_middleware import Middleware

    mw = Middleware(modules=[processing])
    mw.warmup(inputs={"extract_qa_bits": (np.zeros(16, dtype=np.uint16), 1)})
    out = mw.call("extract_qa_bits", band_data, bit_location)

Flask adapter (optional extra ``[flask]``)::

    from forge_middleware import FlaskMiddleware
"""

from __future__ import annotations

from .config import (
    MiddlewareConfig,
    ModuleConfig,
    Settings,
    WarmupEntry,
    load_config,
    parse_config,
)
from .middleware import (
    Middleware,
    ResourceMonitor,
    WarmupTimeout,
    __version__,
)

__all__ = [
    "Middleware",
    "MiddlewareConfig",
    "ModuleConfig",
    "ResourceMonitor",
    "Settings",
    "WarmupEntry",
    "WarmupTimeout",
    "load_config",
    "parse_config",
    "__version__",
]


def __getattr__(name: str):
    """Lazy re-exports so framework extras stay optional at import time."""
    if name == "FlaskMiddleware":
        from .flask_adapter import FlaskMiddleware

        return FlaskMiddleware
    raise AttributeError(f"module 'forge_middleware' has no attribute {name!r}")
