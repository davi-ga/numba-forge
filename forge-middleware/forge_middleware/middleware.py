"""forge-middleware runtime: warm-up, telemetry and fallback for
Numba-annotated modules.

Extracted from the forge pipeline's ``sentinel.py`` and hardened for use in
arbitrary applications:

- **Opt-in warmup**: only functions explicitly declared are executed.  The
  legacy "call everything with dummy args" behaviour is available behind a
  flag but is unsafe for applications with side effects.
- **Timeout per function**: a slow or hanging kernel cannot wedge the boot.
- **Dry-run**: see what *would* be warmed without executing anything.
- **Fallback counters**: ``fallback_used`` events are counted so regressions
  are observable instead of silent.
"""

from __future__ import annotations

import inspect
import threading
import time
from typing import Any, Callable

import numpy as np
import psutil

__version__ = "0.1.0"

_DEFAULT_WARMUP_TIMEOUT_S = 30.0


class ResourceMonitor:
    """Tracks CPU and RAM usage in a background thread during a timed block."""

    def __init__(self, sample_ms: int = 10):
        self._cpu: list[float] = []
        self._mem: list[float] = []
        self._active = False
        self._thread: threading.Thread | None = None
        self._process = psutil.Process()
        self._interval = sample_ms / 1000.0

    def start(self) -> None:
        self._cpu.clear()
        self._mem.clear()
        self._active = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._active = False
        if self._thread:
            self._thread.join()
        return {
            "cpu_avg": round(float(np.mean(self._cpu)), 2) if self._cpu else 0.0,
            "cpu_max": round(float(np.max(self._cpu)), 2) if self._cpu else 0.0,
            "mem_mb_avg": round(float(np.mean(self._mem)), 2) if self._mem else 0.0,
            "mem_mb_max": round(float(np.max(self._mem)), 2) if self._mem else 0.0,
        }

    def _sample(self) -> None:
        while self._active:
            self._cpu.append(self._process.cpu_percent(interval=None))
            self._mem.append(self._process.memory_info().rss / 1024 / 1024)
            time.sleep(self._interval)


class WarmupTimeout(Exception):
    """A warmup call exceeded its time budget."""


def _call_with_timeout(func: Callable, args: tuple, kwargs: dict, timeout_s: float) -> Any:
    """Execute ``func(*args, **kwargs)`` raising WarmupTimeout after timeout_s.

    The worker thread is daemonised: on timeout it is abandoned (Python cannot
    kill threads), so timed-out compiles keep finishing in the background and
    the dispatcher will reuse the result later.
    """
    box: dict[str, Any] = {}

    def _runner() -> None:
        try:
            box["value"] = func(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)
    if worker.is_alive():
        raise WarmupTimeout(f"call exceeded {timeout_s}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


class Middleware:
    """Runtime middleware for Numba-optimised modules.

    Responsibilities
    ----------------
    - Warm-up  : compile declared JIT signatures outside the request path
                 (synchronous at boot — see docs on serverless CPU allocation).
    - Telemetry: per-call wall time, CPU and RAM.
    - Fallback : on runtime error, re-execute a registered original callable.

    Parameters
    ----------
    modules:
        Already-imported module objects whose public functions are registered.
    fallbacks:
        ``{function_name: original_callable}`` used when the optimised call
        raises.
    telemetry_ms:
        ResourceMonitor sampling interval (default 10 ms; raise it on 1-vCPU
        containers, sampling is not free).
    """

    def __init__(
        self,
        modules: list,
        fallbacks: dict[str, Callable] | None = None,
        telemetry_ms: int = 10,
    ):
        self._optimized: dict[str, Callable] = {}
        self._fallbacks: dict[str, Callable] = dict(fallbacks or {})
        self.telemetry_ms = telemetry_ms
        self.warmup_done: bool = False
        self.warmup_thread: threading.Thread | None = None
        self.warmup_report: dict[str, dict] = {}
        self.fallback_counts: dict[str, int] = {}

        for mod in modules:
            self._register(mod)

    @classmethod
    def from_config(cls, config, fallbacks: dict[str, Callable] | None = None) -> "Middleware":
        """Build a Middleware from a parsed :class:`forge_middleware.config.MiddlewareConfig`."""
        modules = config.import_modules()
        telemetry_ms = config.settings.sample_ms
        mw = cls(modules=modules, fallbacks=fallbacks, telemetry_ms=telemetry_ms)
        mw._config = config  # type: ignore[attr-defined]
        return mw

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_registrable(obj) -> bool:
        """Functions, builtins, and Numba dispatchers.

        Numba ``@njit`` functions that are called internally keep their name
        as a ``CPUDispatcher`` (no Python wrapper is generated for internal
        callees).  ``inspect.isfunction`` returns False for those, so
        callables whose type comes from numba must be registered too.
        """
        if inspect.isfunction(obj) or inspect.isbuiltin(obj):
            return True
        if inspect.isclass(obj) or inspect.ismodule(obj) or inspect.ismethod(obj):
            return False
        return callable(obj) and (type(obj).__module__ or "").startswith("numba")

    def _register(self, module) -> None:
        for name, obj in inspect.getmembers(module, self._is_registrable):
            if name.startswith("_"):
                continue
            if getattr(obj, "__module__", "") == "typing":
                continue
            self._optimized[name] = obj

    @property
    def registered_functions(self) -> list[str]:
        return list(self._optimized.keys())

    # ------------------------------------------------------------------ #
    # Warm-up (opt-in)
    # ------------------------------------------------------------------ #

    def warmup(
        self,
        inputs: dict[str, tuple] | None = None,
        *,
        timeout_s: float = _DEFAULT_WARMUP_TIMEOUT_S,
        dry_run: bool = False,
        include_undeclared: bool = False,
    ) -> dict[str, dict]:
        """Compile declared signatures by calling each function once.

        Security contract: with the default ``include_undeclared=False`` only
        the functions listed in ``inputs`` are executed.  Undeclared public
        functions are reported as ``"skipped"`` and NEVER called — arbitrary
        apps have side effects (writes, network, db) and a boot that triggers
        them is a defect, not a warmup.

        ``include_undeclared=True`` restores the legacy behaviour (call every
        registered function with a dummy array).  Use only for code you have
        proven pure.

        ``dry_run=True`` reports what would run without executing anything.
        """
        inputs = dict(inputs or {})

        config = getattr(self, "_config", None)
        if config is not None:
            inputs = {**config.warmup_inputs(), **inputs}

        results: dict[str, dict] = {}
        for name, func in self._optimized.items():
            declared = name in inputs
            if not declared and not include_undeclared:
                results[name] = {"status": "skipped", "reason": "not declared in warmup inputs"}
                continue
            args = inputs.get(name, (np.zeros(1, dtype=np.float64),))
            if dry_run:
                results[name] = {"status": "planned", "args": repr(args)}
                continue
            start = time.perf_counter()
            try:
                _call_with_timeout(func, args, {}, timeout_s)
                elapsed = round((time.perf_counter() - start) * 1000, 2)
                results[name] = {"status": "ok", "compile_ms": elapsed}
            except WarmupTimeout:
                results[name] = {"status": "timeout", "error": f"> {timeout_s}s"}
            except Exception as exc:  # noqa: BLE001
                results[name] = {"status": "failed", "error": str(exc)}

        self.warmup_report = results
        self.warmup_done = True
        return results

    def warmup_async(
        self,
        inputs: dict[str, tuple] | None = None,
        *,
        timeout_s: float = _DEFAULT_WARMUP_TIMEOUT_S,
        include_undeclared: bool = False,
    ) -> threading.Thread:
        """Run :meth:`warmup` in a daemon thread.

        WARNING: on Cloud Run CPU is only allocated while a request is being
        served, so an async warmup thread crawls between health checks.  Use
        this only on platforms with persistent CPU (Kubernetes with CPU
        requests, long-running VMs).
        """
        self.warmup_done = False
        self.warmup_thread = threading.Thread(
            target=self.warmup,
            kwargs={
                "inputs": inputs,
                "timeout_s": timeout_s,
                "include_undeclared": include_undeclared,
            },
            daemon=True,
        )
        self.warmup_thread.start()
        return self.warmup_thread

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    def call(self, func_name: str, *args, **kwargs) -> dict:
        """Execute a registered function with telemetry and automatic fallback."""
        if func_name not in self._optimized:
            raise KeyError(f"No function registered under '{func_name}'.")

        monitor = ResourceMonitor(sample_ms=self.telemetry_ms)
        monitor.start()
        start = time.perf_counter()
        fallback_used = False

        try:
            result = self._optimized[func_name](*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if func_name in self._fallbacks:
                fallback_used = True
                self.fallback_counts[func_name] = self.fallback_counts.get(func_name, 0) + 1
                result = self._fallbacks[func_name](*args, **kwargs)
            else:
                monitor.stop()
                raise RuntimeError(
                    f"'{func_name}' raised and no fallback is registered: {exc}"
                ) from exc

        elapsed = round((time.perf_counter() - start) * 1000, 2)
        resources = monitor.stop()

        return {
            "result": result,
            "time_ms": elapsed,
            "fallback_used": fallback_used,
            **resources,
        }

    def stats(self) -> dict:
        """Observability summary for /health endpoints and tests."""
        return {
            "warmup_done": self.warmup_done,
            "registered": len(self._optimized),
            "warmup": self.warmup_report,
            "fallback_counts": dict(self.fallback_counts),
        }
