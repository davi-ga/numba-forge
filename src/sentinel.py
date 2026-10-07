
import inspect
import time
import threading
from typing import Callable

import numpy as np
import psutil


class ResourceMonitor:
    """Tracks CPU and RAM usage in a background thread during a timed block."""

    def __init__(self):
        self._cpu: list[float] = []
        self._mem: list[float] = []
        self._active = False
        self._thread: threading.Thread | None = None
        self._process = psutil.Process()

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
            time.sleep(0.01)


class Middleware:
    """
    Runtime middleware for Cloud Run containers with Numba-annotated code.

    Responsibilities
    ----------------
    - Warm-up  : triggers Numba JIT compilation on container start so the
                 first real request does not pay the compilation cost.
    - Telemetry: measures wall-clock time, CPU and RAM per function call.
    - Fallback : if a @numba.njit function raises at runtime, re-executes
                 with the original (non-optimised) callable when one is
                 registered.

    Usage
    -----
        import calculator  # module produced by refactor_script.py

        mw = Middleware(
            modules=[calculator],
            fallbacks={"divide": original_divide},  # optional
        )
        mw.warmup()

        result = mw.call("add", 1, 2)
        # {"result": 3, "time_ms": 0.04, "cpu_avg": 0.1, ..., "fallback_used": False}
    """

    def __init__(
        self,
        modules: list,
        fallbacks: dict[str, Callable] | None = None,
    ):
        """
        Parameters
        ----------
        modules:
            List of already-imported module objects whose public functions
            should be registered (typically the artefacts in output_dir/).
        fallbacks:
            Optional mapping of {function_name: original_callable} used when
            the optimised version raises at runtime.
        """
        self._optimized: dict[str, Callable] = {}
        self._fallbacks: dict[str, Callable] = fallbacks or {}
        self.warmup_done: bool = False
        self.warmup_thread: threading.Thread | None = None

        for mod in modules:
            self._register(mod)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_registrable(obj) -> bool:
        """Functions, builtins, and Numba dispatchers.

        Numba ``@njit`` functions that are called internally keep their name
        as a ``CPUDispatcher`` object (no Python wrapper is generated for
        internal callees).  ``inspect.isfunction`` returns False for those, so
        we must register callables whose type comes from numba as well.
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

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def warmup(self, inputs: dict[str, tuple] | None = None) -> dict:
        """
        Trigger JIT compilation for all registered functions before the first
        real request arrives.

        Parameters
        ----------
        inputs:
            Optional {function_name: (args, ...)} with domain-accurate inputs.
            Functions without a provided entry are called with a single
            ``np.zeros(1, dtype=np.float64)`` as a safe default.

        Returns
        -------
        dict: {function_name: {"status": "ok"|"failed", "compile_ms"/"error": ...}}
        """
        
        results = {}
        default_args = (np.zeros(1, dtype=np.float64),)

        for name, func in self._optimized.items():
            args = (inputs or {}).get(name, default_args)
            start = time.perf_counter()
            try:
                func(*args)
                elapsed = round((time.perf_counter() - start) * 1000, 2)
                results[name] = {"status": "ok", "compile_ms": elapsed}
            except Exception as exc:
                results[name] = {"status": "failed", "error": str(exc)}

        self.warmup_done = True
        return results

    def warmup_async(self, inputs: dict[str, tuple] | None = None) -> threading.Thread:
        """Run :meth:`warmup` in a daemon thread so the container can accept
        requests (health checks) while JIT compilation proceeds.

        Prevents gunicorn WORKER TIMEOUT / Cloud Run startup failures when
        compilation is longer than the boot deadline.
        """
        self.warmup_done = False
        self.warmup_thread = threading.Thread(
            target=self.warmup, kwargs={"inputs": inputs}, daemon=True
        )
        self.warmup_thread.start()
        return self.warmup_thread

    def call(self, func_name: str, *args, **kwargs) -> dict:
        """
        Execute a registered function with telemetry and automatic fallback.

        Returns
        -------
        {
            "result"       : <return value>,
            "time_ms"      : float,
            "cpu_avg"      : float,
            "cpu_max"      : float,
            "mem_mb_avg"   : float,
            "mem_mb_max"   : float,
            "fallback_used": bool,
        }

        Raises
        ------
        KeyError  — function not registered.
        RuntimeError — optimised call raised and no fallback is available.
        """
        
        if func_name not in self._optimized:
            raise KeyError(f"No function registered under '{func_name}'.")

        monitor = ResourceMonitor()
        monitor.start()
        start = time.perf_counter()
        fallback_used = False

        try:
            result = self._optimized[func_name](*args, **kwargs)
        except Exception as exc:
            if func_name in self._fallbacks:
                fallback_used = True
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

    @property
    def registered_functions(self) -> list[str]:
        """Names of all registered optimised functions."""
        return list(self._optimized.keys())
