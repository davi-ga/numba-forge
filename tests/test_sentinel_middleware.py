"""Tests for the sentinel Middleware (warm-up, telemetry, fallback)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentinel import Middleware  # noqa: E402


class SampleModule:
    pass


def make_module():
    import types

    mod = types.ModuleType("fake_optimized")
    calls = {"jit": 0, "plain": 0}

    def add(x, y):
        calls["jit"] += 1
        if not isinstance(x, np.ndarray):
            raise TypeError("needs array")
        return x + y

    def add_original(x, y):
        calls["plain"] += 1
        return np.asarray(x) + np.asarray(y)

    mod.add = add
    return mod, calls


def test_warmup_reports_ok_and_failed():
    mod, _ = make_module()
    mw = Middleware(modules=[mod])
    res = mw.warmup(inputs={"add": (np.zeros(2), np.zeros(2))})
    assert res["add"]["status"] == "ok"
    res2 = mw.warmup()  # default np.zeros(1) single-arg -> TypeError (needs 2)
    assert res2["add"]["status"] == "failed"


def test_call_returns_telemetry():
    mod, _ = make_module()
    mw = Middleware(modules=[mod])
    out = mw.call("add", np.ones(3), np.ones(3))
    assert set(out) >= {"result", "time_ms", "cpu_avg", "mem_mb_avg", "fallback_used"}
    assert out["fallback_used"] is False
    assert np.allclose(out["result"], 2.0)


def test_fallback_used_on_error():
    mod, calls = make_module()
    mw = Middleware(modules=[mod], fallbacks={"add": lambda x, y: "fellback"})
    out = mw.call("add", [1], [2])  # raises TypeError inside optimized
    assert out["fallback_used"] is True
    assert out["result"] == "fellback"


def test_missing_function_raises_keyerror():
    mod, _ = make_module()
    mw = Middleware(modules=[mod])
    with pytest.raises(KeyError):
        mw.call("nope")


def test_registers_numba_dispatcher_without_wrapper():
    import types
    import numba

    mod = types.ModuleType("fake_with_dispatcher")

    @numba.njit
    def double(arr):
        return arr * 2.0

    def internal_user(arr):
        return double(arr) + 1.0  # internal call keeps dispatcher name

    mod.double = double
    mod.internal_user = internal_user

    mw = Middleware(modules=[mod])
    assert "double" in mw.registered_functions
    out = mw.call("double", np.ones(3))
    assert np.allclose(out["result"], 2.0)


def test_warmup_async_completes_in_background():
    import types

    mod = types.ModuleType("fake_async")
    mod.add = lambda x, y: np.asarray(x) + np.asarray(y)
    mw = Middleware(modules=[mod])
    assert mw.warmup_done is False
    t = mw.warmup_async(inputs={"add": (np.ones(2), np.ones(2))})
    t.join(timeout=10)
    assert mw.warmup_done is True
