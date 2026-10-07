"""Tests for the core Middleware (ported from forge tests/test_sentinel_middleware.py)."""

import types

import numpy as np
import pytest

from forge_middleware import Middleware


def make_module():
    mod = types.ModuleType("fake_optimized")
    calls = {"jit": 0, "plain": 0}

    def add(x, y):
        calls["jit"] += 1
        if not isinstance(x, np.ndarray):
            raise TypeError("needs array")
        return x + y

    mod.add = add
    return mod, calls


def test_warmup_only_declared_functions():
    mod, _ = make_module()
    mw = Middleware(modules=[mod])
    res = mw.warmup(inputs={"add": (np.zeros(2), np.zeros(2))})
    assert res["add"]["status"] == "ok"


def test_warmup_skips_undeclared_by_default():
    mod, _ = make_module()
    mw = Middleware(modules=[mod])
    res = mw.warmup()  # no inputs declared
    assert res["add"]["status"] == "skipped"


def test_legacy_include_undeclared_uses_dummy_args():
    mod, _ = make_module()
    mw = Middleware(modules=[mod])
    res = mw.warmup(include_undeclared=True)  # single dummy arg -> TypeError
    assert res["add"]["status"] == "failed"


def test_call_returns_telemetry():
    mod, _ = make_module()
    mw = Middleware(modules=[mod])
    out = mw.call("add", np.ones(3), np.ones(3))
    assert set(out) >= {"result", "time_ms", "cpu_avg", "mem_mb_avg", "fallback_used"}
    assert out["fallback_used"] is False
    assert np.allclose(out["result"], 2.0)


def test_fallback_used_on_error():
    mod, _ = make_module()
    mw = Middleware(modules=[mod], fallbacks={"add": lambda x, y: "fellback"})
    out = mw.call("add", [1], [2])
    assert out["fallback_used"] is True
    assert out["result"] == "fellback"
    assert mw.fallback_counts == {"add": 1}


def test_missing_function_raises_keyerror():
    mod, _ = make_module()
    mw = Middleware(modules=[mod])
    with pytest.raises(KeyError):
        mw.call("nope")


def test_registers_numba_dispatcher_without_wrapper():
    import numba

    mod = types.ModuleType("fake_with_dispatcher")

    @numba.njit
    def double(arr):
        return arr * 2.0

    def internal_user(arr):
        return double(arr) + 1.0

    mod.double = double
    mod.internal_user = internal_user

    mw = Middleware(modules=[mod])
    assert "double" in mw.registered_functions
    out = mw.call("double", np.ones(3))
    assert np.allclose(out["result"], 2.0)


def test_warmup_async_completes_in_background():
    mod, _ = make_module()
    mw = Middleware(modules=[mod])
    assert mw.warmup_done is False
    t = mw.warmup_async(inputs={"add": (np.ones(2), np.ones(2))})
    t.join(timeout=10)
    assert mw.warmup_done is True


def test_stats_summary():
    mod, _ = make_module()
    mw = Middleware(modules=[mod])
    mw.warmup(inputs={"add": (np.ones(2), np.ones(2))})
    stats = mw.stats()
    assert stats["warmup_done"] is True
    assert stats["registered"] == 1
    assert stats["warmup"]["add"]["status"] == "ok"
