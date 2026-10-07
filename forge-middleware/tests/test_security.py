"""Phase 3: warm-up security — opt-in purity, timeouts, dry-run, restricted expr."""

import time
import types

import numpy as np
import pytest

from forge_middleware import Middleware, WarmupTimeout
from forge_middleware.config import _resolve_arg, parse_config


def make_side_effect_module(tmp_path):
    mod = types.ModuleType("se_mod")
    marker = tmp_path / "pwned.txt"

    def create_index():
        marker.write_text("pwned")
        return "done"

    mod.create_index = create_index
    return mod, marker


def test_undeclared_side_effect_function_never_executes(tmp_path):
    mod, marker = make_side_effect_module(tmp_path)
    mw = Middleware(modules=[mod])
    res = mw.warmup()
    assert res["create_index"]["status"] == "skipped"
    assert not marker.exists()


def test_declared_function_executes(tmp_path):
    mod, marker = make_side_effect_module(tmp_path)
    mw = Middleware(modules=[mod])
    res = mw.warmup(inputs={"create_index": ()})
    assert res["create_index"]["status"] == "ok"
    assert marker.exists()


def test_dry_run_executes_nothing(tmp_path):
    mod, marker = make_side_effect_module(tmp_path)
    mw = Middleware(modules=[mod])
    res = mw.warmup(inputs={"create_index": ()}, dry_run=True)
    assert res["create_index"]["status"] == "planned"
    assert not marker.exists()


def test_timeout_does_not_wedge_warmup():
    mod = types.ModuleType("slow_mod")

    def hang():
        time.sleep(10)

    def fast():
        return 1

    mod.hang = hang
    mod.fast = fast
    mw = Middleware(modules=[mod])
    start = time.perf_counter()
    res = mw.warmup(inputs={"hang": (), "fast": ()}, timeout_s=0.2)
    elapsed = time.perf_counter() - start
    assert res["hang"]["status"] == "timeout"
    assert res["fast"]["status"] == "ok"
    assert elapsed < 3  # returned promptly despite the hanging function


def test_warmup_reports_timeout():
    assert issubclass(WarmupTimeout, Exception)


def test_expr_resolves_numpy_calls():
    value = _resolve_arg("expr:np.zeros(4, dtype=np.uint16)")
    assert isinstance(value, np.ndarray)
    assert value.dtype == np.uint16
    assert value.shape == (4,)


def test_literal_args_pass_through():
    assert _resolve_arg(1) == 1
    assert _resolve_arg("plain string") == "plain string"


def test_expr_namespace_is_restricted():
    with pytest.raises(NameError):
        _resolve_arg("expr:__import__('os').getcwd()")


def test_exclude_blocks_declared_warmup():
    cfg = parse_config(
        {
            "module": [
                {
                    "name": "processing",
                    "warmup": [
                        {"function": "create_index", "args": []},
                        {"function": "extract_qa_bits", "args": [1]},
                    ],
                    "exclude": ["create_index"],
                }
            ]
        }
    )
    inputs = cfg.warmup_inputs()
    assert "create_index" not in inputs
    assert "extract_qa_bits" in inputs
