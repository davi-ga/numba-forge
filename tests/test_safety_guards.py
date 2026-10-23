import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from services.tester import TesterService as TS  # noqa: E402
from utils.numba_filter import filter_numba_decorators  # noqa: E402


def _decorated_names(code: str) -> set:
    tree = ast.parse(filter_numba_decorators(code))
    out = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            d = dec.func if isinstance(dec, ast.Call) else dec
            name = d.attr if isinstance(d, ast.Attribute) else getattr(d, "id", None)
            if name in ("njit", "jit"):
                out.add(node.name)
    return out


# ------------------------------- dummy tests -------------------------------
def test_dummy_assert_true_is_rejected(tmp_path):
    p = tmp_path / "test_equivalence.py"
    p.write_text("def test_dummy():\n    assert True\n", encoding="utf-8")
    ok, reason = TS._is_meaningful_test(str(p))
    assert not ok
    assert "constant" in reason


def test_test_without_calls_is_rejected(tmp_path):
    p = tmp_path / "test_equivalence.py"
    p.write_text(
        "def test_x():\n    x = 1\n    assert x == 1\n", encoding="utf-8"
    )
    ok, reason = TS._is_meaningful_test(str(p))
    assert not ok
    assert "call" in reason


def test_meaningful_test_is_accepted(tmp_path):
    p = tmp_path / "test_equivalence.py"
    p.write_text(
        "import math\n"
        "def test_sqrt():\n"
        "    v = math.sqrt(4)\n"
        "    assert v == 2.0\n",
        encoding="utf-8",
    )
    ok, reason = TS._is_meaningful_test(str(p))
    assert ok, reason


# ---------------------------- njit safety guard ----------------------------
def test_keeps_safe_numeric_function():
    code = """
import numba
import numpy as np
import math

@numba.njit
def f(arr, k):
    total = 0.0
    for i in range(arr.shape[0]):
        total += np.exp(arr[i]) + math.sqrt(k)
    return total
"""
    assert "f" in _decorated_names(code)


def test_strips_external_module_reference():
    code = """
import numba
import bots.utils as _utils

@numba.njit
def score(user):
    return _utils.scaled_sigmoid(user.followers_count)
"""
    assert "score" not in _decorated_names(code)


def test_strips_object_attribute_access():
    code = """
import numba

@numba.njit
def score(tweet):
    return tweet.favorite_count + 1
"""
    assert "score" not in _decorated_names(code)


def test_strips_logger_assigned_from_external_call():
    code = """
import numba
import logging

logger = logging.getLogger()

@numba.njit
def f(x):
    logger.info("hi")
    return x
"""
    assert "f" not in _decorated_names(code)


def test_strips_fstring():
    code = """
import numba

@numba.njit
def f(x):
    return f"{x}"
"""
    assert "f" not in _decorated_names(code)


def test_strips_nested_function():
    code = """
import numba

@numba.njit
def f(x):
    def g(y):
        return y
    return g(x)
"""
    assert "f" not in _decorated_names(code)


def test_strips_varargs():
    code = """
import numba

@numba.njit
def f(*args):
    return 1
"""
    assert "f" not in _decorated_names(code)
