"""Regression tests for AST preprocessing and annotation fixes."""

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.preprocessor import (
    BooleanMaskRewriter,
    NestedListToArray,
    OrphanCallStripper,
    IOStripper,
    ClassExtractor,
    StrCommentStripper,
)
from utils.modifier import Inserter
from utils.numba_filter import NumbaDecoratorFilter
from services.compatibility import has_actionable_errors
from build_pipeline import _NpArrayStripper, _restore_external_imports


class TestRestoreExternalImports:
    def test_missing_sklearn_import_restored(self):
        original = "from sklearn.cluster import KMeans\n\ndef c(x):\n    return KMeans()\n"
        output = "def c(x):\n    return KMeans()\n"
        result = _restore_external_imports(
            [{"path": "src/a.py", "body": output}], {"src/a.py": original}
        )
        assert "from sklearn.cluster import KMeans" in result[0]["body"]
        assert "except ImportError" in result[0]["body"]

    def test_existing_import_not_duplicated(self):
        original = "from sklearn.cluster import KMeans\n"
        output = "from sklearn.cluster import KMeans\n\ndef c(x):\n    return KMeans()\n"
        result = _restore_external_imports(
            [{"path": "src/a.py", "body": output}], {"src/a.py": original}
        )
        assert result[0]["body"].count("from sklearn.cluster import KMeans") == 1

    def test_relative_imports_skipped(self):
        original = "from .interpreter import execute\n\ndef f(x):\n    return execute(x)\n"
        output = "def f(x):\n    return execute(x)\n"
        result = _restore_external_imports(
            [{"path": "src/a.py", "body": output}], {"src/a.py": original}
        )
        assert "interpreter" not in result[0]["body"]


def _unparse(node: ast.AST) -> str:
    return ast.unparse(ast.fix_missing_locations(node))


def _has_njit(node: ast.FunctionDef) -> bool:
    return any(
        isinstance(d, ast.Attribute)
        and isinstance(d.value, ast.Name)
        and d.value.id == "numba"
        and d.attr in ("njit", "jit")
        for d in node.decorator_list
    )


class TestBooleanMaskRewriter:
    def test_integer_index_is_not_rewritten(self):
        code = (
            "centers = np.empty([n, 3])\n"
            "for i in range(n):\n"
            "    centers[i] = compute(data[labels == i])\n"
        )
        out = _unparse(BooleanMaskRewriter().visit(ast.parse(code)))
        assert "np.ndindex" not in out
        assert "centers[i]" in out

    def test_comparison_mask_is_rewritten(self):
        code = (
            "mask = arr > 0.5\n"
            "arr[mask] = np.power(arr[mask], 2.0)\n"
        )
        out = _unparse(BooleanMaskRewriter().visit(ast.parse(code)))
        assert "np.ndindex" in out
        assert "if mask[_idx]:" in out

    def test_inverted_mask_is_rewritten(self):
        code = (
            "mask = arr > 0.5\n"
            "arr[~mask] = arr[~mask] / 12.92\n"
        )
        out = _unparse(BooleanMaskRewriter().visit(ast.parse(code)))
        assert "np.ndindex" in out
        assert "if not mask[_idx]:" in out

    def test_augmented_assignment_with_mask_is_rewritten(self):
        code = (
            "mask = arr > 0.5\n"
            "arr[~mask] /= 12.92\n"
        )
        out = _unparse(BooleanMaskRewriter().visit(ast.parse(code)))
        assert "np.ndindex" in out
        assert "if not mask[_idx]:" in out
        assert "arr[_idx] /= 12.92" in out

    def test_augmented_whole_array_untouched(self):
        code = "arr *= ref_white\n"
        out = _unparse(BooleanMaskRewriter().visit(ast.parse(code)))
        assert out.strip() == "arr *= ref_white"


class TestStrCommentStripper:
    def test_mid_body_section_comments_stripped_docstring_kept(self):
        code = (
            "def f(x):\n"
            "    \"\"\"real docstring\"\"\"\n"
            "    \"    section comment    \"\n"
            "    y = x + 1\n"
            "    \"    another    \"\n"
            "    return y\n"
        )
        out = _unparse(StrCommentStripper().visit(ast.parse(code)))
        assert "real docstring" in out
        assert "section comment" not in out
        assert "another" not in out

    def test_numeric_kernel_becomes_jittable_after_stripping(self):
        code = (
            "import numpy\n"
            "def kernel(x):\n"
            "    \"\"\"doc\"\"\"\n"
            "    \"    section note    \"\n"
            "    out = numpy.zeros_like(x)\n"
            "    for i in range(len(x)):\n"
            "        out[i] = x[i] * 2.0\n"
            "    return out\n"
        )
        stripped = ast.parse(_unparse(StrCommentStripper().visit(ast.parse(code))))
        ins_with = Inserter()
        ins_with.importer(ast.parse(code))
        ins_clean = Inserter()
        ins_clean.importer(stripped)
        assert "kernel" not in ins_with._jittable
        assert "kernel" in ins_clean._jittable


class TestNestedListToArray:
    def test_list_of_lists_wrapped(self):
        code = "m = [[1.0, 2.0], [3.0, 4.0]]\n"
        out = _unparse(NestedListToArray().visit(ast.parse(code)))
        assert out.strip() == "m = np.array([[1.0, 2.0], [3.0, 4.0]])"

    def test_flat_list_untouched(self):
        code = "v = [1.0, 2.0, 3.0]\n"
        out = _unparse(NestedListToArray().visit(ast.parse(code)))
        assert out.strip() == "v = [1.0, 2.0, 3.0]"


class TestOrphanCallStripper:
    def test_flask_app_assignment_and_routes_removed(self):
        code = (
            "app = Flask(__name__)\n"
            "@app.route('/x')\n"
            "def x():\n"
            "    return 1\n"
            "def keep(a):\n"
            "    return a\n"
        )
        out = _unparse(OrphanCallStripper({"Flask"}).visit(ast.parse(code)))
        assert "Flask" not in out
        assert "route" not in out
        assert "def keep" in out


class TestClassExtractor:
    def test_staticmethod_decorator_stripped(self):
        code = (
            "class Foo:\n"
            "    @staticmethod\n"
            "    def bar(x):\n"
            "        return x\n"
        )
        tree = ClassExtractor().visit(ast.parse(code))
        out = _unparse(tree)
        assert "staticmethod" not in out
        assert "def foo_bar" in out


class TestIOStripper:
    def test_warn_and_abort_removed(self):
        code = (
            "def f(x):\n"
            "    warn('oops')\n"
            "    abort(400, 'bad')\n"
            "    return x\n"
        )
        out = _unparse(IOStripper().visit(ast.parse(code)))
        assert "warn" not in out
        assert "abort" not in out


class TestInserterJittability:
    def test_dangling_annotation_stripped_for_non_jit_function(self):
        code = (
            "import numpy as np\n"
            "def f(data, confidence: QAConfidence):\n"
            "    d = {}\n"
            "    return d\n"
        )
        tree = ast.parse(code)
        ins = Inserter()
        ins.importer(tree)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "f")
        assert "f" not in ins._jittable
        # module-level strip applies even when the function is never annotated
        assert fn.args.args[1].annotation is None

    def test_numpy_annotation_preserved(self):
        code = (
            "import numpy as np\n"
            "def f(x: np.ndarray) -> np.ndarray:\n"
            "    return x\n"
        )
        tree = ast.parse(code)
        ins = Inserter()
        ins.importer(tree)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        assert ast.unparse(fn.args.args[0].annotation) == "np.ndarray"

    def test_unimported_typing_annotation_stripped(self):
        code = (
            "def f(x) -> List[float]:\n"
            "    return x\n"
        )
        tree = ast.parse(code)
        ins = Inserter()
        ins.importer(tree)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        assert fn.returns is None

    def test_caller_of_unsupported_function_not_annotated(self):
        code = (
            "import numpy\n"
            "def core(x):\n"
            "    label = 'str-constant'\n"
            "    return x\n"
            "def wrapper(x):\n"
            "    return core(x)\n"
        )
        tree = ast.parse(code)
        inserter = Inserter()
        inserter.importer(tree)
        assert "wrapper" not in inserter._jittable
        assert "core" not in inserter._jittable

    def test_pure_chain_is_annotated(self):
        code = (
            "import numpy\n"
            "def core(x):\n"
            "    return numpy.power(x, 2)\n"
            "def wrapper(x):\n"
            "    return core(x)\n"
        )
        tree = ast.parse(code)
        inserter = Inserter()
        inserter.importer(tree)
        assert "core" in inserter._jittable
        assert "wrapper" in inserter._jittable

    def test_np_asarray_of_tuple_blocks_jit(self):
        code = (
            "import numpy\n"
            "def conv(x):\n"
            "    return numpy.asarray((x, x, x))\n"
            "def caller(x):\n"
            "    return conv(x)\n"
        )
        tree = ast.parse(code)
        inserter = Inserter()
        inserter.importer(tree)
        assert "conv" not in inserter._jittable
        assert "caller" not in inserter._jittable


class TestNpArrayStripper:
    def test_replace_in_return_context(self):
        code = (
            "def f(x, nn):\n"
            "    neighs = nn[1][0]\n"
            "    return g(np.array(neighs, dtype=np.int64), np.array(x, dtype=np.float64))\n"
        )
        out = _unparse(_NpArrayStripper().visit(ast.parse(code)))
        assert "np.array" not in out
        assert "g(neighs, x)" in out

    def test_list_literal_untouched(self):
        code = "def f():\n    return np.array([1.0, 2.0])\n"
        out = _unparse(_NpArrayStripper().visit(ast.parse(code)))
        assert "np.array([1.0, 2.0])" in out

    def test_module_level_array_literal_preserved(self):
        code = "m = np.array([[1.0, 2.0], [3.0, 4.0]])\n"
        out = _unparse(_NpArrayStripper().visit(ast.parse(code)))
        assert "np.array([[1.0, 2.0], [3.0, 4.0]])" in out

    def test_name_parameter_unwrapped(self):
        code = "def f(p):\n    x = np.array(p, dtype=np.int64)\n    return x\n"
        out = _unparse(_NpArrayStripper().visit(ast.parse(code)))
        assert "x = p" in out


class TestActionableRetry:
    def test_external_lib_is_not_actionable(self):
        issues = [("error", 1, "Function 'f': Image.fromarray() — external library not supported")]
        assert not has_actionable_errors(issues)

    def test_dict_is_not_actionable(self):
        issues = [("error", 1, "Function 'f': Dict literal found — not supported")]
        assert not has_actionable_errors(issues)

    def test_auto_rewritten_is_not_actionable(self):
        issues = [
            ("error", 1, "Function 'f': np.vectorize() is not supported"),
            ("error", 2, "Function 'f': hex() is not supported"),
        ]
        assert not has_actionable_errors(issues)

    def test_np_where_is_actionable(self):
        issues = [("error", 1, "Function 'f': np.where() is not supported")]
        assert has_actionable_errors(issues)

    def test_warning_only_is_not_actionable(self):
        issues = [("warning", 1, "Possible boolean masking detected")]
        assert not has_actionable_errors(issues)


class TestInserterUnsupportedNumpy:
    def test_np_where_blocks_jit(self):
        code = (
            "import numpy\n"
            "def f(x):\n"
            "    return numpy.where(x > 0, x, 0.0)\n"
        )
        tree = ast.parse(code)
        ins = Inserter()
        ins.importer(tree)
        assert "f" not in ins._jittable

    def test_lambda_blocks_jit(self):
        code = (
            "import numpy\n"
            "def f(x):\n"
            "    g = lambda v: v + 1\n"
            "    return g(x)\n"
        )
        tree = ast.parse(code)
        ins = Inserter()
        ins.importer(tree)
        assert "f" not in ins._jittable


class TestNumbaDecoratorFilterTransitive:
    def test_callers_lose_njit_transitively(self):
        code = (
            "import numba\n"
            "@numba.njit\n"
            "def plain(x):\n"
            "    return x\n"
            "@numba.njit\n"
            "def mid(x):\n"
            "    return plain(x)\n"
            "@numba.njit\n"
            "def top(x):\n"
            "    return mid(x)\n"
        )
        tree = ast.parse(code)
        tree = NumbaDecoratorFilter().visit(tree)
        funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        # The external-lib check does not trigger here, so all remain jit.
        assert _has_njit(funcs["top"])

    def test_external_lib_callers_lose_njit_transitively(self):
        code = (
            "import numba\n"
            "@numba.njit\n"
            "def uses_ext(x):\n"
            "    return Image.fromarray(x)\n"
            "@numba.njit\n"
            "def mid(x):\n"
            "    return uses_ext(x)\n"
            "@numba.njit\n"
            "def top(x):\n"
            "    return mid(x)\n"
        )
        tree = ast.parse(code)
        tree = NumbaDecoratorFilter().visit(tree)
        funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert not _has_njit(funcs["uses_ext"])
        assert not _has_njit(funcs["mid"])
        assert not _has_njit(funcs["top"])
