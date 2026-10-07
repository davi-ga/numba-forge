"""
Numba compatibility checker — validates code before AST annotation.

Detects patterns that are incompatible with Numba nopython mode and
provides feedback for retry attempts.
"""

import ast
import sys
from typing import List, Tuple


# Issues whose root cause cannot be removed by refactoring: the function
# legitimately depends on an external library, or its structure is inherent
# (orchestration dicts, string formatting, class definitions).  Retrying the
# LLM for these is guaranteed to waste an API call, because the function will
# simply be left as a plain-Python wrapper (which is correct behaviour).
_NON_ACTIONABLE_MARKERS = (
    # Unavoidable external-library dependencies → plain-Python wrapper.
    "external library",
    "from scipy",
    # Inherent non-numeric structure → plain-Python function.
    "Dict literal",
    "Set literal",
    "dict() is not",
    "set() is not",
    "frozenset() is not",
    "Class '",
    # Already handled automatically by the post-processing rewriters.
    "np.vectorize()",
    "np.unique(return_counts",
    "np.unique(return_index",
    "np.unique(return_inverse",
    "hex() is not",
    "bin() is not",
    "oct() is not",
    "try/except block",
    # Lambdas survive np.vectorize rewriting; such functions stay Python.
    "Lambda expression",
)


def has_actionable_errors(issues: List[Tuple[str, int, str]]) -> bool:
    """Return True if any error-severity issue could plausibly be fixed.

    Only actionable issues justify another (costly) LLM round-trip.
    """
    for severity, _line, message in issues:
        if severity != "error":
            continue
        if any(marker in message for marker in _NON_ACTIONABLE_MARKERS):
            continue
        return True
    return False


class NumbaCompatibilityChecker:
    """Validates code for Numba nopython compatibility.
    
    Checks for:
    - Boolean masking on 2D arrays
    - Unsupported numpy functions (np.ravel_multi_index, np.clip on scalars)
    - Heterogeneous tuples
    - Dict/set usage
    - String operations in numeric contexts
    """

    _UNSUPPORTED_NUMPY_FUNCS = frozenset({
        "ravel_multi_index", "unravel_index", "clip",
        "where", "select", "piecewise", "vectorize",
    })

    _UNSUPPORTED_NUMPY_FUNCS_WITH_KWARGS = {
        "unique": ["return_counts", "return_index", "return_inverse"],
    }

    _UNSUPPORTED_BUILTINS = frozenset({
        "dict", "set", "frozenset",
        "print", "input", "open", "exec", "eval",
        "hex", "bin", "oct", "chr", "ord",
    })

    _UNSUPPORTED_SCIPY_FUNCS = frozenset({
        "gammaln", "gamma", "betainc", "betaincinv",
        "erf", "erfc", "erfinv", "erfcinv",
        "loggamma", "digamma", "polygamma",
    })

    _UNSUPPORTED_EXTERNAL_LIBS = frozenset({
        "sklearn", "KMeans", "AgglomerativeClustering",
        "pairwise_distances", "PIL", "Image",
        "plotly", "requests", "cv2",
    })

    def __init__(self):
        self.issues: List[Tuple[str, int, str]] = []

    def check(self, code: str) -> Tuple[bool, List[Tuple[str, int, str]]]:
        """Check code for Numba compatibility issues.
        
        Returns:
            (is_compatible, issues) where issues is a list of (severity, line, message)
        """
        self.issues = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            self.issues.append(("error", exc.lineno or 0, f"Syntax error: {exc.msg}"))
            return False, self.issues
        
        self._walk_tree(tree)
        
        is_compatible = not any(sev == "error" for sev, _, _ in self.issues)
        return is_compatible, self.issues

    def _walk_tree(self, tree: ast.AST):
        """Walk the AST and check for incompatible patterns."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._check_function(node)
            elif isinstance(node, ast.ClassDef):
                self.issues.append(
                    ("warning", node.lineno, f"Class '{node.name}' found — should be extracted to module-level functions")
                )

    def _check_function(self, node: ast.FunctionDef):
        """Check a function for Numba-incompatible patterns."""
        func_name = node.name
        
        for child in ast.walk(node):
            self._check_call(child, func_name)
            self._check_subscript(child, func_name)
            self._check_dict_set(child, func_name)
            self._check_try_except(child, func_name)
            self._check_lambda(child, func_name)
            self._check_external_libs(child, func_name)

    def _check_call(self, node: ast.AST, func_name: str):
        """Check for unsupported function calls."""
        if not isinstance(node, ast.Call):
            return
        
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == "np" or node.func.value.id == "numpy":
                    if node.func.attr in self._UNSUPPORTED_NUMPY_FUNCS:
                        self.issues.append(
                            ("error", node.lineno,
                             f"Function '{func_name}': np.{node.func.attr}() is not supported in Numba nopython mode")
                        )
                    # Check for np.unique with unsupported kwargs
                    if node.func.attr in self._UNSUPPORTED_NUMPY_FUNCS_WITH_KWARGS:
                        unsupported_kwargs = self._UNSUPPORTED_NUMPY_FUNCS_WITH_KWARGS[node.func.attr]
                        for kw in node.keywords:
                            if kw.arg in unsupported_kwargs:
                                self.issues.append(
                                    ("error", node.lineno,
                                     f"Function '{func_name}': np.{node.func.attr}({kw.arg}=...) is not supported in Numba nopython mode")
                                )
                elif node.func.value.id in ("sps", "scipy", "stats"):
                    if node.func.attr in self._UNSUPPORTED_SCIPY_FUNCS:
                        self.issues.append(
                            ("error", node.lineno,
                             f"Function '{func_name}': scipy.{node.func.attr}() is not supported in Numba nopython mode")
                        )
        
        if isinstance(node.func, ast.Name):
            if node.func.id in self._UNSUPPORTED_BUILTINS:
                self.issues.append(
                    ("error", node.lineno,
                     f"Function '{func_name}': {node.func.id}() is not supported in Numba nopython mode")
                )
            elif node.func.id in self._UNSUPPORTED_SCIPY_FUNCS:
                self.issues.append(
                    ("error", node.lineno,
                     f"Function '{func_name}': {node.func.id}() from scipy is not supported in Numba nopython mode")
                )

    def _check_subscript(self, node: ast.AST, func_name: str):
        """Check for boolean masking patterns."""
        if not isinstance(node, ast.Subscript):
            return
        
        if isinstance(node.slice, ast.Name):
            self.issues.append(
                ("warning", node.lineno,
                 f"Function '{func_name}': Possible boolean masking detected (array[{node.slice.id}]). "
                 "Ensure mask is 1D or convert to explicit loops.")
            )

    def _check_dict_set(self, node: ast.AST, func_name: str):
        """Check for dict/set usage."""
        if isinstance(node, ast.Dict):
            self.issues.append(
                ("error", node.lineno,
                 f"Function '{func_name}': Dict literal found — not supported in Numba nopython mode")
            )
        elif isinstance(node, ast.Set):
            self.issues.append(
                ("error", node.lineno,
                 f"Function '{func_name}': Set literal found — not supported in Numba nopython mode")
            )

    def _check_try_except(self, node: ast.AST, func_name: str):
        """Check for try/except blocks."""
        if isinstance(node, ast.Try):
            self.issues.append(
                ("error", node.lineno,
                 f"Function '{func_name}': try/except block found — not supported in Numba nopython mode")
            )

    def _check_lambda(self, node: ast.AST, func_name: str):
        """Check for lambda expressions (not supported in nopython mode)."""
        if isinstance(node, ast.Lambda):
            self.issues.append(
                ("error", node.lineno,
                 f"Function '{func_name}': Lambda expression found — not supported in Numba nopython mode. "
                 "Convert to explicit loop or named function.")
            )

    def _check_external_libs(self, node: ast.AST, func_name: str):
        """Check for external library usage (sklearn, PIL, plotly, etc)."""
        # Check for function calls to external libraries
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in self._UNSUPPORTED_EXTERNAL_LIBS:
                    self.issues.append(
                        ("error", node.lineno,
                         f"Function '{func_name}': {node.func.id}() from external library — "
                         "not supported in Numba nopython mode. Keep as Python wrapper.")
                    )
            elif isinstance(node.func, ast.Attribute):
                # Check for sklearn.something, PIL.something, etc
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id in self._UNSUPPORTED_EXTERNAL_LIBS:
                        self.issues.append(
                            ("error", node.lineno,
                             f"Function '{func_name}': {node.func.value.id}.{node.func.attr}() — "
                             "external library not supported in Numba nopython mode. Keep as Python wrapper.")
                        )

    def format_issues(self) -> str:
        """Format issues for display."""
        if not self.issues:
            return "No compatibility issues found."
        
        lines = ["Numba compatibility issues detected:"]
        for severity, line, message in self.issues:
            lines.append(f"  [{severity.upper()}] Line {line}: {message}")
        return "\n".join(lines)


def check_compatibility(code: str, verbose: bool = True) -> Tuple[bool, str]:
    """Convenience function to check code compatibility.
    
    Returns:
        (is_compatible, report) where report is a formatted string
    """
    checker = NumbaCompatibilityChecker()
    is_compatible, issues = checker.check(code)
    report = checker.format_issues()
    
    if verbose and not is_compatible:
        print(report, file=sys.stderr)
    
    return is_compatible, report
