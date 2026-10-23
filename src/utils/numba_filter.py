"""
NumbaDecoratorFilter — remove @numba.njit from functions that cannot be
compiled in nopython mode.

It runs after the LLM and after AST annotation. It is intentionally
conservative: when in doubt, the decorator is removed and the function stays
as plain Python. This prevents shipping @njit functions that reference
external modules/objects (boto3, requests, pandas, `self`, arbitrary objects)
or use constructs Numba cannot compile.
"""

import ast


class NumbaDecoratorFilter(ast.NodeTransformer):
    """Remove @numba.njit from functions that are not safely jittable."""

    # Modules that Numba understands natively.
    _SAFE_MODULES = frozenset({"np", "numpy", "numba", "math", "random"})

    # Attributes/methods that are safe on arrays and lists.
    _SAFE_ATTRS = frozenset({
        "shape", "size", "dtype", "ndim", "astype", "reshape", "ravel",
        "flatten", "copy", "sum", "mean", "std", "var", "min", "max",
        "argmin", "argmax", "append", "extend", "sort", "tolist", "T",
        "transpose", "real", "imag", "conj", "conjugate", "item", "fill",
        "nonzero", "take", "repeat", "view", "resize",
    })

    _UNSAFE_BUILTINS = frozenset({
        "print", "input", "open", "exec", "eval", "getattr", "setattr",
        "hasattr", "vars", "dir", "dict", "set", "frozenset",
        "hex", "bin", "oct", "format", "repr",
    })

    _SAFE_CALLS = frozenset({
        "len", "range", "enumerate", "zip", "int", "float", "bool",
        "abs", "min", "max", "sum", "tuple", "list", "round",
    })

    # Names that are external libraries/objects even when they appear without
    # an explicit import in the snippet (e.g. after preprocessing).
    _KNOWN_EXTERNAL = frozenset({
        "Image", "PIL", "sklearn", "KMeans", "AgglomerativeClustering",
        "pairwise_distances", "plotly", "requests", "cv2", "pd", "pandas",
        "boto3", "botocore", "tweepy", "logging", "logger", "os", "json",
        "time", "datetime", "hashlib", "base64", "shutil", "pathlib", "glob",
        "subprocess",
    })

    def __init__(self) -> None:
        self._unsafe_globals: set = set()

    # ------------------------------------------------------------------ scans
    def _collect_unsafe_globals(self, module: ast.Module) -> None:
        """Names that are external modules/objects (or derived from them)."""
        imported: set = set()
        for node in module.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported.add(alias.asname or alias.name)

        unsafe = {name for name in imported if name not in self._SAFE_MODULES}

        for node in module.body:
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            derived_from_unsafe = False
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                derived_from_unsafe = True
            elif isinstance(value, ast.Call):
                func = value.func
                safe_call = (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id in self._SAFE_MODULES
                ) or (
                    isinstance(func, ast.Name) and func.id in self._SAFE_CALLS
                )
                derived_from_unsafe = not safe_call
            if derived_from_unsafe:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        unsafe.add(target.id)

        self._unsafe_globals = unsafe

    def _is_unsafe(self, node: ast.FunctionDef) -> bool:
        """True when the function likely cannot run in nopython mode."""
        params = {
            arg.arg
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        }

        # Variadic arguments are not supported by Numba.
        if node.args.vararg is not None or node.args.kwarg is not None:
            return True

        docstrings: set = set()
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            docstrings.add(id(node.body[0].value))

        # Scan only the body: decorators/annotations live outside it.
        for stmt in node.body:
            for child in ast.walk(stmt):
                if isinstance(child, (ast.Try, ast.JoinedStr, ast.Lambda, ast.Dict, ast.Set)):
                    return True
                if isinstance(child, ast.FunctionDef):
                    return True  # nested function / closure
                if (
                    isinstance(child, ast.Constant)
                    and isinstance(child.value, str)
                    and id(child) not in docstrings
                ):
                    return True
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    if child.func.id in self._UNSAFE_BUILTINS:
                        return True
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    if (
                        child.id in self._unsafe_globals
                        or child.id in self._KNOWN_EXTERNAL
                        or child.id == "self"
                    ):
                        return True
                if isinstance(child, ast.Attribute):
                    receiver = child.value
                    if isinstance(receiver, ast.Name):
                        if (
                            receiver.id in self._unsafe_globals
                            or receiver.id in self._KNOWN_EXTERNAL
                            or receiver.id == "self"
                        ):
                            return True
                        if receiver.id in params and child.attr not in self._SAFE_ATTRS:
                            return True  # attribute access on an arbitrary object
                    elif isinstance(receiver, ast.Call):
                        func = receiver.func
                        allowed = (
                            isinstance(func, ast.Attribute)
                            and isinstance(func.value, ast.Name)
                            and func.value.id in ("np", "numpy")
                        )
                        if not allowed:
                            return True  # method on an arbitrary call result

        return False

    # ------------------------------------------------------------- decorators
    def _has_njit(self, node: ast.FunctionDef) -> bool:
        return any(self._is_numba_decorator(dec) for dec in node.decorator_list)

    def _is_numba_decorator(self, dec: ast.AST) -> bool:
        if isinstance(dec, ast.Attribute):
            if isinstance(dec.value, ast.Name):
                return dec.value.id == "numba" and dec.attr in ("njit", "jit")
        if isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Attribute):
                if isinstance(dec.func.value, ast.Name):
                    return dec.func.value.id == "numba" and dec.func.attr in ("njit", "jit")
            if isinstance(dec.func, ast.Name):
                return dec.func.id in ("njit", "jit")
        if isinstance(dec, ast.Name):
            return dec.id in ("njit", "jit")
        return False

    # ---------------------------------------------------------------- visitor
    def visit_Module(self, node: ast.Module) -> ast.Module:
        self._collect_unsafe_globals(node)

        funcs = {
            n.name: n for n in node.body if isinstance(n, ast.FunctionDef)
        }
        jit_names = {name for name, func in funcs.items() if self._has_njit(func)}

        for name in list(jit_names):
            if self._is_unsafe(funcs[name]):
                jit_names.discard(name)

        # A JIT function cannot call a plain-Python function: propagate the
        # removal transitively through the call graph.
        changed = True
        while changed:
            changed = False
            for name in list(jit_names):
                for child in ast.walk(funcs[name]):
                    if (
                        isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Name)
                        and child.func.id in funcs
                        and child.func.id not in jit_names
                    ):
                        jit_names.discard(name)
                        changed = True
                        break

        for name, func in funcs.items():
            if name not in jit_names:
                func.decorator_list = [
                    dec for dec in func.decorator_list
                    if not self._is_numba_decorator(dec)
                ]

        return node


def filter_numba_decorators(code: str) -> str:
    """Remove @numba.njit from functions that are not safely jittable."""
    tree = ast.parse(code)
    tree = NumbaDecoratorFilter().visit(tree)
    return ast.unparse(ast.fix_missing_locations(tree))
