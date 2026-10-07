import ast
from typing import Union


class _EmptyListPatcher(ast.NodeTransformer):
    """Replaces bare ``x = []`` assignments with ``x = numba.typed.List()``."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        if isinstance(node.value, ast.List) and len(node.value.elts) == 0:
            node.value = ast.Call(
                func=ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Name(id="numba", ctx=ast.Load()),
                        attr="typed",
                        ctx=ast.Load(),
                    ),
                    attr="List",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
            ast.fix_missing_locations(node)
        return node


class _GenExprToListComp(ast.NodeTransformer):
    """Replaces generator expressions with list comprehensions.

    Numba nopython mode does not support generator expressions — they compile
    to closures that use ``yield``, which is unsupported.  List comprehensions
    producing a single homogeneous numeric type are supported and are
    semantically equivalent for every reduction built-in (sum, min, max, …).
    """

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.ListComp:
        self.generic_visit(node)  # recurse into nested generators first
        return ast.fix_missing_locations(
            ast.ListComp(elt=node.elt, generators=node.generators)
        )


class _ZipToRangeLoop(ast.NodeTransformer):
    """Replaces ``for a, b in zip(x, y):`` with an index-based loop.

    Numba nopython mode does not support ``zip()`` over typed lists.  An
    equivalent ``for _zip_i in range(len(x)): a, b = x[_zip_i], y[_zip_i]``
    is fully supported and semantically identical for same-length sequences.
    """

    def visit_For(self, node: ast.For) -> ast.For:
        self.generic_visit(node)  # recurse into nested loops first

        if not (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "zip"
            and len(node.iter.args) >= 2
            and isinstance(node.target, ast.Tuple)
            and len(node.target.elts) == len(node.iter.args)
        ):
            return node

        zip_args = node.iter.args
        idx = "_zip_i"

        range_call = ast.Call(
            func=ast.Name(id="range", ctx=ast.Load()),
            args=[
                ast.Call(
                    func=ast.Name(id="len", ctx=ast.Load()),
                    args=[zip_args[0]],
                    keywords=[],
                )
            ],
            keywords=[],
        )

        unpack = ast.Assign(
            targets=[node.target],
            value=ast.Tuple(
                elts=[
                    ast.Subscript(
                        value=arg,
                        slice=ast.Name(id=idx, ctx=ast.Load()),
                        ctx=ast.Load(),
                    )
                    for arg in zip_args
                ],
                ctx=ast.Load(),
            ),
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

        new_for = ast.For(
            target=ast.Name(id=idx, ctx=ast.Store()),
            iter=range_call,
            body=[unpack] + node.body,
            orelse=node.orelse,
            lineno=node.lineno,
            col_offset=node.col_offset,
        )
        return ast.fix_missing_locations(new_for)


class _NormalizeNumbaAlias(ast.NodeTransformer):
    """Replaces all references to 'nb' (from 'import numba as nb') with 'numba'."""

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id == "nb":
            node.id = "numba"
        return node


class Inserter(ast.NodeTransformer):

    def __init__(self) -> None:
        self._internal_callees: set = set()
        self._known_module_names: set = set()
        self._jittable: set = set()

    def _compute_jittable(self, tree: ast.AST) -> set:
        """Return functions that can safely be JIT-compiled.

        A function is jittable when it has no unsupported calls AND every
        local function it calls is also jittable.  A JIT function cannot
        call a plain Python wrapper, so call-graph consistency is required.
        """
        funcs = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        jittable = {
            name for name, node in funcs.items()
            if not self._has_unsupported_calls(node)
        }
        changed = True
        while changed:
            changed = False
            for name in list(jittable):
                for child in ast.walk(funcs[name]):
                    if (
                        isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Name)
                        and child.func.id in funcs
                        and child.func.id not in jittable
                    ):
                        jittable.discard(name)
                        changed = True
                        break
        return jittable

    def _collect_internal_callees(self, tree: ast.AST) -> set:
        """Return names of module-level functions called by other module-level functions.

        These must stay as plain ``@numba.njit`` functions so that JIT-to-JIT
        calls continue to work.  Only functions that are exclusively called
        from plain Python should receive the sanitising wrapper.
        
        This now detects TRANSITIVE call chains: if A calls B, and B calls C,
        then A, B, and C are all internal callees.
        """
        top_level = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        
        # Build call graph: function_name -> set of functions it calls
        call_graph = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                callees = set()
                for child in ast.walk(node):
                    if (
                        isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Name)
                        and child.func.id in top_level
                        and child.func.id != node.name
                    ):
                        callees.add(child.func.id)
                call_graph[node.name] = callees
        
        # Find all functions that are called (directly or transitively)
        called: set = set()
        
        # First pass: direct callees
        for caller, callees in call_graph.items():
            called.update(callees)
        
        # Second pass: transitive callees (fixed-point iteration)
        # If A calls B, and B is in called, then A should also be in called
        # (because A is called by someone who needs B to be JIT)
        changed = True
        while changed:
            changed = False
            for caller, callees in call_graph.items():
                # If this function calls any internal callee, it should also be internal
                if callees & called and caller not in called:
                    # Check if caller is called by someone
                    is_called = any(
                        caller in other_callees
                        for other_callees in call_graph.values()
                    )
                    if is_called:
                        called.add(caller)
                        changed = True
        
        return called

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        return node

    _UNSUPPORTED_BUILTINS = frozenset(
        {
            "print", "input", "open", "exec", "eval", "raise",
            "abort", "hex", "bin", "oct", "format", "str", "repr",
            "getattr", "setattr", "hasattr", "vars", "dir",
            "dict", "set", "frozenset",
        }
    )

    def _has_numba_decorator(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> bool:
        for dec in node.decorator_list:
            if (
                isinstance(dec, ast.Attribute)
                and isinstance(dec.value, ast.Name)
                and dec.value.id == "numba"
            ):
                return True
            if isinstance(dec, ast.Name) and dec.id in ("njit", "jit"):
                return True
        return False

    def _strip_static_decorators(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> None:
        """Remove @staticmethod / @classmethod decorators from extracted methods.

        ClassExtractor turns methods into module-level functions, but leaves
        @staticmethod / @classmethod behind. Numba cannot decorate those
        descriptor objects, so they must be removed.
        """
        node.decorator_list = [
            dec for dec in node.decorator_list
            if not (isinstance(dec, ast.Name) and dec.id in ("staticmethod", "classmethod", "property"))
        ]

    def _strip_duplicate_numba_decorators(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> None:
        """Remove duplicate or incompatible numba decorators.
        
        If the function has @numba.vectorize or @vectorize, remove any @numba.njit on top.
        If there are multiple @numba.njit, keep only the first one.
        """
        has_vectorize = False
        for dec in node.decorator_list:
            dec_func = dec
            if isinstance(dec, ast.Call):
                dec_func = dec.func
            
            if isinstance(dec_func, ast.Name) and dec_func.id == "vectorize":
                has_vectorize = True
                break
            if (
                isinstance(dec_func, ast.Attribute)
                and isinstance(dec_func.value, ast.Name)
                and dec_func.value.id == "numba"
                and dec_func.attr == "vectorize"
            ):
                has_vectorize = True
                break
        
        if has_vectorize:
            node.decorator_list = [
                dec for dec in node.decorator_list
                if not (
                    (isinstance(dec, ast.Attribute)
                     and isinstance(dec.value, ast.Name)
                     and dec.value.id == "numba"
                     and dec.attr in ("njit", "jit"))
                    or
                    (isinstance(dec, ast.Call)
                     and isinstance(dec.func, ast.Attribute)
                     and isinstance(dec.func.value, ast.Name)
                     and dec.func.value.id == "numba"
                     and dec.func.attr in ("njit", "jit"))
                    or
                    (isinstance(dec, ast.Name) and dec.id in ("njit", "jit"))
                    or
                    (isinstance(dec, ast.Call)
                     and isinstance(dec.func, ast.Name)
                     and dec.func.id in ("njit", "jit"))
                )
            ]
            return
        
        seen_njit = False
        filtered = []
        for dec in node.decorator_list:
            is_njit = (
                (isinstance(dec, ast.Attribute)
                 and isinstance(dec.value, ast.Name)
                 and dec.value.id == "numba"
                 and dec.attr in ("njit", "jit"))
                or
                (isinstance(dec, ast.Call)
                 and isinstance(dec.func, ast.Attribute)
                 and isinstance(dec.func.value, ast.Name)
                 and dec.func.value.id == "numba"
                 and dec.func.attr in ("njit", "jit"))
                or
                (isinstance(dec, ast.Name) and dec.id in ("njit", "jit"))
                or
                (isinstance(dec, ast.Call)
                 and isinstance(dec.func, ast.Name)
                 and dec.func.id in ("njit", "jit"))
            )
            
            if is_njit:
                if not seen_njit:
                    filtered.append(dec)
                    seen_njit = True
            else:
                filtered.append(dec)
        node.decorator_list = filtered

    def _has_str_in_annotations(self, node: ast.FunctionDef) -> bool:
        """Return True if any parameter or return annotation references the str type.

        Functions whose signatures mention str cannot be JIT-compiled by Numba
        in nopython mode (heterogeneous tuples / string scalars are unsupported).
        """
        annotations = [
            arg.annotation
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            if arg.annotation is not None
        ]
        if node.returns is not None:
            annotations.append(node.returns)
        for ann in annotations:
            for child in ast.walk(ann):
                if isinstance(child, ast.Name) and child.id == "str":
                    return True
                if isinstance(child, ast.Constant) and child.value == "str":
                    return True
        return False

    _STANDARD_ANNOTATIONS = frozenset({
        'int', 'float', 'str', 'bool', 'bytes', 'bytearray', 'complex',
        'list', 'tuple', 'dict', 'set', 'frozenset', 'object',
    })

    def _strip_unknown_annotations(self, node: ast.FunctionDef) -> None:
        """Remove annotations that reference names unresolvable in this module.

        Class-extraction and import-stripping leave annotations such as
        ``confidence: QAConfidence`` or ``x: List[...]`` (without a typing
        import) pointing at undefined names.  Python evaluates annotations at
        definition time, so a plain-Python function with a dangling annotation
        crashes on module import.  Drop the whole annotation when any Name it
        references is unknown.
        """
        known = self._STANDARD_ANNOTATIONS | self._known_module_names

        def _unresolvable(ann: ast.expr) -> bool:
            return any(
                isinstance(child, ast.Name) and child.id not in known
                for child in ast.walk(ann)
            )

        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            if arg.annotation is not None and _unresolvable(arg.annotation):
                arg.annotation = None
        if node.returns is not None and _unresolvable(node.returns):
            node.returns = None

    def _strip_str_from_annotations(self, node: ast.FunctionDef) -> None:
        """Replace every ``str`` type reference in annotations with ``float``.

        The wrapper already converts strings to 0.0 via ``_sanitize_for_numba``
        before calling the JIT function, so the JIT signature must only contain
        numeric types that Numba can compile.
        """

        class _StrToFloat(ast.NodeTransformer):
            def visit_Name(self, n: ast.Name) -> ast.Name:  # noqa: ANN001
                if n.id == "str":
                    n.id = "float"
                return n

        replacer = _StrToFloat()
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            if arg.annotation is not None:
                arg.annotation = replacer.visit(arg.annotation)
                ast.fix_missing_locations(arg.annotation)
        if node.returns is not None:
            node.returns = replacer.visit(node.returns)
            ast.fix_missing_locations(node.returns)

    def _has_nested_list_in_annotations(self, node: ast.FunctionDef) -> bool:
        """Return True if any annotation contains a nested list (list[list[...]]).

        Numba cannot reflect nested typed lists back to Python after JIT
        execution.  Functions with such parameter or return annotations must
        remain as plain Python.
        """
        annotations = [
            arg.annotation
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            if arg.annotation is not None
        ]
        if node.returns is not None:
            annotations.append(node.returns)
        for ann in annotations:
            for child in ast.walk(ann):
                if (
                    isinstance(child, ast.Subscript)
                    and isinstance(child.value, ast.Name)
                    and child.value.id == "list"
                ):
                    inner = child.slice
                    if (
                        isinstance(inner, ast.Subscript)
                        and isinstance(inner.value, ast.Name)
                        and inner.value.id == "list"
                    ):
                        return True
        return False

    _STRING_METHODS = frozenset({
        "split", "join", "format", "strip", "lstrip", "rstrip", "replace",
        "lower", "upper", "startswith", "endswith", "encode", "decode",
        "splitlines", "partition", "rpartition", "zfill", "ljust", "rjust",
        "title", "capitalize", "isdigit", "isalpha", "isalnum", "isspace",
        "swapcase", "casefold", "center", "removeprefix", "removesuffix",
        "rfind",
    })

    _DICT_METHODS = frozenset({
        "keys", "values", "items", "get", "update", "setdefault",
        "popitem", "fromkeys",
    })

    _STRING_NAME_HINTS = ("__str__", "__repr__", "__format__")

    _EXTERNAL_LIBS = frozenset({
        "sklearn", "KMeans", "AgglomerativeClustering",
        "pairwise_distances", "PIL", "Image",
        "plotly", "requests", "cv2", "hashlib", "datetime", "random",
        "base64", "json", "shutil", "pathlib", "glob", "subprocess",
    })

    _UNSUPPORTED_NUMPY_ATTRS = frozenset({
        "where", "select", "piecewise", "ravel_multi_index",
        "unravel_index", "clip", "vectorize",
    })

    def _is_np_asarray_of_nonarray(self, call: ast.Call) -> bool:
        """Mirror NumbaDecoratorFilter: np.asarray(tuple/list) is rejected."""
        func = call.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "asarray"
            and isinstance(func.value, ast.Name)
            and func.value.id in ("np", "numpy")
        ):
            return False
        if not call.args:
            return False
        arg = call.args[0]
        is_np_call = (
            isinstance(arg, ast.Call)
            and isinstance(arg.func, ast.Attribute)
            and isinstance(arg.func.value, ast.Name)
            and arg.func.value.id in ("np", "numpy")
        )
        return not is_np_call

    def _has_unsupported_calls(self, node: ast.FunctionDef) -> bool:
        if any(hint in node.name for hint in self._STRING_NAME_HINTS):
            return True

        # Collect docstring constant nodes (they are str constants but harmless)
        docstrings = set()
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            docstrings.add(id(node.body[0].value))

        for child in ast.walk(node):
            if isinstance(child, ast.Try):  # try/except not supported by njit
                return True
            if isinstance(child, ast.JoinedStr):  # f-strings not supported
                return True
            if isinstance(child, ast.Lambda):  # lambdas not supported by njit
                return True
            if isinstance(child, (ast.Dict, ast.Set)):  # dict/set not supported
                return True
            if (
                isinstance(child, ast.Constant)
                and isinstance(child.value, str)
                and id(child) not in docstrings
            ):
                return True
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name):
                    if func.id in self._UNSUPPORTED_BUILTINS:
                        return True
                    if func.id in self._EXTERNAL_LIBS:
                        return True
                if isinstance(func, ast.Attribute):
                    if func.attr in self._STRING_METHODS:
                        return True
                    if func.attr in self._DICT_METHODS:
                        return True
                    if isinstance(func.value, ast.Name):
                        if func.value.id in self._EXTERNAL_LIBS:
                            return True
                        if (
                            func.value.id in ("np", "numpy")
                            and func.attr in self._UNSUPPORTED_NUMPY_ATTRS
                        ):
                            return True
                    if func.attr in self._UNSUPPORTED_NUMPY_ATTRS:
                        return True
                if self._is_np_asarray_of_nonarray(child):
                    return True
            elif isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Name) and child.value.id in self._EXTERNAL_LIBS:
                    return True
        return False

    def _get_empty_list_names(self, func_node: ast.FunctionDef) -> set:
        """Return names of bare [] assignments in the function's direct body."""
        names = set()
        for stmt in func_node.body:
            if (
                isinstance(stmt, ast.Assign)
                and isinstance(stmt.value, ast.List)
                and len(stmt.value.elts) == 0
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                names.add(stmt.targets[0].id)
        return names

    def _returns_typed_list(self, func_node: ast.FunctionDef, list_names: set) -> bool:
        """Return True if any return statement returns a typed.List variable."""
        for node in ast.walk(func_node):
            if (
                isinstance(node, ast.Return)
                and node.value is not None
                and isinstance(node.value, ast.Name)
                and node.value.id in list_names
            ):
                return True
        return False

    def _make_wrapper(
        self, orig_name: str, jit_name: str, args: ast.arguments
    ) -> ast.FunctionDef:
        """Generate a plain Python wrapper that sanitizes inputs and converts typed.List → list."""

        call_args = []
        for a in args.posonlyargs + args.args:
            # For list arguments, convert to numpy array first (to avoid np.array(typed_list) error in JIT)
            call_args.append(
                ast.Call(
                    func=ast.Name(id="_sanitize_for_numba", ctx=ast.Load()),
                    args=[
                        ast.IfExp(
                            test=ast.Call(
                                func=ast.Name(id="isinstance", ctx=ast.Load()),
                                args=[
                                    ast.Name(id=a.arg, ctx=ast.Load()),
                                    ast.Name(id="list", ctx=ast.Load()),
                                ],
                                keywords=[],
                            ),
                            body=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="np", ctx=ast.Load()),
                                    attr="array",
                                    ctx=ast.Load(),
                                ),
                                args=[ast.Name(id=a.arg, ctx=ast.Load())],
                                keywords=[],
                            ),
                            orelse=ast.Name(id=a.arg, ctx=ast.Load()),
                        )
                    ],
                    keywords=[],
                )
            )

        call_kwargs = [
            ast.keyword(
                arg=a.arg,
                value=ast.Call(
                    func=ast.Name(id="_sanitize_for_numba", ctx=ast.Load()),
                    args=[
                        ast.IfExp(
                            test=ast.Call(
                                func=ast.Name(id="isinstance", ctx=ast.Load()),
                                args=[
                                    ast.Name(id=a.arg, ctx=ast.Load()),
                                    ast.Name(id="list", ctx=ast.Load()),
                                ],
                                keywords=[],
                            ),
                            body=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="np", ctx=ast.Load()),
                                    attr="array",
                                    ctx=ast.Load(),
                                ),
                                args=[ast.Name(id=a.arg, ctx=ast.Load())],
                                keywords=[],
                            ),
                            orelse=ast.Name(id=a.arg, ctx=ast.Load()),
                        )
                    ],
                    keywords=[],
                ),
            )
            for a in args.kwonlyargs
        ]

        jit_call = ast.Call(
            func=ast.Name(id=jit_name, ctx=ast.Load()),
            args=call_args,
            keywords=call_kwargs,
        )

        deep_call = ast.Call(
            func=ast.Name(id="_deep_to_list", ctx=ast.Load()),
            args=[jit_call],
            keywords=[],
        )

        wrapper = ast.FunctionDef(
            name=orig_name,
            args=args,
            body=[ast.Return(value=deep_call)],
            decorator_list=[],
            returns=None,
            lineno=0,
            col_offset=0,
        )
        return ast.fix_missing_locations(wrapper)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Union[ast.FunctionDef, list]:

        if node.name in ("_sanitize_for_numba", "_deep_to_list"):
            return node

        # Skip Flask endpoints and functions marked with _skip_numba
        if getattr(node, '_skip_numba', False):
            return node

        self._strip_static_decorators(node)
        self._strip_duplicate_numba_decorators(node)

        if self._has_numba_decorator(node):
            return node
        if node.name not in self._jittable or self._has_nested_list_in_annotations(
            node
        ):
            return node

        if self._has_str_in_annotations(node):
            self._strip_str_from_annotations(node)

        self._strip_unknown_annotations(node)

        _GenExprToListComp().visit(node)
        _ZipToRangeLoop().visit(node)
        _EmptyListPatcher().generic_visit(node)

        decorator = ast.Attribute(
            value=ast.Name(id="numba", ctx=ast.Load()), attr="njit", ctx=ast.Load()
        )
        node.decorator_list.insert(0, decorator)

        self._strip_duplicate_numba_decorators(node)

        if node.name in self._internal_callees or node.name.startswith("_"):
            return ast.fix_missing_locations(node)

        orig_name = node.name
        jit_name = f"_{orig_name}_jit"
        node.name = jit_name
        wrapper = self._make_wrapper(orig_name, jit_name, node.args)

        return [ast.fix_missing_locations(node), ast.fix_missing_locations(wrapper)]

    def importer(self, tree: ast.AST) -> None:
        self._internal_callees = self._collect_internal_callees(tree)

        known = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                known.add(node.name)
            elif isinstance(node, ast.ClassDef):
                known.add(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    known.add(alias.asname or alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    known.add(alias.asname or alias.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        known.add(t.id)
        self._known_module_names = known
        self._jittable = self._compute_jittable(tree)

        # Dangling annotations crash module import even for non-JIT functions
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                self._strip_unknown_annotations(node)

        # Normalize "import numba as nb" to "import numba"
        has_numba_alias = False
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "numba":
                        has_numba_alias = True
                        if alias.asname is not None:
                            alias.asname = None

        if has_numba_alias:
            _NormalizeNumbaAlias().visit(tree)

        if not has_numba_alias:
            import_node = ast.Import(names=[ast.alias(name="numba")])
            tree.body.insert(0, import_node)
            ast.fix_missing_locations(import_node)

        # Ensure "import numpy as np" is present
        has_numpy_with_alias = any(
            isinstance(node, ast.Import) and any(
                n.name == "numpy" and n.asname == "np" for n in node.names
            )
            for node in tree.body
        )
        if not has_numpy_with_alias:
            # Remove any existing "import numpy" without alias
            tree.body = [
                node for node in tree.body
                if not (isinstance(node, ast.Import) and any(
                    n.name == "numpy" for n in node.names
                ))
            ]
            numpy_import = ast.Import(names=[ast.alias(name="numpy", asname="np")])
            tree.body.insert(0, numpy_import)
            ast.fix_missing_locations(numpy_import)
            
            # Normalize all references from "numpy.xxx" to "np.xxx"
            class _NormalizeNumpyRef(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id == "numpy":
                        node.id = "np"
                    return node
            
            _NormalizeNumpyRef().visit(tree)

        if not any(
            isinstance(node, ast.ImportFrom) and node.module == "_numba_helpers"
            for node in tree.body
        ):
            helpers_import = ast.ImportFrom(
                module="_numba_helpers",
                names=[
                    ast.alias(name="_sanitize_for_numba"),
                    ast.alias(name="_deep_to_list"),
                ],
                level=0,
            )
            ast.fix_missing_locations(helpers_import)

            insert_idx = 0
            for i, node in enumerate(tree.body):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    insert_idx = i + 1

            tree.body.insert(insert_idx, helpers_import)
