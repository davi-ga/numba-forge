"""
Preprocessing transformers for forge.

These AST transformers run BEFORE the LLM to prepare code for Numba compilation:
- ClassExtractor: Extracts class methods to module-level functions
- IOStripper: Removes I/O operations (print, logging, file ops, try/except)
- BooleanMaskRewriter: Converts boolean masking to explicit loops
- VectorizeRewriter: Converts np.vectorize to explicit loops
- BuiltinRewriter: Converts hex(), bin(), oct() to Numba-compatible functions
- UniqueRewriter: Converts np.unique(return_counts=True) to manual implementation
"""

import ast
import copy
from typing import Union
from utils.vectorize_rewriter import VectorizeRewriter
from utils.builtin_rewriter import BuiltinRewriter
from utils.unique_rewriter import UniqueRewriter


class ExternalImportStripper(ast.NodeTransformer):
    """Removes imports from external libraries incompatible with Numba.
    
    Keeps only:
    - numpy (as np)
    - numba
    - typing (for type hints)
    - _numba_helpers (internal)
    - os (standard library)
    
    Removes:
    - sklearn
    - multiprocessing
    - Any other external libraries
    """
    
    _ALLOWED_MODULES = frozenset({
        'numpy', 'numba', 'typing', '_numba_helpers', 'os'
    })
    
    def __init__(self):
        super().__init__()
        self.removed_names = set()
    
    def visit_Import(self, node: ast.Import) -> Union[ast.Import, None]:
        # Check if any imported module is not allowed
        for alias in node.names:
            module_name = alias.name.split('.')[0]
            if module_name not in self._ALLOWED_MODULES:
                # Track removed names for OrphanCallStripper
                name = alias.asname or alias.name.split('.')[0]
                self.removed_names.add(name)
                return None
        return node
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> Union[ast.ImportFrom, None]:
        if node.module:
            module_name = node.module.split('.')[0]
            if module_name not in self._ALLOWED_MODULES:
                # Track removed names for OrphanCallStripper
                for alias in node.names:
                    name = alias.asname or alias.name
                    self.removed_names.add(name)
                return None
        return node


class OrphanCallStripper(ast.NodeTransformer):
    """Removes module-level statements that depend on removed imports.
    
    After ExternalImportStripper removes imports, calls or assignments
    at module level (outside functions) that reference those names will
    cause NameError. This transformer removes such orphan statements.
    """
    
    def __init__(self, removed_names: set):
        super().__init__()
        self.removed_names = removed_names
        self._stripped_names: set = set()
    
    def _call_uses_removed_name(self, node: ast.Call) -> bool:
        if isinstance(node.func, ast.Name):
            if node.func.id in self.removed_names:
                return True
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id in self.removed_names:
                    return True
        return False
    
    def visit_Expr(self, node: ast.Expr) -> Union[ast.Expr, None]:
        if isinstance(node.value, ast.Call):
            if self._call_uses_removed_name(node.value):
                return None
        return node
    
    def visit_Assign(self, node: ast.Assign) -> Union[ast.Assign, None]:
        if isinstance(node.value, ast.Call):
            if self._call_uses_removed_name(node.value):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        self._stripped_names.add(t.id)
                return None
        return node
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> Union[ast.FunctionDef, None]:
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Attribute) and isinstance(dec.func.value, ast.Name):
                    if dec.func.value.id in self.removed_names or dec.func.value.id in self._stripped_names:
                        return None
            elif isinstance(dec, ast.Attribute) and isinstance(dec.value, ast.Name):
                if dec.value.id in self.removed_names or dec.value.id in self._stripped_names:
                    return None
        return node


class ExternalFunctionStripper(ast.NodeTransformer):
    """Removes ONLY functions that are pure wrappers of external libraries.
    
    Keeps functions that have substantial logic (loops, computations) even if
    they use external calls. Only removes functions that are trivial wrappers.
    """
    
    _EXTERNAL_CALLS = frozenset({
        'NearestNeighbors', 'Pool', 'StratifiedKFold', 'cross_val_predict',
        'KFold', 'GridSearchCV', 'RandomForestClassifier', 'SVC',
    })
    
    def _is_trivial_wrapper(self, node: ast.FunctionDef) -> bool:
        """Check if function is a trivial wrapper with no substantial logic."""
        # Count non-trivial statements (loops, computations, etc.)
        substantial_statements = 0
        has_external_call = False
        
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While, ast.If)):
                substantial_statements += 1
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    if child.func.id in self._EXTERNAL_CALLS:
                        has_external_call = True
        
        # Remove only if it's a trivial wrapper with external calls
        return has_external_call and substantial_statements < 3
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> Union[ast.FunctionDef, None]:
        self.generic_visit(node)
        if self._is_trivial_wrapper(node):
            return None
        return node


class SelfStrippingTransformer(ast.NodeTransformer):
    """Strips 'self' from method signatures and replaces self.attr with direct params.
    
    After ClassExtractor converts class methods to module-level functions, they
    still have 'self' as first parameter. This transformer:
    1. Removes statements that use external libraries (sklearn, etc.)
    2. Removes 'self' from function signature
    3. Replaces all self.attr references with attr
    4. Adds any undefined names as parameters
    """
    
    _EXTERNAL_ATTRS = frozenset({'NN', 'pool', 'model', 'clf', 'regressor'})
    _BUILTINS = frozenset({
        'range', 'len', 'print', 'int', 'float', 'str', 'list', 'dict',
        'set', 'tuple', 'bool', 'type', 'isinstance', 'enumerate', 'zip',
        'min', 'max', 'sum', 'abs', 'round', 'sorted', 'reversed', 'any',
        'all', 'map', 'filter', 'hasattr', 'getattr', 'setattr', 'None',
        'True', 'False',
    })
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        
        if not node.args.args or node.args.args[0].arg != 'self':
            return node
        
        # Step 1: Remove statements that use external libraries
        new_body = []
        for stmt in node.body:
            uses_external = False
            for child in ast.walk(stmt):
                if (isinstance(child, ast.Attribute) and 
                    isinstance(child.value, ast.Name) and 
                    child.value.id == 'self' and 
                    child.attr in self._EXTERNAL_ATTRS):
                    uses_external = True
                    break
            if not uses_external:
                new_body.append(stmt)
        
        node.body = new_body
        
        # Step 2: Remove 'self' from parameters
        node.args.args = node.args.args[1:]
        
        # Step 3: Replace all self.attr with attr
        class _SelfToParam(ast.NodeTransformer):
            def visit_Attribute(self, attr_node):
                if (isinstance(attr_node.value, ast.Name) and 
                    attr_node.value.id == 'self'):
                    return ast.copy_location(
                        ast.Name(id=attr_node.attr, ctx=attr_node.ctx),
                        attr_node
                    )
                return attr_node
        
        node.body = [_SelfToParam().visit(stmt) for stmt in node.body]
        
        # Step 4: Find all undefined names and add as parameters
        existing_params = {arg.arg for arg in node.args.args}
        defined_names = set()
        used_names = set()
        
        for stmt in node.body:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Name):
                    if isinstance(child.ctx, ast.Store):
                        defined_names.add(child.id)
                    elif isinstance(child.ctx, ast.Load):
                        used_names.add(child.id)
            # Also track augmented assignments targets (+=, etc.)
            for child in ast.walk(stmt):
                if isinstance(child, ast.AugAssign):
                    if isinstance(child.target, ast.Name):
                        defined_names.add(child.target.id)
        
        # Track loop variables
        for stmt in node.body:
            for child in ast.walk(stmt):
                if isinstance(child, (ast.For,)):
                    if isinstance(child.target, ast.Name):
                        defined_names.add(child.target.id)
                    elif isinstance(child.target, ast.Tuple):
                        for elt in child.target.elts:
                            if isinstance(elt, ast.Name):
                                defined_names.add(elt.id)
        
        undefined = used_names - defined_names - existing_params - self._BUILTINS
        
        # Also exclude names that look like module-level calls
        # (e.g., NearestNeighborsFeats, np, os, etc.)
        known_globals = set()
        for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if isinstance(child, ast.Import):
                for alias in child.names:
                    known_globals.add(alias.asname or alias.name)
            elif isinstance(child, ast.ImportFrom):
                for alias in child.names:
                    known_globals.add(alias.asname or alias.name)
        
        undefined -= known_globals
        
        # Remove names that are likely function calls or class references
        # (start with uppercase, or are used only in Call context)
        call_only_names = set()
        module_like_names = set()
        for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    call_only_names.add(child.func.id)
            # Detect module-like usage: np.something, os.something
            if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                module_like_names.add(child.value.id)
        
        # Keep undefined names that are not call-only and not module-like
        param_names = sorted(undefined - call_only_names - module_like_names)
        
        for name in param_names:
            new_param = ast.arg(arg=name, annotation=None)
            node.args.args.append(new_param)
        
        ast.fix_missing_locations(node)
        return node


class ClassExtractor(ast.NodeTransformer):
    """Extracts class methods to module-level functions.
    
    Transforms:
        class Foo:
            def method(self, x):
                return x + 1
    
    Into:
        def foo_method(self, x):
            return x + 1
    """

    def visit_ClassDef(self, node: ast.ClassDef) -> list:
        self.generic_visit(node)
        
        extracted_functions = []
        class_name = node.name.lower()
        
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name == "__init__":
                    continue
                
                new_name = f"{class_name}_{item.name}"
                item.name = new_name
                
                item.decorator_list = [
                    dec for dec in item.decorator_list
                    if not (isinstance(dec, ast.Name) and dec.id in ("staticmethod", "classmethod", "property"))
                ]
                
                if isinstance(item, ast.FunctionDef):
                    extracted_functions.append(item)
        
        return extracted_functions if extracted_functions else []


class IOStripper(ast.NodeTransformer):
    """Removes I/O operations that are incompatible with Numba nopython mode.
    
    Removes:
    - print() calls
    - logging calls
    - file operations (open, read, write)
    - try/except blocks (keeps the try body, removes except)
    - raise statements
    """

    _IO_FUNCTIONS = frozenset({
        "print", "input", "open", "exec", "eval",
        "logging", "logger", "log", "warn", "warnings",
        "abort", "flash", "redirect", "make_response",
    })

    def visit_Expr(self, node: ast.Expr) -> Union[ast.Expr, None]:
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name):
                if node.value.func.id in self._IO_FUNCTIONS:
                    return None
            elif isinstance(node.value.func, ast.Attribute):
                if isinstance(node.value.func.value, ast.Name):
                    if node.value.func.value.id in self._IO_FUNCTIONS:
                        return None
        return node

    def visit_Try(self, node: ast.Try) -> list:
        self.generic_visit(node)
        return node.body

    def visit_Raise(self, node: ast.Raise) -> None:
        return None


class EmptyBlockStripper(ast.NodeTransformer):
    """Removes empty if/else/for/while blocks that cause IndentationError.
    
    After IOStripper or ExternalImportStripper remove statements, some
    blocks may become empty (e.g., `if sls_stage == 'local':` with no body).
    This transformer removes such empty blocks to prevent syntax errors.
    """

    def visit_If(self, node: ast.If) -> Union[ast.If, None]:
        self.generic_visit(node)
        
        # Remove if both body and orelse are empty
        if not node.body and not node.orelse:
            return None
        
        # If body is empty but orelse exists, convert to if not condition
        if not node.body and node.orelse:
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
            node.body = node.orelse
            node.orelse = []
            return node
        
        return node

    def visit_For(self, node: ast.For) -> Union[ast.For, None]:
        self.generic_visit(node)
        
        # Remove if both body and orelse are empty
        if not node.body and not node.orelse:
            return None
        
        return node

    def visit_While(self, node: ast.While) -> Union[ast.While, None]:
        self.generic_visit(node)
        
        # Remove if both body and orelse are empty
        if not node.body and not node.orelse:
            return None
        
        return node


class BooleanMaskRewriter(ast.NodeTransformer):
    """Converts boolean masking operations to explicit loops.
    
    Transforms:
        result[bool_mask] = value
        result[result == 0.0] = 1.0
    
    Into:
        for _idx in np.ndindex(result.shape):
            if bool_mask[_idx]:
                result[_idx] = value
    """

    def __init__(self):
        super().__init__()
        self._bool_mask_names: set = set()
        self._integer_names: set = set()

    def _build_name_defs(self, node: ast.AST) -> None:
        """Pre-scan the AST to classify variable names as boolean masks or integer indices."""
        self._bool_mask_names = set()
        self._integer_names = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(child.value, ast.Compare):
                            self._bool_mask_names.add(target.id)
                        elif isinstance(child.value, (ast.Constant, ast.Call, ast.BinOp)):
                            self._integer_names.add(target.id)
            elif isinstance(child, ast.For):
                if isinstance(child.target, ast.Name):
                    self._integer_names.add(child.target.id)
                elif isinstance(child.target, ast.Tuple):
                    for elt in child.target.elts:
                        if isinstance(elt, ast.Name):
                            self._integer_names.add(elt.id)
                if isinstance(child.iter, ast.Call):
                    if isinstance(child.iter.func, ast.Name) and child.iter.func.id == 'range':
                        if isinstance(child.target, ast.Name):
                            self._integer_names.add(child.target.id)

    def _is_bool_mask_slice(self, slice_node: ast.AST) -> bool:
        """Check if a subscript slice is a known boolean mask."""
        if isinstance(slice_node, ast.Name):
            return slice_node.id in self._bool_mask_names
        if isinstance(slice_node, ast.Compare):
            return True
        if isinstance(slice_node, ast.UnaryOp) and isinstance(slice_node.op, ast.Invert):
            if isinstance(slice_node.operand, ast.Name):
                return slice_node.operand.id in self._bool_mask_names
        return False

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self._build_name_defs(node)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        old_bool = self._bool_mask_names.copy()
        old_int = self._integer_names.copy()
        self._build_name_defs(node)
        self.generic_visit(node)
        self._bool_mask_names = old_bool
        self._integer_names = old_int
        return node

    def _make_indexed_condition(self, condition: ast.AST, array_name: str) -> ast.AST:
        """Replace array references in a Compare condition with indexed versions."""
        idx_tuple = ast.Name(id="_idx", ctx=ast.Load())
        indexed = ast.Subscript(
            value=ast.Name(id=array_name, ctx=ast.Load()),
            slice=idx_tuple,
            ctx=ast.Load(),
        )

        class _ArrayReplacer(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:
                if node.id == array_name:
                    return ast.copy_location(indexed, node)
                return node

        return _ArrayReplacer().visit(condition)

    def visit_Assign(self, node: ast.Assign) -> Union[ast.Assign, ast.For]:
        self.generic_visit(node)
        
        if not (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and self._is_bool_mask_slice(node.targets[0].slice)
        ):
            return node
        
        target_array = node.targets[0].value
        slice_node = node.targets[0].slice
        value = node.value
        
        if not isinstance(target_array, ast.Name):
            return node
        
        array_name = target_array.id
        
        idx_tuple = ast.Name(id="_idx", ctx=ast.Load())
        
        bool_mask_names = self._bool_mask_names

        class _BoolMaskReplacer(ast.NodeTransformer):
            def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
                if not (isinstance(node.value, ast.Name) and node.value.id == array_name):
                    return node
                s = node.slice
                if isinstance(s, ast.Name) and s.id in bool_mask_names:
                    return ast.copy_location(
                        ast.Subscript(value=ast.Name(id=array_name, ctx=ast.Load()), slice=idx_tuple, ctx=ast.Load()),
                        node,
                    )
                if isinstance(s, ast.Compare):
                    return ast.copy_location(
                        ast.Subscript(value=ast.Name(id=array_name, ctx=ast.Load()), slice=idx_tuple, ctx=ast.Load()),
                        node,
                    )
                if isinstance(s, ast.UnaryOp) and isinstance(s.op, ast.Invert):
                    if isinstance(s.operand, ast.Name) and s.operand.id in bool_mask_names:
                        return ast.copy_location(
                            ast.Subscript(value=ast.Name(id=array_name, ctx=ast.Load()), slice=idx_tuple, ctx=ast.Load()),
                            node,
                        )
                return node
        
        value = _BoolMaskReplacer().visit(value)
        
        inner_assign = ast.Assign(
            targets=[
                ast.Subscript(
                    value=ast.Name(id=array_name, ctx=ast.Store()),
                    slice=idx_tuple,
                    ctx=ast.Store(),
                )
            ],
            value=value,
            lineno=node.lineno,
            col_offset=node.col_offset,
        )
        
        if isinstance(slice_node, ast.Name):
            mask_name = slice_node.id
            if_test = ast.Subscript(
                value=ast.Name(id=mask_name, ctx=ast.Load()),
                slice=idx_tuple,
                ctx=ast.Load(),
            )
        elif isinstance(slice_node, ast.Compare):
            if_test = self._make_indexed_condition(slice_node, array_name)
        elif isinstance(slice_node, ast.UnaryOp) and isinstance(slice_node.op, ast.Invert):
            if isinstance(slice_node.operand, ast.Name):
                mask_name = slice_node.operand.id
                if_test = ast.UnaryOp(
                    op=ast.Not(),
                    operand=ast.Subscript(
                        value=ast.Name(id=mask_name, ctx=ast.Load()),
                        slice=idx_tuple,
                        ctx=ast.Load(),
                    ),
                )
            else:
                return node
        else:
            return node
        
        if_stmt = ast.If(
            test=if_test,
            body=[inner_assign],
            orelse=[],
        )
        
        # Use np.ndindex for dimension-agnostic iteration
        ndindex_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="ndindex",
                ctx=ast.Load(),
            ),
            args=[
                ast.Attribute(
                    value=ast.Name(id=array_name, ctx=ast.Load()),
                    attr="shape",
                    ctx=ast.Load(),
                )
            ],
            keywords=[],
        )
        
        loop = ast.For(
            target=ast.Name(id="_idx", ctx=ast.Store()),
            iter=ndindex_call,
            body=[if_stmt],
            orelse=[],
        )
        
        return ast.fix_missing_locations(loop)

    def visit_AugAssign(self, node: ast.AugAssign) -> Union[ast.AugAssign, ast.For]:
        """Rewrite augmented assignments with boolean masks to explicit loops.

        Numba does not support multi-dimensional boolean indices on the LHS,
        e.g. ``arr[~mask] /= 12.92``.
        """
        target = node.target
        if not isinstance(target, ast.Subscript):
            return node
        if not isinstance(target.value, ast.Name):
            return node

        array_name = target.value.id
        slice_node = target.slice

        if not self._is_bool_mask_slice(slice_node):
            return node

        idx_tuple = ast.Name(id="_idx", ctx=ast.Load())

        # Indexed RHS references (e.g. arr[mask] in ``arr[mask] += arr[mask]``)
        bool_mask_names = self._bool_mask_names

        class _BoolMaskReplacer(ast.NodeTransformer):
            def visit_Subscript(self, n: ast.Subscript) -> ast.AST:
                if not (isinstance(n.value, ast.Name) and n.value.id == array_name):
                    return n
                s = n.slice
                if isinstance(s, ast.Name) and s.id in bool_mask_names:
                    return ast.copy_location(
                        ast.Subscript(value=ast.Name(id=array_name, ctx=ast.Load()), slice=idx_tuple, ctx=ast.Load()),
                        n,
                    )
                if isinstance(s, ast.Compare):
                    return ast.copy_location(
                        ast.Subscript(value=ast.Name(id=array_name, ctx=ast.Load()), slice=idx_tuple, ctx=ast.Load()),
                        n,
                    )
                if isinstance(s, ast.UnaryOp) and isinstance(s.op, ast.Invert):
                    if isinstance(s.operand, ast.Name) and s.operand.id in bool_mask_names:
                        return ast.copy_location(
                            ast.Subscript(value=ast.Name(id=array_name, ctx=ast.Load()), slice=idx_tuple, ctx=ast.Load()),
                            n,
                        )
                return n

        value = _BoolMaskReplacer().visit(node.value)

        inner = ast.AugAssign(
            target=ast.Subscript(
                value=ast.Name(id=array_name, ctx=ast.Load()),
                slice=idx_tuple,
                ctx=ast.Store(),
            ),
            op=node.op,
            value=value,
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

        if isinstance(slice_node, ast.Name):
            if_test = ast.Subscript(
                value=ast.Name(id=slice_node.id, ctx=ast.Load()),
                slice=idx_tuple,
                ctx=ast.Load(),
            )
        elif isinstance(slice_node, ast.UnaryOp) and isinstance(slice_node.op, ast.Invert):
            if not isinstance(slice_node.operand, ast.Name):
                return node
            if_test = ast.UnaryOp(
                op=ast.Not(),
                operand=ast.Subscript(
                    value=ast.Name(id=slice_node.operand.id, ctx=ast.Load()),
                    slice=idx_tuple,
                    ctx=ast.Load(),
                ),
            )
        else:
            return node

        if_stmt = ast.If(test=if_test, body=[inner], orelse=[])

        ndindex_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="ndindex",
                ctx=ast.Load(),
            ),
            args=[
                ast.Attribute(
                    value=ast.Name(id=array_name, ctx=ast.Load()),
                    attr="shape",
                    ctx=ast.Load(),
                )
            ],
            keywords=[],
        )

        loop = ast.For(
            target=ast.Name(id="_idx", ctx=ast.Store()),
            iter=ndindex_call,
            body=[if_stmt],
            orelse=[],
        )

        return ast.fix_missing_locations(loop)


class StrCommentStripper(ast.NodeTransformer):
    """Removes bare string-expression statements used as section comments.

    Original code often contains mid-body docstring-like comments::

        def f(x):
            "real docstring"
            "        Section comment.        "
            return x

    After ClassExtractor these stray strings survive inside function bodies.
    They are pure noise, but they make string/annotation-based Numba
    heuristics misclassify numeric kernels as non-JIT-able. Only the first
    statement (the actual docstring) is kept.
    """

    @staticmethod
    def _is_str_expr(stmt: ast.stmt) -> bool:
        return (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )

    def _clean_body(self, body):
        return [s for i, s in enumerate(body) if i == 0 or not self._is_str_expr(s)]

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        node.body = self._clean_body(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.body = self._clean_body(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        node.body = self._clean_body(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        node.body = self._clean_body(node.body)
        return node


class NestedListToArray(ast.NodeTransformer):
    """Wraps module-level list-of-lists literals with np.array().

    Numba cannot index Python list-of-lists with tuple subscripts
    (e.g., x[j, k]). Converting them to numpy arrays at definition time
    resolves this for JIT compilation.
    """

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        self.generic_visit(node)
        if (isinstance(node.value, ast.List) and node.value.elts
                and all(isinstance(elt, ast.List) for elt in node.value.elts)):
            node.value = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr="array",
                    ctx=ast.Load(),
                ),
                args=[node.value],
                keywords=[],
            )
            ast.fix_missing_locations(node)
        return node


def preprocess_code(code: str) -> str:
    """Apply all preprocessing transformers to the code.
    
    Returns the preprocessed code as a string.
    """
    tree = ast.parse(code)
    
    # Apply transformers in order
    tree = ClassExtractor().visit(tree)
    tree = IOStripper().visit(tree)
    tree = BooleanMaskRewriter().visit(tree)
    tree = VectorizeRewriter().visit(tree)
    tree = BuiltinRewriter().visit(tree)
    tree = UniqueRewriter().visit(tree)
    
    return ast.unparse(ast.fix_missing_locations(tree))
