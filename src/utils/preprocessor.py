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
    
    Removes:
    - sklearn
    - multiprocessing
    - Any other external libraries
    """
    
    _ALLOWED_MODULES = frozenset({
        'numpy', 'numba', 'typing', '_numba_helpers'
    })
    
    def visit_Import(self, node: ast.Import) -> Union[ast.Import, None]:
        # Check if any imported module is not allowed
        for alias in node.names:
            module_name = alias.name.split('.')[0]
            if module_name not in self._ALLOWED_MODULES:
                return None
        return node
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> Union[ast.ImportFrom, None]:
        if node.module:
            module_name = node.module.split('.')[0]
            if module_name not in self._ALLOWED_MODULES:
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
        "logging", "logger", "log",
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


class BooleanMaskRewriter(ast.NodeTransformer):
    """Converts boolean masking operations to explicit loops.
    
    Transforms:
        result[bool_mask] = value
        result[result == 0.0] = 1.0
    
    Into:
        for _i in range(result.shape[0]):
            for _j in range(result.shape[1]):
                if bool_mask[_i, _j]:
                    result[_i, _j] = value
    """

    def _is_bool_mask_slice(self, slice_node: ast.AST) -> bool:
        """Check if a subscript slice is a boolean mask (Name or Compare)."""
        if isinstance(slice_node, ast.Name):
            return True
        if isinstance(slice_node, ast.Compare):
            return True
        return False

    def _make_indexed_condition(self, condition: ast.AST, array_name: str) -> ast.AST:
        """Replace array references in a Compare condition with indexed versions."""
        i_idx = ast.Name(id="_i", ctx=ast.Load())
        j_idx = ast.Name(id="_j", ctx=ast.Load())
        indexed = ast.Subscript(
            value=ast.Name(id=array_name, ctx=ast.Load()),
            slice=ast.Tuple(elts=[i_idx, j_idx], ctx=ast.Load()),
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
        
        i_idx = ast.Name(id="_i", ctx=ast.Load())
        j_idx = ast.Name(id="_j", ctx=ast.Load())
        
        inner_assign = ast.Assign(
            targets=[
                ast.Subscript(
                    value=ast.Name(id=array_name, ctx=ast.Store()),
                    slice=ast.Tuple(elts=[i_idx, j_idx], ctx=ast.Load()),
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
                slice=ast.Tuple(elts=[i_idx, j_idx], ctx=ast.Load()),
                ctx=ast.Load(),
            )
        elif isinstance(slice_node, ast.Compare):
            if_test = self._make_indexed_condition(slice_node, array_name)
        else:
            return node
        
        if_stmt = ast.If(
            test=if_test,
            body=[inner_assign],
            orelse=[],
        )
        
        j_loop = ast.For(
            target=ast.Name(id="_j", ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[
                    ast.Subscript(
                        value=ast.Attribute(
                            value=ast.Name(id=array_name, ctx=ast.Load()),
                            attr="shape",
                            ctx=ast.Load(),
                        ),
                        slice=ast.Constant(value=1),
                        ctx=ast.Load(),
                    )
                ],
                keywords=[],
            ),
            body=[if_stmt],
            orelse=[],
        )
        
        i_loop = ast.For(
            target=ast.Name(id="_i", ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[
                    ast.Subscript(
                        value=ast.Attribute(
                            value=ast.Name(id=array_name, ctx=ast.Load()),
                            attr="shape",
                            ctx=ast.Load(),
                        ),
                        slice=ast.Constant(value=0),
                        ctx=ast.Load(),
                    )
                ],
                keywords=[],
            ),
            body=[j_loop],
            orelse=[],
        )
        
        return ast.fix_missing_locations(i_loop)


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
