"""
NumbaDecoratorFilter - Remove @numba.njit de funções incompatíveis

Este transformer roda DEPOIS do LLM e remove decorators @numba.njit
de funções que usam bibliotecas externas (sklearn, PIL, plotly, etc).
"""

import ast


class NumbaDecoratorFilter(ast.NodeTransformer):
    """Remove @numba.njit de funções que usam bibliotecas externas."""

    _EXTERNAL_LIBS = frozenset({
        "sklearn", "KMeans", "AgglomerativeClustering",
        "pairwise_distances", "PIL", "Image",
        "plotly", "requests", "cv2",
    })

    _PIL_OPERATIONS = frozenset({
        "fromarray", "open", "save", "thumbnail", "resize",
        "crop", "rotate", "transpose", "convert",
    })

    def _has_njit(self, node: ast.FunctionDef) -> bool:
        return any(self._is_numba_decorator(dec) for dec in node.decorator_list)

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Remove @numba.njit transitively.

        A JIT function cannot call a plain Python function. If we strip
        @numba.njit from a function (e.g. because it uses an external
        library), every transitive caller must also lose its @numba.njit.
        """
        funcs = {
            n.name: n for n in node.body if isinstance(n, ast.FunctionDef)
        }

        jit_names = {name for name, f in funcs.items() if self._has_njit(f)}
        for name in list(jit_names):
            if self._uses_external_libs(funcs[name]):
                jit_names.discard(name)

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

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Keep per-function traversal for non-module callers."""
        self.generic_visit(node)
        if self._has_njit(node) and self._uses_external_libs(node):
            node.decorator_list = [
                dec for dec in node.decorator_list
                if not self._is_numba_decorator(dec)
            ]
        return node

    def _is_numba_decorator(self, dec: ast.AST) -> bool:
        """Verifica se um decorator é @numba.njit ou @numba.jit."""
        # @numba.njit
        if isinstance(dec, ast.Attribute):
            if isinstance(dec.value, ast.Name):
                if dec.value.id == "numba" and dec.attr in ("njit", "jit"):
                    return True
        
        # @numba.njit(...)
        if isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Attribute):
                if isinstance(dec.func.value, ast.Name):
                    if dec.func.value.id == "numba" and dec.func.attr in ("njit", "jit"):
                        return True
        
        return False

    def _uses_external_libs(self, node: ast.FunctionDef) -> bool:
        """Verifica se a função usa bibliotecas externas."""
        for child in ast.walk(node):
            # Verifica chamadas de função
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    if child.func.id in self._EXTERNAL_LIBS:
                        return True
                elif isinstance(child.func, ast.Attribute):
                    if isinstance(child.func.value, ast.Name):
                        if child.func.value.id in self._EXTERNAL_LIBS:
                            return True
                        if child.func.value.id == "Image" and child.func.attr in self._PIL_OPERATIONS:
                            return True
            
            if isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Name):
                    if child.value.id in self._EXTERNAL_LIBS:
                        return True
                    if child.value.id == "Image" and child.attr in self._PIL_OPERATIONS:
                        return True
            
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if isinstance(child.func.value, ast.Name):
                        if child.func.value.id == "np" and child.func.attr == "asarray":
                            if child.args and not self._is_numpy_array(child.args[0]):
                                return True
        
        return False

    def _is_numpy_array(self, node: ast.AST) -> bool:
        """Verifica se um nó é claramente um array NumPy."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "np":
                        return True
        return False


def filter_numba_decorators(code: str) -> str:
    """Remove @numba.njit de funções incompatíveis.
    
    Returns the filtered code as a string.
    """
    tree = ast.parse(code)
    tree = NumbaDecoratorFilter().visit(tree)
    return ast.unparse(ast.fix_missing_locations(tree))
