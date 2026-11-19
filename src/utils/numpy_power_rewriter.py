"""Restore numpy semantics that the LLM rewrote to the ``**`` operator."""

import ast


class NumpyPowerRewriter(ast.NodeTransformer):
    """Rewrite ``x ** float`` back to numpy's bit-exact functions.

    Numpy's array ufuncs (``np.power``, ``np.cbrt``) are not bit-identical
    to Python's ``**`` operator on scalars.  When the LLM scalarizes a
    vectorized numpy expression and rewrites ``np.power(x, e)`` or
    ``np.cbrt(x)`` into ``x ** e``, the last ulp changes; downstream integer
    casts (``astype``) then amplify that into a ±1 value error.

    This transformer restores:

    - ``x ** (1 / 3)``  ->  ``np.cbrt(x)``
    - ``x ** float``    ->  ``np.power(x, float)``

    Integer exponents (``x ** 2``) are left untouched: they are already
    bit-identical to ``np.power(x, 2)``.
    """

    _CBRT = 1.0 / 3.0

    def _exponent_value(self, node: ast.expr):
        """Best-effort numeric value of a constant or constant division."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return None
            if isinstance(node.value, (int, float)):
                return float(node.value)
            return None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left = self._exponent_value(node.left)
            right = self._exponent_value(node.right)
            if left is not None and right is not None and right != 0.0:
                return left / right
        return None

    def _is_integer_literal(self, node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        )

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.op, ast.Pow):
            return node
        if self._is_integer_literal(node.right):
            return node
        value = self._exponent_value(node.right)
        if value is None:
            return node

        if value == self._CBRT:
            func = ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="cbrt",
                ctx=ast.Load(),
            )
            call = ast.Call(func=func, args=[node.left], keywords=[])
        else:
            func = ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="power",
                ctx=ast.Load(),
            )
            call = ast.Call(func=func, args=[node.left, node.right], keywords=[])

        return ast.copy_location(ast.fix_missing_locations(call), node)


def restore_numpy_power(code: str) -> str:
    """Return *code* with ``**`` float exponents restored to numpy functions."""
    tree = ast.parse(code)
    tree = NumpyPowerRewriter().visit(tree)
    return ast.unparse(ast.fix_missing_locations(tree))
