import ast

from utils.modifier import Inserter
from utils.numba_filter import NumbaDecoratorFilter


def _is_flask_endpoint(node: ast.FunctionDef) -> bool:
    """Check if a function is a Flask route or contains HTTP logic."""
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Attribute) and decorator.func.attr == 'route':
                return True
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name) and child.func.id == 'jsonify':
                return True
    return False


class AnnotatorService:
    def __init__(self):
        self.inserter = Inserter()

    def transform(self, modified_code: str) -> str:
        tree = ast.parse(modified_code)

        self.inserter.importer(tree)

        # Mark Flask endpoints so the Inserter skips them
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and _is_flask_endpoint(node):
                node._skip_numba = True

        modified = self.inserter.visit(tree)

        modified = NumbaDecoratorFilter().visit(modified)

        return ast.unparse(modified)
