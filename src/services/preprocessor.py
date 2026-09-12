"""
Preprocessor service — applies AST transformers before sending code to the LLM.
"""

import ast
import sys

from utils.preprocessor import (
    ClassExtractor,
    IOStripper,
    BooleanMaskRewriter,
    ExternalImportStripper,
    ExternalFunctionStripper,
    SelfStrippingTransformer,
)


class PreprocessorService:
    """Orchestrates preprocessing transformers on source code."""

    def __init__(self, enable_class_extract: bool = True,
                 enable_io_strip: bool = True,
                 enable_bool_mask_rewrite: bool = True,
                 enable_import_strip: bool = True,
                 enable_function_strip: bool = True,
                 enable_self_strip: bool = True):
        self.transformers = []
        if enable_class_extract:
            self.transformers.append(("ClassExtractor", ClassExtractor()))
        if enable_import_strip:
            self.transformers.append(("ExternalImportStripper", ExternalImportStripper()))
        if enable_function_strip:
            self.transformers.append(("ExternalFunctionStripper", ExternalFunctionStripper()))
        if enable_self_strip:
            self.transformers.append(("SelfStrippingTransformer", SelfStrippingTransformer()))
        if enable_io_strip:
            self.transformers.append(("IOStripper", IOStripper()))
        if enable_bool_mask_rewrite:
            self.transformers.append(("BooleanMaskRewriter", BooleanMaskRewriter()))

    def transform(self, code: str) -> str:
        """Apply all enabled transformers to the code.
        
        Returns the preprocessed code as a string.
        Logs each transformer's effect.
        """
        tree = ast.parse(code)
        
        for name, transformer in self.transformers:
            original_count = len(ast.dump(tree))
            tree = transformer.visit(tree)
            tree = ast.fix_missing_locations(tree)
            new_count = len(ast.dump(tree))
            
            if new_count != original_count:
                print(f"[forge] {name}: modified AST ({original_count} → {new_count} nodes)")

        return ast.unparse(tree)

    def transform_documents(self, documents: list[dict]) -> list[dict]:
        """Apply preprocessing to a list of {path, body} documents.
        
        Returns a new list with preprocessed bodies.
        """
        preprocessed = []
        for doc in documents:
            path = doc["path"]
            body = doc["body"]
            
            if "__init__" in path:
                preprocessed.append({"path": path, "body": body})
                continue
            
            try:
                body = self.transform(body)
                print(f"[forge] Preprocessed: {path}")
            except SyntaxError as exc:
                print(
                    f"[forge] WARNING: Skipping preprocessing for '{path}' "
                    f"(SyntaxError): {exc}",
                    file=sys.stderr,
                )
            
            preprocessed.append({"path": path, "body": body})
        
        return preprocessed
