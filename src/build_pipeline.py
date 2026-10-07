"""
forge — LLM-based build-time refactoring pipeline.

Discovers Python source files, sends them to the Gemini LLM for
modularisation, injects @numba.njit decorators via AST, and writes the
optimised artefacts to an output directory ready to be copied into a
Cloud Run container.

CLI usage (after pip install sentinel[build]):
    forge --source-dir my_project/ --output-dir dist/

Environment variables:
    SOURCE_DIR      Override --source-dir default
    OUTPUT_DIR      Override --output-dir default
    GEMINI_API_KEY  Gemini API key (required)
    MODULARIZE_PROMPT   System prompt sent to the LLM (required)
    TEST_PROMPT         Test prompot sent to the LLM (required)
    
"""

import argparse
import ast
import json
import os
import py_compile
import sys
import tempfile

from services.annotator import AnnotatorService
from services.compatibility import NumbaCompatibilityChecker, has_actionable_errors
from services.model import ModelService
from services.patcher import PatcherService
from services.preprocessor import PreprocessorService
from services.tester import TesterService
from utils.vectorize_rewriter import VectorizeRewriter
from utils.builtin_rewriter import BuiltinRewriter
from utils.unique_rewriter import UniqueRewriter
from utils.preprocessor import NestedListToArray, StrCommentStripper


_SAFE_BUILTINS = frozenset({
    "len", "range", "enumerate", "zip", "min", "max", "abs", "sum", "round",
    "int", "float", "bool", "str", "bytes", "list", "tuple", "dict", "set",
    "frozenset", "isinstance", "issubclass", "print", "input", "open", "type",
    "repr", "format", "hex", "bin", "oct", "ord", "chr", "any", "all", "map",
    "filter", "reversed", "iter", "next", "getattr", "setattr", "hasattr",
    "delattr", "id", "callable", "divmod", "pow", "hash", "slice", "object",
    "super", "vars", "dir", "globals", "locals", "exec", "eval", "compile",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "RuntimeError", "StopIteration", "NotImplementedError", "ZeroDivisionError",
    "AssertionError", "AttributeError", "ImportError", "OSError", "IOError",
    "True", "False", "None", "self", "np", "numba", "numpy",
})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="forge",
        description=(
            "LLM-based code refactoring step for CI/CD pipelines. "
            "Transforms plain Python source files into Numba-optimised "
            "artefacts ready for Cloud Run deployment."
        ),
    )
    parser.add_argument(
        "--source-dir",
        default=os.getenv("SOURCE_DIR", "sample_project"),
        help="Directory containing the source Python files (default: sample_project).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "output_dir"),
        help="Directory where the refactored files will be written (default: output_dir).",
    )
    return parser.parse_args()


def _validate_env() -> None:
    missing = [
        v
        for v in ("GEMINI_API_KEY", "MODULARIZE_PROMPT", "TEST_PROMPT")
        if not os.getenv(v)
    ]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}"
        )


def _annotate_documents(
    annotator: AnnotatorService,
    documents: list[dict],
) -> list[dict]:
    annotated = []
    for doc in documents:
        path = doc["path"]
        body = doc["body"]

        if "__init__" in path:
            annotated.append({"path": path, "body": body})
            continue

        try:
            body = annotator.transform(body)
        except SyntaxError as exc:
            print(
                f"[forge] WARNING: Skipping annotation for '{path}' "
                f"(SyntaxError): {exc}",
                file=sys.stderr,
            )

        annotated.append({"path": path, "body": body})
    return annotated


class _NpArrayStripper(ast.NodeTransformer):
    """Rewrites `np.array(param, dtype=...)` to just `param` inside @njit functions.
    
    Numba cannot convert arrays to different dtypes via np.array().
    Since the wrapper already converts lists to numpy arrays before passing
    to the JIT function, np.array() on parameters is redundant.
    """
    
    def _is_np_array_call(self, node: ast.Call) -> bool:
        """Check if a call is np.array(...)."""
        if isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name) and 
                node.func.value.id in ('np', 'numpy') and 
                node.func.attr == 'array'):
                return True
        if isinstance(node.func, ast.Name) and node.func.id == 'array':
            return True
        return False
    
    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        """Replace `x = np.array(param, dtype=...)` with `x = param`.

        Only unwrap when the argument is a plain name (a parameter/variable).
        Module-level literals like ``m = np.array([[1, 2], [3, 4]])`` must be
        preserved, otherwise Numba loses the array.
        """
        if (
            isinstance(node.value, ast.Call)
            and self._is_np_array_call(node.value)
            and len(node.value.args) >= 1
            and isinstance(node.value.args[0], ast.Name)
        ):
            node.value = node.value.args[0]
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Replace `np.array(param, dtype=...)` with `param` in any context.

        The LLM sometimes wraps arguments in ``np.array(name, dtype=...)``
        inside a ``return`` or a nested call.  Numba rejects re-allocating an
        array with a different dtype inside nopython mode, and the wrapper
        already converts inputs, so the conversion is redundant.
        """
        self.generic_visit(node)
        if (
            self._is_np_array_call(node)
            and node.args
            and isinstance(node.args[0], ast.Name)
            and any(kw.arg == "dtype" for kw in node.keywords)
        ):
            return ast.copy_location(node.args[0], node)
        return node


class _UnusedImportCleaner(ast.NodeTransformer):
    """Removes or neutralises imports that preprocessing stripped.

    The LLM may re-add imports (plotly, sklearn, PIL, etc.) based on
    function bodies it preserves.  Rather than deleting such imports
    (which would cause NameErrors at call sites), this transformer
    wraps them in ``try / except ImportError`` so the module still
    loads even when the optional library is absent.  Functions that
    actually need the library will only fail when called — and the
    equivalence tests never call them.
    """

    _SAFE_MODULES = frozenset({
        'numpy', 'numba', 'typing', '_numba_helpers', 'os',
        'math', 'functools', 'itertools', 'collections',
        'copy', 'warnings', 'abc', 'enum', 'dataclasses',
        'json', 'time', 'sys',
    })

    def visit_Import(self, node: ast.Import):
        safe = [a for a in node.names if a.name.split('.')[0] in self._SAFE_MODULES]
        unsafe = [a for a in node.names if a.name.split('.')[0] not in self._SAFE_MODULES]
        if not unsafe:
            return node
        nodes = []
        if safe:
            nodes.append(ast.Import(names=safe))
        for alias in unsafe:
            try_body = [ast.Import(names=[alias])]
            except_body = [ast.Assign(
                targets=[ast.Name(id=alias.asname or alias.name.split('.')[0], ctx=ast.Store())],
                value=ast.Constant(value=None),
            )]
            handler = ast.ExceptHandler(type=ast.Name(id='ImportError', ctx=ast.Load()), name=None, body=except_body)
            nodes.append(ast.Try(body=try_body, handlers=[handler], orelse=[], finalbody=[]))
        return nodes

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module and node.module.split('.')[0] in self._SAFE_MODULES:
            return node
        if node.module and node.module.startswith('_numba_helpers'):
            return node
        try_body = [node]
        fallback_names = [alias.asname or alias.name for alias in node.names]
        except_body = [ast.Assign(
            targets=[ast.Tuple(elts=[ast.Name(id=n, ctx=ast.Store()) for n in fallback_names], ctx=ast.Store())],
            value=ast.Tuple(elts=[ast.Constant(value=None)] * len(fallback_names), ctx=ast.Load()),
        )] if fallback_names else [ast.Pass()]
        handler = ast.ExceptHandler(type=ast.Name(id='ImportError', ctx=ast.Load()), name=None, body=except_body)
        return ast.Try(body=try_body, handlers=[handler], orelse=[], finalbody=[])


def _postprocess_documents(documents: list[dict]) -> list[dict]:
    """Apply post-processing transformers after LLM generation.
    
    This converts patterns that the LLM might generate (like np.vectorize,
    hex(), bin(), np.unique with return_counts) to Numba-compatible code.
    Also strips np.array() calls on parameters that are already arrays.
    """
    import ast
    from utils.unique_rewriter import NUMBA_UNIQUE_HELPER
    from utils.builtin_rewriter import NUMBA_BUILTIN_HELPERS
    
    postprocessed = []
    for doc in documents:
        path = doc["path"]
        body = doc["body"]
        
        if "__init__" in path:
            postprocessed.append({"path": path, "body": body})
            continue
        
        try:
            tree = ast.parse(body)
            
            # Apply transformers
            tree = StrCommentStripper().visit(tree)
            tree = VectorizeRewriter().visit(tree)
            tree = BuiltinRewriter().visit(tree)
            tree = UniqueRewriter().visit(tree)
            tree = _NpArrayStripper().visit(tree)
            tree = NestedListToArray().visit(tree)
            
            # Check if we need to add helper functions
            needs_unique_helper = any(
                isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Name) and 
                node.func.id == '_np_unique_with_counts'
                for node in ast.walk(tree)
            )
            
            needs_builtin_helpers = any(
                isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Name) and 
                node.func.id in ('_int_to_hex', '_int_to_bin', '_int_to_oct')
                for node in ast.walk(tree)
            )
            
            body = ast.unparse(ast.fix_missing_locations(tree))
            
            # Add helper functions if needed (AFTER imports)
            helpers_to_add = []
            if needs_unique_helper:
                helpers_to_add.append(NUMBA_UNIQUE_HELPER)
            if needs_builtin_helpers:
                helpers_to_add.append(NUMBA_BUILTIN_HELPERS)
            
            if helpers_to_add:
                # Find the last import line
                lines = body.split('\n')
                last_import_idx = -1
                has_numba_import = False
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        last_import_idx = i
                        if 'import numba' in line or 'from numba' in line:
                            has_numba_import = True
                
                # Add 'import numba' if not present (helpers use @numba.njit)
                if not has_numba_import:
                    lines.insert(last_import_idx + 1, 'import numba')
                    last_import_idx += 1
                
                # Insert helpers after imports
                if last_import_idx >= 0:
                    lines.insert(last_import_idx + 1, '\n'.join(helpers_to_add))
                else:
                    lines.insert(0, '\n'.join(helpers_to_add))
                
                body = '\n'.join(lines)
            
            print(f"[forge] Post-processed: {path}")
        except SyntaxError as exc:
            print(
                f"[forge] WARNING: Skipping post-processing for '{path}' "
                f"(SyntaxError): {exc}",
                file=sys.stderr,
            )
        
        postprocessed.append({"path": path, "body": body})
    
    return postprocessed


def _collect_import_map(body: str) -> dict[str, ast.stmt]:
    """Map each import-bound name to the import statement that defines it."""
    out: dict[str, ast.stmt] = {}
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return out
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != "*":
                    out[alias.asname or alias.name.split(".")[0]] = node
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            for alias in node.names:
                if alias.name != "*":
                    out[alias.asname or alias.name] = node
    return out


def _restore_external_imports(documents: list[dict], original_by_path: dict[str, str]) -> list[dict]:
    """Re-add external imports that the output references but never defines.

    Preprocessing strips imports (PIL, sklearn, ...).  The pipeline relies on
    the LLM to re-add the ones still needed by plain-Python functions; when the
    LLM forgets one, the module raises ``NameError`` at runtime.  This step
    derives the required import from the ORIGINAL source and inserts it as a
    ``try/except ImportError`` block, making the output self-sufficient.
    """
    restored: list[dict] = []
    for doc in documents:
        body = doc["body"]
        try:
            tree = ast.parse(body)
        except SyntaxError:
            restored.append(doc)
            continue

        imap = _collect_import_map(original_by_path.get(doc["path"], ""))
        if not imap:
            restored.append(doc)
            continue

        defined: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    defined.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    defined.add(alias.asname or alias.name)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)

        needed: dict[str, list[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in defined and node.id in imap:
                    src = ast.unparse(imap[node.id])
                    names = needed.setdefault(src, [])
                    if node.id not in names:
                        names.append(node.id)
                    defined.add(node.id)

        if needed:
            blocks = []
            for src, names in needed.items():
                fallback = " = ".join(f"{n}" for n in names) + " = None"
                blocks.append(f"try:\n    {src}\nexcept ImportError:\n    {fallback}")
            body = "\n".join(blocks) + "\n" + body
            print(
                f"[forge] Restored {len(needed)} missing external import(s) in "
                f"'{doc['path']}': {[n for v in needed.values() for n in v]}",
                file=sys.stderr,
            )
        restored.append({"path": doc["path"], "body": body})
    return restored


def _has_real_body(func_node: ast.FunctionDef) -> bool:
    """Return True if the function has statements beyond ``pass``/docstrings."""
    for stmt in func_node.body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue
        return True
    return False


def _strip_unused_imports(documents: list[dict]) -> list[dict]:
    """Remove import statements whose names are never used in the code body."""
    cleaned = []
    for doc in documents:
        path = doc["path"]
        body = doc["body"]
        if "__init__" in path:
            cleaned.append(doc)
            continue
        try:
            tree = ast.parse(body)
            tree = _UnusedImportCleaner().visit(tree)
            body = ast.unparse(ast.fix_missing_locations(tree))
        except SyntaxError:
            pass
        cleaned.append({"path": path, "body": body})
    return cleaned


def _validate_documents(documents: list) -> None:
    """Raise ValueError if any document is missing required keys."""
    for i, doc in enumerate(documents):
        if not isinstance(doc, dict):
            raise ValueError(f"Item {i} is not an object, got {type(doc).__name__}.")
        missing = [k for k in ("path", "body") if k not in doc]
        if missing:
            raise ValueError(
                f"Item {i} is missing required key(s): {', '.join(missing)}."
            )


def run(
    source_dir: str,
    output_dir: str,
    max_attempts: int | None = None,
) -> list[str]:
    """
    Execute the full build pipeline programmatically.

    Parameters
    ----------
    source_dir:
        Directory containing the raw Python source files to optimise.
    output_dir:
        Directory where the refactored, Numba-annotated files will be written.
    max_attempts:
        Maximum LLM retry attempts when equivalence tests fail.
        Defaults to FORGE_MAX_ATTEMPTS env var, or 3.

    Returns
    -------
    List of absolute paths of the files written to output_dir.

    Raises
    ------
    EnvironmentError — GEMINI_API_KEY, MODULARIZE_PROMPT or TEST_PROMPT not set.
    RuntimeError     — LLM call failed or returned invalid output.
    """

    _validate_env()
    if max_attempts is None:
        max_attempts = int(os.getenv("FORGE_MAX_ATTEMPTS", "3"))

    patcher = PatcherService()
    model = ModelService()
    annotator = AnnotatorService()
    tester = TesterService(model)
    preprocessor = PreprocessorService()
    compat_checker = NumbaCompatibilityChecker()

    # Step 1 — Discover
    print(f"[forge] Discovering Python files in '{source_dir}'...")
    paths = patcher.discover(source_dir, extensions=[".py"])
    if not paths:
        raise RuntimeError(f"No Python files found in '{source_dir}'.")
    print(f"[forge] Found {len(paths)} file(s): {paths}")

    payload = patcher.to_json(source_dir, paths)
    original_docs = json.loads(payload)

    # Step 1.5 — Preprocess (AST transformers before LLM)
    print(f"[forge] Preprocessing {len(original_docs)} file(s)...")
    preprocessed_docs = preprocessor.transform_documents(original_docs)
    payload = json.dumps(preprocessed_docs, ensure_ascii=False, indent=2)

    # Step 1.6 — Write preprocessed code to temp dir for equivalence testing
    input_dir = tempfile.mkdtemp(prefix="forge_input_")
    patcher.to_files(input_dir, preprocessed_docs)
    patcher.ensure_init_files(input_dir, [doc["path"] for doc in preprocessed_docs])

    # Generate equivalence tests from preprocessed docs (matches output structure)
    test_file_path = tester.generate(preprocessed_docs, output_dir)

    prev_created: list[str] = []
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print(
                f"[forge] Retrying LLM pipeline (attempt {attempt}/{max_attempts})..."
            )
            for path in prev_created:
                try:
                    os.remove(path)
                except OSError:
                    pass

        # Step 2 — LLM inference
        print("[forge] Calling LLM for modularization...")
        try:
            model_data = model.modularize(payload)
        except Exception as exc:
            raise RuntimeError(f"LLM call failed: {exc}") from exc

        tokens = model_data.get("tokens", {})
        print(
            f"[forge] LLM response received "
            f"({model_data.get('time', 'n/a')}, {tokens.get('total', '?')} tokens)."
        )

        # Step 3 — Parse LLM output
        raw_text: str = model_data["text"]

        stripped = raw_text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1]
            stripped = stripped.rsplit("```", 1)[0]
            stripped = stripped.strip()
        try:
            documents: list[dict] = json.loads(stripped)
            if not isinstance(documents, list):
                raise ValueError("LLM output is not a JSON array.")
            _validate_documents(documents)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                f"LLM did not return a valid JSON array of files: {exc}\n"
                f"Raw output (first 500 chars): {raw_text}"
            ) from exc

        for doc in documents:
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
                f.write(doc["body"].encode())
                tmp = f.name
            try:
                py_compile.compile(tmp, doraise=True)
            except py_compile.PyCompileError as exc:
                raise RuntimeError(f"Syntax error in '{doc['path']}': {exc}") from exc
            finally:
                os.unlink(tmp)

        # Step 3.1 — Validate file structure preservation
        input_paths = {doc["path"] for doc in preprocessed_docs if "__init__" not in doc["path"]}
        output_paths = {doc["path"] for doc in documents}
        missing_paths = input_paths - output_paths

        # Step 3.2 — Validate function bodies preserved (detect stubs)
        input_funcs: dict[str, set[str]] = {}
        for doc in preprocessed_docs:
            if "__init__" in doc["path"]:
                continue
            try:
                tree = ast.parse(doc["body"])
                input_funcs[doc["path"]] = {
                    n.name for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and _has_real_body(n)
                }
            except SyntaxError:
                pass

        stubbed_funcs: list[str] = []
        for doc in documents:
            path = doc["path"]
            if path not in input_funcs:
                continue
            try:
                output_funcs = {
                    n.name: n for n in ast.walk(ast.parse(doc["body"]))
                    if isinstance(n, ast.FunctionDef)
                }
                for fname in input_funcs[path]:
                    node = output_funcs.get(fname) or output_funcs.get(f"_{fname}_jit")
                    if node is None:
                        stubbed_funcs.append(f"{path}:{fname} (missing)")
                    elif not _has_real_body(node):
                        stubbed_funcs.append(f"{path}:{fname} (stub)")
            except SyntaxError:
                pass

        # Step 3.3 — Validate no hallucinated (undefined) calls were introduced
        input_names_by_path: dict[str, set[str]] = {}
        for doc in preprocessed_docs:
            names: set[str] = set()
            try:
                for child in ast.walk(ast.parse(doc["body"])):
                    if isinstance(child, ast.Name):
                        names.add(child.id)
                    elif isinstance(child, ast.arg):
                        names.add(child.arg)
                    elif isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                        names.add(child.name)
            except SyntaxError:
                pass
            input_names_by_path[doc["path"]] = names

        orphan_calls: list[str] = []
        for doc in documents:
            try:
                tree = ast.parse(doc["body"])
            except SyntaxError:
                continue
            available: set[str] = set(_SAFE_BUILTINS) | input_names_by_path.get(doc["path"], set())
            for child in ast.walk(tree):
                if isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                    available.add(child.name)
                elif isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                    available.add(child.id)
                elif isinstance(child, ast.arg):
                    available.add(child.arg)
                elif isinstance(child, ast.Import):
                    for alias in child.names:
                        available.add(alias.asname or alias.name.split('.')[0])
                elif isinstance(child, ast.ImportFrom):
                    for alias in child.names:
                        available.add(alias.asname or alias.name)
            for child in ast.walk(tree):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id not in available
                ):
                    orphan_calls.append(f"{doc['path']}:{child.func.id}")

        # Step 3.4 — Numba compatibility check
        # Only "actionable" errors justify a retry. Errors caused by unavoidable
        # external-library usage or inherently non-JIT-able structure are left
        # as plain-Python wrappers, so retrying would waste an API call.
        print("[forge] Checking Numba compatibility...")
        actionable_errors = False
        for doc in documents:
            is_compatible, issues = compat_checker.check(doc["body"])
            if not is_compatible:
                print(
                    f"[forge] WARNING: Compatibility issues in '{doc['path']}':",
                    file=sys.stderr,
                )
                for severity, line, message in issues:
                    print(f"  [{severity.upper()}] Line {line}: {message}", file=sys.stderr)
                if has_actionable_errors(issues):
                    actionable_errors = True

        # Step 3.5 — Single retry decision covering all validations
        failures: list[str] = []
        if missing_paths:
            failures.append(f"dropped files: {missing_paths}")
        if stubbed_funcs:
            failures.append(f"{len(stubbed_funcs)} stubbed function(s): {stubbed_funcs[:5]}")
        if orphan_calls:
            failures.append(f"{len(orphan_calls)} undefined call(s): {sorted(set(orphan_calls))[:5]}")
        if actionable_errors:
            failures.append("actionable numba compatibility issues")

        if failures:
            print(
                f"[forge] WARNING: attempt {attempt} validation issues: " + " | ".join(failures),
                file=sys.stderr,
            )
            if attempt < max_attempts:
                print(
                    f"[forge] Reprocessing with LLM (attempt {attempt}/{max_attempts})...",
                    file=sys.stderr,
                )
                continue
            print(
                f"[forge] WARNING: issues remain after {max_attempts} attempt(s). "
                "Proceeding with annotation (some functions may fail at runtime).",
                file=sys.stderr,
            )

        # Step 3.6 — Post-process (convert LLM-generated patterns to Numba-compatible code)
        print(f"[forge] Post-processing {len(documents)} file(s)...")
        documents = _postprocess_documents(documents)

        # Step 4 — Numba annotation
        print(f"[forge] Annotating {len(documents)} file(s) with @numba.njit...")
        annotated_documents = _annotate_documents(annotator, documents)

        # Step 4.5 — Strip unused imports (LLM may re-add imports stripped by preprocessing)
        annotated_documents = _strip_unused_imports(annotated_documents)

        # Step 4.6 — Restore external imports the LLM forgot (NameError guard)
        original_by_path = {doc["path"]: doc["body"] for doc in original_docs}
        annotated_documents = _restore_external_imports(annotated_documents, original_by_path)

        # Step 5 — Write artefacts
        print(f"[forge] Writing optimised files to '{output_dir}'...")
        created = patcher.to_files(output_dir, annotated_documents)
        helpers_path = patcher.emit_helpers_module(output_dir)
        created.append(helpers_path)
        prev_created = list(created)

        # Ensure __init__.py files exist in output_dir for package imports
        patcher.ensure_init_files(output_dir, [doc["path"] for doc in annotated_documents])

        # Step 6 — Run equivalence tests
        if test_file_path:
            if test_file_path not in created:
                created.append(test_file_path)
            input_passed, output_passed = tester.run(
                test_file_path, input_dir, output_dir
            )
            if not (input_passed and output_passed):
                if attempt < max_attempts:
                    if not input_passed:
                        print(
                            f"[forge] WARNING: tests failed against the reference "
                            f"input — the generated test is invalid. Regenerating test...",
                            file=sys.stderr,
                        )
                        regenerated = tester.generate(preprocessed_docs, output_dir)
                        if regenerated:
                            test_file_path = regenerated
                    else:
                        print(
                            f"[forge] WARNING: Tests failed on attempt {attempt}. "
                            "Reprocessing with LLM...",
                            file=sys.stderr,
                        )
                    continue
                else:
                    raise RuntimeError(
                        f"Equivalence tests failed after {max_attempts} attempt(s). "
                        "Aborting."
                    )
        break

    print(f"[forge] Done. {len(created)} file(s) written:")
    for path in created:
        print(f"  - {path}")

    return created


def main() -> None:
    """Entry point for the `forge` CLI command."""
    args = _parse_args()
    try:
        run(args.source_dir, args.output_dir)
    except (EnvironmentError, RuntimeError) as exc:
        print(f"[forge] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
