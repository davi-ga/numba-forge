import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from services.model import ModelService


class TesterService:

    def __init__(self, model: ModelService):
        self.model = model

    def generate(self, docs: list, output_dir: str) -> Optional[str]:
        """Generate a test file from *docs* and write it to *output_dir*.

        *docs* should be the preprocessed documents (functions extracted, no external libs)
        so the LLM generates tests that match the output structure.

        Returns the absolute path of the written test file, or None if generation
        failed.
        """
        print("[forge] Generating equivalence tests...")
        try:
            test_data = self.model.generate_test(
                json.dumps(docs, ensure_ascii=False, indent=2)
            )
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"[forge] WARNING: Test generation failed: {exc}", file=sys.stderr)
            return None

        tokens = test_data.get("tokens", {})
        print(
            f"[forge] Test generation complete "
            f"({test_data.get('time', 'n/a')}, {tokens.get('total', '?')} tokens)."
        )

        raw = test_data["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0].strip()

        try:
            doc = json.loads(raw)
            if isinstance(doc, list):
                doc = doc[0]
            file_name = doc.get("path", "test_equivalence.py")
            file_path = os.path.join(output_dir, file_name)
            os.makedirs(output_dir, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(doc["body"])
            print(f"[forge] Test file written: {file_path}")
            return file_path
        except (json.JSONDecodeError, KeyError) as exc:
            print(
                f"[forge] WARNING: Could not parse test response: {exc}",
                file=sys.stderr,
            )
            return None

    @staticmethod
    def _is_meaningful_test(test_file_path: str) -> tuple[bool, str]:
        """Reject vacuous tests (e.g. ``def test_x(): assert True``).

        A test is meaningful when it has at least one ``test_*`` function, it
        calls at least one function, and it contains a non-constant assertion.
        This prevents the equivalence gate from "passing" on a dummy test.
        """
        try:
            source = Path(test_file_path).read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError) as exc:
            return False, f"could not parse test file: {exc}"

        test_funcs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test")
        ]
        if not test_funcs:
            return False, "no test_* function found"

        real_assertions = 0
        calls = 0
        for func in test_funcs:
            for node in ast.walk(func):
                if isinstance(node, ast.Call):
                    calls += 1
                elif isinstance(node, ast.Assert):
                    test = node.test
                    if not isinstance(test, ast.Constant):
                        real_assertions += 1
                    elif isinstance(test.value, (int, float)) and not isinstance(
                        test.value, bool
                    ):
                        real_assertions += 1

        if real_assertions == 0:
            return False, "tests contain only constant assertions (e.g. assert True)"
        if calls == 0:
            return False, "tests do not call any function"
        return True, ""

    def run(self, test_file_path: str, input_dir: str, output_dir: str) -> tuple[bool, bool]:
        """Run *test_file_path* against both *input_dir* and *output_dir*.

        *input_dir* should contain the preprocessed code (without @njit).
        *output_dir* should contain the annotated code (with @njit).

        Returns ``(input_passed, output_passed)``.  A failure against the
        preprocessed *input* means the generated test itself is invalid (the
        input is the reference), so the caller should regenerate the test
        rather than retry modularisation.
        """
        meaningful, reason = self._is_meaningful_test(test_file_path)
        if not meaningful:
            print(
                f"[forge] WARNING: generated equivalence test is not meaningful "
                f"({reason}); rejecting it.",
                file=sys.stderr,
            )
            return False, False

        runs = [
            ("input (preprocessed)", "input", os.path.abspath(input_dir)),
            ("output (numba-annotated)", "output", os.path.abspath(output_dir)),
        ]
        passed = {"input": True, "output": True}
        for label, key, module_dir in runs:
            print(f"[forge] Running tests against {label}...")
            env = os.environ.copy()
            env["MODULE_DIR"] = module_dir
            timeout = int(os.getenv("FORGE_TEST_TIMEOUT", "120"))
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        test_file_path,
                        "-v",
                        "--tb=short",
                    ],
                    env=env,
                    check=False,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                passed[key] = False
                print(
                    f"[forge] WARNING: Tests timed out against {label} "
                    f"(>{timeout}s).",
                    file=sys.stderr,
                )
                continue
            if result.returncode != 0:
                passed[key] = False
                print(
                    f"[forge] WARNING: Tests failed against {label}.",
                    file=sys.stderr,
                )
        if passed["input"] and passed["output"]:
            print("[forge] All equivalence tests passed for both input and output.")
        return passed["input"], passed["output"]
