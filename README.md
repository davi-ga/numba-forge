# forge

**LLM as an Intelligent Compiler** — uses the Gemini API at **build time** to modularize Python source code and inject `@numba.njit` decorators, then provides a lightweight **runtime middleware** (`sentinel.Middleware`) for Cloud Run containers with warm-up, telemetry and fallback.

> Full system reference (components, contracts, failure modes):
> **[`docs/ARQUITETURA.md`](docs/ARQUITETURA.md)**

---

## How it works

Two planes: a build-time compiler pipeline and a runtime middleware.

```
BUILD TIME (CI/CD)                               RUNTIME (Cloud Run)
──────────────────────────────────────────       ──────────────────────────────────────
forge --source-dir src/ --output-dir dist/        gunicorn boots app.py
  │                                                 │
  ├── 1. PatcherService                             ├── Middleware(modules=[...])
  │     Discovers .py files, serialises to              registers functions AND
  │     [{path, body}] payload                          Numba CPUDispatchers
  │                                                 │
  ├── 2. PreprocessorService (AST, before the LLM)  ├── warmup(inputs={...})  ← SYNC
  │     import strip → orphan-call strip → class        compiles every declared
  │     extraction → self strip → IO strip →            signature AT BOOT (Cloud Run
  │     empty-block strip → boolean-mask rewrite →      only allocates CPU during
  │     nested-list→np.array                            requests, so async warmup
  │                                                     stalls — see docs)
  ├── 3. TesterService (LLM)                        │
  │     generates equivalence test suite            ├── cache JIT na imagem:
  │     (validated against the reference code)      │   @numba.njit(cache=True) +
  │                                                 │   RUN warmup_build.py → boot
  ├── 4. ModelService (LLM modularization)          │   carrega cache (~s, no min)
  │     gemini-3.1-pro-preview; structure, stub,    │
  │     hallucinated-call and compatibility         ├── call("func", *args)
  │     validators gate a retry loop (only            │   telemetry: time_ms,
  │     actionable failures re-invoke the LLM)        │   cpu_avg, mem_mb_avg +
  │                                                 │   automatic fallback
  ├── 5. AnnotatorService (AST)                     └── fallback: if the JIT
  │     jittability fixpoint over the call graph        dispatcher raises, the
  │     (a function is annotated only if its local      registered original
  │     callees are too), @numba.njit injection,          callable re-runs
  │     wrapper generation (_sanitize/_deep_to_list)
  │                                                 First real request never pays
  ├── 6. Postprocess + import guards                LLVM compilation:
  │     vectorize/builtin/unique rewriters,         measured cold 1st request:
  │     unused-import strip, external-import        10.4 ms (yuki-mt) / 0.03 ms
  │     restoration (try/except ImportError)        (cube-builder) vs 1575/586 ms
  │                                                 without the middleware
  └── 7. Equivalence gate (pytest)
        same suite vs reference and annotated
        output; failing reference = bad test →
        regenerate; 3 attempts → abort
```

Measured outcomes of the full system (Cloud Run, 1 vCPU): **yuki-mt ≈ 115×**
compute speedup after LLM modularization; reproducible success rate ≥ 90% per
benchmark — details in `docs/ANALISE_RESULTADOS_EXECUTADOS.md`.

---

## Installation

**Build pipeline (CI/CD — includes Gemini SDK):**
```bash
pip install "forge[build] @ git+https://github.com/davi-ga/forge"
```

**Runtime middleware (Cloud Run container — separate package):**
```bash
pip install "forge-middleware[flask] @ git+https://github.com/davi-ga/forge.git#subdirectory=forge-middleware"
```

> The repository ships **two distributions**: `forge` (build-time pipeline,
> repo root) and `forge-middleware` (runtime, `forge-middleware/`
> subdirectory). The legacy `src/sentinel.py` is kept only for benchmark
> containers migrated so far.

The optimized runtime also needs `numba` + `numpy` in the *application's*
`requirements.txt` (the generated code imports them directly).

---

## Build pipeline

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Gemini API key |
| `MODULARIZE_PROMPT` | Yes | System prompt for code modularization (must return `[{path, body}]` JSON) |
| `TEST_PROMPT` | Yes | System prompt for equivalence-test generation |
| `SOURCE_DIR` / `OUTPUT_DIR` | No | CLI flag defaults |
| `FORGE_TEST_TIMEOUT` | No | Per-side pytest timeout, seconds (default 120) |

### CLI

```bash
forge --source-dir my_project/ --output-dir dist/
```

### Python API

```python
from build_pipeline import run

created_files = run(source_dir="my_project/", output_dir="dist/", max_attempts=3)
```

### Cloud Build (`cloudbuild.yaml`)

```yaml
steps:
  - name: "python:3.12-slim"
    entrypoint: pip
    args: ["install", "--no-cache-dir", "forge[build]@git+https://github.com/davi-ga/forge.git"]

  - name: "python:3.12-slim"
    entrypoint: forge
    args: ["--source-dir", "my_project/", "--output-dir", "dist/"]
    secretEnv: ["GEMINI_API_KEY", "MODULARIZE_PROMPT", "TEST_PROMPT"]

  - name: "gcr.io/cloud-builders/docker"
    args: ["build", "-t", "gcr.io/$PROJECT_ID/my-service", "."]
```

---

## Runtime middleware (`forge-middleware`)

Config-driven, framework-optional. `forge-middleware.toml`:

```toml
[settings]
timeout_s = 30          # per-function warmup budget
sample_ms = 25          # telemetry sampling

[[module]]
name = "processing"
search_path = "optimized"
exclude = ["create_index"]            # side effects: never warm up

[[module.warmup]]
function = "extract_qa_bits"
args = ["expr:np.zeros(16, dtype=np.uint16)", 1]
```

Flask integration (5 lines):

```python
from flask import Flask
from forge_middleware import FlaskMiddleware

app = Flask(__name__)
fm = FlaskMiddleware("forge-middleware.toml").init_app(app)  # sync warmup + /health

# handlers: fm.call("extract_qa_bits", band_data, bit_location)
```

Bare usage (no framework):

```python
from forge_middleware import Middleware

mw = Middleware(modules=[processing])
mw.warmup(inputs={"extract_qa_bits": (np.zeros(16, dtype=np.uint16), 1)})
out = mw.call("extract_qa_bits", band_data, bit_location)
```

Key semantics:

- **Opt-in warmup**: `warmup()` executes **only** the declared functions;
  everything else is reported `"skipped"`. A generic app can have side
  effects (disk, network, DB) — boot must never trigger them. The legacy
  "call everything with dummy args" mode is available via
  `include_undeclared=True` for code proven pure.
- **Registration** covers plain functions, builtins *and* Numba
  `CPUDispatcher` objects (kernels called internally keep their name as the
  dispatcher itself — `inspect.isfunction` would miss them).
- **Timeout per function**: a hanging kernel cannot wedge the boot.
- **`call()`** adds CPU/RAM telemetry and re-executes the registered
  fallback when the optimized path raises (`fallback_used` + counters in
  `mw.stats()`).
- **`expr:`-args** evaluate in a restricted namespace (`np`/`numpy` only —
  no builtins, no imports).
- `warmup_async()` exists for platforms with persistent CPU (e.g.
  Kubernetes); on Cloud Run it starves — use the sync form at import.

Full rationale, trade-offs and measured numbers:
**[`docs/MIDDLEWARE_WARMUP.md`](docs/MIDDLEWARE_WARMUP.md)** (§7 covers the
disadvantages of boot-time warmup).

---

## Dockerfile (Cloud Run, pattern validated in benchmarks)

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY optimized/ optimized/        # forge artefacts + _numba_helpers.py
COPY forge_middleware/ forge_middleware/   # or `pip install forge-middleware[flask]`
COPY forge-middleware.toml .
COPY warmup_build.py .
COPY app.py .

ENV NUMBA_CACHE_DIR=/app/.nv_jit_cache
RUN python warmup_build.py        # JIT cache baked into the image

EXPOSE 8080
# timeout > warmup duration: worker only becomes ready after warming
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "app:app"]
```

`app.py` calls `mw.warmup(inputs=...)` synchronously at import, so the first
real request never pays LLVM compilation.

---

## Project structure

```
src/
  build_pipeline.py     # Build-time pipeline — CLI + Python API (run())
  sentinel.py           # LEGACY runtime middleware (kept for un-migrated deploys;
                        # new code uses forge-middleware/ instead)
  services/
    patcher.py          # File discovery, payload serialisation, artefact I/O
    preprocessor.py     # AST pre-processing orchestration (before the LLM)
    model.py            # Gemini API client (modularize + test generation)
    tester.py           # Equivalence-test generation + input/output runner
    annotator.py        # Numba annotation orchestration
    compatibility.py    # Compatibility checker + actionable-error classifier
  utils/
    preprocessor.py     # AST transformers (class extraction, IO/orphan strip,
                        # boolean-mask rewrite, nested lists, comments…)
    modifier.py         # Inserter: jittability fixpoint + @njit + wrappers
    numba_filter.py     # transitive @njit stripper (external libs)
    vectorize_rewriter.py / builtin_rewriter.py / unique_rewriter.py
benchmarks/
  benchmark_reproducibility.py   # 10 runs × benchmark, success/cost/variance
  gcr/                           # Cloud Run apps, load tests, A/B results
  gcr/run_gcr_requests.py        # N requests per service, warm-up + stats
tests/
  test_pipeline_fixes.py         # regression tests for every pipeline fix
  test_sentinel_middleware.py    # legacy middleware tests (src/sentinel.py)
forge-middleware/                # standalone runtime package
  forge_middleware/
    middleware.py                # Middleware + ResourceMonitor (opt-in warmup)
    config.py                    # TOML schema + restricted arg evaluation
    flask_adapter.py             # FlaskMiddleware (init_app + /health)
  tests/                         # 32 tests: core, config, security, flask
  examples/flask_minimal/        # runnable 5-line integration example
docs/
  ARQUITETURA.md                 # ← system reference (start here)
  MIDDLEWARE_WARMUP.md           # middleware deep-dive + trade-offs
  ANALISE_RESULTADOS_EXECUTADOS.md  # what the experiments establish
  FASE2_GCR_RESULTADOS.md / GCR_REQUESTS_RESULTADOS.md / CUBE_BUILDER_ERROS.md
  COST_ANALYSIS_GEMINI_3.1.md / PIPELINE_CHANGES_GEMINI_3.1.md / ALTERACOES_IMG2PALETTE.md
pyproject.toml          # package metadata, CLI entry point, extras
```

---

## Requirements

- Python ≥ 3.11
- Runtime image: `numba`, `numpy` (+ `psutil` for telemetry; `flask`/`gunicorn` in the app)
- Build extras: `google-genai`, `python-dotenv`, `pyflakes`

## Validation status (Sept/2026)

| Plane | Evidence |
|---|---|
| Pipeline reproducibility | yuki-mt 10/10, img2palette 9/10 on 30-run benchmark (`benchmarks/reproducibility_results.json`) |
| GCR runtime | yuki-mt ≈115× compute; middleware cold-start A/B; 30/30 requests OK |
| Regressions | 40/40 unit tests (`tests/`) |
| Pending | full pipeline with LLM on GCR (blocked by GCP billing); cube-builder rate (API quota) |
