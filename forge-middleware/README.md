# forge-middleware

Runtime middleware for CPU-optimised Python modules (typically produced by the
[forge](../README.md) pipeline, but usable with **any** app).

Three jobs:

1. **Opt-in warm-up** — compile Numba signatures at boot, *outside* the
   request path, without ever calling functions you didn't declare.
2. **Telemetry** — per-call wall time, CPU and RAM.
3. **Fallback** — if an optimised function raises, re-run a registered
   original and count the event.

## Install

```bash
pip install git+https://github.com/<you>/forge.git#subdirectory=forge-middleware
# or with the Flask extra:
pip install "forge-middleware[flask] @ git+https://github.com/<you>/forge.git#subdirectory=forge-middleware"
```

Local/dev:

```bash
cd forge-middleware && pip install -e ".[flask]"
```

## Quickstart (Flask + config file)

`forge-middleware.toml`:

```toml
[settings]
timeout_s = 30
sample_ms = 50

[[module]]
name = "processing"
search_path = "optimized"
exclude = ["create_index"]          # side effects: never warm up

[[module.warmup]]
function = "extract_qa_bits"
args = ["expr:np.zeros(16, dtype=np.uint16)", 1]
```

`app.py`:

```python
from flask import Flask, jsonify
from forge_middleware import FlaskMiddleware

app = Flask(__name__)
fm = FlaskMiddleware("forge-middleware.toml").init_app(app)

@app.route("/qa", methods=["POST"])
def qa():
    out = fm.call("extract_qa_bits", band_data, bit_location)
    return jsonify(result=out["result"], time_ms=out["time_ms"])
```

## Quickstart (bare, no framework)

```python
import numpy as np
from forge_middleware import Middleware

mw = Middleware(modules=[processing])
mw.warmup(inputs={"extract_qa_bits": (np.zeros(16, dtype=np.uint16), 1)})
out = mw.call("extract_qa_bits", data, 3)
```

## Security model

- `warmup()` executes **only** the functions you declare. Everything else is
  reported `"skipped"` and never called. The legacy "warm everything with
  dummy args" mode exists (`include_undeclared=True`) but is unsafe for code
  with side effects (disk, network, DB).
- `expr:`-prefixed args evaluate in a restricted namespace (`np`/`numpy`
  only). No builtins, no imports.
- Each warmup call has a per-function timeout (`[settings] timeout_s`); a
  hanging kernel cannot wedge your boot.

## Platform notes

| Platform | Warmup strategy |
|----------|-----------------|
| Cloud Run (serverless) | **synchronous at boot** — CPU exists only while requests are in flight, background threads stall |
| Kubernetes / VM | synchronous or `warmup_async()` (CPU is persistent) |
| CLI / notebooks | on-demand via `call()`; warmup optional |

Numba compiles **per signature (dtype)**, not per shape — declare warmup args
with the exact dtypes your handlers pass, or the first real request with a new
dtype pays recompilation anyway.

## Tests

```bash
pip install -e ".[flask]" && pytest
```
