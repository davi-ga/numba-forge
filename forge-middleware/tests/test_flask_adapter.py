"""Tests for the Flask adapter (config-driven boot + /health)."""

import textwrap

import numpy as np
import pytest
from flask import Flask

from forge_middleware import FlaskMiddleware

APP_MODULE = """
import numpy as np

def add(x, y):
    return x + y

def dangerous():
    raise AssertionError("must never run during warmup")
"""

CONFIG = textwrap.dedent(
    """
    [settings]
    timeout_s = 10

    [[module]]
    name = "appmod"
    search_path = "{tmp}"

    [[module.warmup]]
    function = "add"
    args = ["expr:np.ones(3)", "expr:np.ones(3)"]
    """
)


@pytest.fixture()
def wired_app(tmp_path):
    (tmp_path / "appmod.py").write_text(APP_MODULE)
    cfg_path = tmp_path / "forge-middleware.toml"
    cfg_path.write_text(CONFIG.format(tmp=tmp_path))

    app = Flask(__name__)
    fm = FlaskMiddleware(str(cfg_path))
    fm.init_app(app)
    return app, fm


def test_init_app_warms_only_declared(wired_app):
    app, fm = wired_app
    report = fm.mw.warmup_report
    assert report["add"]["status"] == "ok"
    assert report["dangerous"]["status"] == "skipped"  # never executed


def test_health_route_registered(wired_app):
    app, fm = wired_app
    client = app.test_client()
    body = client.get("/health").get_json()
    assert body["status"] == "ok"
    assert body["warmup_done"] is True
    assert body["registered"] == 2


def test_call_through_adapter(wired_app):
    app, fm = wired_app
    out = fm.call("add", np.ones(3), np.ones(3))
    assert np.allclose(out["result"], 2.0)
    assert out["fallback_used"] is False


def test_existing_health_route_not_overwritten(tmp_path):
    (tmp_path / "appmod.py").write_text(APP_MODULE)
    cfg_path = tmp_path / "forge-middleware.toml"
    cfg_path.write_text(CONFIG.format(tmp=tmp_path))

    app = Flask(__name__)

    @app.get("/health")
    def my_health():
        return {"custom": True}

    FlaskMiddleware(str(cfg_path)).init_app(app)
    body = app.test_client().get("/health").get_json()
    assert body == {"custom": True}


def test_bare_modules_without_config():
    import types

    mod = types.ModuleType("m")
    mod.f = lambda a: a * 2
    app = Flask(__name__)
    fm = FlaskMiddleware(modules=[mod], add_health_route=False).init_app(app)
    assert fm.warmup_done is True
    assert fm.call("f", 21)["result"] == 42


def test_requires_config_or_modules():
    with pytest.raises(ValueError):
        FlaskMiddleware()
