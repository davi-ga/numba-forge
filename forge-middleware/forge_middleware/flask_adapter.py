"""Flask integration for forge-middleware.

Usage::

    from flask import Flask
    from forge_middleware.flask_adapter import FlaskMiddleware

    app = Flask(__name__)
    fm = FlaskMiddleware("forge-middleware.toml")
    fm.init_app(app)          # warmup síncrono + /health

    @app.route("/qa", methods=["POST"])
    def qa():
        out = fm.call("extract_qa_bits", band_data, bit_location)
        return jsonify(result=out["result"], time_ms=out["time_ms"])

The adapter lives in an optional extra (``pip install forge-middleware[flask]``)
so the core package stays framework-free.
"""

from __future__ import annotations

from typing import Callable

try:
    from flask import Flask, jsonify
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "FlaskMiddleware requires the 'flask' extra: pip install forge-middleware[flask]"
    ) from exc

from .config import MiddlewareConfig, load_config
from .middleware import Middleware


class FlaskMiddleware:
    """Wires a :class:`Middleware` into a Flask app lifecycle."""

    def __init__(
        self,
        config_path: str | None = None,
        config: MiddlewareConfig | None = None,
        modules: list | None = None,
        fallbacks: dict[str, Callable] | None = None,
        add_health_route: bool = True,
        health_path: str = "/health",
        auto_warmup: bool = True,
    ):
        if config is None:
            config = load_config(config_path) if config_path else None
        if config is not None:
            self.mw = Middleware.from_config(config, fallbacks=fallbacks)
            self._config = config
        else:
            if not modules:
                raise ValueError("provide config_path/config or modules")
            self.mw = Middleware(modules=modules, fallbacks=fallbacks)
            self._config = None
        self.add_health_route = add_health_route
        self.health_path = health_path
        self.auto_warmup = auto_warmup

    def init_app(self, app: Flask) -> "FlaskMiddleware":
        """Synchronous warmup at boot + optional /health route.

        Warmup runs here (not in a background thread) on purpose: serverless
        platforms only allocate CPU while a request is in flight, and the
        readiness gate (``warmup_done``) is exposed via ``/health``.
        """
        if self.auto_warmup:
            timeout_s = self._config.settings.timeout_s if self._config else 30.0
            self.mw.warmup(timeout_s=timeout_s)

        app.extensions["forge_middleware"] = self

        if self.add_health_route and not self._app_has_route(app, self.health_path):
            mw = self.mw

            @app.get(self.health_path)
            def _forge_health():  # noqa: ANN202
                return jsonify(status="ok", **mw.stats())

        return self

    @staticmethod
    def _app_has_route(app: Flask, path: str) -> bool:
        return any(rule.rule == path for rule in app.url_map.iter_rules())

    def call(self, func_name: str, *args, **kwargs) -> dict:
        return self.mw.call(func_name, *args, **kwargs)

    @property
    def warmup_done(self) -> bool:
        return self.mw.warmup_done
