"""Minimal Flask app using forge-middleware (6 lines of glue)."""

import numpy as np
from flask import Flask, jsonify, request

from forge_middleware import FlaskMiddleware

app = Flask(__name__)
fm = FlaskMiddleware("forge-middleware.toml").init_app(app)


@app.route("/qa", methods=["POST"])
def qa():
    data = request.get_json()
    band = np.array(data["band_data"], dtype=np.uint16)
    out = fm.call("extract_qa_bits", band, int(data["bit_location"]))
    return jsonify(
        result=out["result"],
        time_ms=out["time_ms"],
        cpu_avg=out["cpu_avg"],
        fallback_used=out["fallback_used"],
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
