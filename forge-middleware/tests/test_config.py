"""Tests for the TOML config parser."""

import textwrap

import numpy as np
import pytest

from forge_middleware.config import MiddlewareConfig, load_config, parse_config


VALID_TOML = textwrap.dedent(
    """
    [settings]
    timeout_s = 45
    sample_ms = 25

    [[module]]
    name = "processing"
    search_path = "optimized"
    exclude = ["create_index"]

    [[module.warmup]]
    function = "extract_qa_bits"
    args = ["expr:np.zeros(16, dtype=np.uint16)", 1]

    [[module.warmup]]
    function = "format_version"
    args = ["1", "v"]
    """
)


def test_load_config(tmp_path):
    p = tmp_path / "forge-middleware.toml"
    p.write_text(VALID_TOML)
    cfg = load_config(str(p))
    assert cfg.settings.timeout_s == 45
    assert cfg.settings.sample_ms == 25
    assert len(cfg.modules) == 1
    mod = cfg.modules[0]
    assert mod.name == "processing"
    assert mod.search_path == "optimized"
    assert mod.exclude == ["create_index"]
    assert [w.function for w in mod.warmup] == ["extract_qa_bits", "format_version"]


def test_warmup_inputs_resolve_expressions(tmp_path):
    p = tmp_path / "forge-middleware.toml"
    p.write_text(VALID_TOML)
    cfg = load_config(str(p))
    inputs = cfg.warmup_inputs()
    args = inputs["extract_qa_bits"]
    assert isinstance(args[0], np.ndarray)
    assert args[0].dtype == np.uint16
    assert args[1] == 1
    assert inputs["format_version"] == ("1", "v")


def test_empty_config_defaults():
    cfg = parse_config({})
    assert cfg.settings.timeout_s == 30.0
    assert cfg.modules == []
    assert cfg.warmup_inputs() == {}


def test_invalid_module_name_rejected():
    with pytest.raises(ValueError):
        parse_config({"module": [{"name": "not an identifier!"}]})


def test_invalid_warmup_function_rejected():
    with pytest.raises(ValueError):
        parse_config(
            {"module": [{"name": "ok", "warmup": [{"function": "bad name", "args": []}]}]}
        )


def test_invalid_args_type_rejected():
    with pytest.raises(ValueError):
        parse_config({"module": [{"name": "ok", "warmup": [{"function": "f", "args": "nope"}]}]})


def test_import_module_with_search_path(tmp_path):
    (tmp_path / "mymod.py").write_text("VALUE = 42\n")
    cfg = parse_config({"module": [{"name": "mymod", "search_path": str(tmp_path)}]})
    mods = cfg.import_modules()
    assert mods[0].VALUE == 42


def test_middlewareconfig_import_modules_empty():
    assert MiddlewareConfig().import_modules() == []
