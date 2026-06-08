from types import SimpleNamespace

from iblgf import ns_amr
from iblgf import ns_amr_2d
from iblgf import poisson


def test_poisson_wrapper_forwards_to_native_bindings(monkeypatch):
    calls = {}

    def fake_run(config_path, cli_overrides):
        calls["config_path"] = config_path
        calls["cli_overrides"] = cli_overrides
        return SimpleNamespace(config_path=config_path)

    fake_bindings = SimpleNamespace(poisson=SimpleNamespace(run=fake_run))
    monkeypatch.setattr(poisson, "load_bindings", lambda: fake_bindings)

    result = poisson.run("cfg", cli_overrides=["--resume"])

    assert result.config_path == "cfg"
    assert calls == {
        "config_path": "cfg",
        "cli_overrides": ["--resume"],
    }


def test_ns_amr_2d_wrapper_forwards_to_native_bindings(monkeypatch):
    calls = {}

    def fake_run(config_path, cli_overrides):
        calls["config_path"] = config_path
        calls["cli_overrides"] = cli_overrides
        return SimpleNamespace(config_path=config_path)

    fake_bindings = SimpleNamespace(ns_amr_2d=SimpleNamespace(run=fake_run))
    monkeypatch.setattr(ns_amr_2d, "load_bindings", lambda: fake_bindings)

    result = ns_amr_2d.run("cfg2", cli_overrides=["--foo"])

    assert result.config_path == "cfg2"
    assert calls == {
        "config_path": "cfg2",
        "cli_overrides": ["--foo"],
    }


def test_ns_amr_wrapper_forwards_to_native_bindings(monkeypatch):
    calls = {}

    def fake_run(config_path, cli_overrides):
        calls["config_path"] = config_path
        calls["cli_overrides"] = cli_overrides
        return SimpleNamespace(config_path=config_path)

    fake_bindings = SimpleNamespace(ns_amr=SimpleNamespace(run=fake_run))
    monkeypatch.setattr(
        ns_amr,
        "load_bindings",
        lambda module_name="iblgf_bindings_ns_amr": fake_bindings,
    )

    result = ns_amr.run("cfg3", cli_overrides=["--bar"])

    assert result.config_path == "cfg3"
    assert calls == {
        "config_path": "cfg3",
        "cli_overrides": ["--bar"],
    }
