from pathlib import Path
from types import SimpleNamespace

from iblgf import ns_amr_2d
from iblgf import poisson
from iblgf._config import stage_config


def test_run_from_template_cleans_generated_config(monkeypatch, tmp_path: Path):
    template = tmp_path / "config.in"
    template.write_text(
        """
simulation_parameters
{
    vortex
    {
        R=0.125;
    }
}
""".strip()
    )

    captured: dict[str, Path] = {}

    def fake_run(config_path, cli_overrides=None):
        path = Path(config_path)
        captured["path"] = path
        assert path.exists()
        return SimpleNamespace(config_path=str(path))

    monkeypatch.setattr(poisson, "run", fake_run)
    result = poisson.run_from_template(
        template,
        vortex_overrides=[{"R": 0.2}],
    )

    generated = captured["path"]
    assert result.config_path == str(generated)
    assert not generated.exists()


def test_prepare_config_without_output_path_uses_runs_pybind(tmp_path: Path):
    template = tmp_path / "config.in"
    template.write_text(
        """
simulation_parameters
{
    vortex
    {
        R=0.125;
    }
}
""".strip()
    )

    generated = poisson.prepare_config(template, vortex_overrides=[{"R": 0.2}])
    try:
        assert "runs_pybind" in str(generated)
        assert generated.exists()
    finally:
        generated.unlink(missing_ok=True)
        generated.parent.rmdir()


def test_ns_amr_2d_run_from_template_cleans_generated_config(
    monkeypatch, tmp_path: Path
):
    template = tmp_path / "config.in"
    template.write_text(
        """
simulation_parameters
{
    Re=1000.0;
}
""".strip()
    )

    captured: dict[str, Path] = {}

    def fake_run(config_path, cli_overrides=None):
        path = Path(config_path)
        captured["path"] = path
        assert path.exists()
        return SimpleNamespace(config_path=str(path))

    monkeypatch.setattr(ns_amr_2d, "run", fake_run)
    result = ns_amr_2d.run_from_template(
        template,
        simulation_overrides={"Re": 250.0},
    )

    generated = captured["path"]
    assert result.config_path == str(generated)
    assert not generated.exists()


def test_stage_config_uses_runs_pybind_and_keeps_generated_file(tmp_path: Path):
    template = tmp_path / "config.in"
    template.write_text(
        """
simulation_parameters
{
    Re=1000.0;
}
""".strip()
    )

    staged = stage_config(template, block_overrides=[("simulation_parameters", 0, {"Re": 250.0})])
    try:
        assert "runs_pybind" in str(staged)
        assert staged.exists()
        assert "Re=250.0;" in staged.read_text()
    finally:
        staged.unlink(missing_ok=True)
        staged.parent.rmdir()
