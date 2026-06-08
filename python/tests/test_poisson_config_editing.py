from pathlib import Path

from iblgf import ns_amr_2d
from iblgf import poisson


def test_prepare_config_rewrites_vortex_radius(tmp_path: Path):
    template = tmp_path / "config.in"
    template.write_text(
        """
simulation_parameters
{
    output
    {
        directory=vortexRings;
    }

    vortex
    {
        R=0.125;
        c1=10000.0;
        center=(0,0,0);
    }
}
""".strip()
    )

    output = tmp_path / "generated.cfg"
    generated = poisson.prepare_config(
        template,
        output_path=output,
        vortex_overrides=[{"R": 0.2, "center": (0.0, 0.0, 0.05)}],
    )

    text = generated.read_text()
    assert "R=0.2;" in text
    assert "center=(0.0,0.0,0.05);" in text


def test_prepare_config_rewrites_simulation_parameters(tmp_path: Path):
    template = tmp_path / "config.in"
    template.write_text(
        """
simulation_parameters
{
    EXP_LInf=1e-2;
}
""".strip()
    )

    generated = poisson.prepare_config(
        template,
        output_path=tmp_path / "generated.cfg",
        simulation_overrides={"EXP_LInf": 0.25},
    )

    assert "EXP_LInf=0.25;" in generated.read_text()


def test_ns_amr_2d_prepare_config_rewrites_simulation_parameters(tmp_path: Path):
    template = tmp_path / "config.in"
    template.write_text(
        """
simulation_parameters
{
    Re=1000.0;
    R=0.5;
}
""".strip()
    )

    generated = ns_amr_2d.prepare_config(
        template,
        output_path=tmp_path / "generated.cfg",
        simulation_overrides={"Re": 250.0, "R": 0.2},
    )

    text = generated.read_text()
    assert "Re=250.0;" in text
    assert "R=0.2;" in text
