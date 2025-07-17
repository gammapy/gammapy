# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.scripts.main import cli
from gammapy.utils.testing import run_cli


def test_cli_tutorial_info():
    result = run_cli(cli, ["tutorial", "setup", "--help"])
    assert "Usage" in result.output


def test_cli_tutorial_info_no_args():
    result = run_cli(cli, ["tutorial", "setup"])
    assert "System" in result.output
    assert "Gammapy package" in result.output
    assert "Other packages" in result.output
    assert "Gammapy environment variables" in result.output
