# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from gammapy.scripts.main import cli
from gammapy.utils.testing import requires_dependency, run_cli


def test_cli_tutorial_info():
    result = run_cli(cli, ["tutorial", "setup", "--help"])
    assert "Usage" in result.output


@requires_dependency("requests")
@requires_dependency("tqdm")
@pytest.mark.remote_data
def test_cli_tutorial_info_no_args(tmp_path):
    result = run_cli(cli, ["tutorial", "setup"])
    assert "System" in result.output
    assert "Gammapy package" in result.output
    assert "Other packages" in result.output
    assert "Gammapy environment variables" in result.output
