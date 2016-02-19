# Licensed under a 3-clause BSD style license - see LICENSE.rst
import filecmp
import os
from astropy.tests.helper import pytest

from gammapy.extern.pathlib import Path
from ...datasets import gammapy_extra
from ...utils.testing import requires_dependency, requires_data
from ..spectrum import cli


@requires_dependency('sherpa')
@requires_dependency('matplotlib')
@requires_data('gammapy-extra')
def test_spectrum(tmpdir):
    os.chdir(str(tmpdir))

    configfile = gammapy_extra.filename(
        'test_datasets/spectrum/spectrum_analysis_example.yaml')
    args = ['all', configfile]
    with pytest.raises(SystemExit) as exc:
        cli(args)

    sref = gammapy_extra.filename(
        'test_datasets/spectrum/total_spectrum_stats_reference.yaml')
    sres = 'total_spectrum_stats.yaml'

    fref = gammapy_extra.filename(
        'test_datasets/spectrum/fit_result_PowerLaw_reference.yaml')
    fres = 'fit_result_PowerLaw.yaml'

    actual = open(sres, 'r').read()
    desired = open(sref, 'r').read()
    assert actual == desired

    actual = open(fres, 'r').read()
    desired = open(fref, 'r').read()
    assert actual == desired


