# Licensed under a 3-clause BSD style license - see LICENSE.rst
import filecmp
import os
from astropy.tests.helper import pytest

from gammapy.extern.pathlib import Path
from ...datasets import gammapy_extra
from ...utils.testing import requires_dependency, requires_data
from ..spectrum import cli


@requires_dependency('sherpa')
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

    assert filecmp.cmp(sref, sres)
    assert filecmp.cmp(fref, fres)

    #test display
    args = ['display']
    with pytest.raises(SystemExit) as exc:
        cli(args)

    # Todo: Implement this
    #args = ['plot']
    #with pytest.raises(SystemExit) as exc:
    #    cli(args)


