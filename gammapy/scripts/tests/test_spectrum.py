# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from astropy.tests.helper import pytest
from astropy.coordinates import SkyCoord, Angle
from ...extern.regions import CircleSkyRegion
from ...datasets import gammapy_extra
from ...data import ObservationList
from ...image import ExclusionMask
from ...utils.testing import data_manager, requires_dependency, requires_data, run_cli
from ...spectrum.results import SpectrumFitResult
from ...spectrum import SpectrumExtraction
from ..spectrum import cli


@pytest.mark.xfail(reason="Command line tool broken")
@requires_dependency('sherpa')
@requires_dependency('matplotlib')
@requires_data('gammapy-extra')
def test_spectrum_cmd(tmpdir):
    # FIXME
    os.chdir(str(tmpdir))

    configfile = gammapy_extra.filename(
        'test_datasets/spectrum/spectrum_analysis_example.yaml')
    args = ['all', configfile]
    run_cli(cli, args)

    sref = gammapy_extra.filename(
        'test_datasets/spectrum/total_spectrum_stats_reference.yaml')
    sres = 'total_spectrum_stats.yaml'

    fref = gammapy_extra.filename(
        'test_datasets/spectrum/fit_result_PowerLaw_reference.yaml')
    fres = 'fit_result_PowerLaw.yaml'

    #actual = SpectrumStats.from_yaml(sres)
    #desired = SpectrumStats.from_yaml(sref)

    #assert str(actual.to_table(format='.3g')) == str(desired.to_table(format='.3g'))

    actual = SpectrumFitResult.from_yaml(fres)
    desired = SpectrumFitResult.from_yaml(fref)

    print('TEMPDIR', tmpdir)
    assert str(actual.to_table(format='.3g')) == str(desired.to_table(format='.3g'))


@requires_dependency('sherpa')
@requires_dependency('matplotlib')
@requires_data('gammapy-extra')
def test_spectrum(tmpdir, data_manager):
    # Minimal version executing all steps
    # This could go into a script accessible to the user and/or an example
    store = data_manager['hess-crab4-hd-hap-prod2']
    obs_id = [23523, 23592]
    obs = ObservationList([store.obs(_) for _ in obs_id])

    center = SkyCoord(83.63, 22.01, unit='deg', frame='icrs')
    radius = Angle('0.3 deg')
    on_region = CircleSkyRegion(center, radius)

    exclusion = ExclusionMask.read('$GAMMAPY_EXTRA/datasets/exclusion_masks/tevcat_exclusion.fits')
    bkg_method = dict(method='reflected', exclusion=exclusion)

    extraction = SpectrumExtraction(target=on_region, obs=obs, background=bkg_method)

    extraction.run(outdir=tmpdir)
