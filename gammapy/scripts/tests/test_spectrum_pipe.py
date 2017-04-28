# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import assert_quantity_allclose
import astropy.units as u
from ...utils.energy import EnergyBounds
from ...utils.testing import requires_dependency, requires_data
from ...background.tests.test_reflected import obs_list, on_region
from ...spectrum.models import PowerLaw
from .. import SpectrumAnalysisIACT


def get_config():
    """Get test config, extend to several scenarios"""
    model = PowerLaw(
        index=2,
        amplitude=1e-11 * u.Unit('cm-2 s-1 TeV-1'),
        reference=1 * u.TeV,
    )
    fp_binning = EnergyBounds.equal_log_spacing(1, 50, 4, 'TeV')
    return dict(
        outdir=None,
        background=dict(on_region=on_region()),
        extraction=dict(),
        fit=dict(model=model),
        fp_binning=fp_binning,
    )


@requires_data('gammapy-extra')
@requires_dependency('scipy')
@requires_dependency('sherpa')
def test_spectrum_analysis_iact(tmpdir):
    conf = get_config()
    conf['outdir'] = tmpdir

    ana = SpectrumAnalysisIACT(
        observations=obs_list(),
        config=conf
    )

    assert 'IACT' in str(ana)
    ana.run()
    actual = ana.flux_point_estimator.flux_points.table[0]['dnde']
    desired = 7.208219787928114e-08
    assert_quantity_allclose(actual, desired)
