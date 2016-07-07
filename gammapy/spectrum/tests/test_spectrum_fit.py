# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.tests.helper import pytest, assert_quantity_allclose
import astropy.units as u
from numpy.testing import assert_allclose 
from ...datasets import gammapy_extra
from ...spectrum.spectrum_extraction import SpectrumObservationList, SpectrumObservation
from ...spectrum.spectrum_fit import SpectrumFit
from ...utils.testing import requires_dependency, requires_data, SHERPA_LT_4_8
from astropy.utils.compat import NUMPY_LT_1_9
import logging

@pytest.mark.skipif('NUMPY_LT_1_9')
@pytest.mark.skipif('SHERPA_LT_4_8')
@requires_dependency('sherpa')
@requires_data('gammapy-extra')
def test_spectral_fit():

    logging.basicConfig(level=logging.INFO)

    pha1 = gammapy_extra.filename("datasets/hess-crab4_pha/pha_obs23592.fits")
    pha2 = gammapy_extra.filename("datasets/hess-crab4_pha/pha_obs23523.fits")
    obs1 = SpectrumObservation.read(pha1)
    obs2 = SpectrumObservation.read(pha2)
    obs_list = SpectrumObservationList([obs1, obs2])

    fit = SpectrumFit(obs_list)
    fit.run()

    assert fit.result[0].fit.spectral_model == 'PowerLaw'
    assert_allclose(fit.result[0].fit.statval, 105.398, rtol=1e-3)
    assert_quantity_allclose(fit.result[0].fit.parameters.index,
                             2.135 * u.Unit(''), rtol=1e-3)

    # Actual fit range can differ from threshold due to binning effects
    # We take the lowest bin that is completely within threshold
    # Sherpa quotes the lincenter of that bin as fitrange
    thres_bin = obs1.on_vector.energy.find_node(obs1.lo_threshold)
    desired = obs1.on_vector.energy.lin_center()[thres_bin + 1]
    actual = fit.result[0].fit.fit_range[0]
    assert_quantity_allclose(actual, desired)
    
    # Compare Npred vectors
    # TODO: Read model from FitResult
    from gammapy.spectrum.models import PowerLaw
    model = PowerLaw(index=fit.result[0].fit.parameters.index,
                     reference=fit.result[0].fit.parameters.reference,
                     amplitude=fit.result[0].fit.parameters.norm
                    )
    npred = obs1.predicted_counts(model)
    assert_allclose(fit.result[0].fit.n_pred, npred.data, rtol=1e-3)


    # Restrict fit range
    fit_range = [4, 20] * u.TeV
    fit.fit_range = fit_range 
    fit.run()

    range_bin = obs1.on_vector.energy.find_node(fit_range[1])
    desired = obs1.on_vector.energy.lin_center()[range_bin]
    actual = fit.result[0].fit.fit_range[1]
    assert_quantity_allclose(actual, desired)
    
    # Make sure fit range is not extended below threshold
    fit_range = [0.001, 10] * u.TeV
    fit.fit_range = fit_range 
    fit.run()
    desired = obs1.on_vector.energy.lin_center()[thres_bin + 1]
    actual = fit.result[0].fit.fit_range[0]
    assert_quantity_allclose(actual, desired)
