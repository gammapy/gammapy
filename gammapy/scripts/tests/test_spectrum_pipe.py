# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.tests.helper import assert_quantity_allclose
from ...spectrum.tests.test_extract import obs, bkg_estimator, extraction
from ...spectrum.tests.test_fit import fit 
from ...utils.energy import EnergyBounds
from .. import SpectrumAnalysisIACT


def test_spectrum_analysis_iact(tmpdir):
    ana = SpectrumAnalysisIACT(outdir=tmpdir,
                               observations=obs(),
                               background_estimator=bkg_estimator(),
                               extraction = extraction(),
                               fit = fit(),
                               fp_binning = EnergyBounds.equal_log_spacing(
                                   1, 50, 4, 'TeV')
                              )

    assert 'IACT' in str(ana)
    ana.run()
    actual = ana.flux_point_estimator.flux_points.table[0]['dnde']
    desired = 8.074225976187008e-08

    assert_quantity_allclose(actual, desired)

