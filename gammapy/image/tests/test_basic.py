# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

from numpy.testing.utils import assert_allclose

from astropy import units as u
from ..basic import FermiLATBasicImageEstimator
from .. import SkyImage
from ...datasets import FermiLATDataset
from ...utils.testing import requires_dependency, requires_data

@requires_data('fermi-lat')
@requires_dependency('healpy')
@requires_dependency('scipy')
@requires_dependency('yaml')
class TestFermiLATBasicImageEstimator:
    def setup(self):
        kwargs = {}
        kwargs['reference'] = SkyImage.empty(nxpix=21, nypix=11, binsz=0.1)
        kwargs['emin'] = 10 * u.GeV
        kwargs['emax'] = 3000 * u.GeV
        self.estimator = FermiLATBasicImageEstimator(**kwargs)

        filename = '$FERMI_LAT_DATA/2fhl/fermi_2fhl_data_config.yaml'
        self.dataset = FermiLATDataset(filename)

    def test_run(self):
        result = self.estimator.run(self.dataset)
        desired = [155.0, 1294.44777903, 7.16714933731e+13, -1139.44777903,
                  -3.67226151181e-09, 1.]
        for image, value in zip(result, desired):
            assert_allclose(image.data.sum(), value)