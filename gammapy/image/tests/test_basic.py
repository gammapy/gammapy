# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

from astropy import units as u
from ..basic import FermiLATBasicImageEstimator
from ...datasets import FermiLATDataset


def TestFermiLATBasicImageEstimator:

    def setup(self):
        kwargs = {}
        kwargs['reference'] = SkyImage.empty(nxpix=21, nypix=11)
        kwargs['emin'] = 10 * u.GeV
        kwargs['emax'] = 100 * u.GeV
        self.estimator = FermiLATBasicImageEstimator(**kwargs)
        self.dataset = FermiLATDataset('')

    def test_run(self):
        result = self.estimator.run(self.dataset)