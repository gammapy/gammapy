# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from collections import OrderedDict
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

        filename = '$GAMMAPY_FERMI_LAT_DATA/2fhl/fermi_2fhl_data_config.yaml'
        self.dataset = FermiLATDataset(filename)

    def test_run(self):
        images = OrderedDict()
        images['counts'] = dict(sum=155.0)
        images['background'] = dict(sum=1294.44777903)
        images['exposure'] = dict(sum=71671503335592.98)
        # Note: excess / flux is negative, because diffuse background
        # is overestimated for the Galactic center region used here.
        images['excess'] = dict(sum=-1139.44777903)
        images['flux'] = dict(sum=-3.67226151181e-09)
        images['psf'] = dict(sum=1)

        if 'FERMI_DIFFUSE_DIR' in os.environ:
            names = list(images.keys())
        else:
            names = ['counts', 'exposure', 'psf']

        results = self.estimator.run(
            self.dataset,
            which=names,
        )

        for name in names:
            assert_allclose(results[name].data.sum(), images[name]['sum'], rtol=1e-3)
