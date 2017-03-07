# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from collections import OrderedDict

from numpy.testing.utils import assert_allclose

from astropy import units as u
from astropy.coordinates import SkyCoord, Angle

from .. import SkyImage, FermiLATBasicImageEstimator, IACTBasicImageEstimator
from ...background import RingBackgroundEstimator
from ...data import DataStore
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


@requires_dependency('scipy')
@requires_dependency('regions')
@requires_data('gammapy-extra')
class TestIACTBasicImageEstimator:

    def setup(self):
        from regions import CircleSkyRegion
        kwargs = {}

        # Define energy range
        kwargs['emin'] = 1 * u.TeV
        kwargs['emax'] = 10 * u.TeV

        # Define reference image
        wcs_spec = {'nxpix': 21,
                    'nypix': 21,
                    'xref': 83.633083,
                    'yref': 22.0145,
                    'coordsys': 'CEL',
                    'proj': 'CAR',
                    'binsz': 0.5}
        kwargs['reference'] = SkyImage.empty(**wcs_spec)

        # Define background estimator
        r_in = 0.3 * u.deg
        width = 0.2 * u.deg
        kwargs['background_estimator'] = RingBackgroundEstimator(r_in=r_in, width=width)

        # Defien exclusion mask
        center = SkyCoord(83.633083, 22.0145, frame='icrs', unit='deg')
        circle = CircleSkyRegion(center, radius=Angle(0.2, 'deg'))
        kwargs['exclusion_mask'] = kwargs['reference'].region_mask(circle)
        kwargs['exclusion_mask'].data = ~kwargs['exclusion_mask'].data

        self.estimator = IACTBasicImageEstimator(**kwargs)

        # setup data store and get list of observations
        data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
        self.observations = data_store.obs_list([23523, 23526, 23559, 23592])

    def test_run(self):
        images = OrderedDict()
        images['counts'] = dict(sum=2620.0)
        images['background'] = dict(sum=1994.79254434631)
        images['exposure'] = dict(sum=83036669325.30281)
        images['excess'] = dict(sum=625.2074556536902)
        images['flux'] = dict(sum=2.524971454563909e-07)
        images['psf'] = dict(sum=1.)
        results = self.estimator.run(self.observations)

        for name in results.names:
            assert_allclose(results[name].data.sum(), images[name]['sum'], rtol=1e-3)
