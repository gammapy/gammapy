# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pytest
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

    @pytest.mark.xfail
    def test_run(self):
        if 'FERMI_DIFFUSE_DIR' in os.environ:
            which = 'all'
        else:
            which = ['counts', 'exposure', 'psf']

        results = self.estimator.run(
            self.dataset,
            which=which,
        )

        assert_allclose(results['counts'].data.sum(), 155, rtol=1e-3)
        assert_allclose(results['background'].data.sum(), 1294.44777903, rtol=1e-3)
        assert_allclose(results['exposure'].data.sum(), 71418159853641.08, rtol=1e-3)
        # Note: excess / flux is negative, because diffuse background
        # is overestimated for the Galactic center region used here.
        assert_allclose(results['excess'].data.sum(), -1139.44777903, rtol=1e-3)
        assert_allclose(results['flux'].data.sum(), -3.686393029491402e-09, rtol=1e-3)
        assert_allclose(results['psf'].data.sum(), 1, rtol=1e-3)


@requires_dependency('scipy')
@requires_dependency('regions')
@requires_data('gammapy-extra')
class TestIACTBasicImageEstimator:

    def setup(self):
        from regions import CircleSkyRegion

        reference = SkyImage.empty(
            nxpix=21, nypix=21, binsz=0.5,
            xref=83.633083, yref=22.0145,
            coordsys='CEL', proj='CAR',
        )

        center = SkyCoord(83.633083, 22.0145, frame='icrs', unit='deg')
        circle = CircleSkyRegion(center, radius=Angle(0.2, 'deg'))
        exclusion_mask = reference.region_mask(circle)
        exclusion_mask.data = ~exclusion_mask.data

        background_estimator = RingBackgroundEstimator(r_in=0.3 * u.deg, width=0.2 * u.deg)

        self.estimator = IACTBasicImageEstimator(
            emin=1 * u.TeV,
            emax=10 * u.TeV,
            reference=reference,
            exclusion_mask=exclusion_mask,
            background_estimator=background_estimator,
        )

        # setup data store and get list of observations
        data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
        self.observations = data_store.obs_list([23523, 23526, 23559, 23592])

    def test_run(self):
        images = self.estimator.run(self.observations)

        assert_allclose(images['counts'].data.sum(), 2222)
        assert_allclose(images['background'].data.sum(), 1885.083333333333, rtol=1e-3)
        assert_allclose(images['exposure'].data.sum(), 83036669325.30281, rtol=1e-3)
        assert_allclose(images['excess'].data.sum(), 336.91666666666674, rtol=1e-3)
        assert_allclose(images['flux'].data.sum(), 5.250525628586507e-07, rtol=1e-3)
        assert_allclose(images['psf'].data.sum(), 1, rtol=1e-3)

        # TODO: there should be asserts as well on lower-level maps, like the `n_off` map
        # Unfortunately it's not really possible, because the following line
        # background = self._background(counts, exposure, observation)['background']
        # is in `IACTBasicImageEstimator.run`, i.e. the lower-level maps are never stored
        # or exposed to the caller.
