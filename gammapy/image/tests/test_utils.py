# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from astropy.tests.helper import pytest
from astropy.io import fits
from astropy.wcs import WCS
from astropy.units import Quantity
from ..utils import _shape_2N
from ...utils.testing import requires_dependency, requires_data
from ...datasets import FermiGalacticCenter
from ...data import DataStore
from ...utils.energy import EnergyBounds
from ...cube import SkyCube
from ...image import (
    binary_disk,
    binary_ring,
    make_header,
    images_to_cube,
    block_reduce_hdu,
    wcs_histogram2d,
    lon_lat_rectangle_mask,
    SkyMap,
    SkyImageList)


def test_binary_disk():
    actual = binary_disk(1)
    desired = np.array([[False, True, False],
                        [True, True, True],
                        [False, True, False]])
    assert_equal(actual, desired)


def test_binary_ring():
    actual = binary_ring(1, 2)
    desired = np.array([[False, False, True, False, False],
                        [False, True, True, True, False],
                        [True, True, False, True, True],
                        [False, True, True, True, False],
                        [False, False, True, False, False]])
    assert_equal(actual, desired)


@pytest.mark.xfail
def test_process_image_pixels():
    """Check the example how to implement convolution given in the docstring"""
    from astropy.convolution import convolve as astropy_convolve

    def convolve(image, kernel):
        """Convolve image with kernel"""
        from ..utils import process_image_pixels
        images = dict(image=np.asanyarray(image))
        kernel = np.asanyarray(kernel)
        out = dict(image=np.empty_like(image))

        def convolve_function(images, kernel):
            value = np.sum(images['image'] * kernel)
            return dict(image=value)

        process_image_pixels(images, kernel, out, convolve_function)
        return out['image']

    random_state = np.testing.RandomState(seed=0)

    image = random_state.uniform(size=(7, 10))
    kernel = random_state.uniform(size=(3, 5))
    actual = convolve(image, kernel)
    desired = astropy_convolve(image, kernel, boundary='fill')
    assert_allclose(actual, desired)


@requires_dependency('skimage')
class TestBlockReduceHDU():
    def setup_class(self):
        # Arbitrarily choose CAR projection as independent from tests
        projection = 'CAR'

        # Create test image
        self.skymap = SkyMap.empty(nxpix=12, nypix=8, proj=projection)
        self.skymap.data = np.ones(self.skymap.data.shape)
        self.image = self.skymap.to_image_hdu()

        # Create test cube
        self.indices = np.arange(4)
        self.cube_skymaps = [self.skymap for _ in self.indices]
        self.cube = SkyImageList(skymaps=self.cube_skymaps, wcs=self.skymap.wcs).to_cube()

    @pytest.mark.parametrize(('operation'), list([np.sum, np.mean]))
    def test_image(self, operation):
        image_1 = block_reduce_hdu(self.image, (2, 4), func=operation)
        if operation == np.sum:
            ref1 = [[8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8]]
        if operation == np.mean:
            ref1 = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
        assert_allclose(image_1.data, ref1)

    @pytest.mark.parametrize(('operation'), list([np.sum, np.mean]))
    def test_cube(self, operation):
        for index in self.indices:
            image = self.cube.sky_image(index)
            layer = self.cube.data[index]
            layer_hdu = fits.ImageHDU(data=layer, header=image.wcs.to_header())
            image_1 = block_reduce_hdu(layer_hdu, (2, 4), func=operation)
            if operation == np.sum:
                ref1 = [[8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8]]
            if operation == np.mean:
                ref1 = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
            assert_allclose(image_1.data, ref1)


@requires_dependency('skimage')
def test_ref_pixel():
    image = SkyMap.empty(nxpix=101, nypix=101, proj='CAR')
    footprint = image.wcs.calc_footprint(center=False)
    image_1 = block_reduce_hdu(image.to_image_hdu(), (10, 10), func=np.sum)
    footprint_1 = WCS(image_1.header).calc_footprint(center=False)
    # Lower left corner shouldn't change
    assert_allclose(footprint[0], footprint_1[0])


def test_shape_2N():
    shape = (34, 89, 120, 444)
    expected_shape = (40, 96, 128, 448)
    assert expected_shape == _shape_2N(shape=shape, N=3)


def test_wcs_histogram2d():
    # A simple test case that can by checked by hand:
    header = make_header(nxpix=2, nypix=1, binsz=10, xref=0, yref=0, proj='CAR')
    # GLON pixel edges: (+10, 0, -10)
    # GLAT pixel edges: (-5, +5)

    EPS = 0.1
    data = [
        (5, 5, 1),  # in image[0, 0]
        (0, 0 + EPS, 2),  # in image[1, 0]
        (5, -5 + EPS, 3),  # in image[0, 0]
        (5, 5 + EPS, 99),  # outside image
        (10 + EPS, 0, 99),  # outside image
    ]
    lon, lat, weights = np.array(data).T
    image = wcs_histogram2d(header, lon, lat, weights)

    print(type(image))

    assert image.data[0, 0] == 1 + 3
    assert image.data[0, 1] == 2


@requires_data('gammapy-extra')
def test_lon_lat_rectangle_mask():
    counts = SkyMap.from_image_hdu(FermiGalacticCenter.counts())
    coordinates = counts.coordinates()
    lons = coordinates.data.lon.wrap_at('180d')
    lats = coordinates.data.lat
    mask = lon_lat_rectangle_mask(lons.degree, lats.degree, lon_min=-1,
                                  lon_max=1, lat_min=-1, lat_max=1)
    assert_allclose(mask.sum(), 400)

    mask = lon_lat_rectangle_mask(lons.degree, lats.degree, lon_min=None,
                                  lon_max=None, lat_min=None,
                                  lat_max=None)
    assert_allclose(mask.sum(), 80601)


@requires_data('gammapy-extra')
def test_bin_events_in_cube():
    dirname = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2'
    data_store = DataStore.from_dir(dirname)

    events = data_store.obs(obs_id=23523).events

    counts = SkyCube.empty(emin=0.5, emax=80, enbins=8, eunit='TeV',
                           nxpix=200, nypix=200, xref=events.meta['RA_OBJ'],
                           yref=events.meta['DEC_OBJ'], dtype='int',
                           coordsys='CEL')
    counts.fill(events)
    assert counts.data.sum().value == 1233


