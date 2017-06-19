# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest
from astropy.io import fits
from astropy import units as u
from ...datasets import FermiGalacticCenter
from ...utils.testing import requires_data, requires_dependency
from ...image import SkyImage, block_reduce_hdu
from ..images import SkyCubeImages


@requires_dependency('scipy')
@requires_data("gammapy-extra")
class TestSkyCubeImages:
    def test_to_cube(self):
        sky_cube_original = FermiGalacticCenter.diffuse_model()
        image_list = sky_cube_original.to_images()
        sky_cube_restored = image_list.to_cube()

        assert_array_equal(image_list.images[0].data, sky_cube_original.data[0])

        assert_array_equal(sky_cube_restored.data, sky_cube_original.data)
        assert sky_cube_restored.name == sky_cube_original.name
        assert sky_cube_restored.wcs == sky_cube_original.wcs


@requires_dependency('skimage')
class TestBlockReduceHDU:
    def setup_class(self):
        # Arbitrarily choose CAR projection as independent from tests
        projection = 'CAR'

        # Create test image
        self.image = SkyImage.empty(nxpix=12, nypix=8, proj=projection)
        self.image.data = np.ones(self.image.data.shape)
        self.image_hdu = self.image.to_image_hdu()

        # Create test cube
        self.energy = [1, 3, 10, 30, 100] * u.TeV
        self.cube_images = [self.image for _ in self.energy]
        self.cube = SkyCubeImages(images=self.cube_images, wcs=self.image.wcs,
                                  energy=self.energy).to_cube()

    @pytest.mark.parametrize(('operation'), list([np.sum, np.mean]))
    def test_cube(self, operation):
        for energy in self.energy:
            image = self.cube.sky_image(energy)
            layer = image.data
            layer_hdu = fits.ImageHDU(data=layer, header=image.wcs.to_header())
            image_1 = block_reduce_hdu(layer_hdu, (2, 4), func=operation)
            if operation == np.sum:
                ref1 = [[8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8]]
            if operation == np.mean:
                ref1 = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
            assert_allclose(image_1.data, ref1)
