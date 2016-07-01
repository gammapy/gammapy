# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from gammapy.utils.testing import requires_data
from gammapy.datasets import FermiGalacticCenter
from numpy.testing import assert_array_equal


@requires_data("gammapy-extra")
def test_to_cube():
    sky_cube_original = FermiGalacticCenter.diffuse_model()
    image_list = sky_cube_original.to_image_list()
    sky_cube_restored = image_list.to_cube()

    assert_array_equal(image_list.skymaps[0].data, sky_cube_original.data[0])

    assert_array_equal(sky_cube_restored.data, sky_cube_original.data)
    assert_array_equal(sky_cube_restored.energy, sky_cube_original.energy)
    assert sky_cube_restored.name == sky_cube_original.name
    assert sky_cube_restored.wcs == sky_cube_original.wcs
