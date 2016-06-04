from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose

from ..utils import cube_to_image
from ...image import images_to_cube, SkyMap


def test_cube_to_image():
    layer = SkyMap.empty(nxpix=101, nypix=101, fill=1.).to_image_hdu()
    hdu_list = [layer, layer, layer, layer]
    cube = images_to_cube(hdu_list)
    case1 = cube_to_image(cube)
    case2 = cube_to_image(cube, slicepos=1)
    # Check that layers are summed if no layer is specified (case1),
    # or only a specified layer is extracted (case2)
    assert_allclose(case1.data, 4 * layer.data)
    assert_allclose(case2.data, layer.data)
