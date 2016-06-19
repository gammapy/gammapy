from __future__ import absolute_import, division, print_function, unicode_literals
from gammapy.datasets import FermiGalacticCenter
import numpy as np


def test_to_cube():
    sky_cube_original = FermiGalacticCenter.diffuse_model()
    image_list = sky_cube_original.to_image_list()
    assert np.array_equal(image_list.skymaps[0].data, sky_cube_original.data[0])
    sky_cube_restored = image_list.to_cube()

    for key in set(sky_cube_original.__dict__.keys() + sky_cube_restored.__dict__.keys()):
        # the __init__ of SkyCube initializes energy_axis from energy, but the original cube may not have this field.
        # Thus the deep equality is not possible and we have to go through it's fields
        if key == 'energy_axis':
            continue

        if isinstance(sky_cube_original.__dict__[key], np.ndarray):
            assert np.array_equal(sky_cube_original.__dict__[key], sky_cube_restored.__dict__[key]),\
                "differs on {} ".format(key)
        else:
            assert sky_cube_original.__dict__[key] == sky_cube_restored.__dict__[key], "differs on {} ".format(key)