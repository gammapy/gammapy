# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.coordinates import Angle
from ...utils.testing import requires_dependency
from ..core import SkyImage
from ..radial_profile import radial_profile, radial_profile_label_image


@requires_dependency('scipy')
def test_radial_profile():
    image = SkyImage.empty()
    image.data.fill(1)
    center = image.center
    radius = Angle([0.1, 0.2, 0.4, 0.5, 1.0], 'deg')

    labels = radial_profile_label_image(image, center, radius)
    assert labels.data.max() == 5

    profile = radial_profile(image, center, radius)
    assert len(profile) == 4
    assert_allclose(profile['MEAN'], 1)
