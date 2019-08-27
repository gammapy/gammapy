# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from gammapy.astro.darkmatter import profiles
from gammapy.utils.testing import assert_quantity_allclose

dm_profiles = [
    profiles.NFWProfile,
    profiles.EinastoProfile,
    profiles.IsothermalProfile,
    profiles.BurkertProfile,
    profiles.MooreProfile,
]


@pytest.mark.parametrize("profile", dm_profiles)
def test_profiles(profile):
    p = profile()
    p.scale_to_local_density()
    actual = p(p.DISTANCE_GC)
    desired = p.LOCAL_DENSITY

    assert_quantity_allclose(actual, desired)
