# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
from numpy.testing import assert_allclose
from regions import CompoundSkyRegion, PolygonSkyRegion
from gammapy.maps import HpxGeom, HpxNDMap, containment_radius, containment_region
from gammapy.modeling.models import GaussianSpatialModel
from gammapy.utils.testing import requires_dependency


def test_containment():

    model = GaussianSpatialModel(sigma="0.15 deg")
    geom = model._get_plot_map(None).geom.upsample(factor=3)
    model_map = model.integrate_geom(geom)

    regions = containment_region(model_map, fraction=0.393, apply_union=True)
    assert isinstance(regions, PolygonSkyRegion)  # because there is only one

    assert_allclose(
        regions.vertices.separation(geom.center_skydir), model.sigma.quantity, rtol=1e-2
    )

    radius = containment_radius(model_map, fraction=0.393)
    assert_allclose(radius, model.sigma.quantity, rtol=1e-2)

    model2 = GaussianSpatialModel(lon_0="-0.5deg", sigma="0.15 deg")
    model_map2 = model_map + model2.integrate_geom(geom)
    regions = containment_region(model_map2, fraction=0.1, apply_union=True)
    assert isinstance(regions, CompoundSkyRegion)

    regions = containment_region(model_map2, fraction=0.1, apply_union=False)
    assert isinstance(regions, list)


@requires_dependency("healpy")
def test_containment_fail_hpx():

    geom_hpx = HpxGeom.create(binsz=10, frame="galactic")
    m = HpxNDMap(geom_hpx)

    with pytest.raises(TypeError):
        containment_region(m, fraction=0.393, apply_union=True)

    with pytest.raises(TypeError):
        containment_radius(m, fraction=0.393)
