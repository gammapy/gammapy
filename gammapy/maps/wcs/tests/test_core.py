# Licensed under a 3-clause BSD style license - see LICENSE.rst

from numpy.testing import assert_allclose
from regions import PolygonSkyRegion
from gammapy.modeling.models import GaussianSpatialModel


def test_containment():

    model = GaussianSpatialModel(sigma="0.15 deg")
    geom = model._get_plot_map(None).geom.upsample(factor=3)
    model_map = model.integrate_geom(geom)

    regions = model_map.containment_region(fraction=0.393, apply_union=True)
    assert isinstance(regions, PolygonSkyRegion)  # because there is only one

    assert_allclose(
        regions.vertices.separation(geom.center_skydir), model.sigma.quantity, rtol=1e-2
    )

    radius = model_map.containment_radius(fraction=0.393)
    assert_allclose(radius, model.sigma.quantity, rtol=1e-2)
