# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.datasets import MapDatasetOnOff
from gammapy.data import GTI
from gammapy.maps import MapAxis, WcsGeom
from gammapy.estimators import ExcessProfileEstimator
from gammapy.utils.regions import (
    make_orthogonal_rectangle_sky_regions,
    make_concentric_annulus_sky_regions,
)


def get_simple_dataset_on_off():
    axis = MapAxis.from_energy_bounds(0.1, 10, 2, unit="TeV")
    geom = WcsGeom.create(npix=40, binsz=0.01, axes=[axis])
    dataset = MapDatasetOnOff.create(geom)
    dataset.mask_safe += 1
    dataset.counts += 5
    dataset.counts_off += 1
    dataset.acceptance += 1
    dataset.acceptance_off += 1
    dataset.exposure += 1000 * u.m ** 2 * u.s
    dataset.gti = GTI.create([0 * u.s], [5 * u.h], reference_time="2010-01-01T00:00:00")
    return dataset


def make_boxes(wcs):
    start_line = SkyCoord(0.08, -0.16, unit="deg", frame="icrs")
    end_line = SkyCoord(359.9, 0.16, unit="deg", frame="icrs")
    return make_orthogonal_rectangle_sky_regions(
        start_line, end_line, wcs, 0.1 * u.deg, 8
    )


def make_horizontal_boxes(wcs):
    start_line = SkyCoord(0.08, 0.1, unit="deg", frame="icrs")
    end_line = SkyCoord(359.9, 0.1, unit="deg", frame="icrs")
    return make_orthogonal_rectangle_sky_regions(
        start_line, end_line, wcs, 0.1 * u.deg, 8
    )


def test_profile_content():
    mapdataset_onoff = get_simple_dataset_on_off()
    wcs = mapdataset_onoff.counts.geom.wcs
    boxes = make_horizontal_boxes(wcs)

    prof_maker = ExcessProfileEstimator(boxes)
    imp_prof = prof_maker.run(mapdataset_onoff)

    assert_allclose(imp_prof[7]["x_min"], 0.1462, atol=1e-4)
    assert_allclose(imp_prof[7]["x_ref"], 0.1575, atol=1e-4)
    assert_allclose(imp_prof[7]["counts"], [100.0, 100.0], atol=1e-2)
    assert_allclose(imp_prof[7]["excess"], [10.0, 10.0], atol=1e-2)
    assert_allclose(imp_prof[7]["sqrt_ts"], [1.41630196, 1.41630196], atol=1e-5)
    assert_allclose(imp_prof[7]["errn"], [-5.26044915, -5.26044915], atol=1e-5)
    assert_allclose(imp_prof[0]["ul"], [93.7604547, 93.7604547], atol=1e-5)
    assert_allclose(imp_prof[0]["flux"], [9.99999985e-07, 9.99999965e-07], atol=1e-3)
    assert_allclose(imp_prof[0]["solid_angle"], [6.853891e-07, 6.853891e-07], atol=1e-5)


def test_radial_profile():
    dataset = get_simple_dataset_on_off()
    geom = dataset.counts.geom
    regions = make_concentric_annulus_sky_regions(
        center=geom.center_skydir, radius_max=0.2 * u.deg,
    )

    prof_maker = ExcessProfileEstimator(regions)
    imp_prof = prof_maker.run(dataset)

    assert_allclose(imp_prof[7]["x_min"], 0.14, atol=1e-4)
    assert_allclose(imp_prof[7]["x_ref"], 0.15, atol=1e-4)
    assert_allclose(imp_prof[7]["counts"], [980.0, 980.0], atol=1e-2)
    assert_allclose(imp_prof[7]["excess"], [98.0, 98.0], atol=1e-2)
    assert_allclose(imp_prof[7]["sqrt_ts"], [4.43372643, 4.43372643], atol=1e-5)
    assert_allclose(imp_prof[7]["errn"], [-5.57970821, -5.57970821], atol=1e-5)
    assert_allclose(imp_prof[0]["ul"], [63.38152373, 63.38152373], atol=1e-5)
    assert_allclose(imp_prof[0]["flux"], [5.99999991e-07, 5.99999979e-07], atol=1e-3)
    assert_allclose(imp_prof[0]["solid_angle"], [6.853891e-07, 6.853891e-07], atol=1e-5)
