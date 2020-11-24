# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.data import GTI
from gammapy.datasets import MapDatasetOnOff
from gammapy.estimators import ExcessProfileEstimator
from gammapy.maps import MapAxis, WcsGeom
from gammapy.utils.regions import (
    make_concentric_annulus_sky_regions,
    make_orthogonal_rectangle_sky_regions,
)


def get_simple_dataset_on_off():
    axis = MapAxis.from_energy_bounds(0.1, 10, 2, unit="TeV")
    geom = WcsGeom.create(npix=40, binsz=0.01, axes=[axis])
    dataset = MapDatasetOnOff.create(geom)
    dataset.mask_safe += True
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

    prof_maker = ExcessProfileEstimator(boxes, energy_edges=[0.1, 1, 10] * u.TeV)
    imp_prof = prof_maker.run(mapdataset_onoff)

    assert_allclose(imp_prof[7]["x_min"], 0.1462, atol=1e-4)
    assert_allclose(imp_prof[7]["x_ref"], 0.1575, atol=1e-4)
    assert_allclose(imp_prof[7]["counts"], [100.0, 100.0], atol=1e-2)
    assert_allclose(imp_prof[7]["excess"], [80.0, 80.0], atol=1e-2)
    assert_allclose(imp_prof[7]["sqrt_ts"], [7.6302447, 7.6302447], atol=1e-5)
    assert_allclose(imp_prof[7]["errn"], [-10.747017, -10.747017], atol=1e-5)
    assert_allclose(imp_prof[0]["ul"], [115.171871, 115.171871], atol=1e-5)
    assert_allclose(imp_prof[0]["flux"], [7.99999987e-06, 8.00000010e-06], atol=1e-3)


def test_radial_profile():
    dataset = get_simple_dataset_on_off()
    geom = dataset.counts.geom
    regions = make_concentric_annulus_sky_regions(
        center=geom.center_skydir, radius_max=0.2 * u.deg,
    )

    prof_maker = ExcessProfileEstimator(regions, energy_edges=[0.1, 1, 10] * u.TeV)
    imp_prof = prof_maker.run(dataset)

    assert_allclose(imp_prof[7]["x_min"], 0.14, atol=1e-4)
    assert_allclose(imp_prof[7]["x_ref"], 0.15, atol=1e-4)
    assert_allclose(imp_prof[7]["counts"], [980.0, 980.0], atol=1e-2)
    assert_allclose(imp_prof[7]["excess"], [784.0, 784.0], atol=1e-2)
    assert_allclose(imp_prof[7]["sqrt_ts"], [23.886444, 23.886444], atol=1e-5)
    assert_allclose(imp_prof[7]["errn"], [-34.075141, -34.075141], atol=1e-5)
    assert_allclose(imp_prof[0]["ul"], [75.834983, 75.834983], atol=1e-5)
    assert_allclose(imp_prof[0]["flux"], [7.99999987e-06, 8.00000010e-06], atol=1e-3)
    assert_allclose(imp_prof[0]["solid_angle"], [6.853891e-07, 6.853891e-07], atol=1e-5)


def test_radial_profile_one_interval():
    dataset = get_simple_dataset_on_off()
    geom = dataset.counts.geom
    regions = make_concentric_annulus_sky_regions(
        center=geom.center_skydir, radius_max=0.2 * u.deg,
    )

    prof_maker = ExcessProfileEstimator(regions)
    imp_prof = prof_maker.run(dataset)

    assert_allclose(imp_prof[7]["counts"], [1960], atol=1e-5)
    assert_allclose(imp_prof[7]["excess"], [1568.0], atol=1e-5)
    assert_allclose(imp_prof[7]["sqrt_ts"], [33.780533], atol=1e-5)
    assert_allclose(imp_prof[7]["errn"], [-48.278367], atol=1e-5)
    assert_allclose(imp_prof[0]["ul"], [134.285969], atol=1e-5)
    assert_allclose(imp_prof[0]["flux"], [16e-06], atol=1e-3)
    assert_allclose(imp_prof[0]["solid_angle"], [6.853891e-07], atol=1e-5)
