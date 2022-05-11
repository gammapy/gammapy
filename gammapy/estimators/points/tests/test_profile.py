# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from gammapy.data import GTI
from gammapy.datasets import MapDatasetOnOff
from gammapy.estimators import FluxPoints, FluxProfileEstimator
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.utils.regions import (
    make_concentric_annulus_sky_regions,
    make_orthogonal_rectangle_sky_regions,
)


def get_simple_dataset_on_off():
    axis = MapAxis.from_energy_bounds(0.1, 10, 2, unit="TeV")
    geom = WcsGeom.create(npix=40, binsz=0.01, axes=[axis])
    dataset = MapDatasetOnOff.create(geom, name="test-on-off")
    dataset.mask_safe += True
    dataset.counts += 5
    dataset.counts_off += 1
    dataset.acceptance += 1
    dataset.acceptance_off += 1
    dataset.exposure += 1000 * u.m**2 * u.s
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

    prof_maker = FluxProfileEstimator(
        regions=boxes,
        energy_edges=[0.1, 1, 10] * u.TeV,
        selection_optional="all",
        n_sigma=1,
        n_sigma_ul=3,
    )
    result = prof_maker.run(mapdataset_onoff)

    imp_prof = result.to_table(sed_type="flux", format="profile")
    assert_allclose(imp_prof[7]["x_min"], 0.1462, atol=1e-4)
    assert_allclose(imp_prof[7]["x_ref"], 0.1575, atol=1e-4)
    assert_allclose(imp_prof[7]["counts"], [[100.0], [100.0]], atol=1e-2)
    assert_allclose(imp_prof[7]["sqrt_ts"], [7.63, 7.63], atol=1e-2)
    assert_allclose(imp_prof[0]["flux"], [8e-06, 8.0e-06], atol=1e-3)

    # TODO: npred quantities are not supported by the table serialisation format
    #  so we rely on the FluxPoints object
    npred_excess = result.npred_excess.data[7].squeeze()
    assert_allclose(npred_excess, [80.0, 80.0], rtol=1e-3)

    errn = result.npred_excess_errn.data[7].squeeze()
    assert_allclose(errn, [10.75, 10.75], atol=1e-2)

    ul = result.npred_excess_ul.data[7].squeeze()
    assert_allclose(ul, [111.32, 111.32], atol=1e-2)


def test_radial_profile():
    dataset = get_simple_dataset_on_off()
    geom = dataset.counts.geom
    regions = make_concentric_annulus_sky_regions(
        center=geom.center_skydir,
        radius_max=0.2 * u.deg,
    )

    prof_maker = FluxProfileEstimator(
        regions,
        energy_edges=[0.1, 1, 10] * u.TeV,
        selection_optional="all",
        n_sigma_ul=3,
    )
    result = prof_maker.run(dataset)

    imp_prof = result.to_table(sed_type="flux", format="profile")

    assert_allclose(imp_prof[7]["x_min"], 0.14, atol=1e-4)
    assert_allclose(imp_prof[7]["x_ref"], 0.15, atol=1e-4)
    assert_allclose(imp_prof[7]["counts"], [[980.0], [980.0]], atol=1e-2)
    assert_allclose(imp_prof[7]["sqrt_ts"], [23.886444, 23.886444], atol=1e-5)
    assert_allclose(imp_prof[0]["flux"], [8e-06, 8.0e-06], atol=1e-3)

    # TODO: npred quantities are not supported by the table serialisation format
    #  so we rely on the FluxPoints object
    npred_excess = result.npred_excess.data[7].squeeze()
    assert_allclose(npred_excess, [784.0, 784.0], rtol=1e-3)

    errn = result.npred_excess_errn.data[7].squeeze()
    assert_allclose(errn, [34.075, 34.075], rtol=2e-3)

    ul = result.npred_excess_ul.data[0].squeeze()
    assert_allclose(ul, [72.074, 72.074], rtol=1e-3)


def test_radial_profile_one_interval():
    dataset = get_simple_dataset_on_off()
    geom = dataset.counts.geom
    regions = make_concentric_annulus_sky_regions(
        center=geom.center_skydir,
        radius_max=0.2 * u.deg,
    )

    prof_maker = FluxProfileEstimator(
        regions,
        selection_optional="all",
        energy_edges=[0.1, 10] * u.TeV,
        n_sigma_ul=3,
        sum_over_energy_groups=True,
    )
    result = prof_maker.run(dataset)

    imp_prof = result.to_table(sed_type="flux", format="profile")

    assert_allclose(imp_prof[7]["counts"], [[1960]], atol=1e-5)
    assert_allclose(imp_prof[7]["npred_excess"], [[1568.0]], rtol=1e-3)
    assert_allclose(imp_prof[7]["sqrt_ts"], [33.780533], rtol=1e-3)
    assert_allclose(imp_prof[0]["flux"], [16e-06], atol=1e-3)

    axis = result.counts.geom.axes["dataset"]
    assert axis.center == ["test-on-off"]

    errn = result.npred_excess_errn.data[7].squeeze()
    assert_allclose(errn, [48.278367], rtol=2e-3)

    ul = result.npred_excess_ul.data[0].squeeze()
    assert_allclose(ul, [130.394824], rtol=1e-3)


def test_serialisation(tmpdir):
    dataset = get_simple_dataset_on_off()
    geom = dataset.counts.geom
    regions = make_concentric_annulus_sky_regions(
        center=geom.center_skydir,
        radius_max=0.2 * u.deg,
    )

    est = FluxProfileEstimator(regions, energy_edges=[0.1, 10] * u.TeV)
    result = est.run(dataset)

    result.write(tmpdir / "profile.fits", format="profile")

    profile = FluxPoints.read(
        tmpdir / "profile.fits",
        format="profile",
        reference_model=PowerLawSpectralModel(),
    )

    assert_allclose(result.norm, profile.norm, rtol=1e-4)
    assert_allclose(result.norm_err, profile.norm_err, rtol=1e-4)
    assert_allclose(result.npred, profile.npred)
    assert_allclose(result.ts, profile.ts)

    assert np.all(result.is_ul == profile.is_ul)


def test_regions_init():
    with pytest.raises(ValueError):
        FluxProfileEstimator(regions=[])

    region = CircleSkyRegion(center=SkyCoord("0d", "0d"), radius=0.1 * u.deg)

    with pytest.raises(ValueError):
        FluxProfileEstimator(regions=[region])
