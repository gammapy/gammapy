# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from gammapy.datasets.evaluator import MapEvaluator
from gammapy.irf import PSFKernel
from gammapy.maps import Map, MapAxis, RegionGeom, RegionNDMap, WcsGeom
from gammapy.modeling.models import (
    ConstantSpectralModel,
    GaussianSpatialModel,
    Models,
    PointSpatialModel,
    SkyModel,
)
from gammapy.utils.gauss import Gauss2DPDF


def test_compute_flux_spatial():
    center = SkyCoord("0 deg", "0 deg", frame="galactic")
    region = CircleSkyRegion(center=center, radius=0.1 * u.deg)

    nbin = 2
    energy_axis_true = MapAxis.from_energy_bounds(
        ".1 TeV", "10 TeV", nbin=nbin, name="energy_true"
    )

    spectral_model = ConstantSpectralModel()
    spatial_model = PointSpatialModel(
        lon_0=0 * u.deg, lat_0=0 * u.deg, frame="galactic"
    )

    models = SkyModel(spectral_model=spectral_model, spatial_model=spatial_model)
    model = Models(models)

    exposure_region = RegionNDMap.create(
        region, axes=[energy_axis_true], binsz_wcs="0.01deg"
    )
    exposure_region.data += 1.0
    exposure_region.unit = "m2 s"

    geom = RegionGeom(region, axes=[energy_axis_true], binsz_wcs="0.01deg")
    psf = PSFKernel.from_gauss(geom.to_wcs_geom(), sigma="0.1 deg")

    evaluator = MapEvaluator(model=model[0], exposure=exposure_region, psf=psf)
    flux = evaluator.compute_flux_spatial()

    g = Gauss2DPDF(0.1)
    reference = g.containment_fraction(0.1)
    assert_allclose(flux.value, reference, rtol=0.003)


def test_compute_flux_spatial_no_psf():
    # check that spatial integration is not performed in the absence of a psf
    center = SkyCoord("0 deg", "0 deg", frame="galactic")
    region = CircleSkyRegion(center=center, radius=0.1 * u.deg)

    nbin = 2
    energy_axis_true = MapAxis.from_energy_bounds(
        ".1 TeV", "10 TeV", nbin=nbin, name="energy_true"
    )

    spectral_model = ConstantSpectralModel()
    spatial_model = GaussianSpatialModel(
        lon_0=0 * u.deg, lat_0=0 * u.deg, frame="galactic", sigma="0.1 deg"
    )

    models = SkyModel(spectral_model=spectral_model, spatial_model=spatial_model)
    model = Models(models)

    exposure_region = RegionNDMap.create(region, axes=[energy_axis_true])
    exposure_region.data += 1.0
    exposure_region.unit = "m2 s"

    evaluator = MapEvaluator(model=model[0], exposure=exposure_region)
    flux = evaluator.compute_flux_spatial()

    assert_allclose(flux, 1.0)


def test_large_oversampling():
    nbin = 2
    energy_axis_true = MapAxis.from_energy_bounds(
        ".1 TeV", "10 TeV", nbin=nbin, name="energy_true"
    )
    geom = WcsGeom.create(width=1, binsz=0.02, axes=[energy_axis_true])

    spectral_model = ConstantSpectralModel()
    spatial_model = GaussianSpatialModel(
        lon_0=0 * u.deg, lat_0=0 * u.deg, sigma=1e-4 * u.deg, frame="icrs"
    )

    models = SkyModel(spectral_model=spectral_model, spatial_model=spatial_model)
    model = Models(models)

    exposure = Map.from_geom(geom, unit="m2 s")
    exposure.data += 1.0

    psf = PSFKernel.from_gauss(geom, sigma="0.1 deg")

    evaluator = MapEvaluator(model=model[0], exposure=exposure, psf=psf)
    flux_1 = evaluator.compute_flux_spatial()

    spatial_model.sigma.value = 0.001
    flux_2 = evaluator.compute_flux_spatial()

    spatial_model.sigma.value = 0.01
    flux_3 = evaluator.compute_flux_spatial()

    spatial_model.sigma.value = 0.03
    flux_4 = evaluator.compute_flux_spatial()

    assert_allclose(flux_1.data.sum(), nbin, rtol=1e-4)
    assert_allclose(flux_2.data.sum(), nbin, rtol=1e-4)
    assert_allclose(flux_3.data.sum(), nbin, rtol=1e-4)
    assert_allclose(flux_4.data.sum(), nbin, rtol=1e-4)
