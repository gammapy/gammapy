# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from regions import (
    CircleAnnulusSkyRegion,
    CircleSkyRegion,
    EllipseSkyRegion,
    PointSkyRegion,
    RectangleSkyRegion,
)
from gammapy.maps import Map, MapAxis, MapCoord, RegionGeom, WcsGeom, WcsNDMap
from gammapy.modeling.models import (
    SPATIAL_MODEL_REGISTRY,
    ConstantSpatialModel,
    DiskSpatialModel,
    FoVBackgroundModel,
    GaussianSpatialModel,
    GeneralizedGaussianSpatialModel,
    PiecewiseNormSpatialModel,
    PointSpatialModel,
    PowerLawSpectralModel,
    Shell2SpatialModel,
    ShellSpatialModel,
    SkyModel,
    TemplateNDSpatialModel,
    TemplateSpatialModel,
)
from gammapy.utils.testing import mpl_plot_check, requires_data, requires_dependency


def test_sky_point_source():
    geom = WcsGeom.create(skydir=(2.4, 2.3), npix=(10, 10), binsz=0.3)
    model = PointSpatialModel(lon_0="2.5 deg", lat_0="2.5 deg", frame="icrs")

    assert model.is_energy_dependent is False

    assert model.evaluation_radius.unit == "deg"
    assert_allclose(model.evaluation_radius.value, 0)

    assert model.frame == "icrs"

    assert_allclose(model.position.ra.deg, 2.5)
    assert_allclose(model.position.dec.deg, 2.5)

    val = model.evaluate_geom(geom)
    assert val.unit == "sr-1"
    assert_allclose(np.sum(val * geom.solid_angle()), 1)

    assert isinstance(model.to_region(), PointSkyRegion)


def test_sky_gaussian():
    # Test symmetric model
    sigma = 1 * u.deg
    model = GaussianSpatialModel(lon_0="5 deg", lat_0="15 deg", sigma=sigma)
    assert model.parameters["sigma"].min == 0
    val_0 = model(5 * u.deg, 15 * u.deg)
    val_sigma = model(5 * u.deg, 16 * u.deg)
    assert val_0.unit == "sr-1"
    ratio = val_0 / val_sigma
    assert_allclose(ratio, np.exp(0.5))
    radius = model.evaluation_radius
    assert radius.unit == "deg"
    assert_allclose(radius.value, 5 * sigma.value)
    assert_allclose(model.evaluation_bin_size_min, (1.0 / 3.0) * u.deg)

    # test the normalization for an elongated Gaussian near the Galactic Plane
    m_geom_1 = WcsGeom.create(
        binsz=0.05, width=(20, 20), skydir=(2, 2), frame="galactic", proj="AIT"
    )
    coords = m_geom_1.get_coord()
    solid_angle = m_geom_1.solid_angle()
    lon = coords.lon
    lat = coords.lat
    sigma = 3 * u.deg
    model_1 = GaussianSpatialModel(
        lon_0=2 * u.deg, lat_0=2 * u.deg, sigma=sigma, e=0.8, phi=30 * u.deg
    )
    vals_1 = model_1(lon, lat)
    assert vals_1.unit == "sr-1"
    assert_allclose(np.sum(vals_1 * solid_angle), 1, rtol=1.0e-3)

    radius = model_1.evaluation_radius
    assert radius.unit == "deg"
    assert_allclose(radius.value, 5 * sigma.value)

    # check the ratio between the value at the peak and on the 1-sigma isocontour
    sigma = 4 * u.deg
    semi_minor = 2 * u.deg
    e = np.sqrt(1 - (semi_minor / sigma) ** 2)
    model_2 = GaussianSpatialModel(
        lon_0=0 * u.deg, lat_0=0 * u.deg, sigma=sigma, e=e, phi=0 * u.deg
    )
    val_0 = model_2(0 * u.deg, 0 * u.deg)
    val_major = model_2(0 * u.deg, 4 * u.deg)
    val_minor = model_2(2 * u.deg, 0 * u.deg)
    assert val_0.unit == "sr-1"
    ratio_major = val_0 / val_major
    ratio_minor = val_0 / val_minor

    assert_allclose(ratio_major, np.exp(0.5))
    assert_allclose(ratio_minor, np.exp(0.5))

    # check the rotation
    model_3 = GaussianSpatialModel(
        lon_0=0 * u.deg, lat_0=0 * u.deg, sigma=sigma, e=e, phi=90 * u.deg
    )
    val_minor_rotated = model_3(0 * u.deg, 2 * u.deg)
    ratio_minor_rotated = val_0 / val_minor_rotated
    assert_allclose(ratio_minor_rotated, np.exp(0.5))

    assert isinstance(model.to_region(), EllipseSkyRegion)


@pytest.mark.parametrize("eta", np.arange(0.1, 1.01, 0.3))
@pytest.mark.parametrize("r_0", np.arange(0.01, 1.01, 0.3))
@pytest.mark.parametrize("e", np.arange(0.0, 0.801, 0.4))
def test_generalized_gaussian(eta, r_0, e):
    # check normalization is robust for a large set of values
    model = GeneralizedGaussianSpatialModel(
        eta=eta, r_0=r_0 * u.deg, e=e, frame="galactic"
    )

    width = np.maximum(2 * model.evaluation_radius.to_value("deg"), 0.5)
    geom = WcsGeom.create(
        skydir=(0, 0),
        binsz=0.02,
        width=width,
        frame="galactic",
    )

    integral = model.integrate_geom(geom)
    assert integral.unit.is_equivalent("")
    assert_allclose(integral.data.sum(), 1.0, atol=5e-3)


def test_generalized_gaussian_io():
    model = GeneralizedGaussianSpatialModel(e=0.5)

    reg = model.to_region()
    assert isinstance(reg, EllipseSkyRegion)
    assert_allclose(reg.width.value, 1.73205, rtol=1e-5)

    new_model = GeneralizedGaussianSpatialModel.from_dict(model.to_dict())
    assert isinstance(new_model, GeneralizedGaussianSpatialModel)


def test_sky_disk():
    # Test the disk case (e=0)
    r_0 = 2 * u.deg
    model = DiskSpatialModel(lon_0="1 deg", lat_0="45 deg", r_0=r_0)
    lon = [1, 5, 359] * u.deg
    lat = 46 * u.deg
    val = model(lon, lat)
    assert val.unit == "sr-1"
    desired = [261.263956, 0, 261.263956]
    assert_allclose(val.value, desired)
    radius = model.evaluation_radius
    assert radius.unit == "deg"
    assert_allclose(radius.to_value("deg"), 2.222)
    assert_allclose(model.evaluation_bin_size_min, 0.198 * u.deg)

    # test the normalization for an elongated ellipse near the Galactic Plane
    m_geom_1 = WcsGeom.create(
        binsz=0.015, width=(20, 20), skydir=(2, 2), frame="galactic", proj="AIT"
    )
    coords = m_geom_1.get_coord()
    solid_angle = m_geom_1.solid_angle()
    lon = coords.lon
    lat = coords.lat
    r_0 = 10 * u.deg
    model_1 = DiskSpatialModel(
        lon_0=2 * u.deg, lat_0=2 * u.deg, r_0=r_0, e=0.4, phi=30 * u.deg
    )
    vals_1 = model_1(lon, lat)
    assert vals_1.unit == "sr-1"
    assert_allclose(np.sum(vals_1 * solid_angle), 1, rtol=1.0e-3)

    radius = model_1.evaluation_radius
    assert radius.unit == "deg"
    assert_allclose(radius.to_value("deg"), 11.11)
    # test rotation
    r_0 = 2 * u.deg
    semi_minor = 1 * u.deg
    eccentricity = np.sqrt(1 - (semi_minor / r_0) ** 2)
    model_rot_test = DiskSpatialModel(
        lon_0=0 * u.deg, lat_0=0 * u.deg, r_0=r_0, e=eccentricity, phi=90 * u.deg
    )
    assert_allclose(model_rot_test(0 * u.deg, 1.5 * u.deg).value, 0)

    # test the normalization for a disk (ellipse with e=0) at the Galactic Pole
    m_geom_2 = WcsGeom.create(
        binsz=0.1, width=(6, 6), skydir=(0, 90), frame="galactic", proj="AIT"
    )
    coords = m_geom_2.get_coord()
    lon = coords.lon
    lat = coords.lat

    r_0 = 5 * u.deg
    disk = DiskSpatialModel(lon_0=0 * u.deg, lat_0=90 * u.deg, r_0=r_0)
    vals_disk = disk(lon, lat)

    solid_angle = 2 * np.pi * (1 - np.cos(5 * u.deg))
    assert_allclose(np.max(vals_disk).value * solid_angle, 1)

    assert isinstance(model.to_region(), EllipseSkyRegion)


def test_sky_disk_edge():
    r_0 = 2 * u.deg
    model = DiskSpatialModel(
        lon_0="0 deg",
        lat_0="0 deg",
        r_0=r_0,
        e=0.5,
        phi="0 deg",
    )
    value_center = model(0 * u.deg, 0 * u.deg)
    value_edge = model(0 * u.deg, r_0)
    assert_allclose((value_edge / value_center).to_value(""), 0.5)

    edge = model.edge_width.value * r_0
    value_edge_pwidth = model(0 * u.deg, r_0 + edge / 2)
    assert_allclose((value_edge_pwidth / value_center).to_value(""), 0.05)

    value_edge_nwidth = model(0 * u.deg, r_0 - edge / 2)
    assert_allclose((value_edge_nwidth / value_center).to_value(""), 0.95)


def test_disk_from_region():
    region = EllipseSkyRegion(
        center=SkyCoord(20, 17, unit="deg"),
        height=0.3 * u.deg,
        width=1.0 * u.deg,
        angle=30 * u.deg,
    )
    disk = DiskSpatialModel.from_region(region, frame="galactic")
    assert_allclose(disk.parameters["lon_0"].value, 132.666, rtol=1e-2)
    assert_allclose(disk.parameters["lat_0"].value, -45.33118067, rtol=1e-2)
    assert_allclose(disk.parameters["r_0"].quantity, 0.5 * u.deg, rtol=1e-2)
    assert_allclose(disk.parameters["e"].value, 0.9539, rtol=1e-2)
    assert_allclose(disk.parameters["phi"].quantity, 110.946048 * u.deg)

    reg1 = disk.to_region()
    assert_allclose(reg1.height, region.width, rtol=1e-2)

    center = SkyCoord(20, 17, unit="deg", frame="galactic")
    region = EllipseSkyRegion(
        center=center,
        height=1 * u.deg,
        width=0.3 * u.deg,
        angle=30 * u.deg,
    )
    disk = DiskSpatialModel.from_region(region, frame="icrs")
    reg1 = disk.to_region()
    assert_allclose(reg1.angle, -30.323 * u.deg, rtol=1e-2)
    assert_allclose(reg1.height, region.height, rtol=1e-3)

    region = CircleSkyRegion(center=region.center, radius=1.0 * u.deg)
    disk = DiskSpatialModel.from_region(region)
    assert_allclose(disk.parameters["e"].value, 0.0, rtol=1e-2)
    assert_allclose(disk.parameters["lon_0"].value, 20, rtol=1e-2)
    assert disk.frame == "galactic"

    geom = WcsGeom.create(skydir=center, npix=(10, 10), binsz=0.3)
    res = disk.evaluate_geom(geom)
    assert_allclose(np.sum(res.value), 50157.904662)

    region = PointSkyRegion(center=region.center)
    with pytest.raises(ValueError):
        DiskSpatialModel.from_region(region)


def test_from_position():
    center = SkyCoord(20, 17, unit="deg")
    spatial_model = GaussianSpatialModel.from_position(
        position=center, sigma=0.5 * u.deg
    )
    geom = WcsGeom.create(skydir=center, npix=(10, 10), binsz=0.3)
    res = spatial_model.evaluate_geom(geom)
    assert_allclose(np.sum(res.value), 36307.440813)
    model = SkyModel(
        spectral_model=PowerLawSpectralModel(), spatial_model=spatial_model
    )
    assert_allclose(model.position.ra.value, center.ra.value, rtol=1e-3)


def test_sky_shell():
    width = 2 * u.deg
    rad = 2 * u.deg
    model = ShellSpatialModel(lon_0="1 deg", lat_0="45 deg", radius=rad, width=width)
    lon = [1, 2, 4] * u.deg
    lat = 45 * u.deg
    val = model(lon, lat)
    assert val.unit == "deg-2"
    desired = [55.979449, 57.831651, 94.919895]
    assert_allclose(val.to_value("sr-1"), desired)
    radius = model.evaluation_radius
    assert radius.unit == "deg"
    assert_allclose(radius.value, rad.value + width.value)
    assert isinstance(model.to_region(), CircleAnnulusSkyRegion)
    assert_allclose(model.evaluation_bin_size_min, 2 * u.deg)


def test_sky_shell2():
    width = 2 * u.deg
    rad = 2 * u.deg
    model = Shell2SpatialModel(lon_0="1 deg", lat_0="45 deg", r_0=rad + width, eta=0.5)
    lon = [1, 2, 4] * u.deg
    lat = 45 * u.deg
    val = model(lon, lat)
    assert val.unit == "deg-2"
    desired = [55.979449, 57.831651, 94.919895]
    assert_allclose(val.to_value("sr-1"), desired)
    radius = model.evaluation_radius
    assert radius.unit == "deg"
    assert_allclose(radius.value, rad.value + width.value)
    assert_allclose(model.r_in.value, rad.value)
    assert isinstance(model.to_region(), CircleAnnulusSkyRegion)
    assert_allclose(model.evaluation_bin_size_min, 2 * u.deg)


def test_sky_diffuse_constant():
    model = ConstantSpatialModel(value="42 sr-1")
    lon = [1, 2] * u.deg
    lat = 45 * u.deg
    val = model(lon, lat)
    assert val.unit == "sr-1"
    assert_allclose(val.value, 42)
    radius = model.evaluation_radius
    assert radius is None
    assert isinstance(model.to_region(), RectangleSkyRegion)


@requires_data()
@requires_dependency("ipywidgets")
def test_sky_diffuse_map(caplog):
    filename = "$GAMMAPY_DATA/catalogs/fermi/Extended_archive_v18/Templates/RXJ1713_2016_250GeV.fits"  # noqa: E501
    model = TemplateSpatialModel.read(filename, normalize=False)
    lon = [258.5, 0] * u.deg
    lat = -39.8 * u.deg
    val = model(lon, lat)

    assert "WARNING" in [_.levelname for _ in caplog.records]
    assert "Missing spatial template unit, assuming sr^-1" in [
        _.message for _ in caplog.records
    ]

    assert val.unit == "sr-1"
    desired = [3265.6559, 0]
    assert_allclose(val.value, desired)

    res = model.evaluate_geom(model.map.geom)
    assert_allclose(np.sum(res.value), 32826159.74707)
    radius = model.evaluation_radius

    assert radius.unit == "deg"
    assert_allclose(radius.value, 0.64, rtol=1.0e-2)
    assert model.frame == "fk5"
    assert isinstance(model.to_region(), RectangleSkyRegion)

    with pytest.raises(TypeError):
        model.plot_interactive()

    with pytest.raises(TypeError):
        model.plot_grid()

    # change central position
    model.lon_0.value = 12.0
    model.lat_0.value = 6
    val = model([11.8, 12.8] * u.deg, 6.1 * u.deg)
    assert_allclose(val.value, [2850.8103, 89.629447], rtol=1e-3)

    # test to and from dict
    dict1 = model.to_dict()
    model2 = TemplateSpatialModel.from_dict(dict1)
    assert_allclose(model2.lon_0.quantity, 12.0 * u.deg, rtol=1e-3)

    # test dict without parameters
    dict1["spatial"]["parameters"] = []
    model3 = TemplateSpatialModel.from_dict(dict1)
    assert_allclose(model3.lon_0.quantity, 258.388 * u.deg, rtol=1e-3)

    # test dict without parameters
    dict1["spatial"].pop("parameters")
    model3 = TemplateSpatialModel.from_dict(dict1)
    assert_allclose(model3.lon_0.quantity, 258.388 * u.deg, rtol=1e-3)


@requires_data()
@requires_dependency("ipywidgets")
def test_sky_diffuse_map_3d():
    filename = "$GAMMAPY_DATA/fermi-3fhl-gc/gll_iem_v06_gc.fits.gz"
    model = TemplateSpatialModel.read(filename, normalize=False)
    lon = [258.5, 0] * u.deg
    lat = -39.8 * u.deg
    energy = 1 * u.GeV
    val = model(lon, lat, energy)

    with pytest.raises(ValueError):
        model(lon, lat)
    assert model.map.unit == "cm-2 s-1 MeV-1 sr-1"

    val = model(lon, lat, energy)
    assert val.unit == "cm-2 s-1 MeV-1 sr-1"

    res = model.evaluate_geom(model.map.geom)
    assert_allclose(np.sum(res.value), 0.11803847221522712)

    with pytest.raises(TypeError):
        model.plot()

    with mpl_plot_check():
        model.plot_grid()

    with mpl_plot_check():
        model.plot_interactive()


def test_sky_diffuse_map_normalize():
    # define model map with a constant value of 1
    model_map = Map.create(map_type="wcs", width=(10, 5), binsz=0.5, unit="sr-1")
    model_map.data += 1.0
    model = TemplateSpatialModel(model_map)

    # define data map with a different spatial binning
    data_map = Map.create(map_type="wcs", width=(10, 5), binsz=1)
    coords = data_map.geom.get_coord()
    solid_angle = data_map.geom.solid_angle()
    vals = model(coords.lon, coords.lat) * solid_angle

    assert vals.unit == ""
    integral = vals.sum()
    assert_allclose(integral.value, 1, rtol=1e-4)


def test_sky_diffuse_map_copy():
    # define model map with a constant value of 1
    model_map = Map.create(map_type="wcs", width=(1, 1), binsz=0.5, unit="sr-1")
    model_map.data += 1.0

    model = TemplateSpatialModel(model_map, normalize=False)
    assert np.all(model.map.data == model_map.data)
    model.map.data += 1
    # Check that the original map is unchanged
    assert np.all(model_map.data == np.ones_like(model_map.data))

    model = TemplateSpatialModel(model_map, normalize=False, copy_data=False)
    assert np.all(model.map.data == model_map.data)
    model.map.data += 1
    # Check that the original map has also been changed
    assert np.all(model.map.data == model_map.data)

    model_copy = model.copy(copy_data=False)
    model_copy.map.data += 1
    # Check that the original map has also been changed
    assert np.all(model.map.data == model_copy.map.data)


def test_sky_diffuse_map_empty(caplog):
    # define model map with 0 values
    model_map = Map.create(map_type="wcs", width=(1, 1), binsz=0.5, unit="sr-1")

    with caplog.at_level(logging.WARNING):
        model = TemplateSpatialModel(model_map, normalize=True)
        assert "Map values are all zeros. Check and fix this!" in [
            _.message for _ in caplog.records
        ]
        assert np.all(np.isfinite(model.map.data))

    axes = [MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=2)]
    model_map = Map.create(
        map_type="wcs", width=(1, 1), binsz=0.5, unit="sr-1", axes=axes
    )
    model_map.data[0, :, :] = 1
    with caplog.at_level(logging.WARNING):
        model = TemplateSpatialModel(model_map, normalize=True)
        assert (
            "Map values are all zeros in at least one energy bin. Check and fix this!"
            in [_.message for _ in caplog.records]
        )
        assert np.all(np.isfinite(model.map.data))


@pytest.mark.parametrize("model_cls", SPATIAL_MODEL_REGISTRY)
def test_model_from_dict(tmpdir, model_cls):
    if model_cls in [TemplateSpatialModel, TemplateNDSpatialModel]:
        default_map = Map.create(map_type="wcs", width=(1, 1), binsz=0.5, unit="sr-1")
        filename = str(tmpdir / "template.fits")
        model = model_cls(default_map, filename=filename)
        model.write()
    elif model_cls is PiecewiseNormSpatialModel:
        geom = WcsGeom.create(skydir=(0, 0), npix=(2, 2), binsz=0.3, frame="galactic")
        default_coords = MapCoord.create(geom.footprint)
        default_coords["lon"] *= u.deg
        default_coords["lat"] *= u.deg
        model = model_cls(default_coords, frame="galactic")
    else:
        model = model_cls()

    data = model.to_dict()
    model_from_dict = model_cls.from_dict(data)
    assert model_from_dict.tag == model_from_dict.tag

    bkg_model = FoVBackgroundModel(spatial_model=model, dataset_name="test")
    bkg_model_dict = bkg_model.to_dict()
    bkg_model_from_dict = FoVBackgroundModel.from_dict(bkg_model_dict)
    assert bkg_model_from_dict.spatial_model is not None


def test_evaluate_on_fk5_map():
    # Check if spatial model can be evaluated on a map with FK5 frame
    # Regression test for GH-2402
    header = {}
    header["CDELT1"] = 1.0
    header["CDELT2"] = 1.0
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["RADESYS"] = "FK5"
    header["CRVAL1"] = 0
    header["CRVAL2"] = 0
    header["CRPIX1"] = 5
    header["CRPIX2"] = 5

    wcs = WCS(header)
    geom = WcsGeom(wcs, npix=(10, 10))
    model = GaussianSpatialModel(lon_0="0 deg", lat_0="0 deg", sigma="1 deg")
    data = model.evaluate_geom(geom)
    assert data.sum() > 0


def test_evaluate_fk5_model():
    geom = WcsGeom.create(width=(5, 5), binsz=0.1, frame="icrs")
    model = GaussianSpatialModel(
        lon_0="0 deg", lat_0="0 deg", sigma="0.1 deg", frame="fk5"
    )
    data = model.evaluate_geom(geom)
    assert data.sum() > 0


def test_spatial_model_plot():
    model = PointSpatialModel()
    model.covariance = np.diag([0.01, 0.01])

    with mpl_plot_check():
        ax = model.plot()

    with mpl_plot_check():
        model.plot_error(ax=ax, which="all")


models_test = [
    (GaussianSpatialModel, "sigma"),
    (GeneralizedGaussianSpatialModel, "r_0"),
    (DiskSpatialModel, "r_0"),
]


@pytest.mark.parametrize(("model_class", "extension_param"), models_test)
def test_spatial_model_plot_error(model_class, extension_param):
    model = model_class(lon="0d", lat="0d", sigma=0.2 * u.deg, frame="galactic")
    model.lat_0.error = 0.04
    model.lon_0.error = 0.02
    model.parameters[extension_param].error = 0.04
    model.e.error = 0.002

    empty_map = Map.create(
        skydir=model.position, frame=model.frame, width=1, binsz=0.02
    )
    with mpl_plot_check():
        ax = empty_map.plot()
        model.plot_error(ax=ax, which="all")
        model.plot_error(ax=ax, which="position")
        model.plot_error(ax=ax, which="extension")


def test_integrate_region_geom():
    center = SkyCoord("0d", "0d", frame="icrs")
    model = GaussianSpatialModel(
        lon_0="0 deg", lat_0="0 deg", sigma=0.1 * u.deg, frame="icrs"
    )

    radius_large = 1 * u.deg
    circle_large = CircleSkyRegion(center, radius_large)
    radius_small = 0.1 * u.deg
    circle_small = CircleSkyRegion(center, radius_small)

    geom_large, geom_small = (
        RegionGeom(region=circle_large),
        RegionGeom(region=circle_small, binsz_wcs="0.01 deg"),
    )

    integral_large, integral_small = (
        model.integrate_geom(geom_large).data,
        model.integrate_geom(geom_small).data,
    )

    assert_allclose(integral_large[0], 1, rtol=0.001)
    assert_allclose(integral_small[0], 0.3953, rtol=0.001)


# TODO: solve issue with small radii (e.g. 1e-5) and improve tolerance
@pytest.mark.parametrize("width", np.geomspace(1e-4, 1e-1, 10) * u.deg)
@pytest.mark.parametrize(
    "model",
    [
        (GaussianSpatialModel, "sigma", 6e-3),
        (DiskSpatialModel, "r_0", 0.4),
        (GeneralizedGaussianSpatialModel, "r_0", 3e-4),
    ],
)
def test_integrate_wcs_geom(width, model):
    model_cls, param_name, tolerance = model
    param_dict = {param_name: width}
    spatial_model = model_cls(
        lon_0="0.234 deg", lat_0="-0.172 deg", frame="icrs", **param_dict
    )
    geom = WcsGeom.create(skydir=(0, 0), npix=100, binsz=0.02)

    integrated = spatial_model.integrate_geom(geom)

    assert_allclose(integrated.data.sum(), 1, atol=tolerance)


def test_integrate_geom_no_overlap():
    center = SkyCoord("0d", "0d", frame="icrs")
    model = GaussianSpatialModel(
        lon_0="10.234 deg", lat_0="-0.172 deg", sigma=1e-2 * u.deg, frame="icrs"
    )
    geom = WcsGeom.create(skydir=center, npix=100, binsz=0.02)

    # This should not fail but return map filled with 0
    integrated = model.integrate_geom(geom)

    assert_allclose(integrated.data, 0.0)


def test_integrate_geom_parameter_issue():
    center = SkyCoord("0d", "0d", frame="icrs")
    model = GaussianSpatialModel(
        lon_0="0.234 deg", lat_0="-0.172 deg", sigma=np.nan * u.deg, frame="icrs"
    )
    geom = WcsGeom.create(skydir=center, npix=100, binsz=0.02)

    # This should not fail but return map filled with nan
    integrated = model.integrate_geom(geom)

    assert_allclose(integrated.data, np.nan)


def test_integrate_geom_energy_axis():
    center = SkyCoord("0d", "0d", frame="icrs")
    model = GaussianSpatialModel(lon="0d", lat="0d", sigma=0.1 * u.deg, frame="icrs")

    radius = 1 * u.deg
    square = RectangleSkyRegion(center, radius, radius)

    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=10)
    geom = RegionGeom(region=square, axes=[axis])

    integral = model.integrate_geom(geom).data

    assert_allclose(integral, 1, rtol=0.0001)


def test_templatemap_clip():
    model_map = Map.create(map_type="wcs", width=(2, 2), binsz=0.5, unit="sr-1")
    model_map.data += 1.0
    model = TemplateSpatialModel(model_map)
    model.map.data = model.map.data * -1

    lon = np.array([0, 0.2, 0.3]) * u.deg
    lat = np.array([0, 0.2, 0.3]) * u.deg

    val = model.evaluate(lon, lat)
    assert_allclose(val, 0, rtol=0.0001)


def test_piecewise_spatial_model_gc():
    geom = WcsGeom.create(skydir=(0, 0), npix=(2, 2), binsz=0.3, frame="galactic")
    coords = MapCoord.create(geom.footprint)
    coords["lon"] *= u.deg
    coords["lat"] *= u.deg

    model = PiecewiseNormSpatialModel(coords, frame="galactic")

    assert_allclose(model(*geom.to_image().center_coord), 1.0)

    norms = np.arange(coords.shape[0])

    model = PiecewiseNormSpatialModel(coords, norms, frame="galactic")

    assert not model.is_energy_dependent

    expected = np.array([[0, 3], [1, 2]])
    assert_allclose(model(*geom.to_image().get_coord()), expected, atol=1e-5)

    assert_allclose(model.evaluate_geom(geom.to_image()), expected, atol=1e-5)

    assert_allclose(model.evaluate_geom(geom), expected, atol=1e-5)

    model_dict = model.to_dict()
    new_model = PiecewiseNormSpatialModel.from_dict(model_dict)
    assert model_dict == new_model.to_dict()

    assert_allclose(new_model.evaluate_geom(geom.to_image()), expected, atol=1e-5)

    assert_allclose(
        model.evaluate(-0.1 * u.deg, 2.3 * u.deg),
        model.evaluate(359.9 * u.deg, 2.3 * u.deg),
    )


def test_piecewise_spatial_model():
    for lon in range(-360, 360):
        geom = WcsGeom.create(
            skydir=(lon, 2.3), npix=(2, 2), binsz=0.3, frame="galactic"
        )

        coords = MapCoord.create(geom.footprint)
        coords["lon"] *= u.deg
        coords["lat"] *= u.deg

        model = PiecewiseNormSpatialModel(coords, frame="galactic")

        assert_allclose(model(*geom.to_image().center_coord), 1.0)

        norms = np.arange(coords.shape[0])

        model = PiecewiseNormSpatialModel(coords, norms, frame="galactic")

        expected = np.array([[0, 3], [1, 2]])
        assert_allclose(model(*geom.to_image().get_coord()), expected, atol=1e-5)

        assert_allclose(model.evaluate_geom(geom.to_image()), expected, atol=1e-5)

        assert_allclose(model.evaluate_geom(geom), expected, atol=1e-5)

        model_dict = model.to_dict()
        new_model = PiecewiseNormSpatialModel.from_dict(model_dict)
        assert model_dict == new_model.to_dict()

        assert_allclose(new_model.evaluate_geom(geom.to_image()), expected, atol=1e-5)


def test_piecewise_spatial_model_3d():
    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=3)
    geom = WcsGeom.create(
        skydir=(2.4, 2.3), npix=(2, 2), binsz=0.3, frame="galactic", axes=[axis]
    )
    coords = geom.get_coord().flat

    with pytest.raises(ValueError):
        PiecewiseNormSpatialModel(coords, frame="galactic")


@requires_data()
def test_template_ND(tmpdir):
    filename = "$GAMMAPY_DATA/catalogs/fermi/Extended_archive_v18/Templates/RXJ1713_2016_250GeV.fits"  # noqa: E501
    map_ = Map.read(filename)
    map_.data[map_.data < 0] = 0
    geom2d = map_.geom
    norm = MapAxis.from_nodes(range(0, 11, 2), interp="lin", name="norm", unit="")
    cste = MapAxis.from_bounds(-1, 1, 3, interp="lin", name="cste", unit="")
    geom = geom2d.to_cube([norm, cste])

    nd_map = WcsNDMap(geom)

    for kn, norm_value in enumerate(norm.center):
        for kp, cste_value in enumerate(cste.center):
            nd_map.data[kp, kn, :, :] = norm_value * map_.data + cste_value

    template = TemplateNDSpatialModel(nd_map, interp_kwargs={"values_scale": "lin"})
    assert len(template.parameters) == 2
    assert_allclose(template.parameters["norm"].value, 5)
    assert_allclose(template.parameters["cste"].value, 0)
    assert_allclose(
        template.evaluate(
            geom2d.center_skydir.ra, geom2d.center_skydir.dec, norm=0, cste=0
        ),
        [0],
    )
    assert_allclose(template.evaluate_geom(geom2d), 5 * map_.data, rtol=0.03, atol=10)

    template.parameters["norm"].value = 2
    template.parameters["cste"].value = 0
    assert_allclose(template.evaluate_geom(geom2d), 2 * map_.data, rtol=0.03, atol=10)

    template.filename = str(tmpdir / "template_ND.fits")
    template.write()

    dict_ = template.to_dict()
    template_new = TemplateNDSpatialModel.from_dict(dict_)
    assert_allclose(template_new.map.data, nd_map.data)
    assert len(template_new.parameters) == 2
    assert template_new.parameters["norm"].value == 2
    assert template_new.parameters["cste"].value == 0


@requires_data()
def test_templatespatial_write(tmpdir):
    filename = "$GAMMAPY_DATA/catalogs/fermi/Extended_archive_v18/Templates/RXJ1713_2016_250GeV.fits"
    map_ = Map.read(filename)
    template = TemplateSpatialModel(map_, filename=filename)

    filename_new = str(tmpdir / "template_test.fits")
    template.write(overwrite=True, filename=filename_new)
    assert Path(filename_new).is_file()


@requires_data()
def test_template_spatial_parameters_copy():
    filename = "$GAMMAPY_DATA/catalogs/fermi/Extended_archive_v18/Templates/RXJ1713_2016_250GeV.fits"
    model = TemplateSpatialModel.read(filename, normalize=False)
    model.position = SkyCoord(0, 0, unit="deg", frame="galactic")
    model_copy = model.copy()
    assert_allclose(model.parameters.value, model_copy.parameters.value)
