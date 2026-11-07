# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from numpy.testing import assert_allclose
from regions import CircleSkyRegion

from gammapy.data import GTI
from gammapy.estimators import FluxPoints
from gammapy.maps import MapAxis, RegionGeom, RegionNDMap
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    SkyModel,
    TemplateSpatialModel,
    TemplateSpectralModel,
)

from gammapy.utils.testing import requires_data

asdf = pytest.importorskip("asdf")
pytest.importorskip("asdf.testing")
asdf.get_config().validate_on_read = True


def assert_fluxpoints_equal(result, expected):
    assert set(result._data) == set(expected._data)
    for key in expected._data:
        assert_allclose(result._data[key].data, expected._data[key].data)
        assert result._data[key].geom == expected._data[key].geom
        assert result._data[key].unit == expected._data[key].unit
        assert result._data[key].data.dtype == expected._data[key].data.dtype

    assert result.filter_success_nan == expected.filter_success_nan
    assert result.meta == expected.meta

    if expected.gti is None:
        assert result.gti is None
    else:
        assert_allclose(
            result.gti.time_sum.to_value("s"),
            expected.gti.time_sum.to_value("s"),
        )

    ref, ref_expected = result.reference_model, expected.reference_model
    assert type(ref) is type(ref_expected)
    assert ref.name == ref_expected.name

    spectral, spectral_expected = ref.spectral_model, ref_expected.spectral_model
    assert type(spectral) is type(spectral_expected)
    if isinstance(spectral_expected, TemplateSpectralModel):
        assert_allclose(spectral.energy, spectral_expected.energy)
        assert_allclose(spectral.values, spectral_expected.values)

    spatial, spatial_expected = ref.spatial_model, ref_expected.spatial_model
    assert type(spatial) is type(spatial_expected)
    if isinstance(spatial_expected, TemplateSpatialModel):
        assert_allclose(spatial.map.data, spatial_expected.map.data)
        assert spatial.map.geom == spatial_expected.map.geom

    for actual, desired in zip(ref_expected.parameters, ref.parameters):
        assert_allclose(actual.value, desired.value)
        assert actual.unit == desired.unit
        assert actual.name == desired.name


def test_fluxpoints_roundtrip(tmp_path):
    file_path = tmp_path / "test.asdf"
    energy_axis = MapAxis.from_energy_edges([1, 3, 10, 30] * u.TeV)
    region = CircleSkyRegion(
        center=SkyCoord(83.63, 22.01, unit="deg", frame="icrs"), radius=0.1 * u.deg
    )
    geom = RegionGeom.create(region, axes=[energy_axis])
    data = {}
    for name, value in [("norm", 1.0), ("norm_err", 0.1), ("norm_ul", 2.0)]:
        m = RegionNDMap.from_geom(geom)
        m.data = np.linspace(0.5, 1.5, m.data.size).reshape(geom.data_shape) * value
        data[name] = m

    success = RegionNDMap.from_geom(geom, dtype=bool)
    success.data[...] = True
    success.data[1] = False
    data["success"] = success

    model = SkyModel(
        spectral_model=PowerLawSpectralModel(index=2.3), name="reference-source"
    )
    time_ref = Time("2010-01-01", scale="tt")
    gti = GTI(
        Table({"START": time_ref + [1, 5] * u.s, "STOP": time_ref + [3, 7] * u.s}),
        reference_time=time_ref,
    )
    meta = {"test-asdf": "fluxpoints"}

    fp = FluxPoints(
        data=data, reference_model=model, meta=meta, gti=gti, filter_success_nan=False
    )
    with asdf.AsdfFile() as af:
        af["fluxpoints"] = fp
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        result = af["fluxpoints"]
        assert_fluxpoints_equal(result, fp)


@requires_data()
def test_fluxpoints_roundtrip_real(tmp_path):
    file_path = tmp_path / "test.asdf"
    fp = FluxPoints.read("$GAMMAPY_DATA/tests/spectrum/flux_points/flux_points.fits")
    with asdf.AsdfFile() as af:
        af["fluxpoints"] = fp
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        result = af["fluxpoints"]
        assert_fluxpoints_equal(result, fp)
