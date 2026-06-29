# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
import numpy as np
import pytest
from gammapy.utils.deprecation import GammapyDeprecationWarning
from astropy.table import Table
from numpy.testing import assert_allclose

from gammapy.astro.darkmatter import (
    ContinuumPrimaryFlux,
    DarkMatterAnnihilationSpectralModel,
    DarkMatterDecaySpectralModel,
    PrimaryFlux,
)
from gammapy.astro.darkmatter.spectra import _PrimaryFluxValidator
from gammapy.modeling.models import Models, SkyModel, SpectralModel
from gammapy.utils.testing import assert_quantity_allclose, requires_data

# ---------------------------------------------------------------------------
# ContinuumPrimaryFlux
# ---------------------------------------------------------------------------


@requires_data()
def test_continuum_primary_flux():
    with pytest.raises(ValueError):
        ContinuumPrimaryFlux(channel="Spam", mDM=1 * u.TeV)

    primflux = ContinuumPrimaryFlux(channel="W", mDM=1 * u.TeV)
    actual = primflux(500 * u.GeV)
    desired = 9.3319318e-05 / u.GeV
    assert_quantity_allclose(actual, desired)


@requires_data()
def test_primary_flux_deprecated():
    with pytest.warns(GammapyDeprecationWarning, match="PrimaryFlux"):
        PrimaryFlux(channel="b", mDM=1 * u.TeV)


@requires_data()
def test_mass_argument_deprecated():
    with pytest.warns(GammapyDeprecationWarning, match="mass"):
        DarkMatterAnnihilationSpectralModel(channel="W", mass=1 * u.TeV)

    with pytest.warns(GammapyDeprecationWarning, match="mass"):
        DarkMatterDecaySpectralModel(channel="W", mass=1 * u.TeV)


@pytest.mark.parametrize(
    "mass, expected_flux, source, expected_exception",
    [
        (1.6, 0.00025037, "pppc4", None),
        (11, 0.00549445, "cosmixs", None),
        (75, None, "nonexistend", ValueError),
        (75, None, "pppc4", ValueError),
    ],
)
@requires_data()
def test_primary_flux_interpolation(mass, expected_flux, source, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            ContinuumPrimaryFlux(
                channel="aZ",
                mDM=mass * u.TeV,
                source=source,
                mapping_dict={"mDM": "mDM"},
            )
        return
    primflux = ContinuumPrimaryFlux(channel="W", mDM=mass * u.TeV, source=source)
    actual = primflux(500 * u.GeV)
    assert_quantity_allclose(actual, expected_flux / u.GeV, rtol=1e-5)


@requires_data()
def test_primary_flux_cosmixs():
    with pytest.raises(ValueError):
        ContinuumPrimaryFlux(channel="Spam", mDM=1 * u.TeV)

    primflux = ContinuumPrimaryFlux(
        channel="W", mDM=1 * u.TeV, source="cosmixs", mapping_dict={"mDM": "mDM"}
    )
    actual = primflux(500 * u.GeV)
    desired = 0.00013085 / u.GeV
    assert_quantity_allclose(actual, desired, rtol=1e-4)

    with pytest.raises(ValueError):
        ContinuumPrimaryFlux(channel="q", mDM=1 * u.TeV, source="cosmixs")
    with pytest.raises(ValueError):
        ContinuumPrimaryFlux(channel="V->e", mDM=1 * u.TeV, source="cosmixs")
    with pytest.raises(ValueError):
        ContinuumPrimaryFlux(channel="V->mu", mDM=1 * u.TeV, source="cosmixs")
    with pytest.raises(ValueError):
        ContinuumPrimaryFlux(channel="V->tau", mDM=1 * u.TeV, source="cosmixs")

    with pytest.raises(ValueError):
        ContinuumPrimaryFlux(channel="d", mDM=1 * u.TeV, source="pppc4")
    with pytest.raises(ValueError):
        ContinuumPrimaryFlux(channel="u", mDM=1 * u.TeV, source="pppc4")
    with pytest.raises(ValueError):
        ContinuumPrimaryFlux(channel="s", mDM=1 * u.TeV, source="pppc4")


@requires_data()
def test_resolve_table_path_unknown_source(monkeypatch):
    flux = ContinuumPrimaryFlux(mDM=1 * u.TeV, channel="b", source="pppc4")
    flux._source = "unknown_predefined"
    flux._source_type = "predefined"
    with pytest.raises(ValueError, match="Unknown source"):
        flux._resolve_table_path()


@requires_data()
def test_continuum_to_from_dict_roundtrip():
    flux = ContinuumPrimaryFlux(channel="W", mDM=1 * u.TeV)
    data = flux.to_dict()
    new_flux = ContinuumPrimaryFlux.from_dict(data)

    assert_quantity_allclose(new_flux.mDM, flux.mDM)
    assert new_flux.channel == flux.channel
    assert new_flux.source == flux.source


def test_custom_source_file_empty(tmp_path):
    empty_file = tmp_path / "empty_spectra.dat"
    empty_file.touch()

    with pytest.raises(ValueError, match="Source file is empty"):
        ContinuumPrimaryFlux(mDM=5 * u.TeV, channel="b", source=str(empty_file))


def test_custom_source_file_bad_extension(tmp_path):
    bad_file = tmp_path / "spectra.dl2"
    bad_file.write_text("mDM Log[10,x] b\n1000 -3 1e-5\n")

    with pytest.raises(KeyError, match="Source file extension"):
        ContinuumPrimaryFlux(mDM=5 * u.TeV, channel="b", source=str(bad_file))


def test_custom_source_invalid_path():
    with pytest.raises(ValueError, match="Invalid source"):
        ContinuumPrimaryFlux(
            mDM=5 * u.TeV, channel="b", source="/nonexistent/path.ecsv"
        )


def test_source_non_string_raises_typeerror():
    with pytest.raises(TypeError, match="source must be a string"):
        ContinuumPrimaryFlux(mDM=1 * u.TeV, channel="b", source=123)


@requires_data()
def test_continuum_source_none_branch():
    with pytest.warns(UserWarning, match="PPPC4 will be used by default"):
        flux = ContinuumPrimaryFlux(mDM=1 * u.TeV, channel="b", source=None)
    assert flux.source == "pppc4"
    assert flux._source_type == "predefined"


@requires_data()
def test_source_none_warns_and_defaults_to_pppc4():
    with pytest.warns(UserWarning, match="PPPC4 will be used by default"):
        flux = ContinuumPrimaryFlux(mDM=1 * u.TeV, channel="b", source=None)
    assert flux.source == "pppc4"


def test_dm_spectral_model_custom_io(tmp_path):
    custom_file = tmp_path / "custom_spectra.ecsv"

    t = Table(
        {
            "mDM": [500.0, 500.0, 1000.0, 1000.0] * u.GeV,
            "Log[10,x]": [-3.0, -2.0, -3.0, -2.0],
            "b": [1e-15, 1e-16, 1e-15, 1e-16] / u.GeV,
        }
    )
    t.write(custom_file, format="ascii.ecsv")

    mapping = {"mDM": "mDM", "Log[10,x]": "Log[10,x]", "b": "b"}

    custom_flux = ContinuumPrimaryFlux(
        mDM=500 * u.GeV, channel="b", source=str(custom_file), mapping_dict=mapping
    )
    assert custom_flux.mapping_dict == mapping

    model = DarkMatterAnnihilationSpectralModel(
        mDM=500 * u.GeV,
        channel="b",
        factor=3.41e19 * u.Unit("GeV2 cm-5"),
        primary_flux=custom_flux,
    )

    sky_model = SkyModel(spectral_model=model, name="skymodel_custom")
    models = Models([sky_model])

    filename = tmp_path / "model_custom.yaml"
    models.write(filename, overwrite=True)
    new_models = Models.read(filename)
    loaded_model = new_models[0].spectral_model

    assert loaded_model.primary_flux.source == str(custom_file)
    # mapping_dict is not serialized by ContinuumPrimaryFlux.to_dict
    assert loaded_model.primary_flux.mapping_dict is None


def test_dm_annihilation_custom_errors(tmp_path):
    file_path = tmp_path / "test_dm.ecsv"
    t = Table()
    t["mDM"] = [1000, 5000] * u.GeV
    t["Log[10,x]"] = [-3, -2]
    t["bbar"] = [1e-5, 1e-4]
    t.write(file_path, format="ascii.ecsv")

    mass = 5 * u.TeV
    with pytest.raises(TypeError, match="mapping_dict must be a dictionary"):
        ContinuumPrimaryFlux(
            mDM=mass,
            channel="b",
            source=str(file_path),
            mapping_dict=["not", "a", "dict"],
        )

    incomplete_mapping = {"Log[10,x]": "Log[10,x]"}
    with pytest.raises(KeyError, match="Mandatory column"):
        ContinuumPrimaryFlux(
            mDM=mass,
            channel="b",
            source=str(file_path),
            mapping_dict=incomplete_mapping,
        )

    wrong_mapping = {"mDM": "mDM", "Log[10,x]": "Log[10,x]", "wrong_col": "tau"}
    with pytest.raises(ValueError, match="is not present in the provided mapping_dict"):
        ContinuumPrimaryFlux(
            mDM=mass, channel="b", source=str(file_path), mapping_dict=wrong_mapping
        )


def test_custom_source_file_without_mapping_and_missing_channel(tmp_path):
    custom_file = tmp_path / "custom_spectra_nomap.ecsv"

    t = Table(
        {
            "mDM": [500.0, 500.0, 1000.0, 1000.0] * u.GeV,
            "Log[10,x]": [-3.0, -2.0, -3.0, -2.0],
            "b": [1e-15, 1e-16, 1e-15, 1e-16],
        }
    )
    t.write(custom_file, format="ascii.ecsv")

    flux = ContinuumPrimaryFlux(mDM=500 * u.GeV, channel="b", source=str(custom_file))
    assert flux.channel == "b"

    with pytest.raises(ValueError, match="is not present in the provided source file"):
        ContinuumPrimaryFlux(mDM=500 * u.GeV, channel="eL", source=str(custom_file))


def test_missing_data_file(monkeypatch):
    monkeypatch.setenv("GAMMAPY_DATA", "/fake/path/to/nowhere")
    with pytest.raises(FileNotFoundError, match="File not found"):
        ContinuumPrimaryFlux(mDM=1 * u.TeV, channel="b")


@requires_data()
def test_mDM_out_of_bounds():
    with pytest.raises(ValueError, match="is out of bounds"):
        ContinuumPrimaryFlux(mDM=1 * u.eV, channel="b")


def test_custom_source_no_mapping_dict(tmp_path):
    custom_file = tmp_path / "spectra.ecsv"
    t = Table(
        {"mDM": [500.0, 1000.0] * u.GeV, "Log[10,x]": [-3.0, -3.0], "b": [1e-15, 1e-15]}
    )
    t.write(custom_file, format="ascii.ecsv")
    flux = ContinuumPrimaryFlux(
        mDM=500 * u.GeV, channel="b", source=str(custom_file), mapping_dict=None
    )
    assert flux.mapping_dict is None


# ---------------------------------------------------------------------------
# _AstrophysicalFactorValidator / _RedshiftValidator
# ---------------------------------------------------------------------------


def test_factor_must_be_positive():
    with pytest.raises(
        ValueError, match="astrophysical factor must be strictly positive"
    ):
        DarkMatterAnnihilationSpectralModel(mDM=1 * u.TeV, channel="b", factor=-1)


def test_redshift_must_be_non_negative():
    with pytest.raises(ValueError, match="Redshift z must be >= 0"):
        DarkMatterAnnihilationSpectralModel(mDM=1 * u.TeV, channel="b", z=-0.1)


def test_redshift_must_be_scalar():
    with pytest.raises(TypeError, match="z must be a dimensionless scalar"):
        DarkMatterAnnihilationSpectralModel(mDM=1 * u.TeV, channel="b", z="bad")


# ---------------------------------------------------------------------------
# _PrimaryFluxValidator (mismatch warnings / type checks)
# ---------------------------------------------------------------------------
@requires_data()
def test_primary_flux_setter_skips_channel_check_for_non_continuum():
    from unittest.mock import patch

    with patch.object(ContinuumPrimaryFlux, "__init__", return_value=None):
        real_flux = ContinuumPrimaryFlux.__new__(ContinuumPrimaryFlux)
        real_flux._mDM = 1 * u.TeV

    model = object.__new__(DarkMatterAnnihilationSpectralModel)
    model._mDM = 1 * u.TeV
    model._z = 0
    model._k = 2
    model._factor = u.Quantity(1)
    model.channel = "b"

    model._primary_flux = real_flux
    assert model._primary_flux is real_flux


def test_primary_flux_validator_missing_impl():
    try:

        class BadModel(_PrimaryFluxValidator, SpectralModel):
            pass

        pytest.fail("Expected TypeError was not raised")
    except TypeError as e:
        assert "must implement" in str(e)


@requires_data()
def test_primary_flux_channel_mismatch_warns():
    mDM = 1 * u.TeV
    cont_flux = ContinuumPrimaryFlux(mDM=mDM, channel="W")

    with pytest.warns(UserWarning, match="does not match"):
        DarkMatterAnnihilationSpectralModel(
            mDM=mDM, channel="b", primary_flux=cont_flux
        )


def test_primary_flux_invalid_type():
    with pytest.raises(TypeError, match="primary_flux must be one of"):
        DarkMatterAnnihilationSpectralModel(
            mDM=1 * u.TeV, channel="b", primary_flux="not_a_flux_model"
        )


@requires_data()
def test_decay_expected_primary_flux_mass_is_half():
    import warnings

    mDM = 2 * u.TeV
    test_flux = ContinuumPrimaryFlux(mDM=1 * u.TeV, channel="b")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model = DarkMatterDecaySpectralModel(
            mDM=mDM, channel="b", primary_flux=test_flux
        )

    assert_quantity_allclose(model.primary_flux.mDM, mDM / 2)


def warnings_should_not_warn(category):
    import contextlib

    @contextlib.contextmanager
    def _cm():
        with pytest.warns(None) as record:
            yield
        for w in record:
            assert not issubclass(w.category, category), (
                f"Unexpected warning: {w.message}"
            )

    return _cm()


# ---------------------------------------------------------------------------
# k parameter (DarkMatterAnnihilationSpectralModel)
# ---------------------------------------------------------------------------


def test_invalid_k_value():
    with pytest.raises(ValueError, match="k must be 2 .Majorana. or 4 .Dirac."):
        DarkMatterAnnihilationSpectralModel(mDM=1 * u.TeV, channel="b", k=3)


@requires_data()
@pytest.mark.parametrize("k", [2, 4])
def test_k_value_roundtrip(k):
    model = DarkMatterAnnihilationSpectralModel(mDM=1 * u.TeV, channel="b", k=k)
    data = model.to_dict()
    new_model = DarkMatterAnnihilationSpectralModel.from_dict(data)
    assert new_model.k == k


# ---------------------------------------------------------------------------
# Full spectral models (annihilation / decay) with default ContinuumPrimaryFlux
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_class, factor_unit, expected_flux, expected_dnde, source",
    [
        (
            DarkMatterAnnihilationSpectralModel,
            "GeV2 cm-5",
            6.19575457e-14,
            2.97831615e-16,
            None,
        ),
        (
            DarkMatterDecaySpectralModel,
            "GeV cm-2",
            0.0480497420860496,
            0.000231,
            "pppc4",
        ),
        (
            DarkMatterAnnihilationSpectralModel,
            "GeV2 cm-5",
            6.03197683e-14,
            3.52065879e-16,
            "cosmixs",
        ),
        (DarkMatterDecaySpectralModel, "GeV cm-2", 0.04676, 0.00027292, "cosmixs"),
    ],
)
@requires_data()
def test_dm_spectral_model(
    tmp_path, factor_unit, model_class, expected_flux, expected_dnde, source
):
    channel = "b"
    mass = 5 * u.TeV
    factor = 3.41e19 * u.Unit(factor_unit)
    energy_min = 0.01 * u.TeV
    energy_max = 10 * u.TeV

    pf = ContinuumPrimaryFlux(mass, channel, source=source)
    model = model_class(mDM=mass, channel=channel, factor=factor, primary_flux=pf)

    flux = model.integral(energy_min=energy_min, energy_max=energy_max).to("cm-2 s-1")

    if model_class is DarkMatterDecaySpectralModel:
        dnde = model.evaluate(
            energy=1 * u.TeV, scale=1, lifetime=model.lifetime.quantity
        ).to("cm-2 s-1 TeV-1")
    else:
        dnde = model.evaluate(energy=1 * u.TeV, scale=1).to("cm-2 s-1 TeV-1")

    sky_model = SkyModel(spectral_model=model, name="skymodel")
    models = Models([sky_model])
    filename = tmp_path / "model.yaml"
    models.write(filename, overwrite=True)
    new_models = Models.read(filename)

    assert_quantity_allclose(flux.value, expected_flux, rtol=1e-2)
    assert_quantity_allclose(dnde.value, expected_dnde, rtol=1e-2)

    loaded = new_models[0].spectral_model
    assert loaded.channel == model.channel
    assert loaded.z == model.z
    assert_allclose(loaded.factor.value, model.factor.value, rtol=1e-2)
    assert_quantity_allclose(loaded.mDM, model.mDM, rtol=1e-2)


@requires_data()
def test_dm_annihilation_to_dict_structure():
    model = DarkMatterAnnihilationSpectralModel(mDM=1 * u.TeV, channel="b")
    data = model.to_dict()

    assert data["spectral"]["channel"] == "b"
    assert data["spectral"]["k"] == 2
    assert "primary_flux" in data["spectral"]
    assert data["spectral"]["primary_flux"]["type"] == "ContinuumPrimaryFlux"

    new_model = DarkMatterAnnihilationSpectralModel.from_dict(data)
    assert new_model.channel == model.channel
    assert new_model.k == model.k


@requires_data()
def test_dm_decay_default_lifetime():
    model = DarkMatterDecaySpectralModel(mDM=1 * u.TeV, channel="b")
    assert_quantity_allclose(model.lifetime.quantity, 4.3e17 * u.s)


@requires_data()
def test_dm_decay_to_from_dict_roundtrip_lifetime():
    model = DarkMatterDecaySpectralModel(
        mDM=1 * u.TeV, channel="b", lifetime=1e20 * u.s
    )
    data = model.to_dict()
    new_model = DarkMatterDecaySpectralModel.from_dict(data)
    assert_quantity_allclose(new_model.lifetime.quantity, model.lifetime.quantity)


@requires_data()
def test_unknown_primary_flux_type_in_from_dict():
    model = DarkMatterAnnihilationSpectralModel(mDM=1 * u.TeV, channel="b")
    data = model.to_dict()
    data["spectral"]["primary_flux"]["type"] = "NotARealFluxType"

    with pytest.raises(ValueError, match="Unknown primary_flux type"):
        DarkMatterAnnihilationSpectralModel.from_dict(data)


@requires_data()
def test_decay_expected_primary_flux_mass_direct():
    model = DarkMatterDecaySpectralModel(mDM=2 * u.TeV, channel="b")
    result = model._expected_primary_flux_mass()
    assert_quantity_allclose(result, 1 * u.TeV)


@requires_data()
def test_unknown_primary_flux_type_in_decay_from_dict():
    model = DarkMatterDecaySpectralModel(mDM=1 * u.TeV, channel="b")
    data = model.to_dict()
    data["spectral"]["primary_flux"]["type"] = "NotARealFluxType"

    with pytest.raises(ValueError, match="Unknown primary_flux type"):
        DarkMatterDecaySpectralModel.from_dict(data)


@requires_data()
def test_dm_annihilation_evaluation_on_dataset():
    """Model can be set on a MapDataset and produces finite positive flux."""
    from gammapy.datasets import MapDataset
    from gammapy.maps import MapAxis, WcsGeom
    from gammapy.modeling.models import PointSpatialModel

    energy_axis = MapAxis.from_edges(
        [0.01, 0.1, 1.0], unit="TeV", name="energy", interp="log"
    )
    geom = WcsGeom.create(
        skydir=(0, 0), binsz=0.1, width=(1, 1), frame="galactic", axes=[energy_axis]
    )
    pf = ContinuumPrimaryFlux(1 * u.TeV, "b")
    spectral_model = DarkMatterAnnihilationSpectralModel(
        mDM=1 * u.TeV,
        channel="b",
        factor=3.41e19 * u.Unit("GeV2 cm-5"),
        primary_flux=pf,
    )
    sky_model = SkyModel(
        spectral_model=spectral_model,
        spatial_model=PointSpatialModel(lon_0="0 deg", lat_0="0 deg", frame="galactic"),
        name="dm_anni",
    )
    dataset = MapDataset.create(geom, name="test_anni")
    dataset.models = [sky_model]

    # npred is zero without exposure, but must not raise
    assert np.all(np.isfinite(dataset.npred().data))

    # energies well below DM mass must give finite positive flux
    flux = spectral_model([0.01, 0.1, 0.5] * u.TeV)
    assert np.all(np.isfinite(flux.value))
    assert np.all(flux.value > 0)


@requires_data()
def test_dm_decay_evaluation_on_dataset():
    """Decay model can be set on a MapDataset and produces finite positive flux."""
    from gammapy.datasets import MapDataset
    from gammapy.maps import MapAxis, WcsGeom
    from gammapy.modeling.models import PointSpatialModel

    energy_axis = MapAxis.from_edges(
        [0.01, 0.1, 1.0], unit="TeV", name="energy", interp="log"
    )
    geom = WcsGeom.create(
        skydir=(0, 0), binsz=0.1, width=(1, 1), frame="galactic", axes=[energy_axis]
    )
    pf = ContinuumPrimaryFlux(0.5 * u.TeV, "b")
    spectral_model = DarkMatterDecaySpectralModel(
        mDM=1 * u.TeV, channel="b", factor=3.41e19 * u.Unit("GeV cm-2"), primary_flux=pf
    )
    sky_model = SkyModel(
        spectral_model=spectral_model,
        spatial_model=PointSpatialModel(lon_0="0 deg", lat_0="0 deg", frame="galactic"),
        name="dm_decay",
    )
    dataset = MapDataset.create(geom, name="test_decay")
    dataset.models = [sky_model]

    # npred is zero without exposure, but must not raise
    assert np.all(np.isfinite(dataset.npred().data))

    # energies well below DM mass must give finite positive flux
    flux = spectral_model([0.01, 0.1, 0.4] * u.TeV)
    assert np.all(np.isfinite(flux.value))
    assert np.all(flux.value > 0)
