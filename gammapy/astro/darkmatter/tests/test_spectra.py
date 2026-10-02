# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings

import astropy.units as u
import numpy as np
import pytest
from gammapy.utils.deprecation import GammapyDeprecationWarning
from astropy.table import Table
from numpy.testing import assert_allclose

from gammapy.astro.darkmatter import (
    ContinuumPrimaryFlux,
    DarkMatterSpectralModel,
    DarkMatterAnnihilationSpectralModel,
    DarkMatterDecaySpectralModel,
    PrimaryFlux,
)
from gammapy.modeling.models import Models, SkyModel, SpectralModel
from gammapy.utils.testing import assert_quantity_allclose, requires_data


# ContinuumPrimaryFlux


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
        DarkMatterSpectralModel(channel="W", mass=1 * u.TeV)


@requires_data()
def test_spectralclasses_deprecated():
    with pytest.warns(
        GammapyDeprecationWarning, match="DarkMatterAnnihilationSpectralModel"
    ):
        DarkMatterAnnihilationSpectralModel(channel="W", mDM=1 * u.TeV)

    with pytest.warns(GammapyDeprecationWarning, match="DarkMatterDecaySpectralModel"):
        DarkMatterDecaySpectralModel(channel="W", mDM=1 * u.TeV)


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
    with pytest.raises(FileNotFoundError, match="File not found"):
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

    with pytest.raises(KeyError, match="Source file is empty"):
        ContinuumPrimaryFlux(mDM=5 * u.TeV, channel="b", source=str(empty_file))


def test_custom_source_invalid_path():
    with pytest.raises(ValueError, match="Invalid source"):
        ContinuumPrimaryFlux(
            mDM=5 * u.TeV, channel="b", source="/nonexistent/path.ecsv"
        )


def test_source_non_string_raises_typeerror():
    with pytest.raises(TypeError, match="source must be"):
        ContinuumPrimaryFlux(mDM=1 * u.TeV, channel="b", source=123)


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

    model = DarkMatterSpectralModel(
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
    with pytest.raises(ValueError, match="is not available"):
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

    with pytest.raises(ValueError, match="is not available"):
        ContinuumPrimaryFlux(mDM=500 * u.GeV, channel="eL", source=str(custom_file))


def test_missing_data_file(monkeypatch):
    monkeypatch.setenv("GAMMAPY_DATA", "/fake/path/to/nowhere")
    with pytest.raises(FileNotFoundError, match="File not found"):
        ContinuumPrimaryFlux(mDM=1 * u.TeV, channel="b")


@requires_data()
def test_mDM_out_of_bounds():
    with pytest.raises(ValueError, match="is out of the bounds"):
        ContinuumPrimaryFlux(mDM=500 * u.TeV, channel="b")


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


@requires_data()
def test_decay_expected_primary_flux_mass_is_half():
    mDM = 2 * u.TeV
    test_flux = ContinuumPrimaryFlux(mDM=1 * u.TeV, channel="b")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model = DarkMatterSpectralModel(
            mDM=mDM, channel="b", primary_flux=test_flux, annihilation=False
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


def test_negative_redshift():
    with pytest.raises(ValueError, match="Redshift z must be >= 0"):
        DarkMatterSpectralModel(mDM=1 * u.TeV, channel="b", z=-1)


# k parameter (DarkMatterSpectralModel)


def test_invalid_k_value():
    with pytest.raises(ValueError, match="k must be 2 .Majorana. or 4 .Dirac."):
        DarkMatterSpectralModel(mDM=1 * u.TeV, channel="b", k=3)


@requires_data()
@pytest.mark.parametrize("k", [2, 4])
def test_k_value_roundtrip(k):
    model = DarkMatterSpectralModel(mDM=1 * u.TeV, channel="b", k=k)
    data = model.to_dict()
    new_model = DarkMatterSpectralModel.from_dict(data)
    assert new_model.k == k


def test_invalid_factor():
    with pytest.raises(
        ValueError, match="The astrophysical factor must be strictly positive."
    ):
        DarkMatterSpectralModel(
            mDM=1 * u.TeV, channel="b", factor=-1 * u.Unit("GeV2 cm-5")
        )


# Full spectral models (annihilation / decay) with default ContinuumPrimaryFlux


@pytest.mark.parametrize(
    "factor_unit, expected_flux, expected_dnde, source, annihilation",
    [
        ("GeV2 cm-5", 6.19575457e-14, 2.97831615e-16, None, True),
        ("GeV cm-2", 3.209234e-2, 2.33485775e-5, "pppc4", False),
        ("GeV2 cm-5", 6.03197683e-14, 3.52065879e-16, "cosmixs", True),
        ("GeV cm-2", 0.031677, 2.77187e-05, "cosmixs", False),
    ],
)
@requires_data()
def test_dm_spectral_model(
    tmp_path, factor_unit, expected_flux, expected_dnde, source, annihilation
):
    channel = "b"
    mass = 5 * u.TeV
    factor = 3.41e19 * u.Unit(factor_unit)
    energy_min = 0.01 * u.TeV
    energy_max = 10 * u.TeV

    pf = ContinuumPrimaryFlux(mass / 2.0, channel, source=source)
    model = DarkMatterSpectralModel(
        mDM=mass,
        channel=channel,
        factor=factor,
        primary_flux=pf,
        annihilation=annihilation,
    )

    flux = model.integral(energy_min=energy_min, energy_max=energy_max).to("cm-2 s-1")

    if annihilation is False:
        dnde = model.evaluate(energy=1 * u.TeV, scale=1).to("cm-2 s-1 TeV-1")
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
    model = DarkMatterSpectralModel(mDM=1 * u.TeV, channel="b", annihilation=True)
    data = model.to_dict()

    assert data["spectral"]["channel"] == "b"
    assert data["spectral"]["k"] == 2
    assert "primary_flux" in data["spectral"]
    assert data["spectral"]["primary_flux"]["type"] == "ContinuumPrimaryFlux"

    new_model = DarkMatterSpectralModel.from_dict(data)
    assert new_model.channel == model.channel
    assert new_model.k == model.k


@requires_data()
def test_unknown_primary_flux_type_in_from_dict():
    model = DarkMatterSpectralModel(mDM=1 * u.TeV, channel="b")
    data = model.to_dict()
    data["spectral"]["primary_flux"]["type"] = "NotARealFluxType"

    with pytest.raises(ValueError, match="Unknown primary_flux type"):
        DarkMatterSpectralModel.from_dict(data)


@requires_data()
def test_decay_expected_primary_flux_mass_direct():
    model = DarkMatterSpectralModel(mDM=2 * u.TeV, channel="b", annihilation=False)
    result = model._expected_primary_flux_mass
    assert_quantity_allclose(result, 1 * u.TeV)


@requires_data()
def test_unknown_primary_flux_type_in_decay_from_dict():
    model = DarkMatterSpectralModel(mDM=1 * u.TeV, channel="b", annihilation=False)
    data = model.to_dict()
    data["spectral"]["primary_flux"]["type"] = "NotARealFluxType"

    with pytest.raises(ValueError, match="Unknown primary_flux type"):
        DarkMatterSpectralModel.from_dict(data)


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
    spectral_model = DarkMatterSpectralModel(
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
    spectral_model = DarkMatterSpectralModel(
        mDM=1 * u.TeV,
        channel="b",
        factor=3.41e19 * u.Unit("GeV cm-2"),
        primary_flux=pf,
        annihilation=False,
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


# Backport compatibility tests for old field names in serialized dicts
@requires_data()
def test_dm_decay_from_dict_missing_primary_flux_key():
    """A dict serialized before 'primary_flux' existed must still be loadable via from_dict, reconstructing the
    primary flux from the legacy flat 'source'/'mapping_dict' fields."""
    model = DarkMatterSpectralModel(mDM=1 * u.TeV, channel="b", annihilation=False)
    data = model.to_dict()
    data["spectral"].pop("primary_flux", None)
    new_model = DarkMatterSpectralModel.from_dict(data)
    assert_quantity_allclose(new_model.mDM, model.mDM)
    assert_allclose(new_model.factor.value, model.factor.value, rtol=1e-2)
    assert new_model.channel == model.channel


@requires_data()
def test_dm_annihilation_from_dict_missing_primary_flux_key():
    """Backward-compatibility check for DarkMatterSpectralModel."""
    model = DarkMatterSpectralModel(mDM=1 * u.TeV, channel="b")
    data = model.to_dict()
    data["spectral"].pop("primary_flux", None)
    new_model = DarkMatterSpectralModel.from_dict(data)
    assert_quantity_allclose(new_model.mDM, model.mDM)
    assert_allclose(new_model.factor.value, model.factor.value, rtol=1e-2)
    assert new_model.channel == model.channel


@requires_data()
def test_dm_decay_from_dict_missing_primary_flux_and_old_field_names():
    """Dict with both no 'primary_flux' key AND old field names ('mass' instead of 'mDM')."""
    model = DarkMatterSpectralModel(mDM=1 * u.TeV, channel="b", annihilation=False)
    data = model.to_dict()
    data["spectral"].pop("primary_flux", None)
    data["spectral"]["mass"] = data["spectral"].pop("mDM")
    with pytest.warns(GammapyDeprecationWarning, match="'mass'"):
        new_model = DarkMatterSpectralModel.from_dict(data)
    assert_quantity_allclose(new_model.mDM, model.mDM)
    assert new_model.channel == model.channel


@requires_data()
def test_dm_annihilation_from_dict_missing_primary_flux_and_old_field_names():
    """Dict with both no 'primary_flux' key AND old field names ('mass' instead of 'mDM')."""
    model = DarkMatterSpectralModel(mDM=1 * u.TeV, channel="b", annihilation=False)
    data = model.to_dict()
    data["spectral"].pop("primary_flux", None)
    data["spectral"]["mass"] = data["spectral"].pop("mDM")
    with pytest.warns(GammapyDeprecationWarning, match="'mass'"):
        new_model = DarkMatterSpectralModel.from_dict(data)
    assert_quantity_allclose(new_model.mDM, model.mDM)
    assert new_model.channel == model.channel


@requires_data()
def test_dm_decay_from_dict_both_old_field_names_warns_and_maps():
    """Dict using both old field names ('mass' and 'jfactor') at once must map both and warn for each."""
    model = DarkMatterSpectralModel(mDM=1 * u.TeV, channel="b", annihilation=False)
    data = model.to_dict()
    data["spectral"]["mass"] = data["spectral"].pop("mDM")
    data["spectral"]["jfactor"] = data["spectral"].pop("factor")
    with pytest.warns(GammapyDeprecationWarning) as record:
        new_model = DarkMatterSpectralModel.from_dict(data)
    messages = [str(w.message) for w in record]
    assert any("'mass'" in m for m in messages)
    assert any("'jfactor'" in m for m in messages)
    assert_quantity_allclose(new_model.mDM, model.mDM)
    assert_allclose(new_model.factor.value, model.factor.value, rtol=1e-2)
    assert new_model.channel == model.channel


@requires_data()
def test_dm_annihilation_from_dict_both_old_field_names_warns_and_maps():
    """Dict using both old field names ('mass' and 'jfactor') at once must map both and warn for each."""
    model = DarkMatterSpectralModel(mDM=1 * u.TeV, channel="b")
    data = model.to_dict()
    data["spectral"]["mass"] = data["spectral"].pop("mDM")
    data["spectral"]["jfactor"] = data["spectral"].pop("factor")
    with pytest.warns(GammapyDeprecationWarning) as record:
        new_model = DarkMatterSpectralModel.from_dict(data)
    messages = [str(w.message) for w in record]
    assert any("'mass'" in m for m in messages)
    assert any("'jfactor'" in m for m in messages)
    assert_quantity_allclose(new_model.mDM, model.mDM)
    assert_allclose(new_model.factor.value, model.factor.value, rtol=1e-2)
    assert new_model.channel == model.channel


@requires_data()
def test_dm_decay_from_dict_unknown_primary_flux_type_raises():
    model = DarkMatterSpectralModel(mDM=1 * u.TeV, channel="b", annihilation=False)
    data = model.to_dict()
    data["spectral"]["primary_flux"]["type"] = "not_a_real_type"
    with pytest.raises(ValueError, match="Unknown primary_flux type"):
        DarkMatterSpectralModel.from_dict(data)


@requires_data()
def test_dm_decay_from_dict_missing_primary_flux_key_custom_source():
    model = DarkMatterSpectralModel(
        mDM=1 * u.TeV, channel="b", source="cosmixs", annihilation=False
    )
    data = model.to_dict()
    data["spectral"].pop("primary_flux", None)
    new_model = DarkMatterSpectralModel.from_dict(data)
    assert new_model.primary_flux.source == "cosmixs"


@requires_data()
def test_backward_compat_old_annihilation_dict_via_registry():
    """Test for serialization of an annihilation dict that was serialized before the
    'annihilation' field existed (old DarkMatterDecaySpectralModel format, pre-unification).
    """
    model = DarkMatterSpectralModel(mDM=1 * u.TeV, channel="b", k=2)
    data = model.to_dict()
    data["spectral"]["type"] = "DarkMatterAnnihilationSpectralModel"
    data["spectral"].pop("annihilation", None)

    new_model = DarkMatterSpectralModel.from_dict(data)

    assert new_model.annihilation is True
    assert new_model.k == 2
    assert_quantity_allclose(new_model.mDM, model.mDM)
    assert new_model.channel == model.channel


@requires_data()
def test_backward_compat_old_decay_dict_direct_base_class():
    """Test for serialization of a decay dict that was serialized before the
    'annihilation' field existed (old DarkMatterDecaySpectralModel format, pre-unification).
    """
    model = DarkMatterSpectralModel(mDM=1 * u.TeV, channel="b", annihilation=False)
    data = model.to_dict()
    data["spectral"]["type"] = "DarkMatterDecaySpectralModel"
    data["spectral"].pop("annihilation", None)
    data["spectral"].pop("k", None)

    new_model = DarkMatterSpectralModel.from_dict(data)

    assert new_model.annihilation is False
    assert new_model.k is None
    assert_quantity_allclose(new_model.mDM, model.mDM)
    assert new_model.channel == model.channel
