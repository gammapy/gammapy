# Licensed under a 3-clause BSD style license - see LICENSE.rst
import sys
import warnings
from types import ModuleType, SimpleNamespace

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
    HDMSpectraPrimaryFlux,
    PrimaryFlux,
)
from gammapy.modeling.models import Models, SkyModel
from gammapy.utils.testing import assert_quantity_allclose, requires_data
from gammapy.astro.darkmatter.utils import add_factor_prior
from gammapy.modeling.models.prior import LogSpaceGaussianPrior


# ContinuumPrimaryFlux


@requires_data()
def test_continuum_primary_flux():
    with pytest.raises(ValueError):
        ContinuumPrimaryFlux(channel="Spam", mDM=1 * u.TeV)

    primflux = ContinuumPrimaryFlux(channel="W", mDM=1 * u.TeV, source="pppc4")
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


def _to_old_format(data):
    """Rewrite a serialized model in the pre-Parameter format."""
    spec = data["spectral"]
    spec["mass"] = spec.pop("mDM")
    pars = spec["parameters"]
    f = next(p for p in pars if p["name"] == "factor")
    pars.remove(f)
    spec["jfactor"] = u.Quantity(f["value"], f.get("unit", "")).to_string()
    return data


@requires_data()
@pytest.mark.parametrize(
    "annihilation, unit", [(True, "GeV2 cm-5"), (False, "GeV cm-2")]
)
def test_from_dict_old_field_names(annihilation, unit):
    model = DarkMatterSpectralModel(
        mDM=1 * u.TeV,
        channel="b",
        factor=3.41e19 * u.Unit(unit),
        annihilation=annihilation,
    )
    data = _to_old_format(model.to_dict())

    with pytest.warns(GammapyDeprecationWarning) as record:
        new_model = DarkMatterSpectralModel.from_dict(data)

    messages = [str(w.message) for w in record]
    assert any("'mass'" in m for m in messages)
    assert any("'jfactor'" in m for m in messages)
    assert_quantity_allclose(new_model.factor.quantity, model.factor.quantity)
    assert_quantity_allclose(new_model.mDM, model.mDM)
    assert new_model.channel == model.channel


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
        annihilation=True,
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
    assert loaded_model.factor.unit == model.factor.unit


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


@requires_data()
def test_invalid_factor():
    with pytest.raises(ValueError, match="factor must be positive"):
        DarkMatterSpectralModel(
            mDM=1 * u.TeV, channel="b", factor=-1 * u.Unit("GeV2 cm-5")
        )


# Full spectral models (annihilation / decay) with default ContinuumPrimaryFlux


@pytest.mark.parametrize(
    "factor_unit, expected_flux, expected_dnde, source, annihilation",
    [
        ("GeV2 cm-5", 6.03197683e-14, 3.52065879e-16, None, True),
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

    dnde = model.evaluate(energy=1 * u.TeV, scale=1, factor=factor).to("cm-2 s-1 TeV-1")

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
    model = DarkMatterSpectralModel(mDM=1 * u.TeV, channel="b", annihilation=True)
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
    with pytest.warns(GammapyDeprecationWarning) as record:
        new_model = DarkMatterSpectralModel.from_dict(data)
    messages = [str(w.message) for w in record]
    assert any("'mass'" in m for m in messages)
    assert_quantity_allclose(new_model.mDM, model.mDM)
    assert_allclose(new_model.factor.value, model.factor.value, rtol=1e-2)
    assert new_model.channel == model.channel


@requires_data()
def test_dm_annihilation_from_dict_both_old_field_names_warns_and_maps():
    """Dict using both old field names ('mass' and 'jfactor') at once must map both and warn for each."""
    model = DarkMatterSpectralModel(mDM=1 * u.TeV, channel="b")
    data = model.to_dict()
    data["spectral"]["mass"] = data["spectral"].pop("mDM")
    with pytest.warns(GammapyDeprecationWarning) as record:
        new_model = DarkMatterSpectralModel.from_dict(data)
    messages = [str(w.message) for w in record]
    assert any("'mass'" in m for m in messages)
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


@requires_data()
def test_factor_wrong_unit_raises():
    with pytest.raises(u.UnitConversionError, match="factor must be convertible"):
        DarkMatterSpectralModel(
            mDM=1 * u.TeV,
            channel="b",
            factor=3.41e19 * u.Unit("GeV cm-2"),
            annihilation=True,
        )


@requires_data()
def test_factor_prior_roundtrip(tmp_path):
    model = DarkMatterSpectralModel(
        mDM=1 * u.TeV, channel="b", factor=3.41e19 * u.Unit("GeV2 cm-5")
    )
    add_factor_prior(model, sigma=0.3)
    model.factor.min = model.factor.value / 10
    model.factor.max = model.factor.value * 10

    filename = tmp_path / "model.yaml"
    Models([SkyModel(spectral_model=model, name="dm")]).write(filename)
    loaded = Models.read(filename)[0].spectral_model

    assert loaded.factor.unit == model.factor.unit
    assert_allclose(loaded.factor.value, model.factor.value)
    assert not loaded.factor.frozen
    assert_allclose(loaded.factor.min, model.factor.min)
    assert_allclose(loaded.factor.max, model.factor.max)

    assert isinstance(loaded.factor.prior, LogSpaceGaussianPrior)
    assert_allclose(loaded.factor.prior.mu.value, model.factor.value)
    assert_allclose(loaded.factor.prior.sigma.value, 0.3 * np.log(10))
    assert loaded.scale.prior is None
    assert not loaded.scale.frozen


def test_dm_spectral_model_generic_primary_flux():
    """Any spectral model returning dN/dE can be used as primary flux."""
    from gammapy.modeling.models import TemplateSpectralModel

    energy = np.geomspace(1, 1000, 20) * u.GeV
    values = 1e-3 * (energy / u.GeV) ** -1.5 / u.GeV
    pf = TemplateSpectralModel(energy=energy, values=values)

    model = DarkMatterSpectralModel(
        mDM=1 * u.TeV,
        channel="b",
        factor=3.41e19 * u.Unit("GeV2 cm-5"),
        primary_flux=pf,
    )

    assert model.primary_flux is pf
    assert not hasattr(pf, "mDM")

    flux = model(10 * u.GeV)
    assert flux.unit.is_equivalent("cm-2 s-1 TeV-1")
    assert flux.value > 0


def test_hdm_primary_flux_masks_unsupported_x(monkeypatch, tmp_path):
    data_file = tmp_path / "HDMSpectra.hdf5"
    data_file.touch()
    calls = {}

    def fake_spec(finalstate, initialstate, x, mass, data, annihilation):
        calls["finalstate"] = finalstate
        calls["initialstate"] = initialstate
        calls["x"] = x.copy()
        calls["mass"] = mass
        calls["data"] = data
        calls["annihilation"] = annihilation
        return np.full_like(x, 2.0)

    hdm_module = ModuleType("HDMSpectra")
    hdm_module.HDMSpectra = SimpleNamespace(spec=fake_spec)
    monkeypatch.setitem(sys.modules, "HDMSpectra", hdm_module)

    mass = 1 * u.TeV
    flux = HDMSpectraPrimaryFlux(
        mDM=mass,
        initialstate="b",
        hdm_data_file=data_file,
    )
    energies = np.array([0.5e-6, 5e-5, 0.5, 1.1]) * mass

    with pytest.warns(UserWarning, match="below 1e-4"):
        actual = flux(energies)

    expected = np.array([0, 2 / 1000, 2 / 1000, 0]) / u.GeV
    assert_quantity_allclose(actual, expected)
    assert_allclose(calls["x"], [5e-5, 0.5])
    assert calls["mass"] == 1000
    assert calls["initialstate"] == "b"
    assert calls["finalstate"] == 22
    assert calls["data"] == data_file
    assert calls["annihilation"] is True


@pytest.mark.parametrize("path_kind", ["missing", "directory"])
def test_hdm_primary_flux_rejects_invalid_data_file(tmp_path, path_kind):
    data_file = tmp_path / "missing.hdf5" if path_kind == "missing" else tmp_path
    with pytest.raises(FileNotFoundError, match="HDMSpectra data file not found"):
        HDMSpectraPrimaryFlux(
            mDM=1 * u.TeV,
            initialstate="b",
            hdm_data_file=data_file,
        )


def test_hdm_primary_flux_units_shape_and_boundary_values(monkeypatch, tmp_path):
    data_file = tmp_path / "HDMSpectra.hdf5"
    data_file.touch()
    calls = {}

    def fake_spec(finalstate, initialstate, x, mass, data, annihilation):
        calls["x"] = x.copy()
        calls["mass"] = mass
        return 2 * x

    hdm_module = ModuleType("HDMSpectra")
    hdm_module.HDMSpectra = SimpleNamespace(spec=fake_spec)
    monkeypatch.setitem(sys.modules, "HDMSpectra", hdm_module)

    flux = HDMSpectraPrimaryFlux(
        mDM=1 * u.TeV,
        initialstate="b",
        hdm_data_file=data_file,
    )

    with pytest.warns(UserWarning, match="below 1e-4"):
        actual = flux(np.array([1e-6, 1]) * u.TeV)

    assert_allclose(calls["x"], [1e-6, 1])
    assert calls["mass"] == 1000
    assert actual.shape == (2,)
    assert_quantity_allclose(actual, np.array([2e-9, 2e-3]) / u.GeV)

    scalar = flux(500 * u.GeV)
    assert scalar.shape == ()
    assert_quantity_allclose(scalar, 1e-3 / u.GeV)


def test_hdm_primary_flux_all_out_of_range_returns_zero(monkeypatch, tmp_path):
    data_file = tmp_path / "HDMSpectra.hdf5"
    data_file.touch()
    calls = []

    def fake_spec(*args, **kwargs):
        calls.append((args, kwargs))
        return np.ones_like(args[2])

    hdm_module = ModuleType("HDMSpectra")
    hdm_module.HDMSpectra = SimpleNamespace(spec=fake_spec)
    monkeypatch.setitem(sys.modules, "HDMSpectra", hdm_module)

    flux = HDMSpectraPrimaryFlux(
        mDM=1 * u.TeV,
        initialstate="b",
        hdm_data_file=data_file,
    )
    actual = flux(np.array([0.5e-6, 1.01]) * u.TeV)

    assert_quantity_allclose(actual, np.zeros(2) / u.GeV)
    assert calls == []


def test_hdm_primary_flux_rejects_non_finite_energy(tmp_path):
    data_file = tmp_path / "HDMSpectra.hdf5"
    data_file.touch()
    flux = HDMSpectraPrimaryFlux(
        mDM=1 * u.TeV,
        initialstate="b",
        hdm_data_file=data_file,
    )

    with pytest.raises(ValueError, match="Energy values must be finite"):
        flux(np.nan * u.GeV)


def test_hdm_primary_flux_rejects_incompatible_energy_units(tmp_path):
    data_file = tmp_path / "HDMSpectra.hdf5"
    data_file.touch()
    flux = HDMSpectraPrimaryFlux(
        mDM=1 * u.TeV,
        initialstate="b",
        hdm_data_file=data_file,
    )

    with pytest.raises(u.UnitConversionError):
        flux(1 * u.s)


def test_hdm_primary_flux_missing_optional_dependency(monkeypatch, tmp_path):
    data_file = tmp_path / "HDMSpectra.hdf5"
    data_file.touch()
    monkeypatch.setitem(sys.modules, "HDMSpectra", None)
    flux = HDMSpectraPrimaryFlux(
        mDM=1 * u.TeV,
        initialstate="b",
        hdm_data_file=data_file,
    )

    with pytest.raises(ImportError, match=r"gammapy\[hdmspectra\]"):
        flux(500 * u.GeV)


def test_hdm_primary_flux_allowed_channels(monkeypatch, tmp_path):
    data_file = tmp_path / "HDMSpectra.hdf5"
    data_file.touch()
    hdm_module = ModuleType("HDMSpectra")
    hdm_module.__path__ = []
    pclid_module = ModuleType("HDMSpectra.pclid")
    pclid_module.str_i = np.array(["b", "W"])
    monkeypatch.setitem(sys.modules, "HDMSpectra", hdm_module)
    monkeypatch.setitem(sys.modules, "HDMSpectra.pclid", pclid_module)

    flux = HDMSpectraPrimaryFlux(
        mDM=1 * u.TeV,
        initialstate="b",
        hdm_data_file=data_file,
    )

    assert flux.allowed_channels() == ["b", "W"]


def test_hdm_dark_matter_model_requires_data_file():
    with pytest.raises(
        ValueError,
        match="hdm_data_file is required when source is 'hdmspectra'",
    ):
        DarkMatterSpectralModel(
            mDM=1 * u.TeV,
            channel="b",
            source="hdmspectra",
            factor=3.41e19 * u.Unit("GeV2 cm-5"),
        )


def test_hdm_primary_flux_serialization_roundtrip(tmp_path):
    data_file = tmp_path / "HDMSpectra.hdf5"
    data_file.touch()
    primary = HDMSpectraPrimaryFlux(
        mDM=1 * u.TeV,
        initialstate="b",
        finalstate="gamma",
        hdm_data_file=data_file,
    )

    restored_primary = HDMSpectraPrimaryFlux.from_dict(primary.to_dict())

    assert_quantity_allclose(restored_primary.mDM, primary.mDM)
    assert restored_primary.initialstate == primary.initialstate
    assert restored_primary.finalstate == primary.finalstate
    assert restored_primary.hdm_data_file == primary.hdm_data_file


def test_hdm_dark_matter_model_serialization_roundtrip(tmp_path):
    data_file = tmp_path / "HDMSpectra.hdf5"
    data_file.touch()
    model = DarkMatterSpectralModel(
        mDM=1 * u.TeV,
        channel="b",
        source="hdmspectra",
        factor=3.41e19 * u.Unit("GeV2 cm-5"),
        hdm_data_file=data_file,
    )
    model.scale.value = 2.5
    model.factor.frozen = False

    restored = DarkMatterSpectralModel.from_dict(model.to_dict())

    assert isinstance(restored.primary_flux, HDMSpectraPrimaryFlux)
    assert_quantity_allclose(restored.primary_flux.mDM, model.primary_flux.mDM)
    assert restored.primary_flux.initialstate == "b"
    assert restored.primary_flux.finalstate == 22
    assert restored.primary_flux.hdm_data_file == data_file
    assert restored.source == "hdmspectra"
    assert_quantity_allclose(restored.factor.quantity, model.factor.quantity)
    assert restored.factor.frozen is False
    assert_allclose(restored.scale.value, 2.5)
