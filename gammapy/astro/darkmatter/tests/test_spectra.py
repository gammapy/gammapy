# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path
from unittest.mock import patch

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from numpy.testing import assert_allclose

from gammapy.astro.darkmatter import (
    DarkMatterAnnihilationSpectralModel,
    DarkMatterDecaySpectralModel,
    PrimaryFlux,
)
from gammapy.modeling.models import GaussianPrior, Models, SkyModel
from gammapy.utils.testing import assert_quantity_allclose, requires_data


@requires_data()
def test_primary_flux():
    with pytest.raises(ValueError):
        PrimaryFlux(channel="Spam", mDM=1 * u.TeV)

    with pytest.raises(ValueError):
        PrimaryFlux(channel="b", mDM=1000 * u.TeV)

    primflux = PrimaryFlux(channel="W", mDM=1 * u.TeV)
    actual = primflux(500 * u.GeV)
    desired = 9.3319318e-05 / u.GeV
    assert_quantity_allclose(actual, desired)


def test_primary_flux_source_case_insensitive():
    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            PrimaryFlux(mDM=1 * u.TeV, channel="b", source="PPPC4")


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
            PrimaryFlux(
                channel="aZ",
                mDM=mass * u.TeV,
                source=source,
                mapping_dict={"mDM": "mDM"},
            )
        return
    primflux = PrimaryFlux(channel="W", mDM=mass * u.TeV, source=source)
    actual = primflux(500 * u.GeV)
    assert_quantity_allclose(actual, expected_flux / u.GeV, rtol=1e-5)


@pytest.mark.parametrize(
    "model_class, jfactor_unit, expected_flux, expected_dnde, source",
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
            3.209234e-2,
            2.33485775e-05,
            "pppc4",
        ),
        (
            DarkMatterAnnihilationSpectralModel,
            "GeV2 cm-5",
            6.03197683e-14,
            3.52065879e-16,
            "cosmixs",
        ),
        (DarkMatterDecaySpectralModel, "GeV cm-2", 0.031677, 2.771877e-05, "cosmixs"),
    ],
)
@requires_data()
def test_dm_spectral_model(
    tmpdir, jfactor_unit, model_class, expected_flux, expected_dnde, source
):
    channel = "b"
    mass = 5 * u.TeV
    jfactor = 3.41e19 * u.Unit(jfactor_unit)
    energy_min = 0.01 * u.TeV
    energy_max = 10 * u.TeV

    model = model_class(mass=mass, channel=channel, jfactor=jfactor, source=source)
    flux = model.integral(energy_min=energy_min, energy_max=energy_max).to("cm-2 s-1")
    dnde = model.evaluate(energy=1 * u.TeV, scale=1).to("cm-2 s-1 TeV-1")

    sky_model = SkyModel(spectral_model=model, name="skymodel")
    models = Models([sky_model])
    filename = tmpdir / "model.yaml"
    models.write(filename, overwrite=True)
    new_models = Models.read(filename)

    assert_quantity_allclose(flux.value, expected_flux, rtol=1e-3)
    assert_quantity_allclose(dnde.value, expected_dnde, rtol=1e-3)

    assert new_models[0].spectral_model.channel == model.channel
    assert new_models[0].spectral_model.z == model.z
    assert_allclose(new_models[0].spectral_model.jfactor.value, model.jfactor.value)
    assert new_models[0].spectral_model.mass.value == 5
    assert new_models[0].spectral_model.mass.unit == u.TeV


@requires_data()
def test_primary_flux_cosmixs():
    with pytest.raises(ValueError):
        PrimaryFlux(channel="Spam", mDM=1 * u.TeV)

    primflux = PrimaryFlux(
        channel="b", mDM=1 * u.TeV, source="cosmixs", mapping_dict={"mDM": "mDM"}
    )
    actual = primflux(500 * u.GeV)
    desired = 1.842842e-05 / u.GeV
    assert_quantity_allclose(actual, desired, rtol=1e-4)

    with pytest.raises(ValueError):
        PrimaryFlux(channel="q", mDM=1 * u.TeV, source="cosmixs")
    with pytest.raises(ValueError):
        PrimaryFlux(channel="V->e", mDM=1 * u.TeV, source="cosmixs")
    with pytest.raises(ValueError):
        PrimaryFlux(channel="V->mu", mDM=1 * u.TeV, source="cosmixs")
    with pytest.raises(ValueError):
        PrimaryFlux(channel="V->tau", mDM=1 * u.TeV, source="cosmixs")

    with pytest.raises(ValueError):
        PrimaryFlux(channel="d", mDM=1 * u.TeV, source="pppc4")
    with pytest.raises(ValueError):
        PrimaryFlux(channel="u", mDM=1 * u.TeV, source="pppc4")
    with pytest.raises(ValueError):
        PrimaryFlux(channel="s", mDM=1 * u.TeV, source="pppc4")


# ─── DarkMatterAnnihilationSpectralModel — nuisance ──────────────────────────


@requires_data()
def test_annihilation_no_sigma_frozen():
    """Without sigma, log10_jfactor must stay frozen."""
    jfactor = 3.41e19 * u.Unit("GeV2 cm-5")
    model = DarkMatterAnnihilationSpectralModel(
        mass=5 * u.TeV, channel="b", jfactor=jfactor
    )
    assert model.log10_jfactor.frozen is True
    assert model.log10_jfactor.prior is None


@requires_data()
def test_annihilation_sigma_zero_equivalent_to_none():
    """sigma=0.0 must behave identically to sigma=None."""
    jfactor = 3.41e19 * u.Unit("GeV2 cm-5")
    model = DarkMatterAnnihilationSpectralModel(
        mass=5 * u.TeV, channel="b", jfactor=jfactor, sigma=0.0
    )
    assert model.log10_jfactor.frozen is True
    assert model.log10_jfactor.prior is None


@requires_data()
def test_annihilation_with_sigma_unfreezes():
    """With sigma, log10_jfactor must be free and have a prior."""
    jfactor = 3.41e19 * u.Unit("GeV2 cm-5")
    model = DarkMatterAnnihilationSpectralModel(
        mass=5 * u.TeV, channel="b", jfactor=jfactor, sigma=0.2
    )
    assert model.log10_jfactor.frozen is False
    assert model.log10_jfactor.prior is not None
    assert_allclose(model.log10_jfactor.value, np.log10(3.41e19), rtol=1e-6)


@requires_data()
def test_annihilation_prior_is_gaussian_with_correct_params():
    """Prior must be GaussianPrior with mu=log10(J_obs) and correct sigma."""
    jfactor = 3.41e19 * u.Unit("GeV2 cm-5")
    sigma = 0.3
    model = DarkMatterAnnihilationSpectralModel(
        mass=5 * u.TeV, channel="b", jfactor=jfactor, sigma=sigma
    )
    assert isinstance(model.log10_jfactor.prior, GaussianPrior)
    assert_allclose(model.log10_jfactor.prior.mu.value, np.log10(3.41e19), rtol=1e-6)
    assert_allclose(model.log10_jfactor.prior.sigma.value, sigma, rtol=1e-6)


@requires_data()
def test_annihilation_prior_bounds():
    """log10_jfactor bounds must be ±5*sigma around the observed value."""
    jfactor = 3.41e19 * u.Unit("GeV2 cm-5")
    sigma = 0.5
    log10_j_obs = np.log10(3.41e19)
    model = DarkMatterAnnihilationSpectralModel(
        mass=5 * u.TeV, channel="b", jfactor=jfactor, sigma=sigma
    )
    assert_allclose(model.log10_jfactor.min, log10_j_obs - 5 * sigma, rtol=1e-6)
    assert_allclose(model.log10_jfactor.max, log10_j_obs + 5 * sigma, rtol=1e-6)


@requires_data()
def test_annihilation_sigma_validates():
    jfactor = 3.41e19 * u.Unit("GeV2 cm-5")
    with pytest.raises(ValueError):
        DarkMatterAnnihilationSpectralModel(
            mass=5 * u.TeV, channel="b", jfactor=jfactor, sigma=-0.1
        )
    with pytest.raises(TypeError):
        DarkMatterAnnihilationSpectralModel(
            mass=5 * u.TeV, channel="b", jfactor=jfactor, sigma="bad"
        )


@requires_data()
def test_annihilation_evaluate_uses_log10_jfactor():
    """evaluate() must scale correctly when log10_jfactor is shifted by 1 dex."""
    jfactor = 3.41e19 * u.Unit("GeV2 cm-5")
    model = DarkMatterAnnihilationSpectralModel(
        mass=5 * u.TeV, channel="b", jfactor=jfactor, sigma=0.5
    )
    log10_j = np.log10(3.41e19)
    flux_nominal = model.evaluate(energy=1 * u.TeV, scale=1, log10_jfactor=log10_j)
    flux_10x = model.evaluate(energy=1 * u.TeV, scale=1, log10_jfactor=log10_j + 1)
    assert_allclose(flux_10x.value / flux_nominal.value, 10.0, rtol=1e-5)


@requires_data()
def test_annihilation_jfactor_array_raises():
    """Passing a map (array) as jfactor must raise ValueError."""
    jfactor_map = np.array([3.41e19, 3.41e18]) * u.Unit("GeV2 cm-5")
    with pytest.raises(ValueError, match="scalar Quantity"):
        DarkMatterAnnihilationSpectralModel(
            mass=5 * u.TeV, channel="b", jfactor=jfactor_map
        )


@requires_data()
def test_annihilation_serialization_with_sigma(tmpdir):
    """sigma must survive a YAML round-trip."""
    jfactor = 3.41e19 * u.Unit("GeV2 cm-5")
    model = DarkMatterAnnihilationSpectralModel(
        mass=5 * u.TeV, channel="b", jfactor=jfactor, sigma=0.3
    )
    sky_model = SkyModel(spectral_model=model, name="skymodel")
    models = Models([sky_model])
    filename = tmpdir / "model_sigma.yaml"
    models.write(filename, overwrite=True)
    new_models = Models.read(filename)
    new_model = new_models[0].spectral_model
    assert_allclose(new_model._sigma, 0.3, rtol=1e-6)
    assert new_model.log10_jfactor.frozen is False
    assert new_model.log10_jfactor.prior is not None


# ─── DarkMatterDecaySpectralModel — nuisance ─────────────────────────────────


@requires_data()
def test_decay_no_sigma_frozen():
    """Without sigma, log10_jfactor must stay frozen."""
    jfactor = 3.41e19 * u.Unit("GeV cm-2")
    model = DarkMatterDecaySpectralModel(mass=5 * u.TeV, channel="b", jfactor=jfactor)
    assert model.log10_jfactor.frozen is True
    assert model.log10_jfactor.prior is None


@requires_data()
def test_decay_sigma_zero_equivalent_to_none():
    """sigma=0.0 must behave identically to sigma=None."""
    jfactor = 3.41e19 * u.Unit("GeV cm-2")
    model = DarkMatterDecaySpectralModel(
        mass=5 * u.TeV, channel="b", jfactor=jfactor, sigma=0.0
    )
    assert model.log10_jfactor.frozen is True
    assert model.log10_jfactor.prior is None


@requires_data()
def test_decay_with_sigma_unfreezes():
    """With sigma, log10_jfactor must be free and have a prior."""
    jfactor = 3.41e19 * u.Unit("GeV cm-2")
    model = DarkMatterDecaySpectralModel(
        mass=5 * u.TeV, channel="b", jfactor=jfactor, sigma=0.2
    )
    assert model.log10_jfactor.frozen is False
    assert model.log10_jfactor.prior is not None
    assert_allclose(model.log10_jfactor.value, np.log10(3.41e19), rtol=1e-6)


@requires_data()
def test_decay_prior_is_gaussian_with_correct_params():
    """Prior must be GaussianPrior with mu=log10(J_obs) and correct sigma."""
    jfactor = 3.41e19 * u.Unit("GeV cm-2")
    sigma = 0.3
    model = DarkMatterDecaySpectralModel(
        mass=5 * u.TeV, channel="b", jfactor=jfactor, sigma=sigma
    )
    assert isinstance(model.log10_jfactor.prior, GaussianPrior)
    assert_allclose(model.log10_jfactor.prior.mu.value, np.log10(3.41e19), rtol=1e-6)
    assert_allclose(model.log10_jfactor.prior.sigma.value, sigma, rtol=1e-6)


@requires_data()
def test_decay_prior_bounds():
    """log10_jfactor bounds must be ±5*sigma around the observed value."""
    jfactor = 3.41e19 * u.Unit("GeV cm-2")
    sigma = 0.2
    log10_j_obs = np.log10(3.41e19)
    model = DarkMatterDecaySpectralModel(
        mass=5 * u.TeV, channel="b", jfactor=jfactor, sigma=sigma
    )
    assert_allclose(model.log10_jfactor.min, log10_j_obs - 5 * sigma, rtol=1e-6)
    assert_allclose(model.log10_jfactor.max, log10_j_obs + 5 * sigma, rtol=1e-6)


@requires_data()
def test_decay_evaluate_uses_log10_jfactor():
    """evaluate() must scale correctly when log10_jfactor is shifted by 1 dex."""
    jfactor = 3.41e19 * u.Unit("GeV cm-2")
    model = DarkMatterDecaySpectralModel(
        mass=5 * u.TeV, channel="b", jfactor=jfactor, sigma=0.5
    )
    log10_j = np.log10(3.41e19)
    flux_nominal = model.evaluate(energy=1 * u.TeV, scale=1, log10_jfactor=log10_j)
    flux_10x = model.evaluate(energy=1 * u.TeV, scale=1, log10_jfactor=log10_j + 1)
    assert_allclose(flux_10x.value / flux_nominal.value, 10.0, rtol=1e-5)


@requires_data()
def test_decay_jfactor_array_raises():
    """Passing a map (array) as jfactor must raise ValueError."""
    jfactor_map = np.array([3.41e19, 3.41e18]) * u.Unit("GeV cm-2")
    with pytest.raises(ValueError, match="scalar Quantity"):
        DarkMatterDecaySpectralModel(mass=5 * u.TeV, channel="b", jfactor=jfactor_map)


@requires_data()
def test_decay_serialization_with_sigma(tmpdir):
    """sigma must survive a YAML round-trip."""
    jfactor = 3.41e19 * u.Unit("GeV cm-2")
    model = DarkMatterDecaySpectralModel(
        mass=5 * u.TeV, channel="b", jfactor=jfactor, sigma=0.3
    )
    sky_model = SkyModel(spectral_model=model, name="skymodel")
    models = Models([sky_model])
    filename = tmpdir / "model_sigma.yaml"
    models.write(filename, overwrite=True)
    new_models = Models.read(filename)
    new_model = new_models[0].spectral_model
    assert_allclose(new_model._sigma, 0.3, rtol=1e-6)
    assert new_model.log10_jfactor.frozen is False
    assert new_model.log10_jfactor.prior is not None
@requires_data()
def test_primary_flux_cosmixs_channel_q_error():
    """Covers the branch raising ValueError for channel='q' with source='cosmixs'."""
    with pytest.raises(ValueError, match="The channel q is not available in cosmixs"):
        PrimaryFlux(channel="q", mDM=1 * u.TeV, source="cosmixs")


def test_custom_source_file_empty(tmp_path):
    """Test that an empty custom source file raises the correct error."""
    empty_file = tmp_path / "empty_spectra.dat"
    empty_file.touch()

    with pytest.raises(KeyError, match="Source file is empty."):
        DarkMatterAnnihilationSpectralModel(
            mass=5 * u.TeV, channel="b", source=str(empty_file)
        )


def test_custom_source_file_unrecognized_format(tmp_path):
    """A file with content that Table.read cannot parse should raise a
    clear error, regardless of its extension (extensions are no longer
    whitelisted; Table.read decides what is readable)."""
    bad_file = tmp_path / "unreadable_spectra.dl2"
    bad_file.write_text("this is not a valid table format §§§ %%% ???")

    with pytest.raises(Exception):
        DarkMatterAnnihilationSpectralModel(
            mass=5 * u.TeV, channel="b", source=str(bad_file)
        )


def test_custom_source_fits_extension_supported(tmp_path):
    """A .fits file should be readable as a custom source (previously
    excluded by the removed extension whitelist)."""
    fits_file = tmp_path / "custom_spectra.fits"
    t = Table(
        {
            "mDM": [500.0, 500.0, 1000.0, 1000.0] * u.GeV,
            "Log[10,x]": [-3.0, -2.0, -3.0, -2.0],
            "b": [1e-15, 1e-16, 1e-15, 1e-16],
        }
    )
    t.write(fits_file, format="fits")

    model = DarkMatterAnnihilationSpectralModel(
        mass=500 * u.GeV, channel="b", source=str(fits_file)
    )
    assert model.channel == "b"


def test_dm_source_as_table():
    """Test that a source can be passed directly as an astropy.table.Table."""
    t = Table(
        {
            "mDM": [500.0, 500.0, 1000.0, 1000.0] * u.GeV,
            "Log[10,x]": [-3.0, -2.0, -3.0, -2.0],
            "b": [1e-15, 1e-16, 1e-15, 1e-16],
        }
    )

    model = DarkMatterAnnihilationSpectralModel(mass=500 * u.GeV, channel="b", source=t)
    assert model.channel == "b"
    assert isinstance(model.source, Table)

    dnde = model.evaluate(energy=200 * u.GeV, scale=1)
    assert dnde.value >= 0
    assert np.isfinite(dnde.value)


def test_dm_source_as_table_with_mapping():
    """Test that a source Table works together with a mapping_dict, exactly
    like a custom file source does."""
    t = Table(
        {
            "mass": [500.0, 500.0, 1000.0, 1000.0] * u.GeV,
            "energy": [-3.0, -2.0, -3.0, -2.0],
            "bbar": [1e-15, 1e-16, 1e-15, 1e-16] / u.GeV,
        }
    )
    mapping = {"energy": "Log[10,x]", "mass": "mDM", "bbar": "b"}

    model = DarkMatterAnnihilationSpectralModel(
        mass=500 * u.GeV, channel="b", source=t, mapping_dict=mapping
    )
    assert model.channel == "b"
    assert model.mapping_dict == mapping


def test_dm_source_as_table_missing_channel():
    """A Table source missing the requested channel column should raise,
    just like a custom file source does."""
    t = Table(
        {
            "mDM": [500.0, 500.0] * u.GeV,
            "Log[10,x]": [-3.0, -2.0],
            "b": [1e-15, 1e-16],
        }
    )

    with pytest.raises(ValueError, match="The channel eL is not available"):
        DarkMatterAnnihilationSpectralModel(mass=500 * u.GeV, channel="eL", source=t)


def test_dm_source_invalid_type():
    """source must be None, a string, or a Table -- anything else should
    raise a clear TypeError."""
    with pytest.raises(TypeError, match="source must be"):
        DarkMatterAnnihilationSpectralModel(mass=500 * u.GeV, channel="b", source=12345)

    with pytest.raises(TypeError, match="source must be"):
        DarkMatterAnnihilationSpectralModel(
            mass=500 * u.GeV, channel="b", source=["not", "a", "table"]
        )


def test_dm_spectral_model_custom_io(tmp_path):
    """Test that source file and mapping_dict survive YAML serialization."""
    custom_file = tmp_path / "custom_spectra.ecsv"

    t = Table(
        {
            "mass": [500.0, 500.0, 1000.0, 1000.0] * u.GeV,
            "energy": [100.0, 200.0, 100.0, 200.0] * u.GeV,
            "b": [1e-15, 1e-16, 1e-15, 1e-16] / u.GeV,
        }
    )
    t.write(custom_file, format="ascii.ecsv")

    mapping = {"energy": "Log[10,x]", "mass": "mDM", "b": "b"}

    model = DarkMatterAnnihilationSpectralModel(
        mass=500 * u.GeV,
        channel="b",
        jfactor=3.41e19 * u.Unit("GeV2 cm-5"),
        source=str(custom_file),
        mapping_dict=mapping,
    )

    sky_model = SkyModel(spectral_model=model, name="skymodel_custom")
    models = Models([sky_model])

    filename = tmp_path / "model_custom.yaml"
    models.write(filename, overwrite=True)
    new_models = Models.read(filename)
    loaded_model = new_models[0].spectral_model

    assert loaded_model.source == str(custom_file)
    assert loaded_model.mapping_dict == mapping


def test_dm_annihilation_custom_errors(tmp_path):
    file_path = tmp_path / "test_dm.ecsv"
    t = Table()
    t["mDM"] = [1000, 5000] * u.GeV
    t["Log[10,x]"] = [-3, -2]
    t["bbar"] = [1e-5, 1e-4]
    t.write(file_path)

    mass = 5 * u.TeV
    with pytest.raises(TypeError, match="mapping_dict must be a dictionary"):
        DarkMatterAnnihilationSpectralModel(
            mass=mass,
            channel="b",
            source=str(file_path),
            mapping_dict=["not", "a", "dict"],
        )

    incomplete_mapping = {"Log[10,x]": "Log[10,x]"}
    with pytest.raises(KeyError, match="Mandatory column"):
        DarkMatterAnnihilationSpectralModel(
            mass=mass,
            channel="b",
            source=str(file_path),
            mapping_dict=incomplete_mapping,
        )

    wrong_mapping = {"mDM": "mDM", "Log[10,x]": "Log[10,x]", "wrong_col": "tau"}
    with pytest.raises(
        ValueError,
        match="The channel b is not available \
                            in the provided mapping dictionary. Please choose another \
                            channel or check the mapping_dict provided.\n",
    ):
        DarkMatterAnnihilationSpectralModel(
            mass=mass, channel="b", source=str(file_path), mapping_dict=wrong_mapping
        )


def test_missing_data_file(monkeypatch):
    """Test that FileNotFoundError is raised if the data path does not exist."""
    monkeypatch.setenv("GAMMAPY_DATA", "/fake/path/to/nowhere")
    with pytest.raises(FileNotFoundError, match="File not found"):
        PrimaryFlux(mDM=1 * u.TeV, channel="b")


@requires_data()
def test_mDM_out_of_bounds():
    """Test that ValueError is raised if the mass is out of the interpolation table bounds."""
    with pytest.raises(ValueError, match="is out of the bounds of the model"):
        PrimaryFlux(mDM=1 * u.eV, channel="b")


def test_custom_source_file_without_mapping_and_missing_channel(tmp_path):
    """Test the use of custom files without mapping_dict and with missing channels."""
    custom_file = tmp_path / "custom_spectra_nomap.ecsv"

    t = Table(
        {
            "mDM": [500.0, 500.0, 1000.0, 1000.0] * u.GeV,
            "Log[10,x]": [100.0, 200.0, 100.0, 200.0],
            "b": [1e-15, 1e-16, 1e-15, 1e-16],
        }
    )
    t.write(custom_file, format="ascii.ecsv")

    model = DarkMatterAnnihilationSpectralModel(
        mass=500 * u.GeV, channel="b", source=str(custom_file)
    )
    assert model.channel == "b"

    with pytest.raises(ValueError, match="The channel eL is not available"):
        DarkMatterAnnihilationSpectralModel(
            mass=500 * u.GeV, channel="eL", source=str(custom_file)
        )
