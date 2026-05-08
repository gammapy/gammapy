# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
import pytest
from astropy.table import Table
from numpy.testing import assert_allclose

from gammapy.astro.darkmatter import (
    DarkMatterAnnihilationSpectralModel,
    DarkMatterDecaySpectralModel,
    PrimaryFlux,
)
from gammapy.modeling.models import Models, SkyModel
from gammapy.utils.testing import assert_quantity_allclose, requires_data


@requires_data()
def test_primary_flux():
    with pytest.raises(ValueError):
        PrimaryFlux(channel="Spam", mDM=1 * u.TeV)

    primflux = PrimaryFlux(channel="W", mDM=1 * u.TeV)
    actual = primflux(500 * u.GeV)
    desired = 9.3319318e-05 / u.GeV
    assert_quantity_allclose(actual, desired)


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

    sky_model = SkyModel(
        spectral_model=model,
        name="skymodel",
    )
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

    primflux = PrimaryFlux(channel="W", mDM=1 * u.TeV, source="cosmixs")
    actual = primflux(500 * u.GeV)
    desired = 0.00013085 / u.GeV
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


def test_custom_source_file_empty(tmp_path):
    """Test that an empty custom source file raises the correct error."""
    empty_file = tmp_path / "empty_spectra.dat"
    empty_file.touch()

    with pytest.raises(KeyError, match="Source file is empty."):
        DarkMatterAnnihilationSpectralModel(
            mass=5 * u.TeV, channel="b", source=str(empty_file)
        )


def test_custom_source_file_bad_extension(tmp_path):
    """Test that a custom file has a bad extension raises the correct error."""
    empty_file = tmp_path / "empty_spectra.dl2"
    empty_file.touch()

    with pytest.raises(KeyError, match="Source file extension "):
        DarkMatterAnnihilationSpectralModel(
            mass=5 * u.TeV, channel="b", source=str(empty_file)
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
