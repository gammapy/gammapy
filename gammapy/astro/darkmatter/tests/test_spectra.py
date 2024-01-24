# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
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
    "mass, expected_flux", [(1.6, 0.00025037), (11, 0.00502079), (75, 0.02028309)]
)
@requires_data()
def test_primary_flux_interpolation(mass, expected_flux):
    primflux = PrimaryFlux(channel="W", mDM=mass * u.TeV)
    actual = primflux(500 * u.GeV)
    assert_quantity_allclose(actual, expected_flux / u.GeV, rtol=1e-5)


@requires_data()
def test_dm_annihilation_spectral_model(tmpdir):
    channel = "b"
    massDM = 5 * u.TeV
    jfactor = 3.41e19 * u.Unit("GeV2 cm-5")
    energy_min = 0.01 * u.TeV
    energy_max = 10 * u.TeV

    model = DarkMatterAnnihilationSpectralModel(
        mass=massDM, channel=channel, jfactor=jfactor
    )
    integral_flux = model.integral(energy_min=energy_min, energy_max=energy_max).to(
        "cm-2 s-1"
    )
    differential_flux = model.evaluate(energy=1 * u.TeV, scale=1).to("cm-2 s-1 TeV-1")

    sky_model = SkyModel(
        spectral_model=model,
        name="skymodel",
    )
    models = Models([sky_model])
    filename = tmpdir / "model.yaml"
    models.write(filename, overwrite=True)
    new_models = Models.read(filename)

    assert_quantity_allclose(integral_flux.value, 6.19575457e-14, rtol=1e-3)
    assert_quantity_allclose(differential_flux.value, 2.97831615e-16, rtol=1e-3)

    assert new_models[0].spectral_model.channel == model.channel
    assert new_models[0].spectral_model.z == model.z
    assert_allclose(new_models[0].spectral_model.jfactor.value, model.jfactor.value)
    assert new_models[0].spectral_model.mass.value == 5
    assert new_models[0].spectral_model.mass.unit == u.TeV


@requires_data()
def test_dm_decay_spectral_model(tmpdir):
    channel = "b"
    massDM = 5 * u.TeV
    jfactor = 3.41e19 * u.Unit("GeV cm-2")
    energy_min = 0.01 * u.TeV
    energy_max = 10 * u.TeV

    model = DarkMatterDecaySpectralModel(mass=massDM, channel=channel, jfactor=jfactor)
    integral_flux = model.integral(energy_min=energy_min, energy_max=energy_max).to(
        "cm-2 s-1"
    )
    differential_flux = model.evaluate(energy=1 * u.TeV, scale=1).to("cm-2 s-1 TeV-1")

    sky_model = SkyModel(
        spectral_model=model,
        name="skymodel",
    )
    models = Models([sky_model])
    filename = tmpdir / "model.yaml"
    models.write(filename, overwrite=True)
    new_models = Models.read(filename)

    assert_quantity_allclose(integral_flux.value, 4.80283595e-2, rtol=1e-3)
    assert_quantity_allclose(differential_flux.value, 2.3088e-4, rtol=1e-3)

    assert new_models[0].spectral_model.channel == model.channel
    assert new_models[0].spectral_model.z == model.z
    assert_allclose(new_models[0].spectral_model.jfactor.value, model.jfactor.value)
    assert new_models[0].spectral_model.mass.value == 5
    assert new_models[0].spectral_model.mass.unit == u.TeV
