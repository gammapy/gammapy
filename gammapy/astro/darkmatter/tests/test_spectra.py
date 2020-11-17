# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import astropy.units as u
from gammapy.astro.darkmatter import DarkMatterAnnihilationSpectralModel, PrimaryFlux
from gammapy.utils.testing import assert_quantity_allclose, requires_data


@requires_data()
def test_primary_flux():
    with pytest.raises(ValueError):
        PrimaryFlux(channel="Spam", mDM=1 * u.TeV)

    primflux = PrimaryFlux(channel="W", mDM=1 * u.TeV)
    actual = primflux.table_model(500 * u.GeV)
    desired = 9.328234e-05 / u.GeV
    assert_quantity_allclose(actual, desired)


@requires_data()
def test_DMAnnihilation():
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

    assert_quantity_allclose(integral_flux.value, 6.19575457e-14, rtol=1e-3)
    assert_quantity_allclose(differential_flux.value, 2.97506768e-16, rtol=1e-3)
