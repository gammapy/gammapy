import pytest
import astropy.units as u
from ...utils.testing import assert_quantity_allclose
from ...spectrum import cosmic_ray_spectrum

Cosmic_rays_SPECTRA = [
    dict(
        name="proton",
        dnde=u.Quantity(0.014773732960138994, "m-2 s-1 TeV-1 sr-1"),
        flux=u.Quantity(0.056470139673467444, "m-2 s-1 sr-1"),
        index=2.70,
    ),
    dict(
        name="N",
        dnde=u.Quantity(0.0115347902543466, "m-2 s-1 TeV-1 sr-1"),
        flux=u.Quantity(0.043840936324311894, "m-2 s-1 sr-1"),
        index=2.64,
    ),
    dict(
        name="Si",
        dnde=u.Quantity(0.0044934359085944935, "m-2 s-1 TeV-1 sr-1"),
        flux=u.Quantity(0.017108254587645998, "m-2 s-1 sr-1"),
        index=2.66,
    ),
    dict(
        name="Fe",
        dnde=u.Quantity(0.0021646909913177995, "m-2 s-1 TeV-1 sr-1"),
        flux=u.Quantity(0.008220752990527653, "m-2 s-1 sr-1"),
        index=2.63,
    ),
]


@pytest.mark.parametrize("spec", Cosmic_rays_SPECTRA, ids=lambda _: _["name"])
def test_cosmic_ray_spectrum(spec):
    energy = 2 * u.TeV
    emin, emax = [1, 1e3] * u.TeV

    cr_spectrum = cosmic_ray_spectrum(particle=spec["name"])

    dnde = cr_spectrum(energy)
    assert_quantity_allclose(dnde, spec["dnde"])

    flux = cr_spectrum.integral(emin, emax)
    assert_quantity_allclose(flux, spec["flux"])

    index = cr_spectrum.spectral_index(energy)
    assert_quantity_allclose(index, spec["index"], rtol=1e-5)


def test_invalid_format():
    with pytest.raises(ValueError):
        cosmic_ray_spectrum("spam")
