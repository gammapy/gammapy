"""Spectrum model API for Gammapy.
"""
from astropy.units import Quantity as Q
from gammapy.spectrum.models import (
    SpectralModel, PowerLaw, ExponentialCutoffPowerLaw
)


def test_spectrum_model():
    model = SpectralModel()


def test_power_law():
    model = PowerLaw(
        amplitude=Q(1e-11, 'cm^-2 s^-1 TeV^-1'),
        e_0=Q(1, 'TeV'),
        alpha=2,
    )
    print(model)
    print(model(energy=Q(10, 'TeV')))

    # plot

    # butterfly

    # npred

    # fit -- to_sherpa, do later

    # flux points -- later


def test_exponential_cutoff_power_law():
    model = ExponentialCutoffPowerLaw(
        amplitude=Q(1e-11, 'cm^-2 s^-1 TeV^-1'),
        e_0=Q(1, 'TeV'),
        alpha=2,
        e_cutoff=Q('3 TeV'),
    )
    print(model)
    print(model(energy=Q(10, 'TeV')))


if __name__ == '__main__':
    test_spectrum_model()
    test_power_law()
    test_exponential_cutoff_power_law()
