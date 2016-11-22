"""Spectrum model API for Gammapy.
"""
import numpy as np
from astropy.units import Quantity as Q
from gammapy.spectrum.models import (
    SpectralModel, PowerLaw, ExponentialCutoffPowerLaw
)
from gammapy.irf import (
    EffectiveAreaTable,
    EnergyDispersion
)
from gammapy.utils.energy import EnergyBounds
from gammapy.spectrum import calculate_predicted_counts


# TODO clean up and add to gammapy.irf
def make_perfect_resolution(e_true, e_reco):
    # have to take log vals here
    diff  = e_reco[1:, np.newaxis] - e_true[1:]
    idx = np.argmin(np.abs(diff), axis=0)
    data = np.zeros((len(e_true)-1, len(e_reco)-1))
    for i in range(len(data)):
        data[i][idx[i]] = 1

    return data

def test_spectrum_model():
    model = SpectralModel()


def test_model(model):
    print(model)
    print(model(energy=Q(10, 'TeV')))
    print(model.integral(emin=Q(1, 'TeV'), emax=Q(2, 'TeV')))

    # plot
    # butterfly
    # npred
    reco_bins = 5
    true_bins = 10
    e_reco = Q(np.logspace(-1,1,reco_bins+1), 'TeV')
    e_true = Q(np.logspace(-1.5, 1.5, true_bins+1), 'TeV')
    livetime = Q(26, 'min')
    aeff_data = Q(np.ones(true_bins) * 1e5, 'cm2')
    aeff = EffectiveAreaTable(energy=e_true, data=aeff_data)
    edisp_data = make_perfect_resolution(e_true, e_reco)
    edisp = EnergyDispersion(edisp_data, EnergyBounds(e_true),
                             EnergyBounds(e_reco))
    npred = calculate_predicted_counts(model=model,
                                       livetime=livetime,
                                       aeff=aeff,
                                       edisp=edisp)
    print(npred.data)
    # fit -- to_sherpa, do later
    # flux points -- later



def test_power_law():
    model = PowerLaw(
        amplitude=Q(1e-11, 'cm^-2 s^-1 TeV^-1'),
        reference=Q(1, 'TeV'),
        index=2,
    )
    test_model(model)


def test_exponential_cutoff_power_law():
    model = ExponentialCutoffPowerLaw(
        amplitude=Q(1e-11, 'cm^-2 s^-1 TeV^-1'),
        e_0=Q(1, 'TeV'),
        alpha=2,
        e_cutoff=Q('3 TeV'),
    )

if __name__ == '__main__':
    test_spectrum_model()
    test_power_law()
    #test_exponential_cutoff_power_law()
