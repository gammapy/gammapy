"""Plot of x position depending on spectral index estimation
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
#from gammapy.spectrum.flux_point import _energy_lafferty_power_law
from gammapy.spectrum.powerlaw import power_law_evaluate, power_law_integral_flux
from flux_point_demo import get_flux_tables

SPECTRAL_INDEX = 4


def make_x_plot():
    energy_min = np.array([300])
    energy_max = np.array([1000])
    energies = np.array(_energy_lafferty_power_law(energy_min, energy_max,
                                                   SPECTRAL_INDEX))
    diff_flux = power_law_evaluate(energies, 1, SPECTRAL_INDEX, 1)
    # `True' differential & integral fluxes
    int_flux = power_law_integral_flux(diff_flux, SPECTRAL_INDEX,
                                       energies, energy_min, energy_max)
    # Put data into table
    table = Table()
    table['ENERGY_MIN'] = energy_min
    table['ENERGY_MAX'] = energy_max
    table['INT_FLUX'] = int_flux
    lafferty_array = []
    log_array = []
    spectral_indices = np.arange(1.1, 6, 0.01)
    for spectral_index in spectral_indices:
        lafferty_flux, log_flux = get_flux_tables(table, 'power_law', None,
                                                  spectral_index)
        dlog_energy = np.log(energy_max) - np.log(energy_min)
        residuals_lafferty = np.log(lafferty_flux['ENERGY'] - np.log(energy_min)) / dlog_energy
        residuals_log = np.log(log_flux['ENERGY'] - np.log(energy_min)) / dlog_energy
        lafferty_array.append(residuals_lafferty[0])
        log_array.append(residuals_log[0])
    plt.plot(spectral_indices, lafferty_array,
             linewidth=1, ms=0, label='Lafferty Method')
    plt.plot(spectral_indices, log_array,
             linewidth=1, ms=0, label='Log Center Method')
    plt.legend()
    plt.ylabel('X position in bin')
    plt.xlabel('Guessed spectral Index')
