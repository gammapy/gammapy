"""Spectral plotting with gammapy.spectrum.flux_point
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
#from gammapy.spectrum.flux_point import (compute_differential_flux_points,
#                                         _x_lafferty, _integrate)
#from gammapy.spectrum.powerlaw import power_law_integral_flux

SPECTRAL_INDEX = 4


def my_spectrum(x):
    E_1 = 1
    E_2 = 10
    E_3 = 100
    E_4 = 1000
    E_5 = 10000
    g1 = -1
    g2 = -2
    g3 = -3
    g4 = -4
    g5 = -5
    im = np.select([x <= E_1, x <= E_2, x <= E_3, x <= E_4, x <= E_5, x > E_5],
                   [(x / E_1) ** g1, 1e-2 * (x / E_2) ** g2,
                    1e-5 * (x / E_3) ** g3, 1e-9 * (x / E_4) ** g4,
                    1e-14 * (x / E_5) ** g5, 0])
    return im


def get_flux_tables(table, y_method, function, spectral_index):
    table1 = table.copy()
    lafferty_flux = compute_differential_flux_points('lafferty', y_method, table1,
                                                     function, spectral_index)
    table2 = table1.copy()
    log_flux = compute_differential_flux_points('log_center', y_method, table2,
                                                function, spectral_index)
    return lafferty_flux, log_flux


def plot_flux_points(table, x, y, function, energy_min, y_method):
    f, ax = plt.subplots(2, sharex=True)
    lafferty_flux, log_flux = get_flux_tables(table, y_method, function,
                                              SPECTRAL_INDEX)
    ax[0].loglog(lafferty_flux['ENERGY'],
                 ((lafferty_flux['ENERGY'] ** 2) * lafferty_flux['DIFF_FLUX']),
                 marker='D', linewidth=0, ms=5,
                 label='Lafferty Method')
    residuals_lafferty = (lafferty_flux['DIFF_FLUX']
                          - function(lafferty_flux['ENERGY'])) / function(lafferty_flux['ENERGY']) * 100
    ax[0].loglog(log_flux['ENERGY'],
                 (log_flux['ENERGY'] ** 2) * log_flux['DIFF_FLUX'],
                 marker='D', linewidth=0, ms=5,
                 label='Log Center Method')
    ax[0].legend(loc='lower left', fontsize=10)
    residuals_log = (log_flux['DIFF_FLUX'] -
                     function(log_flux['ENERGY'])) / function(log_flux['ENERGY']) * 100
    ax[1].semilogx(lafferty_flux['ENERGY'], residuals_lafferty, marker='D',
                   linewidth=0, ms=5)
    ax[1].semilogx(log_flux['ENERGY'], residuals_log, marker='D',
                   linewidth=0, ms=5)
    ax[0].loglog(x, (x ** 2) * y, color='k')
    indices = np.arange(len(energy_min))
    for index in indices:
        ax[0].axvline(energy_min[index], 0, 1e6, color='k',
                      linestyle=':')
        ax[1].axvline(energy_min[index], 0, 1e6, color='k',
                      linestyle=':')
    ax[1].axhline(0, 0, 10, color='k')
    ax[0].set_ylabel('E^2 * Differential Flux')
    ax[1].set_ylabel('Residuals/%')
    ax[1].set_xlabel('Energy')
    ax[0].set_xlim([0.1, 10000])
    ax[0].set_ylim([1e-6, 1e1])
    return f


def plot_power_law():
    # Define the function
    x = np.arange(0.1, 100000, 0.1)
    spectral_model = my_spectrum(x)
    spectral_model_function = lambda x: my_spectrum(x)
    y = spectral_model
    # Set the x-bins
    energy_min = [0.1, 1, 10, 100, 1000]
    energy_max = [1, 10, 100, 1000, 10000]
    energies = np.array(_x_lafferty(energy_min, energy_max,
                                    spectral_model_function))
    diff_fluxes = spectral_model_function(energies)
    indices = np.array([0, 1, 2, 3, 4])
    int_flux = power_law_integral_flux(diff_fluxes, (indices + 1),
                                       energies, energy_min, energy_max)
    special = lambda x: np.log(x)
    int_flux[0] = np.abs(_integrate([energy_min[0]],
                                    [energy_max[0]], special)[0])
    # Put data into table
    table = Table()
    table['ENERGY_MIN'] = energy_min
    table['ENERGY_MAX'] = energy_max
    table['INT_FLUX'] = int_flux
    plot_flux_points(table, x, y, spectral_model_function,
                     energy_min, 'power_law')
    plt.tight_layout()
    plt.legend()
