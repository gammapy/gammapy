"""Create residuals image based on the two flux point methods
"""
import numpy as np
from astropy.table import Table
from gammapy.spectrum.flux_point import (compute_differential_flux_points,
                                         _energy_lafferty_power_law,
                                         _x_lafferty, _integrate)
from gammapy.spectrum.powerlaw import power_law_eval, power_law_integral_flux
import matplotlib.pyplot as plt


def compute_flux_error(gamma_true, gamma_reco, method):
    # Let's assume a concrete true spectrum and energy bin.
    # Note that the residuals computed below do *not* depend on
    # these parameters.
    energy_min, energy_max = 1, 10
    energy_ref, diff_flux_ref = 1, 1
    # Compute integral flux in the energy band assuming `gamma_true`
    int_flux = power_law_integral_flux(diff_flux_ref, gamma_true,
                                       energy_ref, energy_min, energy_max)
    # Compute flux point
    table = compute_differential_flux_points(method, 'power_law',
                                     spectral_index=gamma_reco,
                                     energy_min=energy_min, energy_max=energy_max,
                                     int_flux=int_flux)
    # Compute relative error of the flux point
    energy = table['ENERGY'].data
    flux_reco = table['DIFF_FLUX'].data
    flux_true = power_law_eval(energy, diff_flux_ref * np.ones_like(energy),
                               np.array(gamma_true).reshape(energy.shape),
                               energy_ref * np.ones_like(energy))
    flux_true = flux_true.reshape(gamma_true.shape)
    flux_reco = flux_reco.reshape(gamma_true.shape)
    flux_error = (flux_reco - flux_true) / flux_true
    return flux_error


def residuals_image():
    gamma_true = np.arange(1.01, 7, 1)
    gamma_reco = np.arange(1.01, 7, 1)
    gamma_true, gamma_reco = np.meshgrid(gamma_true, gamma_reco)
    flux_error_lafferty = compute_flux_error(gamma_true, gamma_reco,
                                             method='lafferty')
    flux_error_log_center = compute_flux_error(gamma_true, gamma_reco,
                                               method='log_center')
    flux_error_ratio = np.log10(flux_error_lafferty / flux_error_log_center)
    extent = [0.5, 6.5, 0.5, 6.5]
    vmin, vmax = -3, 3
    fig, axes = plt.subplots(nrows=1, ncols=3)
    im = axes.flat[0].imshow(np.array(flux_error_lafferty),
                             interpolation='nearest', extent=extent,
                             origin="lower", vmin=vmin, vmax=vmax,
                             cmap=plt.get_cmap('seismic'))
    axes.flat[0].set_ylabel('Assumed Spectral Index', fontsize=14)
    axes.flat[0].set_title('Lafferty Method', fontsize=12)
    im = axes.flat[1].imshow(np.array(flux_error_log_center),
                             interpolation='nearest', extent=extent,
                             origin="lower", vmin=vmin, vmax=vmax,
                             cmap=plt.get_cmap('seismic'))
    axes.flat[1].set_xlabel('True Spectral Index', fontsize=14)
    axes.flat[1].set_title('Log-Center Method', fontsize=12)
    im = axes.flat[2].imshow(np.array(flux_error_ratio),
                             interpolation='nearest', extent=extent,
                             origin="lower", vmin=vmin, vmax=vmax,
                             cmap=plt.get_cmap('seismic'))
    axes.flat[2].set_title('Residual Log Ratio: \n Log(Lafferty/Log Center)',
                           fontsize=12)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.3, 0.025, 0.4])
    fig.colorbar(im, cax=cbar_ax)
    return fig
