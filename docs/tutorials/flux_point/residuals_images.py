"""Create residuals image based on the two flux point methods
"""
import numpy as np
import matplotlib.pyplot as plt
#from gammapy.spectrum.flux_point import compute_differential_flux_points
from gammapy.spectrum.powerlaw import power_law_evaluate, power_law_integral_flux


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
    flux_true = power_law_evaluate(energy, diff_flux_ref * np.ones_like(energy),
                                   np.array(gamma_true).reshape(energy.shape),
                                   energy_ref * np.ones_like(energy))
    flux_true = flux_true.reshape(gamma_true.shape)
    flux_reco = flux_reco.reshape(gamma_true.shape)
    flux_error = (flux_reco - flux_true) / flux_true
    return flux_error


def residuals_image():
    fig = plt.figure(figsize=(15, 5))
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
    axes_1 = fig.add_subplot(131)
    axes_1.imshow(np.array(flux_error_lafferty),
                  interpolation='nearest', extent=extent,
                  origin="lower", vmin=vmin, vmax=vmax,
                  cmap='RdBu')
    axes_1.set_ylabel('Assumed Spectral Index', fontsize=14)
    axes_1.set_title('Lafferty Method', fontsize=12)

    axes_2 = fig.add_subplot(132)
    axes_2.imshow(np.array(flux_error_log_center),
                  interpolation='nearest', extent=extent,
                  origin="lower", vmin=vmin, vmax=vmax,
                  cmap='RdBu')
    axes_2.set_xlabel('True Spectral Index', fontsize=14)
    axes_2.set_title('Log-Center Method', fontsize=12)

    axes_3 = fig.add_subplot(133)
    im = axes_3.imshow(np.array(flux_error_ratio),
                       interpolation='nearest', extent=extent,
                       origin="lower", vmin=vmin, vmax=vmax,
                       cmap='RdBu')
    axes_3.set_title('Residual Log Ratio: \n Log(Lafferty/Log Center)',
                     fontsize=12)
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.11, 0.025, 0.78])
    fig.colorbar(im, cax=cbar_ax)
    return fig
