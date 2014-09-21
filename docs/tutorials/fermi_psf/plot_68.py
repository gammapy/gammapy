"""Plots 68 Percent PSF Containment Radius with Energy for Galactic Center
and Vela region.
"""
from fermi_psf_study import plot_containment_radii

plt = plot_containment_radii(fraction=68)
plt.ylim([0, 0.5])
plt.title('Fermi-LAT PSF \n 68% Containment Radius with Energy')
