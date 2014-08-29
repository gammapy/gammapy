"""Plots 95 Percent PSF Containment Radius with Energy for Galactic Center
and Vela region.
"""
from fermi_psf_study import plot_containment_radii

plt = plot_containment_radii(fraction=95)
plt.ylim([0, 2])
plt.title('Fermi-LAT PSF \n 95% Containment Radius with Energy')
