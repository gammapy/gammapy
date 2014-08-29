"""Produces table to study the size of the Fermi-LAT PSF with Energy.
"""
import numpy as np
from astropy.units import Quantity
from astropy.table import Table
from gammapy.spectrum import energy_bounds_equal_log_spacing 
from gammapy.datasets import FermiGalacticCenter, FermiVelaRegion
from gammapy.datasets import load_lat_psf_performance
import matplotlib.pyplot as plt

__all__ = ['get_psf_table', 'plot_containment_radii']


def get_psf_table(psf, emin, emax, bins):
    """Returns a table of energy and containment radius
    from an EnergyDependentTablePSF object."""

    # Container for data
    data = []

    # Loop over energies and determine PSF containment radius
    for energy in energy_bounds_equal_log_spacing(Quantity((emin, emax), 'MeV'), bins):
        energy_psf = psf.table_psf_at_energy(energy)
    
        containment_68 = energy_psf.containment_radius(0.68)
        containment_95 = energy_psf.containment_radius(0.95)

        row = dict(ENERGY=energy.value,
                   CONT_68=containment_68.value,
                   CONT_95=containment_95.value)

        data.append(row)

    # Construct table and add correct units to columns
    table = Table(data)
    table['ENERGY'].units = energy.unit
    table['CONT_68'].units = containment_68.unit
    table['CONT_95'].units = containment_95.unit
    
    return table


def plot_containment_radii(fraction):
    """Plotting script for 68% and 95% containment radii."""

    psf_gc = FermiGalacticCenter.psf()
    gtpsf_table_gc = get_psf_table(psf_gc,  10000, 300000, 15)

    psf_vela = FermiVelaRegion.psf()
    gtpsf_table_vela = get_psf_table(psf_vela, 10000, 300000, 15)
    
    if fraction == 68:
        true_table_rep = load_lat_psf_performance('P7REP_SOURCE_V15_68')
        true_table = load_lat_psf_performance('P7SOURCEV6_68')
        rad = 'CONT_68'
    elif fraction == 95:
        true_table_rep = load_lat_psf_performance('P7REP_SOURCE_V15_95')
        true_table = load_lat_psf_performance('P7SOURCEV6_95')
        rad = 'CONT_95'
    
    plt.plot(gtpsf_table_gc['ENERGY'], gtpsf_table_gc[rad],
             color='red',label='Fermi Tools PSF @ Galactic Center')
    plt.plot(gtpsf_table_vela['ENERGY'], gtpsf_table_vela[rad],
             color='blue', label='Fermi Tools PSF @ Vela Region')
    plt.plot(true_table_rep['energy'], true_table_rep['containment_angle'],
             color='green', linestyle='--', label='P7REP_SOURCE_V15')
    plt.plot(true_table['energy'], true_table['containment_angle'],
             color='black', linestyle='--', label='P7SOURCEV6')

    plt.xlim([10000, 300000])

    plt.legend()
    plt.semilogx()
    plt.xlabel('Energy/MeV')
    plt.ylabel('PSF Containment Radius/deg')

    return plt
