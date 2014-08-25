# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from astropy import log
from astropy.io import fits
from astropy.units import Quantity
from astropy.table import Table
from ..extern.validator import validate_physical_type
from ..utils.array import array_stats_str

__all__ = ['abramowski_effective_area', 'EffectiveAreaTable']


def abramowski_effective_area(energy, instrument='HESS'):
    """Simple IACT effective area parametrizations from Abramowski et al. (2010).

    TODO: give formula

    Parametrizations of the effective areas of Cherenkov telescopes
    taken from Appendix B of http://adsabs.harvard.edu/abs/2010MNRAS.402.1342A .

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Energy
    instrument : {'HESS', 'HESS2', 'CTA'}
        Instrument name

    Returns
    -------
    effective_area : `~astropy.units.Quantity`
        Effective area in cm^2
    """
    # Put the parameters g in a dictionary.
    # Units: g1 (cm^2), g2 (), g3 (MeV)
    # Note that whereas in the paper the parameter index is 1-based,
    # here it is 0-based
    pars = {'HESS': [6.85e9, 0.0891, 5e5],
            'HESS2': [2.05e9, 0.0891, 1e5],
            'CTA': [1.71e11, 0.0891, 1e5]}

    if not isinstance(energy, Quantity):
        raise ValueError("energy must be a Quantity object.")

    energy = energy.to('MeV').value

    if not instrument in pars.keys():
        ss = 'Unknown instrument: {0}\n'.format(instrument)
        ss += 'Valid instruments: HESS, HESS2, CTA'
        raise ValueError(ss)

    g1 = pars[instrument][0]
    g2 = pars[instrument][1]
    g3 = -pars[instrument][2]
    value = g1 * energy ** (-g2) * np.exp(g3 / energy)
    return Quantity(value, 'cm^2')


class EffectiveAreaTable(object):
    """
    Effective area table class.

    Not interpolation is used.

    Parameters
    ----------
    energy_lo : `~astropy.units.Quantity`
        Lower energy boundary of the energy bin.
    energy_hi : `~astropy.units.Quantity`
        Upper energy boundary of the energy bin.
    effective_area : `~astropy.units.Quantity`
        Effective area at the given energy bins.
    energy_thresh_lo : `~astropy.units.Quantity`
        Lower save energy threshold of the psf.
    energy_thresh_hi : `~astropy.units.Quantity`
        Upper save energy threshold of the psf.

    Examples
    --------
    Plot effective area vs. energy:

     .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from gammapy.irf import EffectiveAreaTable
        from gammapy.datasets import load_arf_fits_table
        arf = EffectiveAreaTable.from_fits(load_arf_fits_table())
        arf.plot_area_vs_energy(show_save_energy=False)
        plt.show()
    """
    def __init__(self, energy_lo, energy_hi, effective_area,
                 energy_thresh_lo=Quantity(0.1, 'TeV'),
                 energy_thresh_hi=Quantity(100, 'TeV')):

        # Validate input
        validate_physical_type('energy_lo', energy_lo, 'energy')
        validate_physical_type('energy_hi', energy_hi, 'energy')
        validate_physical_type('effective_area', effective_area, 'area')
        validate_physical_type('energy_thresh_lo', energy_thresh_lo, 'energy')
        validate_physical_type('energy_thresh_hi', energy_thresh_hi, 'energy')

        # Set attributes
        self.energy_hi = energy_hi.to('TeV')
        self.energy_lo = energy_lo.to('TeV')
        self.effective_area = effective_area.to('m^2')
        self.energy_thresh_lo = energy_thresh_lo.to('TeV')
        self.energy_thresh_hi = energy_thresh_hi.to('TeV')

    def to_fits(self, header=None, **kwargs):
        """
        Convert ARF to FITS HDU list format.

        Parameters
        ----------
        header : `~astropy.io.fits.header.Header`
            Header to be written in the fits file.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            ARF in HDU list format.

        Notes
        -----
        For more info on the ARF FITS file format see:
        http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/summary/cal_gen_92_002_summary.html

        Recommended units for ARF tables are keV and cm^2, but TeV and m^2 are chosen here
        as the more natural units for IACTs
        """
        hdu = fits.new_table(
            [fits.Column(name='ENERG_LO',
                          format='1E',
                          array=self.energy_lo,
                          unit='TeV'),
             fits.Column(name='ENERG_HI',
                          format='1E',
                          array=self.energy_hi,
                          unit='TeV'),
             fits.Column(name='SPECRESP',
                          format='1E',
                          array=self.effective_area,
                          unit='m^2')
             ]
            )

        if header is None:
            from ..datasets import load_arf_fits_table
            header = load_arf_fits_table()[1].header

        if header == 'pyfact':
            header = hdu.header

            # Write FITS extension header
            header['EXTNAME'] = 'SPECRESP', 'Name of this binary table extension'
            header['TELESCOP'] = 'DUMMY', 'Mission/satellite name'
            header['INSTRUME'] = 'DUMMY', 'Instrument/detector'
            header['FILTER'] = 'NONE', 'Filter information'
            header['HDUCLASS'] = 'OGIP', 'Organisation devising file format'
            header['HDUCLAS1'] = 'RESPONSE', 'File relates to response of instrument'
            header['HDUCLAS2'] = 'SPECRESP', 'Effective area data is stored'
            header['HDUVERS '] = '1.1.0', 'Version of file format'

            header['PHAFILE'] = ('', 'PHA file for which ARF was produced')

            # Obsolete ARF headers, included for the benefit of old software
            header['ARFVERSN'] = '1992a', 'Obsolete'
            header['HDUVERS1'] = '1.0.0', 'Obsolete'
            header['HDUVERS2'] = '1.1.0', 'Obsolete'

        for key, value in kwargs.items():
            header[key] = value

        header['LO_THRES'] = self.energy_thresh_lo.value
        header['HI_THRES'] = self.energy_thresh_hi.value

        prim_hdu = fits.PrimaryHDU()
        hdu.header = header
        hdu.add_checksum()
        hdu.add_datasum()
        return fits.HDUList([prim_hdu, hdu])

    def write(self, filename, *args, **kwargs):
        """Write ARF to FITS file.

        Calls `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_fits().writeto(filename, *args, **kwargs)

    @staticmethod
    def read(filename):
        """Read FITS format ARF file.

        Parameters
        ----------
        filename : str
            File name

        Returns
        -------
        parf : `EnergyDependentARF`
            ARF object.
        """
        hdu_list = fits.open(filename)
        return EffectiveAreaTable.from_fits(hdu_list)

    @staticmethod
    def from_fits(hdu_list):
        """
        Create EnergyDependentARF from HDU list.

        Parameters
        ----------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list with ``SPECRESP`` extensions.

        Returns
        -------
        arf : `EnergyDependentARF`
            ARF object.

        Notes
        -----
        For more info on the ARF FITS file format see:
        http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/summary/cal_gen_92_002_summary.html

        Recommended units for ARF tables are keV and cm^2, but TeV and m^2 are chosen here
        as the more natural units for IACTs.
        """
        energy_lo = Quantity(hdu_list['SPECRESP'].data['ENERG_LO'], 'TeV')
        energy_hi = Quantity(hdu_list['SPECRESP'].data['ENERG_HI'], 'TeV')
        effective_area = Quantity(hdu_list['SPECRESP'].data['SPECRESP'], 'm^2')
        try:
            energy_thresh_lo = Quantity(hdu_list['SPECRESP'].header['LO_THRES'], 'TeV')
            energy_thresh_hi = Quantity(hdu_list['SPECRESP'].header['HI_THRES'], 'TeV')
            return EffectiveAreaTable(energy_lo, energy_hi, effective_area,
                                      energy_thresh_lo, energy_thresh_hi)
        except KeyError:
            log.warn('No safe energy thresholds found. Setting to default')
            return EffectiveAreaTable(energy_lo, energy_hi, effective_area)

    def effective_area_at_energy(self, energy):
        """
        Get effective area for given energy.

        Not interpolation is used.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy where to return effective area.

        Returns
        -------
        effective_area : `~astropy.units.Quantity`
            Effective area at given energy.
        """
        if not isinstance(energy, Quantity):
            raise ValueError("energy must be a Quantity object.")

        # TODO: Use some kind of interpolation here
        i = np.argmin(np.abs(self.energy_hi - energy))
        return self.effective_area[i]

    def info(self, energies=Quantity([1, 10], 'TeV')):
        """
        Print some ARF info.

        Parameters
        ----------
        energies : `~astropy.units.Quantity`
            Energies for which to print effective areas.
        """
        ss = "\nSummary ARF info\n"
        ss += "----------------\n"
        # Summarise data members
        ss += array_stats_str(self.energy_lo, 'Energy lo')
        ss += array_stats_str(self.energy_hi, 'Energy hi')
        ss += array_stats_str(self.effective_area.to('m^2'), 'Effective area')
        ss += 'Safe energy threshold lo: {0:6.3f}\n'.format(self.energy_thresh_lo)
        ss += 'Safe energy threshold hi: {0:6.3f}\n'.format(self.energy_thresh_hi)

        for energy in energies:
            eff_area = self.effective_area_at_energy(energy)
            ss += ("Effective area at E = {0:4.1f}: {1:10.0f}\n".format(energy, eff_area))
        return ss

    def plot_area_vs_energy(self, filename=None, show_save_energy=True):
        """
        Plot effective area vs. energy.
        """
        import matplotlib.pyplot as plt

        energy_hi = self.energy_hi.value
        effective_area = self.effective_area.value
        plt.plot(energy_hi, effective_area)
        if show_save_energy:
            plt.vlines(self.energy_thresh_hi.value, 1E3, 1E7, 'k', linestyles='--')
            plt.text(self.energy_thresh_hi.value - 1, 3E6,
                     'Safe energy threshold: {0:3.2f}'.format(self.energy_thresh_hi),
                     ha='right')
            plt.vlines(self.energy_thresh_lo.value, 1E3, 1E7, 'k', linestyles='--')
            plt.text(self.energy_thresh_lo.value + 0.1, 3E3,
                     'Safe energy threshold: {0:3.2f}'.format(self.energy_thresh_lo))
        plt.xlim(0.1, 100)
        plt.ylim(1E3, 1E7)
        plt.loglog()
        plt.xlabel('Energy [TeV]')
        plt.ylabel('Effective Area [m^2]')
        if filename is not None:
            plt.savefig(filename)
            log.info('Wrote {0}'.format(filename))
