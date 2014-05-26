# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from astropy.io import fits
from astropy.units import Quantity

__all__ = ['abramowski_effective_area',
           'np_to_arf', 'arf_to_np']


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


def np_to_arf(effective_area, energy_bounds, telescope='DUMMY',
              phafile=None, instrument='DUMMY', filter='NONE') :
    """Create ARF FITS table extension from numpy arrays.

    Parameters
    ----------
    effective_area : array_like
       1-dim effective area array (m^2)
    energy_bounds : array_like
       1-dim energy bounds array (TeV)

    Returns
    -------
    arf : `~astropy.io.fits.BinTableHDU`
        ARF in FITS table HDU format

    Notes
    -----
    For more info on the ARF FITS file format see:
    http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/summary/cal_gen_92_002_summary.html
    
    Recommended units for ARF tables are keV and cm^2, but TeV and m^2 are chosen here
    as the more natural units for IACTs
    """
    effective_area = np.asarray(effective_area)
    energy_bounds = np.asarray(energy_bounds)

    hdu = fits.new_table(
        [fits.Column(name='ENERG_LO',
                      format='1E',
                      array=energy_bounds[:-1],
                      unit='TeV'),
         fits.Column(name='ENERG_HI',
                      format='1E',
                      array=energy_bounds[1:],
                      unit='TeV'),
         fits.Column(name='SPECRESP',
                      format='1E',
                      array=effective_area,
                      unit='m^2')
         ]
        )
    
    header = hdu.header

    # Write FITS extension header
    header['EXTNAME'] = 'SPECRESP', 'Name of this binary table extension'
    header['TELESCOP'] = telescope, 'Mission/satellite name'
    header['INSTRUME'] = instrument, 'Instrument/detector'
    header['FILTER'] = filter, 'Filter information'
    header['HDUCLASS'] = 'OGIP', 'Organisation devising file format'
    header['HDUCLAS1'] = 'RESPONSE', 'File relates to response of instrument'
    header['HDUCLAS2'] = 'SPECRESP', 'Effective area data is stored'
    header['HDUVERS '] = '1.1.0', 'Version of file format'

    if phafile != None:
        header['PHAFILE'] = (phafile, 'PHA file for which ARF was produced')

    # Obsolete ARF headers, included for the benefit of old software
    header['ARFVERSN'] = '1992a', 'Obsolete'
    header['HDUVERS1'] = '1.0.0', 'Obsolete'
    header['HDUVERS2'] = '1.1.0', 'Obsolete'

    return hdu


def arf_to_np(arf) :
    """Extract ARF FITS table extension to numpy arrays.

    Parameters
    ----------
    arf : `astropy.io.fits.BinTableHDU`
        ARF in FITS table HDU format

    Returns
    -------
    effective_area : numpy.array
       1-dim effective area array (m^2)
    energy_bounds : numpy.array
       1-dim energy bounds array (TeV)

    Notes
    -----
    For more info on the ARF FITS file format see:
    http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/summary/cal_gen_92_002_summary.html
    
    Recommended units for ARF tables are keV and cm^2, but TeV and m^2 are chosen here
    as the more natural units for IACTs
    """
    # TODO: read units and convert to m^2 and TeV if necessary.
    data = arf.data
    effective_area = data['SPECRESP']
    energy_bounds = np.hstack([data['ENERG_LO'], data['ENERG_HI'][-1]])
    return effective_area, energy_bounds
