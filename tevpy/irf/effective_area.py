# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np

__all__ = ['effective_area']

def effective_area(energy, instrument='HESS'):
    """Calculate the effective area for a given instrument.

    Parameters
    ----------
    energy : array-like
        Energy in TeV
    instruments : {'HESS', 'HESS2', 'CTA'}
        Name of the instrument
    Returns
    -------
    Effective area in cm^2
        
    Parametrizations of the effective areas of Cherenkov
    telescopes taken from Appendix B of
    http://adsabs.harvard.edu/abs/2010MNRAS.402.1342A
    """
    # Put the parameters g in a dictionary.
    # Units: g1 (cm^2), g2 (), g3 (MeV)
    # Note that whereas in the paper the parameter index is 1-based,
    # here it is 0-based
    pars = {'HESS': [6.85e9, 0.0891, 5e5],
            'HESS2': [2.05e9, 0.0891, 1e5],
            'CTA': [1.71e11, 0.0891, 1e5]}

    energy = np.asanyarray(energy, dtype=np.float64)
    # Convert TeV to MeV
    energy = 1e6 * energy

    if not instrument in pars.keys():
        raise ValueError('Unknown instrument: {0}'.format(instrument))
   
    g1 = pars[instrument][0]
    g2 = pars[instrument][1]
    g3 = -pars[instrument][2]
    value = g1 * energy ** (-g2) * np.exp(g3 / energy)
    return value
