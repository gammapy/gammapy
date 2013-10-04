# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Cosmic ray spectra at Earth 

Gaisser (2011)
"Spectrum of cosmic-ray nucleons and the atmospheric muon charge ratio"
http://adsabs.harvard.edu/abs/2011arXiv1111.6675G
See Equation 21 and Table 1. 
"""
from __future__ import print_function, division

__all__ = ['cosmic_ray_flux']

# TODO
def cosmic_ray_flux(energy, particle='proton'):
    """Cosmic ray flux as measured at Earth.
    
    Parameters
    ----------
    energy : array-like
        Cosmic ray particle energy in TeV
    particle : {'proton', 'electron'}
        Particle type
    
    Returns
    -------
    flux : array
        Cosmic ray flux in unit TODO.
    """
    raise NotImplementedError