# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Galactic interstellar radiation field (ISRF) models"""
from __future__ import print_function, division
from os.path import join
from astropy.io import fits

__all__ = ['Galprop', 'Schlickeiser']


class Schlickeiser(object):
    """ISRF model for the solar neighborhood.

    Reference: Book by Schlickeiser "Cosmic Ray Astrophysics", Section 2.3

    Note: component 1 (spectral type B) from Schlickeiser
    is not very important and was ignored so that there is a 1:1 match
    to the three Galprop ISRF components.

    Note: the form of the ISRF component spectra is like a
    blackbody, which is characterized by only one number,
    the temperature T.
    But the energy density of the ISRF infrared component
    is different than the corresponding energy density of the blackbody,
    so each component has one more energy density parameter W
    and is called a "greybody" distribution by Schlickeiser."""
    components = ['Optical', 'Infrared', 'CMB']
    # Temperature (K) and energy density (eV cm^-3)
    # of the greybody components
    component_infos = {'Optical': {'T': 5000, 'W': 0.3, 'color': 'blue'},
                       'Infrared': {'T': 20, 'W': 0.4, 'color': 'green'},
                       'CMB': {'T': 2.7, 'W': 0.25, 'color': 'red'},
                       'Total': {'color': 'black'}}

    def _get_component_info(self, component):
        T = self.component_infos[component]['T']
        W = self.component_infos[component]['W']
        return T, W

    def __call__(self, energy, component='Total'):
        """Evaluate model.

        Parameters
        ----------
        energy : float or array
            Photon energy (eV)
        component : {'Optical', 'Infrared', 'CMB', 'Total'}
            Radiation field component

        Returns
        -------
        Photon number density of a given component (cm^-3 eV^-1)
        """
        if component == 'Total':
            total = 0
            for component in self.components:
                total += self._density(energy, component)
            return total
        elif component in self.components:
            return self._density(energy, component)
        else:
            raise IndexError('Component {0} not available.'
                             ''.format(component))

    def _density(self, energy, component):
        raise NotImplementedError


class Galprop(object):
    """A cylindrically symmetric model for the distribution
    of the optical and infrared ISRF in the Milky Way.

    The CMB component is isotropic, so it's implementation
    is different from the two non-isotropic components.

    http://vizier.cfa.harvard.edu/vizier/ftp/cats/J/A+A/534/A54/galprop_v54.pdf
    """
    components = ['Optical', 'Infrared', 'CMB']

    def __init__(self):
        """Initialize the lookup"""
        pass

    def __read_lookup__(self):
        """Read lookup table from FITS file."""
        # TODO: use astropy, not kapteyn here
        from kapteyn.maputils import FITSimage
        dirname = '/Users/deil/work/workspace/galpop/data/GALPROP'
        filename = join(dirname, 'MilkyWay_DR0.5_DZ0.1_DPHI10_RMAX20_ZMAX5_galprop_format.fits')
        # The FITS header is missing the CRPIX keywords, so FITSimage would complain.
        # That's why we read it with pyfits, add the keywords, and then create the FITSimage.
        data = fits.getdata(filename)
        header = fits.getheader(filename)
        for axis in [1, 2, 3, 4]:
            header['CRPIX{0}'.format(axis)] = 1
        self.lookup = FITSimage(externaldata=data, externalheader=header)

    def _density(self, energy, R, z, component):
        """This method is wrapped by __call__. See description there."""
        if component == 'CMB':
            return Schlickeiser()(energy, 'CMB')
        return NotImplementedError

    def __call__(self, energy, R, z, component='Total'):
        """Evaluate model.

        Parameters
        ----------
        energy : float or array
            Photon energy (eV)
        R : float
            Galactic radius (kpc)
        z : float
            Galactic height (kpc)
        component : {'Optical', 'Infrared', 'CMB', 'Total'}
            Radiation field component

        Returns
        -------
        Photon number density of a given component (cm^-3 eV^-1)
        """
        if component == 'Total':
            total = 0
            for component in self.components:
                total += self._density(energy, R, z, component)
            return total
        elif component in self.components:
            return self._density(energy, R, z, component)
        else:
            raise IndexError('Component {0} not available.'
                             ''.format(component))
