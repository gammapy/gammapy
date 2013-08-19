# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Galactic interstellar radiation field (ISRF) models"""
from os.path import join
from astropy.io import fits

class Schlickeiser:
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

    def __call__(self, e, component='Total'):
        """Evaluate model.

        @param e: photon energy (eV)
        @param component: radiation field component.
        One of 'Optical', 'Infrared', 'CMB', 'Total'
        @return: Photon number density of a given component (cm^-3 eV^-1)"""
        if component == 'Total':
            total = 0
            for component in self.components:
                total += self._density(e, component)
            return total
        elif component in self.components:
            return self._density(e, component)
        else:
            raise IndexError('Component {0} not available.'
                             ''.format(component))

    def _density(self):
        raise NotImplementedError

class Galprop:
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

    def _density(self, e, R, z, component):
        """This method is wrapped by __call__. See description there."""
        if component == 'CMB':
            return Schlickeiser()(e, 'CMB')
        # @todo: implement lookup
        return NotImplementedError

    def __call__(self, e, R, z, component='Total'):
        """Evaluate model.

        Parameters:
        e: photon energy (eV)
        R: Galactic radius (kpc)
        z: Galactic height (kpc)
        component = radiation field component
        ('Optical', 'Infrared', 'CMB' or 'Total')
        @return: Photon number density of a given component (cm^-3 eV^-1)"""
        if component == 'Total':
            total = 0
            for component in self.components:
                total += self._density(e, R, z, component)
            return total
        elif component in self.components:
            return self._density(e, R, z, component)
        else:
            raise IndexError('Component {0} not available.'
                             ''.format(component))
