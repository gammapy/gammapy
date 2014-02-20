# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.io import fits

__all__ = ['FermiPSF']

class FermiPSF(object):
    """Fermi PSF I/O and computations (``gtpsf`` format).
    
    TODO: linear interpolation in theta and energy?
    
    Parameters
    ----------
    TODO
    """
    
    def __init__(self, hdu_list):
        self.hdu_list = hdu_list
        self.theta = hdu_list['THETA'].data['Theta']
        self.energy = hdu_list['PSF'].data['Energy']
        self.exposure = hdu_list['PSF'].data['Exposure']
        self.psf_data = hdu_list['PSF'].data['PSF']
    
    @staticmethod
    def read(filename):
        """Read FITS format PSF file (``gtpsf`` output).
        
        Parameters
        ----------
        TODO
        
        Returns
        -------
        TODO
        """
        hdu_list = fits.open(filename)
        return FermiPSF(hdu_list)

    def containment_radius(self, energy, fraction):
        """Compute containment radius at a given energy and containment fraction.
        
        Parameters
        ----------
        energy : float
            Energy (GeV)
        fraction : float
            Containment fraction in %
        
        Returns
        -------
        radius : float
            Containment radius in deg
        """
        #psf = self._psf(energy)
        #radius = 
        #return radius
        pass

    def containment_fraction(self, energy, theta):
        """Compute containment fraction for a given containment radius.
        
        Parameters
        ----------
        energy : float
            Energy (GeV)
        radius : float
            Containment radius in deg        
        
        Returns
        -------
        fraction : float
            Containment fraction in %
        """
        psf = self._psf(energy)
        theta_max = self._theta_index(theta)
        t = np.radians(self.theta)
        fraction_per_bin = 2 * np.pi * t[:-1] * psf[:-1] * np.diff(t) 
        fraction = fraction_per_bin[0:theta_max].sum()
        return fraction

    def _energy_index(self, energy):
        return np.searchsorted(self.energy, energy)

    def _theta_index(self, theta):
        return np.searchsorted(self.theta, theta)

    def _psf(self, energy):
        energy_index = self._energy_index(energy)
        psf = self.psf_data[energy_index, :]
        return psf
    
    def info(self):
        """Print into about energy and theta binning."""
        r68 = self.containment_radius(energy=10, fraction=0.68)
        print('68% containment radius at 10 GeV: {0}'.format(r68))

    def plot_theta(self, filename):
        """Plot PSF vs theta."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        for energy in [1e4, 1e5, 1e6]:
            energy_index = self._energy_index(energy)
            psf = self.psf_data[energy_index, :]
            label = '{0} GeV'.format(1e-3 * energy)
            x = np.hstack([-self.theta[::-1], self.theta])
            y = 1e-6 * np.hstack([psf[::-1], psf])
            plt.plot(x, y, lw=2, label=label)
        #plt.semilogy()
        #plt.loglog()
        plt.legend()
        plt.xlim(-0.2, 0.5)
        plt.xlabel('Position (deg)')
        plt.ylabel('PSF probability density (1e-6 sr^-1)')
        plt.tight_layout()
        plt.savefig(filename)
    
    def plot_containment(self, filename):
        """Plot containment versus energy."""
        raise NotImplementedError
        import matplotlib.pyplot as plt
        plt.clf()
        plt.savefig(filename)

    def plot_exposure(self, filename):
        """Plot exposure versus energy."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 3))
        plt.plot(self.energy, self.exposure, color='black', lw=3)
        plt.semilogx()
        plt.xlabel('Energy (MeV)')
        plt.ylabel('Exposure (cm^2 s)')
        plt.xlim(1e4 / 1.3, 1.3 * 1e6)
        plt.ylim(0, 1.5e11)
        plt.tight_layout()
        plt.savefig(filename)
