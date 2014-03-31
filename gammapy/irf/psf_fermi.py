# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.io import fits
from astropy.units import Quantity
from ..morphology import Gauss2DPDF

__all__ = ['TablePSF',
           'EnergyDependentTablePSF',
           'make_table_psf',
           ]


class TablePSF(object):
    """Radially-symmetric table PSF.
    
    TODO: explain format.
    """
    def __init__(self, offset, psf_value):
        if not isinstance(offset, Quantity):
            raise ValueError("offset must be a Quantity object.")
        if not isinstance(psf_value, Quantity):
            raise ValueError("psf_value must be a Quantity object.")

        self.offset = offset
        self.psf_value = psf_value

    def eval(self, offset):
        """Evaluate PSF.

        The PSF is defined as dP / (dx dy),
        i.e. the 2-dimensional probability density.

        Parameters
        ----------
        offset : `~astropy.units.Quantity`
            Offset

        Returns
        -------
        psf : `~astropy.units.Quantity`
            PSF = dP / (dx dy) (sr^-1)
        """
        if not isinstance(offset, Quantity):
            raise ValueError("offset must be a Quantity object.")

        offset = offset.to('deg')

        offset_index = self._offset_index(offset)
        psf_value = self.psf_value[offset_index]

        return psf_value

    def _offset_index(self, offset):
        """Find offset array index.
        """
        return np.searchsorted(self.offset, offset)


class EnergyDependentTablePSF(object):
    """Energy-dependent radially-symmetric table PSF (``gtpsf`` format).
    
    TODO: add references and explanations.
    
    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Energy (1-dim)
    offset : `~astropy.units.Quantity`
        Offset (1-dim)
    exposure : `~astropy.units.Quantity`
        Exposure (1-dim)
    psf_value : `~astropy.units.Quantity`
        PSF data (TODO: describe format)
    """
    def __init__(self, energy, offset, exposure, psf_value):
        if not isinstance(energy, Quantity):
            raise ValueError("energy must be a Quantity object.")
        if not isinstance(offset, Quantity):
            raise ValueError("offset must be a Quantity object.")
        if not isinstance(exposure, Quantity):
            raise ValueError("exposure must be a Quantity object.")
        if not isinstance(psf_value, Quantity):
            raise ValueError("psf_value must be a Quantity object.")

        self.energy = energy
        self.offset = offset
        self.exposure = exposure
        self.psf_value = psf_value

    @staticmethod
    def from_fits(hdu_list):
        """Create EnergyDependentTablePSF from ``gtpsf`` format HDU list.

        Parameters
        ----------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list with `THETA` and `PSF` extensions.

        Returns
        -------
        psf : `EnergyDependentTablePSF`
            PSF object.
        """
        offset = Quantity(hdu_list['THETA'].data['Theta'], 'deg')
        energy = Quantity(hdu_list['PSF'].data['Energy'], 'GeV')
        exposure = Quantity(hdu_list['PSF'].data['Exposure'], 'cm^2 s')
        psf_value = Quantity(hdu_list['PSF'].data['PSF'], 'sr^-1')

        return EnergyDependentTablePSF(energy, offset, exposure, psf_value)

    def to_fits(self):
        """Convert PSF to FITS HDU list format.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            PSF in HDU list format.
        """
        # TODO: write HEADER keywords as gtpsf

        data = self.offset
        theta_hdu = fits.BinTableHDU(data=data, name='Theta')

        data = [self.energy, self.exposure, self.psf_value]
        psf_hdu = fits.BinTableHDU(data=data, name='PSF')

        hdu_list = fits.HDUList([theta_hdu, psf_hdu])
        return hdu_list

    @staticmethod
    def read(filename):
        """Read FITS format PSF file (``gtpsf`` output).
        
        Parameters
        ----------
        filename : str
            File name
        
        Returns
        -------
        psf : `EnergyDependentTablePSF`
            PSF object.
        """
        hdu_list = fits.open(filename)
        return EnergyDependentTablePSF.from_fits(hdu_list)

    def write(self, *args, **kwargs):
        """Write PSF to FITS file.

        Calls `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_fits().writeto(*args, **kwargs)

    def eval(self, energy, offset):
        """Evaluate PSF.

        The PSF is defined as dP / (dx dy),
        i.e. the 2-dimensional probability density.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy
        offset : `~astropy.units.Quantity`
            Offset

        Returns
        -------
        psf : `~astropy.units.Quantity`
            PSF = dP / (dx dy) (sr^-1)
        """
        if not isinstance(energy, Quantity):
            raise ValueError("energy must be a Quantity object.")
        if not isinstance(offset, Quantity):
            raise ValueError("offset must be a Quantity object.")

        energy = energy.to('GeV')
        offset = offset.to('deg')

        energy_index = self._energy_index(energy)
        offset_index = self._offset_index(offset)
        psf = self.psf_value[energy_index, offset_index]

        return Quantity(psf, 'sr^-1')

    def table_psf(self, energy):
        """PSF at a given energy.
        
        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy
        """
        psf_value = self._psf(energy)
        return TablePSF(self.offset, psf_value)


    def average_psf(self, energy_band, spectrum):
        """Average PSF in a given energy band.
        
        Parameters
        ----------
        spectrum : callable
            Spectrum (callable with energy as parameter)
        energy_band : `~astropy.units.Quantity`
            Energy band
        """
        energy_indices = self._energy_indices(energy_band)
        energies = self.energy[energy_indices]
        weights = spectrum(energies)
        psfs = []
        for energy, weight in zip(energies, weights):
            psf = self.table_psf(energy)

    def containment_radius(self, energy, fraction):
        """Containment radius.
        
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
        # psf = self._psf(energy)
        # radius = 
        # return radius
        pass

    def containment_fraction(self, energy, offset):
        """Containment fraction.
        
        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy
        offset : `~astropy.units.Quantity`
            Offset
        
        Returns
        -------
        fraction : array_like
            Containment fraction (in range 0 .. 1)
        """
        psf = self._psf(energy)
        offset_max = self._offset_index(offset)
        t = np.radians(self.theta)
        fraction_per_bin = 2 * np.pi * t[:-1] * psf[:-1] * np.diff(t) 
        fraction = fraction_per_bin[0:offset_max].sum()
        return fraction

    def info(self):
        """Print basic info."""
        r68 = self.containment_radius(energy=10, fraction=0.68)
        ss = '68% containment radius at 10 GeV: {0}'.format(r68)
        return ss

    def plot_psf_vs_theta(self, filename, energies=[1e4, 1e5, 1e6]):
        """Plot PSF vs theta."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        for energy in energies:
            energy_index = self._energy_index(energy)
            psf = self.psf_value[energy_index, :]
            label = '{0} GeV'.format(1e-3 * energy)
            x = np.hstack([-self.theta[::-1], self.theta])
            y = 1e-6 * np.hstack([psf[::-1], psf])
            plt.plot(x, y, lw=2, label=label)
        # plt.semilogy()
        # plt.loglog()
        plt.legend()
        plt.xlim(-0.2, 0.5)
        plt.xlabel('Position (deg)')
        plt.ylabel('PSF probability density (1e-6 sr^-1)')
        plt.tight_layout()
        plt.savefig(filename)
    
    def plot_containment_vs_energy(self, filename):
        """Plot containment versus energy."""
        raise NotImplementedError
        import matplotlib.pyplot as plt
        plt.clf()
        plt.savefig(filename)

    def plot_exposure_vs_energy(self, filename):
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

    def _energy_index(self, energy):
        """Find energy array index.
        """
        return np.searchsorted(self.energy, energy)

    def _offset_index(self, offset):
        """Find offset array index.
        """
        return np.searchsorted(self.offset, offset)

    def _psf(self, energy):
        """PSF values.
        TODO: describe better
        """
        energy_index = self._energy_index(energy)
        psf = self.psf_value[energy_index, :]
        return psf


def make_table_psf(shape, width, offset):
    """Make TablePSF objects with commonly used shapes.
    
    This function is mostly useful for examples and testing. 
    
    Parameters
    ----------
    shape : {'disk', 'gauss'}
        PSF shape.
    width : `~astropy.unit.Quantity`
        PSF width (radius for disk, sigma for Gauss).
    offset : `~astropy.units.Quantity`
        Offset angle
    
    Returns
    -------
    psf : `TablePSF`
        Table PSF
    
    Examples
    --------
    >>> import numpy as np
    >>> from gammapy.irf import make_table_psf
    >>> make_table_psf(shape='gauss', width=0.2,
    ...                offset=np.linspace(0, 0.7, 100))
    """
    if not isinstance(width, Quantity):
        raise ValueError("width must be a Quantity object.")
    if not isinstance(offset, Quantity):
        raise ValueError("offset must be a Quantity object.")

    if shape == 'disk':
        amplitude = 1 / (np.pi * width ** 2)
        psf_value = np.where(offset < width, amplitude, 0)
    elif shape == 'gauss':
        gauss2d_pdf = Gauss2DPDF(sigma=width)
        psf_value = gauss2d_pdf(offset.value)
    
    psf_value = Quantity(psf_value, 'sr^-1')

    return TablePSF(offset, psf_value)
