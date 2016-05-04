# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

from astropy import log
from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity
from astropy.coordinates import Angle

from . import EnergyDependentTablePSF
from ..extern.validator import validate_physical_type
from ..utils.array import array_stats_str
from ..utils.energy import Energy, EnergyBounds
from ..utils.fits import table_to_fits_table
from ..utils.scripts import make_path
from ..irf import HESSMultiGaussPSF


__all__ = ['EnergyDependentMultiGaussPSF']


# TODO: Improve and add functionality from psf_core to this class
class EnergyDependentMultiGaussPSF(object):
    """
    Triple Gauss analytical PSF depending on energy and theta.

    Parameters
    ----------
    energy_lo : `~astropy.units.Quantity`
        Lower energy boundary of the energy bin.
    energy_hi : `~astropy.units.Quantity`
        Upper energy boundary of the energy bin.
    theta : `~astropy.units.Quantity`
        Center values of the theta bins.
    sigmas : list of 'numpy.ndarray'
        Triple Gauss sigma parameters, where every entry is
        a two dimensional 'numpy.ndarray' containing the sigma
        value for every given energy and theta.
    norms : list of 'numpy.ndarray'
        Triple Gauss norm parameters, where every entry is
        a two dimensional 'numpy.ndarray' containing the norm
        value for every given energy and theta. Norm corresponds
        to the value of the Gaussian at theta = 0.
    energy_thresh_lo : `~astropy.units.Quantity`
        Lower save energy threshold of the psf.
    energy_thresh_hi : `~astropy.units.Quantity`
        Upper save energy threshold of the psf.

    Examples
    --------
    Plot R68 of the PSF vs. theta and energy:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from gammapy.irf import EnergyDependentMultiGaussPSF
        filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/irfs/psf.fits'
        psf = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')
        psf.plot_containment(0.68, show_safe_energy=False)
        plt.show()
    """

    def __init__(self, energy_lo, energy_hi, theta, sigmas, norms,
                 energy_thresh_lo=Quantity(0.1, 'TeV'),
                 energy_thresh_hi=Quantity(100, 'TeV'),
                 ):

        # Validate input
        validate_physical_type('energy_lo', energy_lo, 'energy')
        validate_physical_type('energy_hi', energy_hi, 'energy')
        validate_physical_type('theta', theta, 'angle')
        validate_physical_type('energy_thresh_lo', energy_thresh_lo, 'energy')
        validate_physical_type('energy_thresh_hi', energy_thresh_hi, 'energy')

        # Set attributes
        self.energy_lo = energy_lo.to('TeV')
        self.energy_hi = energy_hi.to('TeV')
        self.theta = theta.to('deg')
        self.sigmas = sigmas
        self.norms = norms
        self.energy_thresh_lo = energy_thresh_lo.to('TeV')
        self.energy_thresh_hi = energy_thresh_hi.to('TeV')

    @classmethod
    def read(cls, filename, hdu='psf_3gauss'):
        """Create `EnergyDependentMultiGaussPSF` from FITS file.

        Parameters
        ----------
        filename : str
            File name
        """
        filename = make_path(filename)
        hdu_list = fits.open(str(filename))
        hdu = hdu_list[hdu]
        return cls.from_fits(hdu)

    @classmethod
    def from_fits(cls, hdu):
        """Create `EnergyDependentMultiGaussPSF` from HDU list.

        Parameters
        ----------
        hdu : `~astropy.io.fits.BintableHDU`
            HDU
        """
        energy_lo = Quantity(hdu.data['ENERG_LO'][0], 'TeV')
        energy_hi = Quantity(hdu.data['ENERG_HI'][0], 'TeV')
        theta = Angle(hdu.data['THETA_LO'][0], 'deg')

        # Get sigmas
        shape = (len(theta), len(energy_hi))
        sigmas = []
        for key in ['SIGMA_1', 'SIGMA_2', 'SIGMA_3']:
            sigmas.append(hdu.data[key].reshape(shape))

        # Get amplitudes
        norms = []
        for key in ['SCALE', 'AMPL_2', 'AMPL_3']:
            norms.append(hdu.data[key].reshape(shape))
        try:
            energy_thresh_lo = Quantity(hdu.header['LO_THRES'], 'TeV')
            energy_thresh_hi = Quantity(hdu.header['HI_THRES'], 'TeV')
            return cls(energy_lo, energy_hi, theta, sigmas,
                       norms, energy_thresh_lo, energy_thresh_hi)
        except KeyError:
            log.warning('No safe energy thresholds found. Setting to default')
            return cls(energy_lo, energy_hi, theta, sigmas, norms)

    def to_fits(self):
        """
        Convert psf table data to FITS hdu list.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            PSF in HDU list format.
        """
        # Set up data
        names = ['ENERG_LO', 'ENERG_HI', 'THETA_LO', 'THETA_HI',
                 'SCALE', 'SIGMA_1', 'AMPL_2', 'SIGMA_2', 'AMPL_3', 'SIGMA_3']
        units = ['TeV', 'TeV', 'deg', 'deg',
                 '', 'deg', '', 'deg', '', 'deg']

        data = [self.energy_lo, self.energy_hi, self.theta, self.theta,
                self.norms[0], self.sigmas[0],
                self.norms[1], self.sigmas[1],
                self.norms[2], self.sigmas[2]]

        table = Table()
        for name_, data_, unit_ in zip(names, data, units):
            table[name_] = [data_]
            table[name_].unit = unit_

        # TODO: add units!?
        # Create hdu and hdu list
        hdu = table_to_fits_table(table)
        hdu.header['LO_THRES'] = self.energy_thresh_lo.value
        hdu.header['HI_THRES'] = self.energy_thresh_hi.value

        return fits.HDUList([fits.PrimaryHDU(), hdu])

    def write(self, filename, *args, **kwargs):
        """Write PSF to FITS file.

        Calls `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_fits().writeto(filename, *args, **kwargs)

    def psf_at_energy_and_theta(self, energy, theta):
        """
        Get `~gammapy.morphology.MultiGauss2D` model for given energy and theta.

        No interpolation is used.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy at which a PSF is requested.
        theta : `~astropy.coordinates.Angle`
            Offset angle at which a PSF is requested.

        Returns
        -------
        psf : `~gammapy.morphology.MultiGauss2D`
            Multigauss PSF object.
        """
        energy = Energy(energy)
        theta = Angle(theta)

        # Find nearest energy value
        i = np.argmin(np.abs(self.energy_hi - energy))
        j = np.argmin(np.abs(self.theta - theta))

        # TODO: Use some kind of interpolation to get PSF
        # parameters for every energy and theta

        # Select correct gauss parameters for given energy and theta
        sigmas = [_[j][i] for _ in self.sigmas]
        norms = [_[j][i] for _ in self.norms]

        pars = {}
        pars['scale'], pars['A_2'], pars['A_3'] = norms
        pars['sigma_1'], pars['sigma_2'], pars['sigma_3'] = sigmas
        psf = HESSMultiGaussPSF(pars)
        return psf.to_MultiGauss2D(normalize=True)

    def containment_radius(self, energy, theta, fraction=0.68):
        """Compute containment for all energy and theta values"""
        energy = Energy(energy).flatten()
        theta = Angle(theta).flatten()
        radius = np.empty((theta.size, energy.size))
        for idx_energy in range(len(energy)):
            for idx_theta in range(len(theta)):
                try:
                    psf = self.psf_at_energy_and_theta(energy[idx_energy], theta[idx_theta])
                    radius[idx_theta, idx_energy] = psf.containment_radius(fraction)
                except ValueError:
                    log.debug("Computing containment failed for E = {0:.2f}"
                              " and Theta={1:.2f}".format(energy[idx_energy], theta[idx_theta]))
                    log.debug("Sigmas: {0} Norms: {1}".format(psf.sigmas, psf.norms))
                    radius[idx_theta, idx_energy] = np.nan

        return Angle(radius, 'deg')

    def plot_containment(self, fraction=0.68, ax=None, show_safe_energy=False,
                         add_cbar=True, **kwargs):
        """
        Plot containment image with energy and theta axes.

        Parameters
        ----------
        fraction : float
            Containment fraction between 0 and 1.
        add_cbar : bool
            Add a colorbar
        """
        from matplotlib.colors import PowerNorm
        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        kwargs.setdefault('cmap', 'afmhot')
        kwargs.setdefault('norm', PowerNorm(gamma=0.5))
        kwargs.setdefault('origin', 'lower')
        kwargs.setdefault('interpolation', 'nearest')
        # kwargs.setdefault('vmin', 0.1)
        # kwargs.setdefault('vmax', 0.2)

        # Set up and compute data
        containment = self.containment_radius(self.energy_hi, self.theta, fraction)

        extent = [
            self.theta[0].value, self.theta[-1].value,
            self.energy_lo[0].value, self.energy_hi[-1].value,
        ]

        # Plotting
        ax.imshow(containment.T.value, extent=extent, **kwargs)

        if show_safe_energy:
            # Log scale transformation for position of energy threshold
            e_min = self.energy_hi.value.min()
            e_max = self.energy_hi.value.max()
            e = (self.energy_thresh_lo.value - e_min) / (e_max - e_min)
            x = (np.log10(e * (e_max / e_min - 1) + 1) / np.log10(e_max / e_min)
                 * (len(self.energy_hi) + 1))
            ax.vlines(x, -0.5, len(self.theta) - 0.5)
            ax.text(x + 0.5, 0, 'Safe energy threshold: {0:3.2f}'.format(self.energy_thresh_lo))

        # Axes labels and ticks, colobar
        ax.semilogy()
        ax.set_xlabel('Offset (deg)')
        ax.set_ylabel('Energy (TeV)')

        if add_cbar:
            ax_cbar = plt.colorbar(fraction=0.1, pad=0.01, shrink=0.9,
                                   mappable=ax.images[0], ax=ax)
            label = 'Containment radius R{0:.0f} (deg)'.format(100 * fraction)
            ax_cbar.set_label(label)

        return ax

    def plot_containment_vs_energy(self, fractions=[0.68, 0.95],
                                   thetas=Angle([0, 1], 'deg'), ax=None, **kwargs):
        """Plot containment fraction as a function of energy.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        energy = Energy.equal_log_spacing(
            self.energy_lo[0], self.energy_hi[-1], 100)

        for theta in thetas:
            for fraction in fractions:
                radius = self.containment_radius(energy, theta, fraction).squeeze()
                label = '{} deg, {:.1f}%'.format(theta, 100 * fraction)
                ax.plot(energy.value, radius.value, label=label)

        ax.semilogx()
        ax.legend(loc='best')
        ax.set_xlabel('Energy (TeV)')
        ax.set_ylabel('Containment radius (deg)')

    def peek(self, figsize=(15, 5)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)

        self.plot_containment(fraction=0.68, ax=axes[0])
        self.plot_containment(fraction=0.95, ax=axes[1])
        self.plot_containment_vs_energy(ax=axes[2])

        # TODO: implement this plot
        # psf = self.psf_at_energy_and_theta(energy='1 TeV', theta='1 deg')
        # psf.plot_components(ax=axes[2])

        plt.tight_layout()
        plt.show()

    def info(self, fractions=[0.68, 0.95], energies=Quantity([1., 10.], 'TeV'),
             thetas=Quantity([0.], 'deg')):
        """
        Print PSF summary info.

        The containment radius for given fraction, energies and thetas is
        computed and printed on the command line.

        Parameters
        ----------
        fractions : list
            Containment fraction to compute containment radius for.
        energies : `~astropy.units.Quantity`
            Energies to compute containment radius for.
        thetas : `~astropy.units.Quantity`
            Thetas to compute containment radius for.

        Returns
        -------
        ss : string
            Formatted string containing the summary info.
        """
        ss = "\nSummary PSF info\n"
        ss += "----------------\n"
        # Summarise data members
        ss += array_stats_str(self.theta.to('deg'), 'Theta')
        ss += array_stats_str(self.energy_hi, 'Energy hi')
        ss += array_stats_str(self.energy_lo, 'Energy lo')
        ss += 'Safe energy threshold lo: {0:6.3f}\n'.format(self.energy_thresh_lo)
        ss += 'Safe energy threshold hi: {0:6.3f}\n'.format(self.energy_thresh_hi)

        for fraction in fractions:
            containment = self.containment_radius(energies, thetas, fraction)
            for i, energy in enumerate(energies):
                for j, theta in enumerate(thetas):
                    radius = containment[j, i]
                    ss += ("{0:2.0f}% containment radius at theta = {1} and "
                           "E = {2:4.1f}: {3:5.8f}\n"
                           "".format(100 * fraction, theta, energy, radius))
        return ss



    def to_table_psf(self, theta=None, offset=None, exposure=None):
        """
        Convert triple Gaussian PSF ot table PSF.

        Parameters
        ----------
        theta : `~astropy.coordinates.Angle`
            Offset in the field of view. Default theta = 0 deg
        offset : `~astropy.coordinates.Angle`
            Offset from PSF center used for evaluating the PSF on a grid.
            Default offset = [0, 0.005, ..., 1.495, 1.5] deg.
        exposure : `~astropy.units.Quantity`
            Energy dependent exposure. Should be in units equivalent to 'cm^2 s'.
            Default exposure = 1.

        Returns
        -------
        tabe_psf : `~gammapy.irf.EnergyDependentTablePSF`
            Instance of `EnergyDependentTablePSF`.
        """
        # Convert energies to log center
        ebounds = EnergyBounds.from_lower_and_upper_bounds(self.energy_lo, self.energy_hi)
        energies = ebounds.log_centers

        # Defaults
        theta = theta or Angle(0, 'deg')
        offset = offset or Angle(np.arange(0, 1.5, 0.005), 'deg')
        psf_value = Quantity(np.empty((len(energies), len(offset))), 'deg^-2')

        for i, energy in enumerate(energies):
            psf_gauss = self.psf_at_energy_and_theta(energy, theta)
            psf_value[i] = Quantity(psf_gauss(offset, np.zeros_like(offset)), 'deg^-2')

        return EnergyDependentTablePSF(energy=energies, offset=offset,
                                       exposure=exposure, psf_value=psf_value)


