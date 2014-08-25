# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from astropy import log
from astropy.io import fits
from astropy.units import Quantity
from astropy.coordinates import Angle
from ..extern.validator import validate_physical_type
from ..utils.array import array_stats_str
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
        from gammapy.datasets import load_psf_fits_table
        psf = EnergyDependentMultiGaussPSF.from_fits(load_psf_fits_table())
        psf.plot_containment(0.68, show_save_energy=False)
        plt.show()

    """
    def __init__(self, energy_lo, energy_hi, theta, sigmas, norms,
                 energy_thresh_lo=Quantity(0.1, 'TeV'),
                 energy_thresh_hi=Quantity(100, 'TeV'),
                 azimuth=Quantity([0], 'deg'), zenith=Quantity([0], 'deg')):

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
        self._azimuth = azimuth
        self._zenith = zenith

    @staticmethod
    def read(filename):
        """Read FITS format PSF file.

        Parameters
        ----------
        filename : str
            File name

        Returns
        -------
        psf : `EnergyDependentMultiGaussPSF`
            PSF
        """
        hdu_list = fits.open(filename)
        return EnergyDependentMultiGaussPSF.from_fits(hdu_list)

    @staticmethod
    def from_fits(hdu_list):
        """
        Create EnergyDependentMultiGaussPSF from HDU list.

        Parameters
        ----------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list with correct extensions.

        Returns
        -------
        psf : `EnergyDependentMultiGaussPSF`
            PSF
        """
        extension = 'POINT SPREAD FUNCTION'
        energy_lo = Quantity(hdu_list[extension].data['ENERG_LO'][0], 'TeV')
        energy_hi = Quantity(hdu_list[extension].data['ENERG_HI'][0], 'TeV')
        theta = Angle(hdu_list[extension].data['THETA_LO'][0], 'degree')

        # Get sigmas
        shape = (len(theta), len(energy_hi))
        sigmas = []
        for key in ['SIGMA_1', 'SIGMA_2', 'SIGMA_3']:
            sigmas.append(hdu_list[extension].data[key].reshape(shape))

        # Get amplitudes
        norms = []
        for key in ['SCALE', 'AMPL_2', 'AMPL_3']:
            norms.append(hdu_list[extension].data[key].reshape(shape))
        try:
            energy_thresh_lo = Quantity(hdu_list[extension].header['LO_THRES'], 'TeV')
            energy_thresh_hi = Quantity(hdu_list[extension].header['HI_THRES'], 'TeV')
            return EnergyDependentMultiGaussPSF(energy_lo, energy_hi, theta, sigmas,
                                                norms, energy_thresh_lo, energy_thresh_hi)
        except KeyError:
            log.warn('No safe energy thresholds found. Setting to default')
            return EnergyDependentMultiGaussPSF(energy_lo, energy_hi, theta, sigmas, norms)

    def to_fits(self, header=None, **kwargs):
        """
        Convert psf table data to FITS hdu list.

        Any FITS header keyword can be passed to the function and will be
        changed in the header.

        Parameters
        ----------
        header : `~astropy.io.fits.header.Header`
            Header to be written in the fits file.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            PSF in HDU list format.
        """
        # Set up header
        if header is None:
            from ..datasets import load_psf_fits_table
            header = load_psf_fits_table()[1].header
        header['LO_THRES'] = self.energy_thresh_lo.value
        header['HI_THRES'] = self.energy_thresh_hi.value

        for key, value in kwargs.items():
            header[key] = value

        # Set up data
        names = ['ENERG_LO', 'ENERG_HI', 'THETA_LO', 'THETA_HI', 'AZIMUTH_LO',
                 'AZIMUTH_HI', 'ZENITH_LO', 'ZENITH_HI', 'SCALE', 'SIGMA_1',
                 'AMPL_2', 'SIGMA_2', 'AMPL_3', 'SIGMA_3']
        formats = ['15E', '15E', '12E', '12E', '1E', '1E', '1E', '1E', '180E',
                   '180E', '180E', '180E', '180E', '180E']
        data = [self.energy_lo, self.energy_hi, self.theta, self.theta,
                self._azimuth, self._azimuth, self._zenith, self._zenith,
                self.norms[0].flatten(), self.sigmas[0].flatten(),
                self.norms[1].flatten(), self.sigmas[1].flatten(),
                self.norms[2].flatten(), self.sigmas[2].flatten()]
        units = ['TeV', 'TeV', 'deg', 'deg', 'deg', 'deg', 'deg', 'deg',
                 '', 'deg', '', 'deg', '', 'deg']

        # Set up columns
        columns = []
        for name_, format_, data_, unit_ in zip(names, formats, data, units):
            if isinstance(data_, Quantity):
                data_ = data_.value
            columns.append(fits.Column(name=name_, format=format_,
                                       array=[data_], unit=unit_))
        # Create hdu and hdu list
        prim_hdu = fits.PrimaryHDU()
        hdu = fits.BinTableHDU.from_columns(columns)
        hdu.header = header
        hdu.add_checksum()
        hdu.add_datasum()
        return fits.HDUList([prim_hdu, hdu])

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
        validate_physical_type('energy', energy, 'energy')
        # TODO: validate `theta`

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

    def _containment_radius_array(self, energies, thetas, fraction):
        """Compute containment for all energy and theta values"""
        containment = np.empty((len(thetas), len(energies)))
        for j, energy in enumerate(energies):
            for i, theta in enumerate(thetas):
                psf = self.psf_at_energy_and_theta(energy, theta)
                try:
                    containment[i, j] = psf.containment_radius(fraction)
                except ValueError:
                    log.debug("Computing containment failed for E = {0:.2f}"
                              " and Theta={1:.2f}".format(energy, theta))
                    log.debug("Sigmas: {0} Norms: {1}".format(psf.sigmas, psf.norms))
                    containment[i, j] = np.nan
        return Quantity(containment, 'deg')

    def plot_containment(self, fraction, filename=None, show_save_energy=True):
        """
        Plot containment image with energy and theta axes.

        Parameters
        ----------
        fraction : float
            Containment fraction between 0 and 1.
        filename : string
            Filename under which the plot is saved.
        """
        import matplotlib.pyplot as plt

        # Set up and compute data
        containment = self._containment_radius_array(self.energy_hi, self.theta, fraction)
        # Plotting
        plt.figure(figsize=(8, 5))
        plt.imshow(containment.value, origin='lower', interpolation='None',
                   vmin=0.05, vmax=0.3)

        if show_save_energy:
            # Log scale transformation for position of energy threshold
            e_min = self.energy_hi.value.min()
            e_max = self.energy_hi.value.max()
            e = (self.energy_thresh_lo.value - e_min) / (e_max - e_min)
            x = (np.log10(e * (e_max / e_min - 1) + 1) / np.log10(e_max / e_min)
                 * (len(self.energy_hi) + 1))
            plt.vlines(x, -0.5, len(self.theta) - 0.5)
            plt.text(x + 0.5, 0, 'Safe energy threshold: {0:3.2f}'.format(self.energy_thresh_lo))

        # Axes labels and ticks, colobar
        plt.xlabel('E [TeV]')
        xticks = ["{0:3.2g}".format(_) for _ in self.energy_hi.value]
        plt.xticks(np.arange(len(self.energy_hi)) + 0.5, xticks, size=9)
        plt.ylabel('Theta [deg]')
        yticks = ["{0:3.2g}".format(_) for _ in self.theta.value]
        plt.yticks(np.arange(len(self.theta)), yticks, size=9)
        cbar = plt.colorbar(fraction=0.1, pad=0.01)
        cbar.set_label('Containment radius R{0:.0f} [deg]'.format(100 * fraction),
                        labelpad=20)

        if filename is not None:
            log.info('Wrote {0}'.format(filename))
            plt.savefig(filename)

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
        ss += array_stats_str(self.theta.to('degree'), 'Theta')
        ss += array_stats_str(self.energy_hi, 'Energy hi')
        ss += array_stats_str(self.energy_lo, 'Energy lo')
        ss += 'Safe energy threshold lo: {0:6.3f}\n'.format(self.energy_thresh_lo)
        ss += 'Safe energy threshold hi: {0:6.3f}\n'.format(self.energy_thresh_hi)

        for fraction in fractions:
            containment = self._containment_radius_array(energies, thetas, fraction)
            for i, energy in enumerate(energies):
                for j, theta in enumerate(thetas):
                    radius = containment[j, i]
                    ss += ("{0:2.0f}% containment radius at theta = {1} and "
                           "E = {2:4.1f}: {3:5.8f}\n"
                           "".format(100 * fraction, theta, energy, radius))
        return ss
