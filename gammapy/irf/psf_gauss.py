# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import logging
from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity, Unit
from astropy.coordinates import Angle
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from ..extern.validator import validate_physical_type
from ..utils.array import array_stats_str
from ..utils.energy import Energy, EnergyBounds
from ..utils.scripts import make_path
from ..utils.gauss import MultiGauss2D
from .psf_3d import PSF3D
from . import EnergyDependentTablePSF

__all__ = ["EnergyDependentMultiGaussPSF"]

log = logging.getLogger(__name__)


class EnergyDependentMultiGaussPSF(object):
    """
    Triple Gauss analytical PSF depending on energy and theta.

    To evaluate the PSF call the ``to_energy_dependent_table_psf`` or ``psf_at_energy_and_theta`` methods.

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

    def __init__(
        self,
        energy_lo,
        energy_hi,
        theta,
        sigmas,
        norms,
        energy_thresh_lo=Quantity(0.1, "TeV"),
        energy_thresh_hi=Quantity(100, "TeV"),
    ):

        # Validate input
        validate_physical_type("energy_lo", energy_lo, "energy")
        validate_physical_type("energy_hi", energy_hi, "energy")
        validate_physical_type("theta", theta, "angle")
        validate_physical_type("energy_thresh_lo", energy_thresh_lo, "energy")
        validate_physical_type("energy_thresh_hi", energy_thresh_hi, "energy")

        # Set attributes
        self.energy_lo = energy_lo.to("TeV")
        self.energy_hi = energy_hi.to("TeV")
        ebounds = EnergyBounds.from_lower_and_upper_bounds(
            self.energy_lo, self.energy_hi
        )
        self.energy = ebounds.log_centers
        self.theta = theta.to("deg")
        sigmas[0][sigmas[0] == 0] = 1
        sigmas[1][sigmas[1] == 0] = 1
        sigmas[2][sigmas[2] == 0] = 1
        self.sigmas = sigmas

        self.norms = norms
        self.energy_thresh_lo = energy_thresh_lo.to("TeV")
        self.energy_thresh_hi = energy_thresh_hi.to("TeV")

    @classmethod
    def read(cls, filename, hdu="PSF_2D_GAUSS"):
        """Create `EnergyDependentMultiGaussPSF` from FITS file.

        Parameters
        ----------
        filename : str
            File name
        """
        filename = make_path(filename)
        with fits.open(str(filename), memmap=False) as hdulist:
            psf = cls.from_fits(hdulist[hdu])

        return psf

    @classmethod
    def from_fits(cls, hdu):
        """Create `EnergyDependentMultiGaussPSF` from HDU list.

        Parameters
        ----------
        hdu : `~astropy.io.fits.BintableHDU`
            HDU
        """
        energy_lo = Quantity(hdu.data["ENERG_LO"][0], "TeV")
        energy_hi = Quantity(hdu.data["ENERG_HI"][0], "TeV")
        theta = Angle(hdu.data["THETA_LO"][0], "deg")

        # Get sigmas
        shape = (len(theta), len(energy_hi))
        sigmas = []
        for key in ["SIGMA_1", "SIGMA_2", "SIGMA_3"]:
            sigma = hdu.data[key].reshape(shape).copy()
            sigmas.append(sigma)

        # Get amplitudes
        norms = []
        for key in ["SCALE", "AMPL_2", "AMPL_3"]:
            norm = hdu.data[key].reshape(shape).copy()
            norms.append(norm)

        opts = {}
        try:
            opts["energy_thresh_lo"] = Quantity(hdu.header["LO_THRES"], "TeV")
            opts["energy_thresh_hi"] = Quantity(hdu.header["HI_THRES"], "TeV")
        except KeyError:
            pass

        return cls(energy_lo, energy_hi, theta, sigmas, norms, **opts)

    def to_fits(self):
        """
        Convert psf table data to FITS hdu list.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            PSF in HDU list format.
        """
        # Set up data
        names = [
            "ENERG_LO",
            "ENERG_HI",
            "THETA_LO",
            "THETA_HI",
            "SCALE",
            "SIGMA_1",
            "AMPL_2",
            "SIGMA_2",
            "AMPL_3",
            "SIGMA_3",
        ]
        units = ["TeV", "TeV", "deg", "deg", "", "deg", "", "deg", "", "deg"]

        data = [
            self.energy_lo,
            self.energy_hi,
            self.theta,
            self.theta,
            self.norms[0],
            self.sigmas[0],
            self.norms[1],
            self.sigmas[1],
            self.norms[2],
            self.sigmas[2],
        ]

        table = Table()
        for name_, data_, unit_ in zip(names, data, units):
            table[name_] = [data_]
            table[name_].unit = unit_

        # Create hdu and hdu list
        hdu = fits.BinTableHDU(table)
        hdu.header["LO_THRES"] = self.energy_thresh_lo.value
        hdu.header["HI_THRES"] = self.energy_thresh_hi.value

        return fits.HDUList([fits.PrimaryHDU(), hdu])

    def write(self, filename, *args, **kwargs):
        """Write PSF to FITS file.

        Calls `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_fits().writeto(filename, *args, **kwargs)

    def psf_at_energy_and_theta(self, energy, theta):
        """
        Get `~gammapy.image.models.MultiGauss2D` model for given energy and theta.

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
        i = np.argmin(np.abs(self.energy - energy))
        j = np.argmin(np.abs(self.theta - theta))

        # TODO: Use some kind of interpolation to get PSF
        # parameters for every energy and theta

        # Select correct gauss parameters for given energy and theta
        sigmas = [_[j][i] for _ in self.sigmas]
        norms = [_[j][i] for _ in self.norms]

        pars = {}
        pars["scale"], pars["A_2"], pars["A_3"] = norms
        pars["sigma_1"], pars["sigma_2"], pars["sigma_3"] = sigmas
        psf = HESSMultiGaussPSF(pars)
        return psf.to_MultiGauss2D(normalize=True)

    def containment_radius(self, energy, theta, fraction=0.68):
        """Compute containment for all energy and theta values"""
        # This is a false positive from pylint
        # See https://github.com/PyCQA/pylint/issues/2435
        energy = Energy(energy).flatten()  # pylint:disable=assignment-from-no-return
        theta = Angle(theta).flatten()
        radius = np.empty((theta.size, energy.size))

        for idx_energy in range(len(energy)):
            for idx_theta in range(len(theta)):
                try:
                    psf = self.psf_at_energy_and_theta(
                        energy[idx_energy], theta[idx_theta]
                    )
                    radius[idx_theta, idx_energy] = psf.containment_radius(fraction)
                except ValueError:
                    log.debug(
                        "Computing containment failed for E = {:.2f}"
                        " and Theta={:.2f}".format(energy[idx_energy], theta[idx_theta])
                    )
                    log.debug("Sigmas: {} Norms: {}".format(psf.sigmas, psf.norms))
                    radius[idx_theta, idx_energy] = np.nan

        return Angle(radius, "deg")

    def plot_containment(
        self, fraction=0.68, ax=None, show_safe_energy=False, add_cbar=True, **kwargs
    ):
        """
        Plot containment image with energy and theta axes.

        Parameters
        ----------
        fraction : float
            Containment fraction between 0 and 1.
        add_cbar : bool
            Add a colorbar
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        energy = self.energy_hi
        offset = self.theta

        # Set up and compute data
        containment = self.containment_radius(energy, offset, fraction)

        # plotting defaults
        kwargs.setdefault("cmap", "GnBu")
        kwargs.setdefault("vmin", np.nanmin(containment.value))
        kwargs.setdefault("vmax", np.nanmax(containment.value))

        # Plotting
        x = energy.value
        y = offset.value
        caxes = ax.pcolormesh(x, y, containment.value, **kwargs)

        # Axes labels and ticks, colobar
        ax.semilogx()
        ax.set_ylabel("Offset ({unit})".format(unit=offset.unit))
        ax.set_xlabel("Energy ({unit})".format(unit=energy.unit))
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

        if show_safe_energy:
            self._plot_safe_energy_range(ax)

        if add_cbar:
            label = "Containment radius R{0:.0f} ({1})".format(
                100 * fraction, containment.unit
            )
            ax.figure.colorbar(caxes, ax=ax, label=label)

        return ax

    def _plot_safe_energy_range(self, ax):
        """add safe energy range lines to the plot"""
        esafe = self.energy_thresh_lo
        omin = self.offset.value.min()
        omax = self.offset.value.max()
        ax.hlines(y=esafe.value, xmin=omin, xmax=omax)
        label = "Safe energy threshold: {0:3.2f}".format(esafe)
        ax.text(x=0.1, y=0.9 * esafe.value, s=label, va="top")

    def plot_containment_vs_energy(
        self, fractions=[0.68, 0.95], thetas=Angle([0, 1], "deg"), ax=None, **kwargs
    ):
        """Plot containment fraction as a function of energy.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        energy = Energy.equal_log_spacing(self.energy_lo[0], self.energy_hi[-1], 100)

        for theta in thetas:
            for fraction in fractions:
                radius = self.containment_radius(energy, theta, fraction).squeeze()
                label = "{} deg, {:.1f}%".format(theta, 100 * fraction)
                ax.plot(energy.value, radius.value, label=label)

        ax.semilogx()
        ax.legend(loc="best")
        ax.set_xlabel("Energy (TeV)")
        ax.set_ylabel("Containment radius (deg)")

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

    def info(
        self,
        fractions=[0.68, 0.95],
        energies=Quantity([1., 10.], "TeV"),
        thetas=Quantity([0.], "deg"),
    ):
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
        ss += array_stats_str(self.theta.to("deg"), "Theta")
        ss += array_stats_str(self.energy_hi, "Energy hi")
        ss += array_stats_str(self.energy_lo, "Energy lo")
        ss += "Safe energy threshold lo: {0:6.3f}\n".format(self.energy_thresh_lo)
        ss += "Safe energy threshold hi: {0:6.3f}\n".format(self.energy_thresh_hi)

        for fraction in fractions:
            containment = self.containment_radius(energies, thetas, fraction)
            for i, energy in enumerate(energies):
                for j, theta in enumerate(thetas):
                    radius = containment[j, i]
                    ss += (
                        "{0:2.0f}% containment radius at theta = {1} and "
                        "E = {2:4.1f}: {3:5.8f}\n"
                        "".format(100 * fraction, theta, energy, radius)
                    )
        return ss

    def to_energy_dependent_table_psf(self, theta=None, rad=None, exposure=None):
        """
        Convert triple Gaussian PSF ot table PSF.

        Parameters
        ----------
        theta : `~astropy.coordinates.Angle`
            Offset in the field of view. Default theta = 0 deg
        rad : `~astropy.coordinates.Angle`
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
        energies = self.energy
        # Defaults and input handling
        if theta is None:
            theta = Angle(0, "deg")
        else:
            theta = Angle(theta)

        if rad is None:
            rad = Angle(np.arange(0, 1.5, 0.005), "deg")
        else:
            rad = Angle(rad).to("deg")

        psf_value = Quantity(np.zeros((energies.size, rad.size)), "deg^-2")

        for idx, energy in enumerate(energies):
            psf_gauss = self.psf_at_energy_and_theta(energy, theta)
            psf_value[idx] = Quantity(psf_gauss(rad), "deg^-2")

        return EnergyDependentTablePSF(
            energy=energies, rad=rad, exposure=exposure, psf_value=psf_value
        )

    def to_psf3d(self, rad):
        """ Creates a PSF3D from an analytical PSF.

        Parameters
        ----------
        rad : `~astropy.units.Quantity` or `~astropy.coordinates.Angle`
            the array of position errors (rad) on which the PSF3D will be defined

        Returns
        -------
        psf3d : `~gammapy.irf.PSF3D`
            the PSF3D. It will be defined on the same energy and offset values than the input psf.
        """
        offsets = self.theta
        energy = self.energy
        energy_lo = self.energy_lo
        energy_hi = self.energy_hi
        rad_lo = rad[:-1]
        rad_hi = rad[1:]

        psf_values = np.zeros(
            (rad_lo.shape[0], offsets.shape[0], energy_lo.shape[0])
        ) * Unit("sr-1")

        for i, offset in enumerate(offsets):
            psftable = self.to_energy_dependent_table_psf(offset)
            psf_values[:, i, :] = psftable.evaluate(energy, 0.5 * (rad_lo + rad_hi)).T

        return PSF3D(
            energy_lo,
            energy_hi,
            offsets,
            rad_lo,
            rad_hi,
            psf_values,
            self.energy_thresh_lo,
            self.energy_thresh_hi,
        )


class HESSMultiGaussPSF(object):
    """Multi-Gauss PSF as represented in the HESS software.

    The 2D Gaussian is represented as a 1D exponential
    probability density function per offset angle squared:
    dp / dtheta**2 = [0]*(exp(-x/(2*[1]*[1]))+[2]*exp(-x/(2*[3]*[3]))

    @param source: either a dict of a filename

    The following two parameters control numerical
    precision / speed. Usually the defaults are fine.
    @param theta_max: Maximum offset in numerical computations
    @param npoints: Number of points in numerical computations
    @param eps: Allowed tolerance on normalization of total P to 1
    """

    def __init__(self, source):
        if isinstance(source, dict):
            # Assume source is a dict with correct format
            self.pars = source
        else:
            # Assume source is a filename with correct format
            self.pars = self._read_ascii(source)
        # Scale will be computed from normalization anyways,
        # so any default is fine here
        self.pars["scale"] = self.pars.get("scale", 1)
        # This avoids handling the first PSF as a special case
        self.pars["A_1"] = self.pars.get("A_1", 1)

    def _read_ascii(self, filename):
        """Parse file with parameters."""
        fh = open(filename)  # .readlines()
        pars = {}
        for line in fh:
            try:
                key, value = line.strip().split()[:2]
                if key.startswith("#"):
                    continue
                else:
                    pars[key] = float(value)
            except ValueError:
                pass
        fh.close()
        return pars

    def n_gauss(self):
        """Count number of Gaussians."""
        return len([_ for _ in self.pars.keys() if "sigma" in _])

    def dpdtheta2(self, theta2):
        """dp / dtheta2 at position theta2 = theta ^ 2."""
        theta2 = np.asarray(theta2, "f")
        total = np.zeros_like(theta2)
        for ii in range(1, self.n_gauss() + 1):
            A = self.pars["A_{}".format(ii)]
            sigma = self.pars["sigma_{}".format(ii)]
            total += A * np.exp(-theta2 / (2 * sigma ** 2))
        return self.pars["scale"] * total

    def to_MultiGauss2D(self, normalize=True):
        """Use this to compute containment angles and fractions.

        Note: We have to set norm = 2 * A * sigma ^ 2, because in
        MultiGauss2D norm represents the integral, and in HESS A
        represents the amplitude at 0."""
        sigmas, norms = [], []
        for ii in range(1, self.n_gauss() + 1):
            A = self.pars["A_{}".format(ii)]
            sigma = self.pars["sigma_{}".format(ii)]
            norm = self.pars["scale"] * 2 * A * sigma ** 2
            sigmas.append(sigma)
            norms.append(norm)
        m = MultiGauss2D(sigmas, norms)
        if normalize:
            m.normalize()
        return m


def multi_gauss_psf_kernel(psf_parameters, BINSZ=0.02, NEW_BINSZ=0.02, **kwargs):
    """Create multi-Gauss PSF kernel.

    The Gaussian PSF components are specified via the
    amplitude at the center and the FWHM.
    See the example for the exact format.

    Parameters
    ----------
    psf_parameters : dict
        PSF parameters
    BINSZ : float (0.02)
        Pixel size used for the given parameters in deg.
    NEW_BINSZ : float (0.02)
        New pixel size in deg. USed to change the resolution of the PSF.

    Returns
    -------
    psf_kernel : `astropy.convolution.Kernel2D`
        PSF kernel

    Examples
    --------
    >>> psf_pars = dict()
    >>> psf_pars['psf1'] = dict(ampl=1, fwhm=2.5)
    >>> psf_pars['psf2'] = dict(ampl=0.06, fwhm=11.14)
    >>> psf_pars['psf3'] = dict(ampl=0.47, fwhm=5.16)
    >>> psf_kernel = multi_gauss_psf_kernel(psf_pars, x_size=51)
    """
    psf = None
    for ii in range(1, 4):
        # Convert sigma and amplitude
        pars = psf_parameters["psf{}".format(ii)]
        sigma = gaussian_fwhm_to_sigma * pars["fwhm"] * BINSZ / NEW_BINSZ
        ampl = 2 * np.pi * sigma ** 2 * pars["ampl"]
        if psf is None:
            psf = float(ampl) * Gaussian2DKernel(sigma, **kwargs)
        else:
            psf += float(ampl) * Gaussian2DKernel(sigma, **kwargs)
    psf.normalize()
    return psf
