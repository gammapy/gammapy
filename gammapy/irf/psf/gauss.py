# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from gammapy.maps import MapAxis
from gammapy.utils.gauss import MultiGauss2D, Gauss2DPDF
from .table import PSF3D, EnergyDependentTablePSF
from .core import ParametricPSF

__all__ = ["EnergyDependentMultiGaussPSF"]

log = logging.getLogger(__name__)


class EnergyDependentMultiGaussPSF(ParametricPSF):
    """Triple Gauss analytical PSF depending on energy and theta.

    To evaluate the PSF call the ``to_energy_dependent_table_psf`` or ``psf_at_energy_and_theta`` methods.

    Parameters
    ----------
    energy_axis_true : `MapAxis`
        True energy axis
    offset_axis : `MapAxis`
        Offset axis.
    sigmas : list of 'numpy.ndarray'
        Triple Gauss sigma parameters, where every entry is
        a two dimensional 'numpy.ndarray' containing the sigma
        value for every given energy and theta.
    norms : list of 'numpy.ndarray'
        Triple Gauss norm parameters, where every entry is
        a two dimensional 'numpy.ndarray' containing the norm
        value for every given energy and theta. Norm corresponds
        to the value of the Gaussian at theta = 0.
    meta : dict
        Meta data

    Examples
    --------
    Plot R68 of the PSF vs. theta and energy:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from gammapy.irf import EnergyDependentMultiGaussPSF
        filename = '$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits'
        psf = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')
        psf.plot_containment(0.68)
        plt.show()
    """
    tag = "psf_3gauss"
    required_axes = ["energy_true", "offset"]
    required_parameters = ["SIGMA_1", "SIGMA_2", "SIGMA_3", "SCALE", "AMPL_2", "AMPL_3"]

    def evaluate(self, energy, offset):
        """"""
        energy = u.Quantity(energy)
        offset = u.Quantity(offset)

        sigmas, norms = [], []

        pars = {"AMPL_1": 1}

        for name in ["SIGMA_1", "SIGMA_2", "SIGMA_3"]:
            interp = self._interpolators[name]
            sigmas.append(interp((energy, offset)))

        for name in ["SCALE", "AMPL_2", "AMPL_3"]:
            interp = self._interpolators[name]
            pars[name] = interp((energy, offset))

        for idx, sigma in enumerate(sigmas):
            a = pars[f"AMPL_{idx + 1}"]
            norm = pars["SCALE"] * 2 * a * sigma ** 2 * u.Unit("deg-2")
            norms.append(norm)

        return {"norms": norms, "sigmas": sigmas}

    @staticmethod
    def evaluate_direct(rad, norms, sigmas):
        """Evaluate psf model"""
        value = np.zeros(rad.shape) * u.Unit("deg-2")

        for norm, sigma in zip(norms, sigmas):
            value += norm / (2 * np.pi * sigma ** 2) * np.exp(-0.5 * (rad / sigma) ** 2)

        return value

    def psf_at_energy_and_theta(self, energy, offset):
        """
        Get `~gammapy.modeling.models.MultiGauss2D` model for given energy and theta.

        No interpolation is used.

        Parameters
        ----------
        energy : `~astropy.units.u.Quantity`
            Energy at which a PSF is requested.
        theta : `~astropy.coordinates.Angle`
            Offset angle at which a PSF is requested.

        Returns
        -------
        psf : `~gammapy.utils.gauss.MultiGauss2D`
            Multigauss PSF object.
        """
        pars = self.evaluate(energy=energy, offset=offset)
        m = MultiGauss2D(pars["sigmas"], pars["norms"])
        m.normalize()
        return m

    def containment_radius(self, energy, theta, fraction=0.68):
        """Compute containment for all energy and theta values"""
        # This is a false positive from pylint
        # See https://github.com/PyCQA/pylint/issues/2435
        energies = u.Quantity(
            energy
        ).flatten()  # pylint:disable=assignment-from-no-return
        thetas = Angle(theta).flatten()
        radius = np.empty((theta.size, energy.size))

        for idx, energy in enumerate(energies):
            for jdx, theta in enumerate(thetas):
                try:
                    psf = self.psf_at_energy_and_theta(energy, theta)
                    radius[jdx, idx] = psf.containment_radius(fraction)
                except ValueError:
                    log.debug(
                        f"Computing containment failed for energy = {energy:.2f}"
                        f" and theta={theta:.2f}"
                    )
                    log.debug(f"Sigmas: {psf.sigmas} Norms: {psf.norms}")
                    radius[jdx, idx] = np.nan
        return Angle(radius, "deg")

    def to_psf3d(self, rad=None):
        """Create a PSF3D from an analytical PSF.

        Parameters
        ----------
        rad : `~astropy.units.u.Quantity` or `~astropy.coordinates.Angle`
            the array of position errors (rad) on which the PSF3D will be defined

        Returns
        -------
        psf3d : `~gammapy.irf.PSF3D`
            the PSF3D. It will be defined on the same energy and offset values than the input psf.
        """
        offset_axis = self.axes["offset"]
        energy_axis_true = self.axes["energy_true"]

        if rad is None:
            rad = np.linspace(0, 0.66, 67) * u.deg

        rad_axis = MapAxis.from_edges(rad, name="rad")

        shape = (energy_axis_true.nbin, offset_axis.nbin, rad_axis.nbin)
        psf_value = np.zeros(shape) * u.Unit("sr-1")

        for idx, offset in enumerate(offset_axis.center):
            table_psf = self.to_energy_dependent_table_psf(offset)
            psf_value[:, idx, :] = table_psf.evaluate(
                energy_true=energy_axis_true.center[:, np.newaxis], rad=rad_axis.center
            )

        return PSF3D(
            axes=[energy_axis_true, offset_axis, rad_axis],
            data=psf_value.value,
            unit=psf_value.unit,
            meta=self.meta.copy()
        )
