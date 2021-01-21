# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from astropy import units as u
from gammapy.maps import MapAxis
from gammapy.utils.gauss import MultiGauss2D
from .core import ParametricPSF

__all__ = ["EnergyDependentMultiGaussPSF"]

log = logging.getLogger(__name__)


class EnergyDependentMultiGaussPSF(ParametricPSF):
    """Triple Gauss analytical PSF depending on energy and theta.

    Parameters
    ----------
    axes : list of `MapAxis`
        Required axes are ["energy_true", "offset"]
    data : `~numpy.recarray`
        Data array
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

    def containment(self, rad, **kwargs):
        """Containment of the PSF at given axes coordinates

        Parameters
        ----------
        rad : `~astropy.units.Quantity`
            Rad value
        **kwargs : dict
            Other coordinates

        Returns
        -------
        containment : `~numpy.ndarray`
            Containment
        """
        pars = self.evaluate_parameters(**kwargs)
        m = MultiGauss2D(**pars)
        m.normalize()
        containment = m.containment_fraction(rad)
        return containment

    def evaluate_parameters(self, energy_true, offset):
        """"""
        energy = u.Quantity(energy_true)
        offset = u.Quantity(offset)

        sigmas, norms = [], []

        pars = {"A_1": 1}

        for name in ["SIGMA_1", "SIGMA_2", "SIGMA_3"]:
            sigma = self._interpolators[name]((energy, offset))
            sigmas.append(sigma)

        for name, interp_name in zip(["scale", "A_2", "A_3"], ["SCALE", "AMPL_2", "AMPL_3"]):
            interp = self._interpolators[interp_name]
            pars[name] = interp((energy, offset))

        for idx, sigma in enumerate(sigmas):
            a = pars[f"A_{idx + 1}"]
            norm = (pars["scale"] * 2 * a * sigma ** 2).to_value(sigma.unit ** 2)
            norms.append(norm)

        return {"norms": norms, "sigmas": sigmas}

    def evaluate(self, rad, energy_true, offset):
        """Evaluate psf model"""
        pars = self.evaluate_parameters(energy_true=energy_true, offset=offset)

        m = MultiGauss2D(**pars)
        m.normalize()
        return m(rad)
