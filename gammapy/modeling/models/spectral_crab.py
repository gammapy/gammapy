# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy import units as u
from gammapy.modeling import Parameter
from .spectral import (
    ExpCutoffPowerLawSpectralModel,
    LogParabolaSpectralModel,
    PowerLawSpectralModel,
    SpectralModel,
)

__all__ = [
    "create_crab_spectral_model",
    "MeyerCrabSpectralModel",
]


class MeyerCrabSpectralModel(SpectralModel):
    """Meyer 2010 log polynomial Crab spectral model.

    Reference: https://ui.adsabs.harvard.edu/abs/2010A%26A...523A...2M, Appendix D
    """

    norm = Parameter("norm", value=1, frozen=True, is_norm=True)
    coefficients = [-0.00449161, 0, 0.0473174, -0.179475, -0.53616, -10.2708]

    @staticmethod
    def evaluate(energy, norm):
        polynomial = np.poly1d(MeyerCrabSpectralModel.coefficients)
        log_energy = np.log10(energy.to_value("TeV"))
        log_flux = polynomial(log_energy)
        flux = u.Quantity(np.power(10, log_flux), "erg / (cm2 s)", copy=False)
        return norm * flux / energy**2


def create_crab_spectral_model(reference="meyer"):
    """Create a Crab nebula reference spectral model.

    The Crab nebula is often used as a standard candle in gamma-ray astronomy.
    Fluxes and sensitivities are often quoted relative to the Crab spectrum.

    The following references are available:

    * 'meyer', https://ui.adsabs.harvard.edu/abs/2010A%26A...523A...2M, Appendix D
    * 'hegra', https://ui.adsabs.harvard.edu/abs/2000ApJ...539..317A
    * 'hess_pl' and 'hess_ecpl': https://ui.adsabs.harvard.edu/abs/2006A%26A...457..899A
    * 'magic_lp' and 'magic_ecpl': https://ui.adsabs.harvard.edu/abs/2015JHEAp...5...30A

    Parameters
    ----------
    reference : {'meyer', 'hegra', 'hess_pl', 'hess_ecpl', 'magic_lp', 'magic_ecpl'}
        Which reference to use for the spectral model.

    Examples
    --------
    Let's first import what we need::

        import astropy.units as u
        from gammapy.modeling.models import PowerLaw, create_crab_spectral_model

    Plot the 'hess_ecpl' reference Crab spectrum between 1 TeV and 100 TeV::

        crab_hess_ecpl = create_crab_spectral_model('hess_ecpl')
        crab_hess_ecpl.plot([1, 100] * u.TeV)

    Use a reference crab spectrum as unit to measure a differential flux (at 10 TeV)::

        >>> pwl = PowerLawSpectralModel(
                index=2.3, amplitude=1e-12 * u.Unit('1 / (cm2 s TeV)'), reference=1 * u.TeV
            )
        >>> crab = create_crab_spectral_model('hess_pl')
        >>> energy = 10 * u.TeV
        >>> dnde_cu = (pwl(energy) / crab(energy)).to('%')
        >>> print(dnde_cu)
        6.196991563774588 %

    And the same for integral fluxes (between 1 and 10 TeV)::

        >>> # compute integral flux in crab units
        >>> emin, emax = [1, 10] * u.TeV
        >>> flux_int_cu = (pwl.integral(emin, emax) / crab.integral(emin, emax)).to('%')
        >>> print(flux_int_cu)
        3.535058216604496 %
    """
    if reference == "meyer":
        return MeyerCrabSpectralModel()
    elif reference == "hegra":
        return PowerLawSpectralModel(
            amplitude=2.83e-11 * u.Unit("1 / (cm2 s TeV)"),
            index=2.62,
            reference=1 * u.TeV,
        )
    elif reference == "hess_pl":
        return PowerLawSpectralModel(
            amplitude=3.45e-11 * u.Unit("1 / (cm2 s TeV)"),
            index=2.63,
            reference=1 * u.TeV,
        )
    elif reference == "hess_ecpl":
        return ExpCutoffPowerLawSpectralModel(
            amplitude=3.76e-11 * u.Unit("1 / (cm2 s TeV)"),
            index=2.39,
            lambda_=1 / (14.3 * u.TeV),
            reference=1 * u.TeV,
        )
    elif reference == "magic_lp":
        return LogParabolaSpectralModel(
            amplitude=3.23e-11 * u.Unit("1 / (cm2 s TeV)"),
            alpha=2.47,
            beta=0.24 / np.log(10),
            reference=1 * u.TeV,
        )
    elif reference == "magic_ecpl":
        return ExpCutoffPowerLawSpectralModel(
            amplitude=3.80e-11 * u.Unit("1 / (cm2 s TeV)"),
            index=2.21,
            lambda_=1 / (6.0 * u.TeV),
            reference=1 * u.TeV,
        )
    else:
        raise ValueError(f"Invalid reference: {reference!r}")
