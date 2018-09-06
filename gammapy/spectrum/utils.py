# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.units import Quantity

__all__ = ["CountsPredictor", "integrate_spectrum"]


class CountsPredictor(object):
    """Calculate number of predicted counts (``npred``).

    The true and reconstructed energy binning are inferred from the provided IRFs.

    Parameters
    ----------
    model : `~gammapy.spectrum.models.SpectralModel`
        Spectral model
    aeff : `~gammapy.irf.EffectiveAreaTable`
        EffectiveArea
    edisp : `~gammapy.irf.EnergyDispersion`, optional
        EnergyDispersion
    livetime : `~astropy.units.Quantity`
        Observation duration (may be contained in aeff)
    e_true : `~astropy.units.Quantity`, optional
        Desired energy axis of the prediced counts vector if no IRFs are given

    Examples
    --------
    Calculate prediced counts in a desired reconstruced energy binning

    .. plot::
        :include-source:

        from gammapy.irf import EnergyDispersion, EffectiveAreaTable
        from gammapy.spectrum import models, CountsPredictor
        import numpy as np
        import astropy.units as u
        import matplotlib.pyplot as plt

        e_true = np.logspace(-2,2.5,109) * u.TeV
        e_reco = np.logspace(-2,2,73) * u.TeV

        aeff = EffectiveAreaTable.from_parametrization(energy=e_true)
        edisp = EnergyDispersion.from_gauss(e_true=e_true, e_reco=e_reco,
                                            sigma=0.3, bias=0)

        model = models.PowerLaw(index=2.3,
                                amplitude=2.5 * 1e-12 * u.Unit('cm-2 s-1 TeV-1'),
                                reference=1*u.TeV)

        livetime = 1 * u.h

        predictor = CountsPredictor(model=model,
                                    aeff=aeff,
                                    edisp=edisp,
                                    livetime=livetime)
        predictor.run()
        predictor.npred.plot_hist()
        plt.show()
    """

    def __init__(self, model, aeff=None, edisp=None, livetime=None, e_true=None):
        self.model = model
        self.aeff = aeff
        self.edisp = edisp
        self.livetime = livetime
        self.e_true = e_true
        self.e_reco = None

        self.true_flux = None
        self.true_counts = None
        self.npred = None

    def run(self):
        self.integrate_model()
        self.apply_aeff()
        self.apply_edisp()

    def integrate_model(self):
        """Integrate model in true energy space"""
        if self.aeff is not None:
            # TODO: True energy is converted to model amplitude unit. See issue 869
            ref_unit = None
            try:
                for unit in self.model.parameters["amplitude"].quantity.unit.bases:
                    if unit.is_equivalent("eV"):
                        ref_unit = unit
            except IndexError:
                ref_unit = "TeV"
            self.e_true = self.aeff.energy.bins.to(ref_unit)
        else:
            if self.e_true is None:
                raise ValueError("No true energy binning given")

        self.true_flux = self.model.integral(
            emin=self.e_true[:-1], emax=self.e_true[1:], intervals=True
        )

    def apply_aeff(self):
        if self.aeff is not None:
            cts = self.true_flux * self.aeff.data.data
        else:
            cts = self.true_flux

        # Multiply with livetime if not already contained in aeff or model
        if cts.unit.is_equivalent("s-1"):
            cts *= self.livetime

        self.true_counts = cts.to("")

    def apply_edisp(self):
        from . import CountsSpectrum

        if self.edisp is not None:
            cts = self.edisp.apply(self.true_counts)
            self.e_reco = self.edisp.e_reco.bins
        else:
            cts = self.true_counts
            self.e_reco = self.e_true

        self.npred = CountsSpectrum(
            data=cts, energy_lo=self.e_reco[:-1], energy_hi=self.e_reco[1:]
        )


def integrate_spectrum(func, xmin, xmax, ndecade=100, intervals=False):
    """
    Integrate 1d function using the log-log trapezoidal rule. If scalar values

    for xmin and xmax are passed an oversampled grid is generated using the
    ``ndecade`` keyword argument. If xmin and xmax arrays are passed, no
    oversampling is performed and the integral is computed in the provided
    grid.

    Parameters
    ----------
    func : callable
        Function to integrate.
    xmin : `~astropy.units.Quantity` or array-like
        Integration range minimum
    xmax : `~astropy.units.Quantity` or array-like
        Integration range minimum
    ndecade : int, optional
        Number of grid points per decade used for the integration.
        Default : 100.
    intervals : bool, optional
        Return integrals in the grid not the sum, default: False
    """
    is_quantity = False
    if isinstance(xmin, Quantity):
        unit = xmin.unit
        xmin = xmin.value
        xmax = xmax.to(unit).value
        is_quantity = True

    if np.isscalar(xmin):
        logmin = np.log10(xmin)
        logmax = np.log10(xmax)
        n = (logmax - logmin) * ndecade
        x = np.logspace(logmin, logmax, n)
    else:
        x = np.append(xmin, xmax[-1])

    if is_quantity:
        x = x * unit

    y = func(x)

    val = _trapz_loglog(y, x, intervals=intervals)

    return val


# This function is copied over from https://github.com/zblz/naima/blob/master/naima/utils.py#L261
# and slightly modified to allow use with the uncertainties package


def _trapz_loglog(y, x, axis=-1, intervals=False):
    """
    Integrate along the given axis using the composite trapezoidal rule in
    loglog space.

    Integrate `y` (`x`) along given axis in loglog space.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        Independent variable to integrate over.
    axis : int, optional
        Specify the axis.
    intervals : bool, optional
        Return array of shape x not the total integral, default: False

    Returns
    -------
    trapz : float
        Definite integral as approximated by trapezoidal rule in loglog space.
    """
    log10 = np.log10

    try:
        y_unit = y.unit
        y = y.value
    except AttributeError:
        y_unit = 1.
    try:
        x_unit = x.unit
        x = x.value
    except AttributeError:
        x_unit = 1.

    y = np.asanyarray(y)
    x = np.asanyarray(x)

    slice1 = [slice(None)] * y.ndim
    slice2 = [slice(None)] * y.ndim
    slice1[axis] = slice(None, -1)
    slice2[axis] = slice(1, None)
    slice1, slice2 = tuple(slice1), tuple(slice2)

    # arrays with uncertainties contain objects
    if y.dtype == "O":
        from uncertainties.unumpy import log10

        # uncertainties.unumpy.log10 can't deal with tiny values see
        # https://github.com/gammapy/gammapy/issues/687, so we filter out the values
        # here. As the values are so small it doesn't affect the final result.
        # the sqrt is taken to create a margin, because of the later division
        # y[slice2] / y[slice1]
        valid = y > np.sqrt(np.finfo(float).tiny)
        x, y = x[valid], y[valid]

    if x.ndim == 1:
        shape = [1] * y.ndim
        shape[axis] = x.shape[0]
        x = x.reshape(shape)

    with np.errstate(invalid="ignore", divide="ignore"):
        # Compute the power law indices in each integration bin
        b = log10(y[slice2] / y[slice1]) / log10(x[slice2] / x[slice1])

        # if local powerlaw index is -1, use \int 1/x = log(x); otherwise use normal
        # powerlaw integration
        trapzs = np.where(
            np.abs(b + 1.) > 1e-10,
            (y[slice1] * (x[slice2] * (x[slice2] / x[slice1]) ** b - x[slice1]))
            / (b + 1),
            x[slice1] * y[slice1] * np.log(x[slice2] / x[slice1]),
        )

    tozero = (y[slice1] == 0.) + (y[slice2] == 0.) + (x[slice1] == x[slice2])
    trapzs[tozero] = 0.

    if intervals:
        return trapzs * x_unit * y_unit

    ret = np.add.reduce(trapzs, axis) * x_unit * y_unit

    return ret
