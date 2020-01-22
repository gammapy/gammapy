# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from .interpolation import LogScale

__all__ = ["evaluate_integral_pwl", "trapz_loglog"]


def evaluate_integral_pwl(emin, emax, index, amplitude, reference):
    """Evaluate pwl integral (static function)."""
    val = -1 * index + 1

    prefactor = amplitude * reference / val
    upper = np.power((emax / reference), val)
    lower = np.power((emin / reference), val)
    integral = prefactor * (upper - lower)

    mask = np.isclose(val, 0)

    if mask.any():
        integral[mask] = (amplitude * reference * np.log(emax / emin))[mask]

    return integral


def trapz_loglog(y, x, axis=-1):
    """Integrate using the composite trapezoidal rule in log-log space.

    Integrate `y` (`x`) along given axis in loglog space.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        Independent variable to integrate over.
    axis : int, optional
        Specify the axis.

    Returns
    -------
    trapz : float
        Definite integral as approximated by trapezoidal rule in loglog space.
    """
    # see https://stackoverflow.com/a/56840428
    x, y = np.moveaxis(x, axis, 0), np.moveaxis(y, axis, 0)

    emin, emax = x[:-1], x[1:]
    vals_emin, vals_emax = y[:-1], y[1:]

    # log scale has the build-in zero clipping
    log = LogScale()
    index = -log(vals_emin / vals_emax) / log(emin / emax)
    index[np.isnan(index)] = np.inf

    return evaluate_integral_pwl(
        emin=emin,
        emax=emax,
        index=index,
        reference=emin,
        amplitude=vals_emin
    )
