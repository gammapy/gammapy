# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from .interpolation import LogScale

__all__ = ["trapz_loglog"]


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
    from gammapy.modeling.models import PowerLawSpectralModel as pl

    # see https://stackoverflow.com/a/56840428
    x, y = np.moveaxis(x, axis, 0), np.moveaxis(y, axis, 0)

    energy_min, energy_max = x[:-1], x[1:]
    vals_energy_min, vals_energy_max = y[:-1], y[1:]

    # log scale has the built-in zero clipping
    log = LogScale()

    with np.errstate(invalid="ignore", divide="ignore"):
        index = -log(vals_energy_min / vals_energy_max) / log(energy_min / energy_max)

    index[np.isnan(index)] = np.inf

    return pl.evaluate_integral(
        energy_min=energy_min,
        energy_max=energy_max,
        index=index,
        reference=energy_min,
        amplitude=vals_energy_min,
    )
