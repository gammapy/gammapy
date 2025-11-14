# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from .interpolation import LogScale

__all__ = ["trapz_loglog", "integrate_histogram"]


def integrate_histogram(hist, bin_edges, a, b, axis=0):
    """
    Integrate histogram along a chosen axis between values a and b.

    Parameters
    ----------
    hist : ndarray
        Histogram counts (or densities).
    bin_edges : array_like
        Edges of the bins along the chosen axis (length = hist.shape[axis] + 1).
    a, b : float
        Integration limits.
    axis : int, optional
        Axis of `hist` to integrate over (default=0).

    Returns
    -------
    ndarray
        Integral of the histogram along the given axis between [a, b].
        Shape is `hist.shape` with that axis removed.
    """
    # Ensure a < b
    if a > b:
        a, b = b, a

    # Clip to histogram range
    a = max(a, bin_edges[0])
    b = min(b, bin_edges[-1])
    if a >= b:
        out_shape = list(hist.shape)
        del out_shape[axis]
        return np.zeros(out_shape, dtype=float)

    # Compute overlaps of [a,b] with each bin
    lefts = bin_edges[:-1]
    rights = bin_edges[1:]
    overlap_left = np.maximum(lefts, a)
    overlap_right = np.minimum(rights, b)
    widths = np.clip(overlap_right - overlap_left, 0, None)  # (nbins,)

    # Bring axis to front
    hist_moved = np.moveaxis(hist, axis, 0)  # shape (nbins, ...)

    # Weighted sum along first axis
    total = np.tensordot(widths, hist_moved, axes=(0, 0))  # shape (...)
    total = total[np.newaxis, ...]

    return total


def trapz_loglog(y, x, axis=-1):
    """Integrate using the composite trapezoidal rule in log-log space.

    Integrate `y` (`x`) along given axis in loglog space.

    Parameters
    ----------
    y : `~numpy.ndarray`
        Input array to integrate.
    x : `~numpy.ndarray`
        Independent variable to integrate over.
    axis : int, optional
        Specify the axis. Default is -1.

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
