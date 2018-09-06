# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""On-off bin stats computations."""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from ..utils.random import get_random_state

__all__ = ["Stats", "make_stats", "combine_stats", "compute_total_stats"]


class Stats(object):
    """Container for an on-off observation.

    Parameters
    ----------
    n_on : array_like
        Observed number of counts in the on region
    n_off : array_like
        Observed number of counts in the off region
    a_on : array_like
        Relative background exposure of the on region
    a_off : array_like
        Relative background exposure in the off region
    """

    # TODO: use numpy arrays and properties
    # TODO: add gamma exposure

    def __init__(self, n_on, n_off, a_on, a_off):
        self.n_on = n_on
        self.n_off = n_off
        self.a_on = a_on
        self.a_off = a_off

    @property
    def alpha(self):
        r"""Background exposure ratio (float)

        .. math:: \alpha = a_\mathrm{on} / a_\mathrm{off}
        """
        return self.a_on / self.a_off

    @property
    def background(self):
        r"""Background estimate (float)

        .. math:: \mu_\mathrm{bg} = \alpha\ n_\mathrm{off}
        """
        return self.alpha * self.n_off

    @property
    def excess(self):
        r"""Excess (float)

        .. math:: n_\mathrm{ex} = n_\mathrm{on} - \mu_\mathrm{bg}
        """
        return self.n_on - self.background

    def __str__(self):
        keys = ["n_on", "n_off", "a_on", "a_off", "alpha", "background", "excess"]
        values = [
            self.n_on,
            self.n_off,
            self.a_on,
            self.a_off,
            self.alpha,
            self.background,
            self.excess,
        ]
        return "\n".join(["%s = %s" % (k, v) for (k, v) in zip(keys, values)])


def make_stats(
    signal,
    background,
    area_factor,
    weight_method="background",
    poisson_fluctuate=False,
    random_state="random-seed",
):
    """Fill using some weight method for the exposure.

    Parameters
    ----------
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.

    Returns
    -------
    """
    random_state = get_random_state(random_state)

    # Compute counts
    n_on = signal + background
    n_off = area_factor * background
    if poisson_fluctuate:
        n_on = random_state.poisson(n_on)
        n_off = random_state.poisson(n_off)

    # Compute weight
    if weight_method == "none":
        weight = 1
    elif weight_method == "background":
        weight = background
    elif weight_method == "n_off":
        weight = n_off
    else:
        raise ValueError("Invalid weight_method: {}".format(weight_method))

    # Compute exposure
    a_on = weight
    a_off = weight * area_factor

    return Stats(n_on, n_off, a_on, a_off)


def combine_stats(stats_1, stats_2, weight_method="none"):
    """Combine using some weight method for the exposure.

    Parameters
    ----------
    stats_1 : `Stats`
        Observation 1
    stats_2 : `Stats`
        Observation 2
    weight_method : {'none', 'background', 'n_off'}
        Observation weighting method.

    Returns
    -------
    stats : `Stats`
        Combined Observation 1 and 2
    """
    # Compute counts
    n_on = stats_1.n_on + stats_2.n_on
    n_off = stats_1.n_off + stats_2.n_off

    # Compute weights
    if weight_method == "none":
        weight_1 = 1
        weight_2 = 1
    elif weight_method == "background":
        weight_1 = stats_1.background()
        weight_2 = stats_2.background()
    elif weight_method == "n_off":
        weight_1 = stats_1.n_off
        weight_2 = stats_2.n_off
    else:
        raise ValueError("Invalid weight_method: {}".format(weight_method))

    # Compute exposure
    a_on = weight_1 * stats_1.a_on + weight_2 * stats_2.a_on
    a_off = weight_1 * stats_1.a_off + weight_2 * stats_2.a_off

    return Stats(n_on, n_off, a_on, a_off)


def compute_total_stats(counts, exposure, background=None, solid_angle=None, mask=None):
    r"""Compute total stats for arrays of per-bin stats.

    The ``result`` dictionary contains a ``flux`` entry computed as

    .. math:: F = N / E = \sum{N_i} / \sum{E_i}

    as well as a ``flux3`` entry computed as

    .. math:: F = \sum{F_i} = \sum{\left(N_i / E_i\right)}

    where ``F`` is flux, ``N`` is excess and ``E`` is exposure.

    The output ``flux`` units are the inverse of the input ``exposure`` units, e.g.

    * ``exposure`` in cm^2 s -> ``flux`` in cm^-2 s^-1
    * ``exposure`` in cm^2 s TeV -> ``flux`` in cm^-2 s^-1 TeV-1

    The output ``surface_brightness`` units in addition depend on the ``solid_angle`` units, e.g.

    * ``exposure`` in cm^2 s and ``solid_angle`` in deg^2 -> ``surface_brightness`` in cm^-2 s^-1 deg^-2

    TODOs:

    * integrate this with the `Stats` class.
    * add statistical errors on excess, flux, surface brightness

    Parameters
    ----------
    counts, exposure : array_like
        Input arrays
    background, solid_angle, mask : array_like
        Optional input arrays

    Returns
    -------
    result : dict
        Dictionary of total stats (for now, see the code for which entries it has).
    """
    counts = np.asanyarray(counts)
    exposure = np.asanyarray(exposure)

    if solid_angle is None:
        background = np.zeros_like(counts)
    else:
        background = np.asanyarray(background)

    if solid_angle is None:
        solid_angle = np.ones_like(counts)
    else:
        solid_angle = np.asanyarray(solid_angle)

    if mask is None:
        mask = np.ones_like(counts)
    else:
        mask = np.asanyarray(mask)

    t = dict()
    t["n_pix_map"] = mask.size
    t["n_pix_mask"] = mask.sum()
    t["n_pix_fraction"] = t["n_pix_mask"] / float(t["n_pix_map"])
    t["counts"] = counts[mask].sum(dtype=np.float64)
    t["background"] = background[mask].sum(dtype=np.float64)
    # Note that we use mean exposure (not sum) here!!!
    t["exposure"] = exposure[mask].mean(dtype=np.float64)
    t["solid_angle"] = solid_angle[mask].sum(dtype=np.float64)

    excess = counts - background
    t["excess"] = t["counts"] - t["background"]
    t["excess_2"] = excess[mask].sum(dtype=np.float64)

    flux = excess / exposure
    t["flux"] = (t["excess"]) / t["exposure"]
    t["flux_2"] = t["excess_2"] / t["exposure"]
    t["flux_3"] = flux[mask].sum(dtype=np.float64)

    surface_brightness = flux / solid_angle
    t["surface_brightness"] = t["flux"] / t["solid_angle"]
    t["surface_brightness_2"] = t["flux_2"] / t["solid_angle"]
    # Note that we use mean exposure (not sum) here!!!
    t["surface_brightness_3"] = surface_brightness[mask].mean(dtype=np.float64)

    return t
