# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.units import Quantity
from ..data import ObservationStats
from .observation import SpectrumObservation

__all__ = [
    'calculate_flux_point_binning',
]


def calculate_flux_point_binning(obs_list, min_signif):
    """Compute energy binning.

    This is useful to get an energy binning to use with
    :func:`~gammapy.spectrum.DifferentialFluxPoints.compute` Each bin in the
    resulting energy binning will include a ``min_signif`` source detection.

    TODO: It is required that at least two fine bins be included in one
    flux point interval, otherwise the sherpa covariance method breaks
    down.

    Parameters
    ----------
    obs_list : `~gammapy.spectrum.SpectrumObservationList`
        Observations
    min_signif : float
        Required significance for each bin

    Returns
    -------
    binning : `~astropy.units.Quantity`
        Energy binning
    """
    # NOTE: Results may vary from FitSpectrum since there the rebin
    # parameter can only have fixed values, here it grows linearly. Also it
    # has to start at 2 here (see docstring)

    # rebin_factor = 1
    rebin_factor = 2

    obs = SpectrumObservation.stack(obs_list)

    # First first bin above low threshold and last bin below high threshold
    current_ebins = obs.on_vector.energy
    current_bin = (current_ebins.find_node(obs.lo_threshold) + 1)[0]
    max_bin = (current_ebins.find_node(obs.hi_threshold))[0]

    # List holding final energy binning
    binning = [current_ebins.data[current_bin]]

    # Precompute ObservationStats for each bin
    obs_stats = [obs.stats(i) for i in range(current_ebins.nbins)]
    while current_bin + rebin_factor <= max_bin:
        # Merge bins together until min_signif is reached
        stats_list = obs_stats[current_bin:current_bin + rebin_factor:1]
        stats = ObservationStats.stack(stats_list)
        sigma = stats.sigma
        if sigma < min_signif or np.isnan(sigma):
            rebin_factor += 1
            continue

        # Append upper bin edge of good energy bin to final binning
        binning.append(current_ebins.data[current_bin + rebin_factor])
        current_bin += rebin_factor

    binning = Quantity(binning)
    # Replace highest bin edge by high threshold
    binning[-1] = obs.hi_threshold

    return binning
