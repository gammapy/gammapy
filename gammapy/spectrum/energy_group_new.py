# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spectrum energy bin grouping.
"""
import numpy as np
import astropy.units as u

class FluxPointBins:
    def __init__(self, indices):
        self.indices = indices

    def lower_bounds(self):
        return self.indices[:-1]

    def upper_bounds(self):
        return self.indices[1:] - 1

    def bin(self, idx):
        return np.arange(self.lower_bounds[idx],
                         self.upper_bounds[idx])



class FluxPointBinMaker:
    def __init__(self, obs):
        self.obs = obs
        self.indices = None

    def compute_bins_fixed(self, energy_binning):
        energy_binning_offset = energy_binning - 1 * u.MeV
        diff = energy_binning_offset[:, np.newaxis] - self.obs.e_reco.lower_bounds
        lower_indices = np.argmin(np.sign(diff), axis=1)

        if lower_indices[-1] == 0:
            lower_indices[-1] = self.obs.e_reco.nbins + 1

        self.indices = FluxPointBins(lower_indices)

