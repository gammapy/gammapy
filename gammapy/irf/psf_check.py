# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import math
import numpy as np

__all__ = []


# TODO: change this checker to the structure of the other checkers
# we have; see gammapy.utils.testing.Checker and examples in gammapy.data
class PSF3DChecker(object):
    """Automated quality checks for `gammapy.irf.PSF3D`.

    At the moment used for HESS HAP HD.

    Parameters
    ----------
    psf : `~gammapy.irf.PSF3D`
        PSF to check
    d_norm : float
        Config option: maximum norm deviation from 1
    containment_fraction : float
        Config option: containment fraction to check
    d_rel_containment : float
        Config option: maximum relative difference of containment
        radius between neighboring bins

    Examples
    --------
    To check a PSF, load it, run the checker and look at the results dict::

        from gammapy.irf import PSF3D, PSF3DChecker
        filename = '$GAMMAPY_EXTRA/datasets/hess-hap-hd-prod3/psf_table.fits.gz'
        psf = PSF3D.read(filename)
        checker = PSF3DChecker(psf)
        print('config: ', checker.config)
        checker.check_all()
        print('results: ', checker.results)
    """

    def __init__(
        self, psf, d_norm=0.01, containment_fraction=0.68, d_rel_containment=0.7
    ):
        self.psf = psf

        self.config = OrderedDict(
            d_norm=d_norm,
            containment_fraction=containment_fraction,
            d_rel_containment=d_rel_containment,
        )

        self.results = OrderedDict()

    def check_all(self):
        """Run all checks.
        """
        self.check_nan()
        self.check_normalise()
        self.check_containment()

        # Aggregate status of all checks
        status = "ok"
        for key in ["nan", "normalise", "containment"]:
            if self.results[key]["status"] == "failed":
                status = "failed"
        self.results["status"] = status

    def check_nan(self):
        """Check for `NaN` values in PSF.
        """
        # generate array for easier handling
        values = np.swapaxes(self.psf.psf_value, 0, 2)
        fail_count = 0

        # loop over energies
        for i, arr in enumerate(values):
            energy_hi = self.psf.energy_hi[i]
            energy_lo = self.psf.energy_lo[i]

            # check if bin is outside of safe energy threshold
            if self.psf.energy_thresh_lo > energy_hi:
                continue
            if self.psf.energy_thresh_hi < energy_lo:
                continue

            # loop over offsets
            for arr2 in arr:

                # loop over deltas
                for v in arr2:

                    # check for nan
                    if math.isnan(v.value):
                        # add to fail counter
                        fail_count += 1
                        break

        results = OrderedDict()
        if fail_count == 0:
            results["status"] = "ok"
        else:
            results["status"] = "failed"
            results["n_failed_bins"] = fail_count

        self.results["nan"] = results

    def check_normalise(self):
        """Check PSF normalisation.

        For each energy / offset, the PSF should integrate to 1.
        """
        # generate array for easier handling
        values = np.swapaxes(self.psf.psf_value, 0, 2)

        # init fail count
        fail_count = 0

        # loop over energies
        for i, arr in enumerate(values):
            energy_hi = self.psf.energy_hi[i]
            energy_lo = self.psf.energy_lo[i]

            # check if energy is outside of safe energy threshold
            if self.psf.energy_thresh_lo > energy_hi:
                continue
            if self.psf.energy_thresh_hi < energy_lo:
                continue

            # loop over offsets
            for arr2 in arr:

                # init integral
                sum = 0

                # loop over deltas
                for j, v in enumerate(arr2):
                    # calculate contribution to integral
                    width = self.psf.rad_hi[j].rad - self.psf.rad_lo[j].rad
                    rad = 0.5 * (self.psf.rad_hi[j].rad + self.psf.rad_lo[j].rad)
                    sum += v.value * width * rad * 2 * np.pi

                # check if integral is close enough to 1
                if np.abs(sum - 1.0) > self.config["d_norm"]:
                    # add to fail counter
                    fail_count += 1

        # write results to dict
        results = OrderedDict()
        if fail_count == 0:
            results["status"] = "ok"
        else:
            results["status"] = "failed"
            results["n_failed_bins"] = fail_count
        self.results["normalise"] = results

    def check_containment(self):
        """Check PSF containment.

        TODO: describe what this actually does!?
        """
        # set fraction to check for
        fraction = self.config["containment_fraction"]

        # set maximum relative difference between neighboring bins
        rel_diff = self.config["d_rel_containment"]

        # generate array for easier handling
        values = np.swapaxes(self.psf.psf_value, 0, 2)

        # init containment radius array
        radii = np.zeros(values[:, :, 0].shape)

        # init fail count
        fail_count = 0

        # loop over energies
        for i, arr in enumerate(values):
            energy_hi = self.psf.energy_hi[i]
            energy_lo = self.psf.energy_lo[i]

            # loop over offsets
            for k, arr2 in enumerate(arr):

                # if energy is outside safe energy threshold,
                # set containment radius to None
                if self.psf.energy_thresh_lo > energy_hi:
                    radii[i, k] = None
                    continue
                if self.psf.energy_thresh_hi < energy_lo:
                    radii[i, k] = None
                    continue

                # init integral and containment radius
                sum = 0
                r = None

                # loop over deltas
                for j, v in enumerate(arr2):

                    # calculate contribution to integral
                    width = self.psf.rad_hi[j].rad - self.psf.rad_lo[j].rad
                    rad = 0.5 * (self.psf.rad_hi[j].rad + self.psf.rad_lo[j].rad)
                    sum += v.value * width * rad * 2 * np.pi

                    # check if conainmant radius is reached
                    if sum >= fraction:
                        # convert radius to degrees
                        r = rad * 180. / np.pi
                        break

                # store containment radius in array
                radii[i, k] = r

        # generate an array of radii with stripped edges so that each
        # element has 9 neighbors
        inner = radii[1:-1, 1:-1]

        # loop over energies
        for i, arr in enumerate(inner):

            # loop over offsets
            for j, v in enumerate(inner[i]):

                # check if radius is nan
                if math.isnan(v):
                    continue

                # set distance to neighbors
                d = 1

                # calculate corresponding indices in whole radii array
                ii = i + 1
                jj = j + 1

                # retrieve array of neighbors
                nb = radii[ii - d : ii + d + 1, jj - d : jj + d + 1].flatten()

                # loop over neighbors
                for n in nb:

                    # check if neighbor is nan
                    if math.isnan(n):
                        continue

                    # calculate relative difference to neighbor
                    diff = np.abs(v - n) / v

                    # check if difference is to big
                    if diff > rel_diff:
                        # add to fail counter
                        fail_count += 1

        # write results to dict
        results = OrderedDict()
        if fail_count == 0:
            results["status"] = "ok"
        else:
            results["status"] = "failed"
            results["n_failed_bins"] = fail_count
        self.results["containment"] = results


def check_all_table_psf(data_store):
    """Check all `gammapy.irf.PSF3D` for a given `gammapy.data.DataStore`.
    """
    config = OrderedDict(d_norm=0.01, containment_fraction=0.68, d_rel_containment=0.7)

    obs_ids = data_store.obs_table["OBS_ID"].data

    for obs_id in obs_ids[:10]:
        obs = data_store.obs(obs_id=obs_id)
        psf = obs.load(hdu_class="psf_table")
        checker = PSF3DChecker(psf=psf, **config)
        checker.check_all()
        print(checker.results)


if __name__ == "__main__":
    import sys
    from ..data.data_store import DataStore

    data_store = DataStore.from_dir(sys.argv[1])
    check_all_table_psf(data_store)
