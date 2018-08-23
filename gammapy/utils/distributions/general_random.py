# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from ...utils.random import get_random_state

__all__ = ["GeneralRandom"]


class GeneralRandom(object):
    """Fast random number generation with an arbitrary pdf
    of a continuous variable x.
    Linear interpolation is applied between points pdf(x)
    at which the pdf is specified.

    I started with the recipy 576556, removed some unnecessary stuff
    and added some useful stuff.
    Recipe 576556: Generating random numbers with arbitrary distribution
    http://code.activestate.com/recipes/576556/

    Note: This class can only handle 1D distributions.

    Note: Should it be required the cdf could be deleted
    after computing to inversecdf to free memory since it is
    not required for random number generation."""

    def __init__(self, pdf, min_range, max_range, ninversecdf=None, ran_res=1e3):
        """Initialize the lookup table

        Inputs:
        x: random number values
        pdf: probability density profile at that point
        ninversecdf: number of reverse lookup values

        Lookup is computed and stored in:
        cdf: cumulative pdf
        inversecdf: the inverse lookup table
        delta_inversecdf: difference of inversecdf
        ran_res: Resolution of the PDF
        """
        self.ran_res = ran_res
        x = np.linspace(min_range, max_range, ran_res)
        # This is a good default for the number of reverse
        # lookups to not loose much information in the pdf
        if ninversecdf is None:
            ninversecdf = 5 * x.size

        self.nx = x.size
        self.x = x
        self.pdf = pdf(x)

        self.cdf = np.empty(self.nx, dtype=float)
        self.cdf[0] = 0
        for i in range(1, self.nx):
            temp = (self.pdf[i] + self.pdf[i - 1]) * (self.x[i] - self.x[i - 1]) / 2
            self.cdf[i] = self.cdf[i - 1] + temp

        self.pdf = self.pdf / self.cdf.max()  # normalize pdf
        self.cdf = self.cdf / self.cdf.max()  # normalize cdf

        self.ninversecdf = ninversecdf
        y = np.arange(ninversecdf) / float(ninversecdf)
        self.inversecdf = np.empty(ninversecdf)
        self.inversecdf[0] = self.x[0]
        cdf_idx = 0
        for n in range(1, self.ninversecdf):
            while self.cdf[cdf_idx] < y[n] and cdf_idx < ninversecdf:
                cdf_idx += 1

            self.inversecdf[n] = self.x[cdf_idx - 1] + (
                self.x[cdf_idx] - self.x[cdf_idx - 1]
            ) * (y[n] - self.cdf[cdf_idx - 1]) / (
                self.cdf[cdf_idx] - self.cdf[cdf_idx - 1]
            )
            if cdf_idx >= ninversecdf:
                break
        self.delta_inversecdf = np.concatenate((np.diff(self.inversecdf), [0]))

    def draw(self, N=1000, random_state="random-seed"):
        """Returns an array of random numbers with the requested distribution.

        The random numbers x are generated using the lookups
        inversecdf and delta_inversecdf.

        Parameters
        ----------
        N : int
            array length
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`.

        Returns
        -------
        x : `~numpy.ndarray`
            random numbers
        """
        random_state = get_random_state(random_state)

        # Generate uniform random float index in range [0, ninversecdf-1]
        idx_f = random_state.uniform(size=N, high=self.ninversecdf - 1)
        # Round down to next integer
        idx = np.array(idx_f, "i")
        # Use the inversecdf lookup to get the corresponding x
        # and the delta_inversecdf lookup for linear interpolation
        x = self.inversecdf[idx] + (idx_f - idx) * self.delta_inversecdf[idx]
        return x

    def make_plots(self, N=1e5):
        """Plot the pdf, cdf and inversecdf
        and a random distribution of sample size N.

        Useful for illustrating the interpolation and debugging."""
        import matplotlib.pyplot as plt

        # Plot the cdf
        plt.figure()
        plt.plot(self.x, self.cdf)
        plt.title("cdf(x)")

        # Plot the inverse cdf
        plt.figure()
        y = np.arange(self.ninversecdf) / float(self.ninversecdf)
        plt.plot(y, self.inversecdf)
        plt.title("inversecdf(y)")

        # Plot the pdf and a random sample distribution
        plt.figure()
        x = self.draw(N)

        # Use the same binning as self.x
        binedges = self.x
        plt.hist(x, bins=binedges, normed=True)
        #    x1 = 0.5*(edges[0:-1] + edges[1:])
        #    plot(x1, p1/p1.sum(),label='hist of random draws')
        plt.plot(self.x, self.pdf, label="pdf")
