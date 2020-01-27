# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from .utils import get_random_state

__all__ = ["InverseCDFSampler"]


class InverseCDFSampler:
    """Inverse CDF sampler.

    It determines a set of random numbers and calculate the cumulative
    distribution function.

    Parameters
    ----------
    pdf : `~gammapy.maps.Map`
        Map of the predicted source counts.
    axis : int
        Axis along which sampling the indexes.
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.
    """

    def __init__(self, pdf, axis=None, random_state=0):
        self.random_state = get_random_state(random_state)
        self.axis = axis

        if axis is not None:
            self.cdf = np.cumsum(pdf, axis=self.axis)
            self.cdf /= self.cdf[:, [-1]]
        else:
            self.pdf_shape = pdf.shape

            pdf = pdf.ravel() / pdf.sum()
            self.sortindex = np.argsort(pdf, axis=None)

            self.pdf = pdf[self.sortindex]
            self.cdf = np.cumsum(self.pdf)

    def sample_axis(self):
        """Sample along a given axis.

        Returns
        -------
        index : tuple of `~numpy.ndarray`
            Coordinates of the drawn sample.
        """
        choices = self.random_state.uniform(high=1, size=len(self.cdf))
        shape_cdf = self.cdf.shape

        cdf_all = np.insert(self.cdf, 0, 0, axis=1)
        edges = np.arange(shape_cdf[1] + 1) - 0.5

        pix_coords = []

        for cdf, choice in zip(cdf_all, choices):
            pix = np.interp(choice, cdf, edges)
            pix_coords.append(pix)

        return np.array(pix_coords)

    def sample(self, size):
        """Draw sample from the given PDF.

        Parameters
        ----------
        size : int
            Number of samples to draw.

        Returns
        -------
        index : tuple of `~numpy.ndarray`
            Coordinates of the drawn sample.
        """
        # pick numbers which are uniformly random over the cumulative distribution function
        choice = self.random_state.uniform(high=1, size=size)

        # find the indices corresponding to this point on the CDF
        index = np.searchsorted(self.cdf, choice)
        index = self.sortindex[index]

        # map back to multi-dimensional indexing
        index = np.unravel_index(index, self.pdf_shape)
        index = np.vstack(index)

        index = index + self.random_state.uniform(low=-0.5, high=0.5, size=index.shape)
        return index
