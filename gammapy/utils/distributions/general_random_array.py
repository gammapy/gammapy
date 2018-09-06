# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Implementation of the GeneralRandomArray class"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from ...utils.random import get_random_state

__all__ = ["GeneralRandomArray"]


class GeneralRandomArray(object):
    """Draw random indices from a discrete probability distribution
    given by a numpy array.
    The array dimension can be arbitrary.

    Note that drawing a random number from an array with N entries
    is not a constant cost operation. Even the efficient implementation
    using a binary tree used here costs log(N).

    For a general description of the method see the end of the following page:
    http://www.cs.utk.edu/~parker/Courses/CS302-Fall06/Notes/PQueues/random_num_gen.html
    """

    def __init__(self, pdf):
        # Computes the cdf from the pdf.

        # Note that numpy flattens the array automatically,
        # i.e. cdf is a 1D array (normalization not necessary)
        self.cdf = pdf.cumsum()
        self.cdfmax = self.cdf.max()

        # Remember the dimension and shape for unravel_index()
        self.ndim = pdf.ndim
        self.shape = pdf.shape

    def draw(self, n=1, return_flat_index=False, random_state="random-seed"):
        """Returns n draws from the pdf.

        If return_flat_index == true, a linearized index is returned.

        Parameters
        ----------
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`.
        """
        random_state = get_random_state(random_state)

        u = random_state.uniform(0, self.cdfmax, size=n)
        indices = self.cdf.searchsorted(u)
        if return_flat_index:
            return indices
        else:
            # TODO: vectorize unravel_index
            # This for loop is a dirty hack and most likely is very slow.
            unraveled_indices = np.empty((n, self.ndim), dtype=np.int64)
            for i in np.arange(n):
                unraveled_indices[i] = np.unravel_index(indices[i], self.shape)
            return unraveled_indices
