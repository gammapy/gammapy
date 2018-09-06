# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility functions."""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

__all__ = ["cov_to_corr"]


def cov_to_corr(covariance):
    """Compute correlation matrix from covariance matrix.

    The correlation matrix :math:`c` is related to the covariance matrix :math:`\\sigma` by:

    .. math::

        c_{ij} = \\frac{\\sigma_{ij}}{\\sqrt{\\sigma_{ii} \sigma_{jj}}}

    Parameters
    ----------
    covariance : `~numpy.ndarray`
        Covariance matrix

    Returns
    -------
    correlation : `~numpy.ndarray`
        Correlation matrix
    """
    diagonal = np.sqrt(covariance.diagonal())
    return (covariance.T / diagonal).T / diagonal
