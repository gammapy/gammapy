# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

__all__ = ["exptest"]


def exptest(time_delta):
    """Compute the level of variability for a certain period of time.

    The level of variability is quantified by ``mr``, as defined in Prahl (1999).

    For constant rate events, it follows a standard normal distribution,
    i.e. it can be used directly as the significance of variability.

    For example usage, see :ref:`time-variability`.

    Parameters
    ----------
    time_delta : array-like
        Time differences between consecutive events

    Returns
    -------
    mr : float
        Level of variability

    References
    ----------
    .. [1] Prahl (1999), "A fast unbinned test on event clustering in Poisson processes",
       `Link <http://adsabs.harvard.edu/abs/1999astro.ph..9399P>`_
    """
    mean_time = np.mean(time_delta)
    normalized_time_delta = time_delta / mean_time
    mask = normalized_time_delta < 1
    sum_time = 1 - normalized_time_delta[mask] / 1
    sum_time_all = np.sum([sum_time])
    m_value = sum_time_all / len(time_delta)
    # the numbers are from Prahl(1999), derived from simulations
    term1 = m_value - (1 / 2.71828 - 0.189 / len(time_delta))
    term2 = 0.2427 / np.sqrt(len(time_delta))
    mr = term1 / term2
    return mr
