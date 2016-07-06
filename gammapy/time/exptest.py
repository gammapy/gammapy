# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

__all__ = [
    'exptest',
]

def exptest(time_delta):
    """Compute Mr value, the level of variability for a certain period of time.

     A single Mr value can be calculated, which shows the level of variability
     for the whole period, or the Mr value for each run can be shown.


    Ref:Prah(1999),http://adsabs.harvard.edu/abs/1999astro.ph..9399P

    Parameters
    ----------
    time_delta : list of float
        the time difference between two adjacent events

    Returns
    -------
    mr : float
        Level of variability
    """
    mean_time = np.mean(time_delta)
    normalized_time_delta = time_delta / mean_time
    sum_time = []

    mask = normalized_time_delta < 1
    sum_time = 1 - normalized_time_delta[mask] / 1
    mean_normalized_time=np.mean(normalized_time_delta)
    sum_time_all = np.sum([sum_time])
    m_value = sum_time_all / len(time_delta)
    # the numbers are from Prahl(1999), derived from simulations
    mr = (m_value - (1 / 2.71828 - 0.189 / len(time_delta))) / (0.2427 / np.sqrt(len(time_delta)))
    return mr
