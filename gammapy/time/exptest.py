# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

__all__ = [
    'exptest_for_run',
]

def exptest_for_run(time_delta=[]):
    """Compute Mr value, the level of variability for a certain period of time.

    Longer description: A single Mr value can be calculated, which shows the level of variability
                        for the whole period, or the Mr value for each run can be shown.


    Ref:Prah(1999),http://adsabs.harvard.edu/abs/1999astro.ph..9399P

    Parameters
    ----------
    run : list of int
        Run number for each event
    event_time : list of float
        times for each event
    expCount: list of float
        the acceptance for each run according to the observation conditions

    Returns
    -------
    Mr : float
        Level of variability
    """
    mean_time = np.mean(time_delta)
    normalized_time_delta = time_delta / mean_time
    sum_time = []
    for i in range(len(normalized_time_delta)):
        if normalized_time_delta[i] < 1:
            sum_time.append(1 - normalized_time_delta[i] / 1.0)
    mean_normalized_time=np.mean(normalized_time_delta)
    M_value=0
    Mr=0
    sum_time_all = np.sum([sum_time])
    if len(time_delta)!=0:
        M_value = sum_time_all / len(time_delta)
        Mr = (M_value - (1 / 2.71828 - 0.189 / len(time_delta))) / (0.2427 / np.sqrt(len(time_delta)))
    return Mr
