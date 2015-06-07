# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['plot_time_difference_distribution',
           ]


def plot_time_difference_distribution(time, ax=None):
    """Plot event time difference distribution.

    Parameters
    ----------
    time : `~astropy.time.Time`
        Event times (must be sorted)
    ax : `~matplotlib.axes.Axes` or None
        Axes

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
        Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gcf()

    td = time[1:] - time[:-1]

    # TODO: implement!
    raise NotImplementedError
