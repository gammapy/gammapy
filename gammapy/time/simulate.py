# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.time import TimeDelta
from ..utils.random import get_random_state

__all__ = ["random_times"]


def random_times(
    size,
    rate,
    dead_time=TimeDelta(0, format="sec"),
    return_diff=False,
    random_state="random-seed",
):
    """Make random times assuming a Poisson process.

    This function can be used to test event time series,
    to have a comparison what completely random data looks like.

    Can be used in two ways (in either case the return type is `~astropy.time.TimeDelta`):

    * ``return_delta=False`` - Return absolute times, relative to zero (default)
    * ``return_delta=True`` - Return time differences between consecutive events.

    Parameters
    ----------
    size : int
        Number of samples
    rate : `~astropy.units.Quantity`
        Event rate (dimension: 1 / TIME)
    dead_time : `~astropy.units.Quantity` or `~astropy.time.TimeDelta`, optional
        Dead time after event (dimension: TIME)
    return_diff : bool
        Return time difference between events? (default: no, return absolute times)
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.

    Returns
    -------
    time : `~astropy.time.TimeDelta`
        Time differences (second) after time zero.

    Examples
    --------
    Example how to simulate 100 events at a rate of 10 Hz.
    As expected the last event occurs after about 10 seconds.

    >>> from astropy.units import Quantity
    >>> from gammapy.time import random_times
    >>> rate = Quantity(10, 'Hz')
    >>> times = random_times(size=100, rate=rate, random_state=0)
    >>> times[-1]
    <TimeDelta object: scale='None' format='sec' value=9.186484131475076>
    """
    random_state = get_random_state(random_state)

    dead_time = TimeDelta(dead_time)
    scale = (1 / rate).to("second").value
    time_delta = random_state.exponential(scale=scale, size=size)
    time_delta += dead_time.to("second").value

    if return_diff:
        return TimeDelta(time_delta, format="sec")
    else:
        time = time_delta.cumsum()
        return TimeDelta(time, format="sec")
