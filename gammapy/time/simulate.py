# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.time import TimeDelta
from ..utils.random import get_random_state

__all__ = [
    'make_random_times_poisson_process',
]


def make_random_times_poisson_process(size, rate, dead_time=TimeDelta(0, format='sec'),
                                      random_state='random-seed'):
    """Make random times assuming a Poisson process.

    This function can be used to test event time series,
    to have a comparison what completely random data looks like.

    For the implementation see
    `here <http://preshing.com/20111007/how-to-generate-random-timings-for-a-poisson-process/>`__ and
    `here <http://stackoverflow.com/questions/1155539/how-do-i-generate-a-poisson-process>`__,
    as well as `numpy.random.exponential`.

    TODO: I think usually one has a given observation duration,
    not a given number of events to generate.
    Implementing this is more difficult because then the number
    of samples to generate is variable.

    Parameters
    ----------
    size : int
        Number of samples
    rate : `~astropy.units.Quantity`
        Event rate (dimension: 1 / TIME)
    dead_time : `~astropy.units.Quantity` or `~astropy.time.TimeDelta`, optional
        Dead time after event (dimension: TIME)
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.

    Returns
    -------
    time : `~astropy.time.TimeDelta`
        Time differences (second) after time zero.
    """
    # initialise random number generator
    random_state = get_random_state(random_state)

    dead_time = TimeDelta(dead_time)
    scale = 1
#    scale = (1 / rate).to('second').value
    time_delta = random_state.exponential(scale=scale, size=size)

    time_delta = TimeDelta(time_delta, format='sec')
    time_delta += dead_time

    return time_delta
