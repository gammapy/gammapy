from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.time import TimeDelta
from gammapy.utils.random import get_random_state
import matplotlib . pyplot as plt
from numpy import linspace
from scipy.stats import expon,norm
#import scipy.stats as ss
import matplotlib.mlab as mlab
import math
from pylab import plot,show,hist,figure,title
from ..exptest import exptest_for_run


__all__ = [
    'make_random_times_poisson_process',
]

Mr_array=[]

def make_random_times_poisson_process(size, rate, dead_time=TimeDelta(0, format='sec'),
                                      random_state='random-seed'):
    random_state = get_random_state(random_state)
    dead_time = TimeDelta(dead_time)
    scale = 1
    time_delta = random_state.exponential(scale=scale, size=size)
    time_delta = TimeDelta(time_delta, format='sec')
    time_delta += dead_time
    mean_time = np.mean(time_delta)
    normalized_time_delta = time_delta / mean_time
    max_normalized_time_delta=max(normalized_time_delta)

    M_value=exptest_for_run(time_delta)
    Mr_array.append(M_value)
  
    return Mr_array

#Make a simulation of 100 events 1000 times
for i in range(0,1000):
    make_random_times_poisson_process(100, 1.0)


