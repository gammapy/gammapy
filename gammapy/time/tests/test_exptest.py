import numpy as np
from numpy.testing import assert_allclose
from astropy.units import Quantity
from gammapy.time.exptest import exptest_for_run
from gammapy.time import make_random_times_poisson_process


def test_exptest_for_run():
    # setup
    rate = Quantity(100, 's^-1')
    time_delta = make_random_times_poisson_process(1000, rate=rate, random_state=0)
    print('time_delta = {}'.format(time_delta))

    # execution
    mr = exptest_for_run(time_delta)
    print('mr = {}'.format(mr))

    # checks on results
    assert_allclose(mr, 1.3790634240947202)


