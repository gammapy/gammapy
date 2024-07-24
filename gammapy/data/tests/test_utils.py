# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.time import Time
from gammapy.data.utils import check_time_intervals


def test_check_time_intervals():
    assert check_time_intervals(1.0) is False
    assert check_time_intervals([]) is False
    assert check_time_intervals([53090.130, 53090.140]) is False

    aa = [
        Time(53090.130, format="mjd", scale="tt"),
        Time(53090.140, format="mjd", scale="tt"),
    ]
    bb = [
        Time(53090.150, format="mjd", scale="tt"),
        Time(53090.160, format="mjd", scale="tt"),
    ]
    ti = [aa, bb]
    assert check_time_intervals(ti) is True

    cc = [
        Time(53090.140, format="mjd", scale="tt"),
        Time(53090.160, format="mjd", scale="tt"),
    ]
    ti = [aa, cc]
    assert check_time_intervals(ti) is True

    dd = [
        Time(53080.150, format="mjd", scale="tt"),
        Time(53091.160, format="mjd", scale="tt"),
    ]
    ti = [aa, bb, dd]
    assert check_time_intervals(ti) is False

    from gammapy.data import utils

    utils.CHECK_OVERLAPPING_TI = False
    assert check_time_intervals(ti) is True
