# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from astropy.coordinates import Angle
from astropy.time import Time
from ...obs import ObservationTable
from ...datasets import make_test_observation_table
from ...time import absolute_time


# def test_Observation():
#     Observation(GLON=42, GLAT=43)


def test_ObservationTable():
    ObservationTable()

def test_filter_observations():
    # create random observation table
    observatory_name='HESS'
    n_obs = 10
    obs_table = make_test_observation_table(observatory_name, n_obs)

    # test no selection: input and output tables should be the same
    filtered_obs_table = obs_table.filter_observations()
    assert len(filtered_obs_table) == len(obs_table)

    # filter some pars and check the correspoding values in the columns

    # test box selection in obs_id
    variable = 'OBS_ID'
    value_min = 2
    value_max = 5
    selection = dict(shape='box', variable=variable,
                     value_min=value_min, value_max=value_max)
    filtered_obs_table = obs_table.filter_observations(selection)
    assert (value_min < filtered_obs_table[variable]).all()
    assert (filtered_obs_table[variable] < value_max).all()

    # test box selection in obs_id inverted
    selection = dict(shape='box', variable=variable,
                     value_min=value_min, value_max=value_max, inverted=True)
    filtered_obs_table = obs_table.filter_observations(selection)
    selection = dict(shape='box', variable=variable,
                     value_min=value_min, value_max=value_max, inverted=True)
    filtered_obs_table = obs_table.filter_observations(selection)

    # test circle selection in obs_id
    variable = 'OBS_ID'
    center = 4
    radius = 2
    selection = dict(shape='circle', variable=variable,
                     center=center, radius=radius)
    filtered_obs_table = obs_table.filter_observations(selection)
    assert (center - radius < filtered_obs_table[variable]).all()
    assert (filtered_obs_table[variable] < center + radius).all()

    # test box selection in alt
    variable = 'ALT'
    value_min = Angle(Angle(70., 'degree').radian, 'radian')
    value_max = Angle(Angle(80., 'degree').radian, 'radian')
    selection = dict(shape='box', variable=variable,
                     value_min=value_min, value_max=value_max)
    filtered_obs_table = obs_table.filter_observations(selection)
    assert (value_min < Angle(filtered_obs_table[variable])).all()
    assert (Angle(filtered_obs_table[variable]) < value_max).all()

    # test box selection in zenith angle
    variable = 'zenith'
    value_min = Angle(20., 'degree')
    value_max = Angle(30., 'degree')
    selection = dict(shape='box', variable=variable,
                     value_min=value_min, value_max=value_max)
    filtered_obs_table = obs_table.filter_observations(selection)
    zenith = Angle(90., 'degree') - filtered_obs_table['ALT']
    assert (value_min < zenith).all()
    assert (zenith < value_max).all()

    # test box selection in time_start
    variable = 'TIME_START'
    value_min = Time('2012-01-01 00:00:00', format='iso', scale='utc')
    value_max = Time('2014-01-01 00:00:00', format='iso', scale='utc')
    selection = dict(shape='box', variable=variable,
                     value_min=value_min, value_max=value_max)
    filtered_obs_table = obs_table.filter_observations(selection)
    time_start = absolute_time(filtered_obs_table['TIME_START'],
                               filtered_obs_table.meta)
    assert (value_min < time_start).all()
    assert (time_start < value_max).all()

    # test box selection in time: (time_start, time_stop) within (value_min,
    # value_min)
    # new obs table with very close (in time) observations (and times in
    # absolute times)
    datestart = Time('2012-01-01 00:03:00', format='iso', scale='utc')
    dateend = Time('2012-01-01 02:03:00', format='iso', scale='utc')
    obs_table_time = make_test_observation_table(observatory_name, n_obs,
                                                 datestart, dateend, True)
    variable = 'time'
    value_min = Time('2012-01-01 01:00:00', format='iso', scale='utc')
    value_max = Time('2012-01-01 02:00:00', format='iso', scale='utc')
    selection = dict(shape='box', variable=variable,
                     value_min=value_min, value_max=value_max)
    filtered_obs_table = obs_table_time.filter_observations(selection)
    time_start = filtered_obs_table['TIME_START']
    time_stop = filtered_obs_table['TIME_STOP']
    assert (value_min < time_start).all()
    assert (time_start < value_max).all()
    assert (value_min < time_stop).all()
    assert (time_stop < value_max).all()
