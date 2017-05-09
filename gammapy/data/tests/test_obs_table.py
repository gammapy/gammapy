# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from ...datasets import make_test_observation_table
from ...catalog import skycoord_from_table


def common_sky_region_select_test_routines(obs_table, selection):
    """Common routines for the tests of sky_box/sky_circle selection of obs tables."""
    type = selection['type']

    if type not in ['sky_box', 'sky_circle']:
        raise ValueError("Invalid type: {}".format(type))

    if type == 'sky_box':
        lon_range_eff = (selection['lon'][0] - selection['border'],
                         selection['lon'][1] + selection['border'])
        lat_range_eff = (selection['lat'][0] - selection['border'],
                         selection['lat'][1] + selection['border'])
    elif type == 'sky_circle':
        lon_cen = selection['lon']
        lat_cen = selection['lat']
        center = SkyCoord(lon_cen, lat_cen, frame=selection['frame'])
        radius_eff = selection['radius'] + selection['border']

    do_wrapping = False
    # not needed in the case of sky_circle
    if type == 'sky_box' and any(l < Angle(0., 'deg') for l in lon_range_eff):
        do_wrapping = True

    # test on the selection
    selected_obs_table = obs_table.select_observations(selection)
    skycoord = skycoord_from_table(selected_obs_table)
    if type == 'sky_box':
        skycoord = skycoord.transform_to(selection['frame'])
        lon = skycoord.data.lon
        lat = skycoord.data.lat
        if do_wrapping:
            lon = lon.wrap_at(Angle(180, 'deg'))
        assert ((lon_range_eff[0] < lon) & (lon < lon_range_eff[1]) &
                (lat_range_eff[0] < lat) & (lat < lat_range_eff[1])).all()
    elif type == 'sky_circle':
        ang_distance = skycoord.separation(center)
        assert (ang_distance < radius_eff).all()

    # test on the inverted selection
    selection['inverted'] = True
    inv_selected_obs_table = obs_table.select_observations(selection)
    skycoord = skycoord_from_table(inv_selected_obs_table)
    if type == 'sky_box':
        skycoord = skycoord.transform_to(selection['frame'])
        lon = skycoord.data.lon
        lat = skycoord.data.lat
        if do_wrapping:
            lon = lon.wrap_at(Angle(180, 'deg'))
        assert ((lon_range_eff[0] >= lon) | (lon >= lon_range_eff[1]) |
                (lat_range_eff[0] >= lat) | (lat >= lat_range_eff[1])).all()
    elif type == 'sky_circle':
        ang_distance = skycoord.separation(center)
        assert (ang_distance >= radius_eff).all()

    # the sum of number of entries in both selections should be the total
    # number of entries
    assert len(selected_obs_table) + len(inv_selected_obs_table) == len(obs_table)


def test_select_parameter_box():
    # create random observation table
    random_state = np.random.RandomState(seed=0)
    obs_table = make_test_observation_table(n_obs=10, random_state=random_state)

    # select some pars and check the corresponding values in the columns

    # test box selection in obs_id
    variable = 'OBS_ID'
    value_range = [2, 5]
    selection = dict(type='par_box', variable=variable, value_range=value_range)
    selected_obs_table = obs_table.select_observations(selection)
    assert len(selected_obs_table) == 3
    assert (value_range[0] <= selected_obs_table[variable]).all()
    assert (selected_obs_table[variable] < value_range[1]).all()

    # test box selection in obs_id inverted
    selection = dict(type='par_box', variable=variable,
                     value_range=value_range, inverted=True)
    selected_obs_table = obs_table.select_observations(selection)
    assert len(selected_obs_table) == 7
    assert ((value_range[0] > selected_obs_table[variable]) |
            (selected_obs_table[variable] >= value_range[1])).all()

    # test box selection in alt
    variable = 'ALT'
    value_range = Angle([60., 70.], 'deg')
    selection = dict(type='par_box', variable=variable, value_range=value_range)
    selected_obs_table = obs_table.select_observations(selection)
    assert (value_range[0] < Angle(selected_obs_table[variable])).all()
    assert (Angle(selected_obs_table[variable]) < value_range[1]).all()


def test_select_time_box():
    # create random observation table with very close (in time)
    # observations (and times in absolute times)
    datestart = Time('2012-01-01T00:30:00')
    dateend = Time('2012-01-01T02:30:00')
    random_state = np.random.RandomState(seed=0)
    obs_table_time = make_test_observation_table(n_obs=10,
                                                 date_range=(datestart, dateend),
                                                 use_abs_time=True,
                                                 random_state=random_state)

    # test box selection in time: (time_start, time_stop) within value_range
    value_range = Time(['2012-01-01T01:00:00', '2012-01-01T02:00:00'])
    selection = dict(type='time_box', time_range=value_range)
    selected_obs_table = obs_table_time.select_observations(selection)
    time_start = selected_obs_table['TSTART']
    assert (value_range[0] < time_start).all()
    assert (time_start < value_range[1]).all()


def test_select_sky_regions():
    # create random observation table with many entries
    random_state = np.random.RandomState(seed=0)
    obs_table = make_test_observation_table(n_obs=100, random_state=random_state)

    # test sky box selection in gal coordinates
    lon_range = Angle([-100., 50.], 'deg')
    lat_range = Angle([-25., 25.], 'deg')
    frame = 'galactic'
    border = Angle(2., 'deg')
    selection = dict(type='sky_box', frame=frame,
                     lon=lon_range,
                     lat=lat_range,
                     border=border)
    common_sky_region_select_test_routines(obs_table, selection)

    # test sky box selection in radec coordinates
    lon_range = Angle([150., 300.], 'deg')
    lat_range = Angle([-50., 0.], 'deg')
    frame = 'icrs'
    border = Angle(2., 'deg')
    selection = dict(type='sky_box', frame=frame,
                     lon=lon_range,
                     lat=lat_range,
                     border=border)
    common_sky_region_select_test_routines(obs_table, selection)

    # test sky circle selection in gal coordinates
    lon_cen = Angle(0., 'deg')
    lat_cen = Angle(0., 'deg')
    radius = Angle(50., 'deg')
    frame = 'galactic'
    border = Angle(2., 'deg')
    selection = dict(type='sky_circle', frame=frame,
                     lon=lon_cen, lat=lat_cen,
                     radius=radius, border=border)
    common_sky_region_select_test_routines(obs_table, selection)

    # test sky circle selection in radec coordinates
    lon_cen = Angle(130., 'deg')
    lat_cen = Angle(-40., 'deg')
    radius = Angle(50., 'deg')
    frame = 'icrs'
    border = Angle(2., 'deg')
    selection = dict(type='sky_circle', frame=frame,
                     lon=lon_cen, lat=lat_cen,
                     radius=radius, border=border)
    common_sky_region_select_test_routines(obs_table, selection)
