# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from ...obs import ObservationTable
from ...datasets import make_test_observation_table
from ...time import absolute_time
from ...catalog import skycoord_from_table


# def test_Observation():
#     Observation(GLON=42, GLAT=43)


def test_ObservationTable():
    ObservationTable()


def common_sky_region_select_test_routines(obs_table, selection):
    """Common routines for the tests of sky_box/sky_circle selection of obs tables"""
    type = selection['type']
    if type not in ['sky_box', 'sky_circle']:
        raise ValueError("Invalid type: {}".format(type))

    if type == 'sky_box':
        lon_range_eff = (selection['lon'][0] - selection['border'], selection['lon'][1] + selection['border'])
        lat_range_eff = (selection['lat'][0] - selection['border'], selection['lat'][1] + selection['border'])
    elif type == 'sky_circle':
        lon_cen = selection['lon']
        lat_cen = selection['lat']
        center = SkyCoord(lon_cen, lat_cen, frame=selection['frame'])
        radius_eff = selection['radius'] + selection['border']

    do_wrapping = False
    # not needed in the case of sky_circle
    if (type == 'sky_box' and
        any(l < Angle(0., 'degree') for l in lon_range_eff)):
        do_wrapping = True

    # observation table
    skycoord = skycoord_from_table(obs_table)

    # test on the selection
    selected_obs_table = obs_table.select_observations(selection)
    skycoord = skycoord_from_table(selected_obs_table)
    if type == 'sky_box':
        skycoord = skycoord.transform_to(selection['frame'])
        lon = skycoord.data.lon
        lat = skycoord.data.lat
        if do_wrapping:
            lon = lon.wrap_at(Angle(180, 'degree'))
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
            lon = lon.wrap_at(Angle(180, 'degree'))
        assert ((lon_range_eff[0] >= lon) | (lon >= lon_range_eff[1]) |
                (lat_range_eff[0] >= lat) | (lat >= lat_range_eff[1])).all()
    elif type == 'sky_circle':
        ang_distance = skycoord.separation(center)
        assert (ang_distance >= radius_eff).all()

    # the sum of number of entries in both selections should be the total number of entries
    assert len(selected_obs_table) + len(inv_selected_obs_table) == len(obs_table)


def test_select_parameter_box():
    # create random observation table
    observatory_name='HESS'
    n_obs = 10
    obs_table = make_test_observation_table(observatory_name, n_obs)

    # test no selection: input and output tables should be the same
    selected_obs_table = obs_table.select_observations()
    assert len(selected_obs_table) == len(obs_table)

    # select some pars and check the correspoding values in the columns

    # test box selection in obs_id
    variable = 'OBS_ID'
    value_min = 2
    value_max = 5
    selection = dict(type='par_box', variable=variable,
                     value_min=value_min, value_max=value_max)
    selected_obs_table = obs_table.select_observations(selection)
    assert (value_min < selected_obs_table[variable]).all()
    assert (selected_obs_table[variable] < value_max).all()

    # test box selection in obs_id inverted
    selection = dict(type='par_box', variable=variable,
                     value_min=value_min, value_max=value_max, inverted=True)
    selected_obs_table = obs_table.select_observations(selection)
    assert len(selected_obs_table) == 8
    assert ((value_min >= selected_obs_table[variable]) |
            (selected_obs_table[variable] >= value_max)).all()

    # test box selection in alt
    variable = 'ALT'
    value_min = Angle(60., 'degree')
    value_max = Angle(70., 'degree')
    selection = dict(type='par_box', variable=variable,
                     value_min=value_min, value_max=value_max)
    selected_obs_table = obs_table.select_observations(selection)
    assert (value_min < Angle(selected_obs_table[variable])).all()
    assert (Angle(selected_obs_table[variable]) < value_max).all()


def test_select_time_box():
    # create random observation table with very close (in time)
    # observations (and times in absolute times)
    observatory_name='HESS'
    n_obs = 10
    datestart = Time('2012-01-01T00:30:00', format='isot', scale='utc')
    dateend = Time('2012-01-01T02:30:00', format='isot', scale='utc')
    obs_table_time = make_test_observation_table(observatory_name, n_obs,
                                                 datestart, dateend, True)

    # test box selection in time: (time_start, time_stop) within (value_min, value_max)
    print()
    print("Test box selection in time")
    value_min = Time('2012-01-01T01:00:00', format='isot', scale='utc')
    value_max = Time('2012-01-01T02:00:00', format='isot', scale='utc')
    selection = dict(type='time_box',
                     time_min=value_min, time_max=value_max)
    selected_obs_table = obs_table_time.select_observations(selection)
    time_start = selected_obs_table['TIME_START']
    time_stop = selected_obs_table['TIME_STOP']
    assert (value_min < time_start).all()
    assert (time_start < value_max).all()
    assert (value_min < time_stop).all()
    assert (time_stop < value_max).all()


def test_select_sky_regions():

    # create random observation table with many entries
    observatory_name='HESS'
    n_obs = 100
    obs_table = make_test_observation_table(observatory_name, n_obs)

    # test sky box selection in gal coordinates
    lon_range = Angle([-100., 50.], 'degree')
    lat_range = Angle([-25., 25.], 'degree')
    frame = 'galactic'
    border = Angle(2., 'degree')
    selection = dict(type='sky_box', frame=frame,
                     lon=lon_range,
                     lat=lat_range,
                     border=border)
    common_sky_region_select_test_routines(obs_table, selection)

    # test sky box selection in radec coordinates
    lon_range = Angle([150., 300.], 'degree')
    lat_range = Angle([-50., 0.], 'degree')
    frame = 'icrs'
    border = Angle(2., 'degree')
    selection = dict(type='sky_box', frame=frame,
                     lon=lon_range,
                     lat=lat_range,
                     border=border)
    common_sky_region_select_test_routines(obs_table, selection)

    # test sky circle selection in gal coordinates
    lon_cen = Angle(0., 'degree')
    lat_cen = Angle(0., 'degree')
    radius = Angle(50., 'degree')
    frame = 'galactic'
    border = Angle(2., 'degree')
    selection = dict(type='sky_circle', frame=frame,
                     lon=lon_cen, lat=lat_cen,
                     radius=radius, border=border)
    common_sky_region_select_test_routines(obs_table, selection)

    # test sky circle selection in radec coordinates
    lon_cen = Angle(130., 'degree')
    lat_cen = Angle(-40., 'degree')
    radius = Angle(50., 'degree')
    frame = 'icrs'
    border = Angle(2., 'degree')
    selection = dict(type='sky_circle', frame=frame,
                     lon=lon_cen, lat=lat_cen,
                     radius=radius, border=border)
    common_sky_region_select_test_routines(obs_table, selection)
