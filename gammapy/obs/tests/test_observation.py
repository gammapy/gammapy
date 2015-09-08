# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from astropy.tests.helper import remote_data
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from ...obs import (ObservationTable, ObservationGroups,
                    ObservationGroupAxis)
from ... import datasets
from ...datasets import make_test_observation_table
from ...catalog import skycoord_from_table


def test_ObservationTable():
    obs_table = ObservationTable()
    # TODO: implement some useful test with asserts


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

    # the sum of number of entries in both selections should be the total
    # number of entries
    assert len(selected_obs_table) + len(inv_selected_obs_table) == len(obs_table)


def test_select_parameter_box():
    # create random observation table
    random_state = np.random.RandomState(seed=0)
    obs_table = make_test_observation_table(n_obs=10, random_state=random_state)

    # test no selection: input and output tables should be the same
    selected_obs_table = obs_table.select_observations()
    assert len(selected_obs_table) == len(obs_table)

    # select some pars and check the correspoding values in the columns

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
    value_range = Angle([60., 70.], 'degree')
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
    time_start = selected_obs_table['TIME_START']
    time_stop = selected_obs_table['TIME_STOP']
    assert (value_range[0] < time_start).all()
    assert (time_start < value_range[1]).all()
    assert (value_range[0] < time_stop).all()
    assert (time_stop < value_range[1]).all()


def test_select_sky_regions():
    # create random observation table with many entries
    random_state = np.random.RandomState(seed=0)
    obs_table = make_test_observation_table(n_obs=100, random_state=random_state)

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


@remote_data
def test_ObservationGroups(tmpdir):
    """Test create obs groups"""
    alt = Angle([0, 30, 60, 90], 'degree')
    az = Angle([-90, 90, 270], 'degree')
    ntels = np.array([3, 4])
    list_obs_group_axis = [ObservationGroupAxis('ALT', alt, 'bin_edges'),
                           ObservationGroupAxis('AZ', az, 'bin_edges'),
                           ObservationGroupAxis('N_TELS', ntels, 'bin_values')]
    obs_groups = ObservationGroups(list_obs_group_axis)
    assert ((0 <= obs_groups.obs_groups_table['GROUP_ID']) &
            (obs_groups.obs_groups_table['GROUP_ID'] < obs_groups.n_groups)).all()

    # write
    obs_groups_1 = obs_groups
    outfile = str(tmpdir.join('obs_groups.ecsv'))
    obs_groups_1.write(outfile)

    # read
    obs_groups_2 = ObservationGroups.read(outfile)

    # test that obs groups read from file match the ones defined
    assert (obs_groups_1.obs_groups_table == obs_groups_2.obs_groups_table).all()

    # test group obs list
    # using file in gammapy-extra (I also could create a dummy table)
    infile = datasets.get_path('../test_datasets/obs/test_observation_table.fits',
                               location='remote')
    obs_table = ObservationTable.read(infile)

    # wrap azimuth angles to [-90, 270) deg
    # to match definition of azimuth grouping axis
    obs_table['AZ'] = Angle(obs_table['AZ']).wrap_at(Angle(270., 'degree'))

    # group obs list
    obs_table_grouped = obs_groups.group_observation_table(obs_table)

    # assert consistency of the grouping
    assert len(obs_table) == len(obs_table_grouped)
    assert ((0 <= obs_table_grouped['GROUP_ID']) &
            (obs_table_grouped['GROUP_ID'] < obs_groups.n_groups)).all()
    # check grouping for 1 group
    group_id = 10
    obs_table_grouped_10 = obs_groups.get_group_of_observations(obs_table_grouped,
                                                                group_id)
    alt_min = obs_groups.obs_groups_table['ALT_MIN'][group_id]
    alt_max = obs_groups.obs_groups_table['ALT_MAX'][group_id]
    az_min = obs_groups.obs_groups_table['AZ_MIN'][group_id]
    az_max = obs_groups.obs_groups_table['AZ_MAX'][group_id]
    n_tels = obs_groups.obs_groups_table['N_TELS'][group_id]
    assert ((alt_min <= obs_table_grouped_10['ALT']) &
            (obs_table_grouped_10['ALT'] < alt_max)).all()
    assert ((az_min <= obs_table_grouped_10['AZ']) &
            (obs_table_grouped_10['AZ'] < az_max)).all()
    assert (n_tels == obs_table_grouped_10['N_TELS']).all()
    # check on inverse mask (i.e. all other groups)
    obs_table_grouped_not10 = obs_groups.get_group_of_observations(obs_table_grouped,
                                                                   group_id,
                                                                   inverted=True)
    assert (((alt_min > obs_table_grouped_not10['ALT']) |
             (obs_table_grouped_not10['ALT'] >= alt_max)) |
            ((az_min > obs_table_grouped_not10['AZ']) |
             (obs_table_grouped_not10['AZ'] >= az_max)) |
            (n_tels != obs_table_grouped_not10['N_TELS'])).all()
    # check sum of selections
    assert (len(obs_table_grouped_10) + len(obs_table_grouped_not10)
            == len(obs_table_grouped))


def test_ObservationGroupAxis():
    """Test create a few obs group axis objects"""

    alt = Angle([0, 30, 60, 90], 'degree')
    alt_obs_group_axis = ObservationGroupAxis('ALT', alt, 'bin_edges')
    assert alt_obs_group_axis.n_bins == len(alt) - 1

    az = Angle([-90, 90, 270], 'degree')
    az_obs_group_axis = ObservationGroupAxis('AZ', az, 'bin_edges')
    assert az_obs_group_axis.n_bins == len(az) - 1

    ntels = np.array([3, 4])
    ntels_obs_group_axis = ObservationGroupAxis('N_TELS', ntels, 'bin_values')
    assert ntels_obs_group_axis.n_bins == len(ntels)
