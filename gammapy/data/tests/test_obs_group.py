# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import Angle
from ...utils.testing import requires_data, requires_dependency
from ...datasets import gammapy_extra
from ..obs_table import ObservationTable
from ..obs_group import ObservationGroups, ObservationGroupAxis


def make_test_obs_groups():
    zenith = Angle([0, 30, 60, 90], 'deg')
    ntels = [3, 4]
    obs_groups = ObservationGroups([
        ObservationGroupAxis('ZENITH', zenith, fmt='edges'),
        ObservationGroupAxis('N_TELS', ntels, fmt='values'),
    ])
    return obs_groups


def make_test_obs_table():
    # test group obs list
    infile = gammapy_extra.filename('test_datasets/obs/test_observation_table.fits')
    obs_table = ObservationTable.read(infile)

    # wrap azimuth angles to [-90, 270) deg
    # to match definition of azimuth grouping axis
    obs_table['AZ'] = Angle(obs_table['AZ']).wrap_at(Angle(270., 'deg'))
    obs_table['ZENITH'] = Angle(90, 'deg') - obs_table['ALT']

    return obs_table


@requires_data('gammapy-extra')
def test_obsgroup():
    obs_groups = make_test_obs_groups()
    obs_table = make_test_obs_table()

    assert ((0 <= obs_groups.obs_groups_table['GROUP_ID']) &
            (obs_groups.obs_groups_table['GROUP_ID'] < obs_groups.n_groups)).all()

    # group obs list
    obs_table_grouped = obs_groups.apply(obs_table)

    # assert consistency of the grouping
    assert len(obs_table) == len(obs_table_grouped)
    assert ((0 <= obs_table_grouped['GROUP_ID']) &
            (obs_table_grouped['GROUP_ID'] < obs_groups.n_groups)).all()

    # check grouping for one group
    group_id = 5
    obs_table_group_5 = obs_groups.get_group_of_observations(obs_table_grouped, group_id)
    zenith_min = obs_groups.obs_groups_table['ZENITH_MIN'][group_id]
    zenith_max = obs_groups.obs_groups_table['ZENITH_MAX'][group_id]
    n_tels = obs_groups.obs_groups_table['N_TELS'][group_id]
    assert ((zenith_min <= obs_table_group_5['ZENITH']) &
            (obs_table_group_5['ZENITH'] < zenith_max)).all()
    assert (n_tels == obs_table_group_5['N_TELS']).all()
    # check on inverse mask (i.e. all other groups)
    obs_table_grouped_not5 = obs_groups.get_group_of_observations(obs_table_grouped,
                                                                  group_id,
                                                                  inverted=True)
    assert (((zenith_min > obs_table_grouped_not5['ZENITH']) |
             (obs_table_grouped_not5['ZENITH'] >= zenith_max)) |
            (n_tels != obs_table_grouped_not5['N_TELS'])).all()

    # check sum of selections
    assert len(obs_table_group_5) + len(obs_table_grouped_not5) == len(obs_table_grouped)


@requires_dependency('pyyaml')
@requires_data('gammapy-extra')
def test_obsgroup_io():
    obs_groups = make_test_obs_groups()

    filename = 'obs_groups.ecsv'
    obs_groups.write(filename)
    obs_groups2 = ObservationGroups.read(filename)

    # test that obs groups read from file match the ones defined
    assert obs_groups.n_groups == obs_groups2.n_groups
    assert obs_groups.axes[1].name == obs_groups2.axes[1].name
    assert_allclose(obs_groups.obs_groups_table['ZENITH_MAX'], obs_groups2.obs_groups_table['ZENITH_MAX'])


def test_obsgroup_axis():
    """Test create a few obs group axis objects"""

    alt = Angle([0, 30, 60, 90], 'deg')
    alt_obs_group_axis = ObservationGroupAxis('ALT', alt, fmt='edges')
    assert alt_obs_group_axis.n_bins == len(alt) - 1

    az = Angle([-90, 90, 270], 'deg')
    az_obs_group_axis = ObservationGroupAxis('AZ', az, fmt='edges')
    assert az_obs_group_axis.n_bins == len(az) - 1

    ntels = np.array([3, 4])
    ntels_obs_group_axis = ObservationGroupAxis('N_TELS', ntels, fmt='values')
    assert ntels_obs_group_axis.n_bins == len(ntels)
