# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from ...obs import ObservationTable
from ...datasets import make_test_observation_table


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
    # TODO: more!!! (implement some filters and test them!!!
