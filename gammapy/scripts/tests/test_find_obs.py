# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest, remote_data
from ... import datasets
from ..find_obs import main as find_obs_main
from ...obs import ObservationTable


@pytest.mark.parametrize("extra_options,n_selected_obs", [
    (["--x", "0", "--y", "0", "--r", "50", "--system", "galactic"], 31),
    (["--x", "0", "--y", "0", "--dx", "20", "--dy", "3", "--system", "galactic"], 1),
    (["--t_start", "2012-04-20", "--t_stop", "2012-04-30T12:42"], 3),
    (["--par_name", "OBS_ID", "--par_min", "42", "--par_max", "101"], 59),
    (["--par_name", "ALT", "--par_min", "70", "--par_max", "90"], 13),
    ])
@remote_data
def test_find_obs_main(extra_options, n_selected_obs, tmpdir):
    infile = datasets.get_path('../test_datasets/obs/test_observation_table.fits',
                                 location='remote')
    outfile = str(tmpdir.join('find_obs_test.fits'))
    find_obs_main([infile, outfile] + extra_options)
    observation_table = ObservationTable.read(outfile)
    assert len(observation_table) == n_selected_obs
