# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity
from astropy.tests.helper import assert_quantity_allclose
from ...datasets import (make_test_psf, make_test_observation_table,
                         make_test_bg_cube_model, make_test_dataset)
from ...obs import ObservationTable, DataStore


def test_make_test_psf_fits_table():
    # TODO
    psf = make_test_psf()
    energy = Quantity(100, 'GeV')
    theta = Angle(0.1, 'deg')
    psf2 = psf.psf_at_energy_and_theta(energy, theta)
    fraction = psf2.containment_fraction(0.1)
    assert fraction == 0.44485638998490795


def test_make_test_observation_table():
    observatory_name = 'HESS'
    n_obs = 10
    random_state = np.random.RandomState(seed=0)
    obs_table = make_test_observation_table(observatory_name=observatory_name,
                                            n_obs=n_obs,
                                            random_state=random_state)

    # test: assert if the length of the table is n_obs:
    assert len(obs_table) == n_obs

    # test: assert if the header 'OBSERVATORY_NAME' is observatory_name
    assert obs_table.meta['OBSERVATORY_NAME'] == observatory_name

    # test: assert if the TIME_START > 0:
    assert (obs_table['TIME_START'] > 0).all()

    # test: assert if TIME_STOP > TIME_START:
    assert (obs_table['TIME_STOP'] > obs_table['TIME_START']).all()

    # test: assert if RA is in the interval (0, 360) deg:
    ra_min = Angle(0, 'degree')
    ra_max = Angle(360, 'degree')
    assert (ra_min < obs_table['RA']).all()
    assert (obs_table['RA'] < ra_max).all()

    # test: assert if dec is inthe interval (-90, 90) deg:
    dec_min = Angle(-90, 'degree')
    dec_max = Angle(90, 'degree')
    assert (dec_min < obs_table['DEC']).all()
    assert (obs_table['DEC'] < dec_max).all()


def test_make_test_bg_cube_model():
    # make a cube bg model with non-equal axes
    ndetx_bins = 1
    ndety_bins = 2
    nenergy_bins = 3
    bg_cube_model = make_test_bg_cube_model(ndetx_bins=ndetx_bins,
                                            ndety_bins=ndety_bins,
                                            nenergy_bins=nenergy_bins)

    # test shape of cube bg model
    assert len(bg_cube_model.data.shape) == 3
    assert bg_cube_model.data.shape == (nenergy_bins, ndety_bins, ndetx_bins)

    # make masked bg model
    bg_cube_model = make_test_bg_cube_model(apply_mask=True)

    # test that values with (x, y) > (0, 0) are zero
    x_points = Angle(np.arange(5) + 0.01, 'degree')
    y_points = Angle(np.arange(5) + 0.01, 'degree')
    e_points = bg_cube_model.energy_bin_centers
    x_points, y_points, e_points = np.meshgrid(x_points, y_points, e_points,
                                               indexing='ij')
    det_bin_index = bg_cube_model.find_coord_bin(Angle([x_points, y_points]))
    e_bin_index = bg_cube_model.find_energy_bin(e_points)
    bg = bg_cube_model.data[e_bin_index, det_bin_index[1], det_bin_index[0]]

    # assert that values are 0
    assert_quantity_allclose(bg, Quantity(0., bg.unit))


def test_make_test_dataset(tmpdir):
    # create a dataset
    fits_path = str(tmpdir.join('test_dataset'))
    observatory_name = 'HESS'
    scheme = 'HESS'
    n_obs = 2
    random_state = np.random.RandomState(seed=0)

    make_test_dataset(fits_path=fits_path,
                      observatory_name=observatory_name,
                      n_obs=n_obs,
                      random_state=random_state)

    # test number of files created
    n_event_list_files = sum(len([f for f in fs if f.lower().endswith('.fits.gz')])
                             for _, _, fs in os.walk(fits_path))
    assert n_event_list_files == 2*n_obs

    # test length of created observation list table
    data_store = DataStore(dir=fits_path, scheme=scheme)
    observation_table = data_store.make_observation_table()
    assert len(observation_table) == n_obs

    # test number of event list and effective area table files created
    event_list_files = data_store.make_table_of_files(observation_table,
                                                      filetypes=['events'])
    aeff_table_files = data_store.make_table_of_files(observation_table,
                                                      filetypes=['effective area'])
    assert len(event_list_files) == n_obs
    assert len(aeff_table_files) == n_obs
