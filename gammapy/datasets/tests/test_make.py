# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.coordinates import Angle
from astropy.units import Quantity
from ...utils.testing import assert_quantity
from ...datasets import make_test_psf, make_test_observation_table
from ...obs import ObservationTable


def test_make_test_psf_fits_table():
    # TODO
    psf = make_test_psf()
    energy = Quantity(100, 'GeV')
    theta = Angle(0.1, 'deg')
    psf2 = psf.psf_at_energy_and_theta(energy, theta)
    fraction = psf2.containment_fraction(0.1)
    assert fraction == 0.44485638998490795


def test_make_test_observation_table():
    observatory_name='HESS'
    n_obs = 10
    obs_table = make_test_observation_table(observatory_name, n_obs)

    # test: assert if the length of the table is n_obs:
    assert len(obs_table) == n_obs

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
