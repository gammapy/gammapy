# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
import pytest
from ...utils.testing import requires_data
from ..pointing import PointingInfo


@requires_data('gammapy-extra')
class TestPointingInfo:
    @classmethod
    def setup_class(cls):
        filename = '$GAMMAPY_EXTRA/test_datasets/hess_event_list.fits'
        cls.pointing_info = PointingInfo.read(filename)

    def test_str(self):
        ss = str(self.pointing_info)
        assert 'Pointing info' in ss

    def test_location(self):
        lon, lat, height = self.pointing_info.location.geodetic
        assert_allclose(lon.deg, 16.5002222222222)
        assert_allclose(lat.deg, -23.2717777777778)
        assert_allclose(height.value, 1834.999999999783)

    def test_time_ref(self):
        assert self.pointing_info.time_ref.fits == '2001-01-01T00:01:04.184(TT)'

    def test_table(self):
        assert len(self.pointing_info.table) == 100

    def test_time(self):
        time = self.pointing_info.time
        assert len(time) == 100
        assert time.fits[0] == '2004-01-21T19:50:02.184(TT)'

    def test_duration(self):
        duration = self.pointing_info.duration
        assert duration.sec == 1586.0000000044238

    def test_radec(self):
        pos = self.pointing_info.radec[0]
        assert_allclose(pos.ra.deg, 83.633333333333)
        assert_allclose(pos.dec.deg, 24.51444444)
        assert pos.name == 'icrs'

    def test_altaz(self):
        pos = self.pointing_info.altaz[0]
        assert_allclose(pos.az.deg, 11.20432353385406)
        assert_allclose(pos.alt.deg, 41.37921408774436)
        assert pos.name == 'altaz'

    @pytest.mark.xfail
    def test_position_consistency(self):
        """
        Test if ALT / AZ in the table is consistent
        with ALT / AZ computed from TIME, LOCATION, RA / DEC
        and Astropy coordinates.
        """
        altaz_hess = self.pointing_info.altaz
        altaz_astropy = self.pointing_info.radec.transform_to(self.pointing_info.altaz_frame)
        sky_diff = altaz_astropy.separation(altaz_hess).to('arcsec')

        # TODO: positions currently don't match ... should be very close to zero.
        # Investigation ongoing here:
        # https://github.com/cdeil/astrometric_checks/blob/master/test_pointing.py
        # https://github.com/gammasky/hess-host-analyses/issues/47
        assert_allclose(sky_diff.max().arcsec, 697.9085394803815)
