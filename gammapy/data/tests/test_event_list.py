# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from regions import CircleSkyRegion
from numpy.testing import assert_allclose
from ...utils.testing import requires_dependency, requires_data, mpl_plot_check
from ...data.event_list import EventListBase, EventList, EventListLAT
from ...maps import WcsGeom, Map, MapAxis

@requires_data("gammapy-data")
class TestEventListBase:
    def setup(self):
        filename = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
        self.events = EventListBase.read(filename)

    @pytest.mark.parametrize(
        "parameter, band", [("ENERGY", (0.8 * u.TeV, 5.0 * u.TeV))]
    )
    def test_select_parameter(self, parameter, band):
        selected_events = self.events.select_parameter(parameter, band)
        assert np.all(
            (selected_events.table[parameter].quantity >= band[0])
            & (selected_events.table[parameter].quantity < band[1])
        )


@requires_data("gammapy-data")
class TestEventListHESS:
    def setup(self):
        filename = (
            "$GAMMAPY_DATA/tests/unbundled/hess/run_0023037_hard_eventlist.fits.gz"
        )
        self.events = EventList.read(filename)

    def test_basics(self):
        assert "EventList" in str(self.events)

        assert len(self.events.table) == 49
        assert self.events.time[0].iso == "2004-10-14 00:08:39.214"
        assert self.events.radec[0].to_string() == "82.7068 19.8186"
        assert self.events.galactic[0].to_string(precision=2) == "185.96 -7.69"
        assert self.events.altaz[0].to_string() == "46.5793 30.8799"
        assert_allclose(self.events.offset[0].value, 1.904497742652893, rtol=1e-5)
        assert "{:1.5f}".format(self.events.energy[0]) == "11.64355 TeV"

        lon, lat, height = self.events.observatory_earth_location.to_geodetic()
        assert "{:1.5f}".format(lon) == "16.50022 deg"
        assert "{:1.5f}".format(lat) == "-23.27178 deg"
        assert "{:1.5f}".format(height) == "1835.00000 m"

    def test_altaz(self):
        altaz = self.events.altaz
        assert_allclose(altaz[0].az.deg, 46.579258, atol=1e-3)
        assert_allclose(altaz[0].alt.deg, 30.879939, atol=1e-3)

        altaz = self.events.altaz_from_table
        assert_allclose(altaz[0].az.deg, 46.205875, atol=1e-3)
        assert_allclose(altaz[0].alt.deg, 31.200132, atol=1e-3)
        # TODO: add asserts for frame properties

    def test_stack(self):
        event_lists = [self.events] * 3
        stacked_list = EventList.stack(event_lists)
        assert len(stacked_list.table) == 49 * 3

    @requires_dependency("matplotlib")
    def test_plot_time(self):
        with mpl_plot_check():
            self.events.plot_time()

    @requires_dependency("matplotlib")
    def test_plot_energy(self):
        with mpl_plot_check():
            self.events.plot_energy()

    @requires_dependency("matplotlib")
    def test_plot_offset2_distribution(self):
        with mpl_plot_check():
            self.events.plot_offset2_distribution()

    @requires_dependency("matplotlib")
    def test_plot_energy_offset(self):
        with mpl_plot_check():
            self.events.plot_energy_offset()

    @requires_dependency("matplotlib")
    def test_plot_image(self):
        with mpl_plot_check():
            self.events.plot_image()

    @requires_dependency("matplotlib")
    def test_peek(self):
        with mpl_plot_check():
            self.events.peek()


@requires_data("gammapy-data")
class TestEventListFermi:
    def setup(self):
        filename = "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz"
        self.events = EventListLAT.read(filename)

    def test_basics(self):
        assert "EventList" in str(self.events)
        assert len(self.events.table) == 32843

    @requires_dependency("matplotlib")
    def test_plot_image(self):
        with mpl_plot_check():
            self.events.plot_image()


@requires_data("gammapy-data")
class TestEventListChecker:
    def setup(self):
        self.event_list = EventList.read(
            "$GAMMAPY_DATA/cta-1dc/data/baseline/gps/gps_baseline_111140.fits"
        )

    def test_check_all(self):
        records = list(self.event_list.check())
        assert len(records) == 3

class TestEventSelection:
    def setup(self):
        ra = [0., 0., 0., 10.]*u.deg
        dec = [0.,0.9, 10., 10.]*u.deg
        energy = [1.,1.5, 1.5, 10.]*u.TeV

        evt_table = Table([ra,dec,energy], names=['RA','DEC','ENERGY'])
        self.evt_list = EventListBase(evt_table)

        center = SkyCoord(0.,0., frame='icrs', unit='deg')
        self.on_region = CircleSkyRegion(center,radius=1.0*u.deg)

    def test_map_select(self):
        axis = MapAxis.from_edges((0.5, 2.), unit='TeV', name='ENERGY')
        geom = WcsGeom.create(skydir=(0,0), binsz=0.2, width=4. * u.deg, proj='TAN', axes=[axis])

        mask_data = geom.region_mask(regions=[self.on_region], inside=True).astype(float)
        mask = Map.from_geom(geom, data=mask_data)
        new_list = self.evt_list.select_map_mask(mask)
        print(new_list.table)
        assert len(new_list.table) == 2