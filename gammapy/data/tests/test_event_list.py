# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from regions import CircleSkyRegion, RectangleSkyRegion
from gammapy.data import EventList, EventListBase, EventListLAT
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.utils.testing import mpl_plot_check, requires_data, requires_dependency


@requires_data()
class TestEventListBase:
    def setup_class(self):
        self.events = EventListBase.read(
            "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
        )

    def test_select_parameter(self):
        events = self.events.select_parameter("ENERGY", (0.8 * u.TeV, 5.0 * u.TeV))
        assert len(events.table) == 2716


@requires_data()
class TestEventListHESS:
    def setup_class(self):
        self.events = EventList.read(
            "$GAMMAPY_DATA/tests/unbundled/hess/run_0023037_hard_eventlist.fits.gz"
        )

    def test_basics(self):
        assert "EventList" in str(self.events)

        assert len(self.events.table) == 49
        assert self.events.time[0].iso == "2004-10-14 00:08:39.214"
        assert self.events.radec[0].to_string() == "82.7068 19.8186"
        assert self.events.galactic[0].to_string(precision=2) == "185.96 -7.69"
        assert self.events.altaz[0].to_string() == "46.5793 30.8799"
        assert_allclose(self.events.offset[0].value, 1.904497742652893, rtol=1e-5)
        assert f"{self.events.energy[0]:1.5f}" == "11.64355 TeV"

        lon, lat, height = self.events.observatory_earth_location.to_geodetic()
        assert f"{lon:1.5f}" == "16.50022 deg"
        assert f"{lat:1.5f}" == "-23.27178 deg"
        assert f"{height:1.5f}" == "1835.00000 m"

    def test_observation_time_duration(self):
        dt = self.events.observation_time_duration
        assert dt.unit == "s"
        assert_allclose(dt.value, 1577)

    def test_observation_live_time_duration(self):
        dt = self.events.observation_live_time_duration
        assert dt.unit == "s"
        assert_allclose(dt.value, 1510.959106)

    def test_observation_dead_time_fraction(self):
        deadc = self.events.observation_dead_time_fraction
        assert_allclose(deadc, 0.035763, rtol=1e-3)

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


@requires_data()
class TestEventListFermi:
    def setup_class(self):
        self.events = EventListLAT.read(
            "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz"
        )

    def test_basics(self):
        assert "EventList" in str(self.events)
        assert len(self.events.table) == 32843

    @requires_dependency("matplotlib")
    def test_plot_image(self):
        with mpl_plot_check():
            self.events.plot_image()


@requires_data()
class TestEventListChecker:
    def setup_class(self):
        self.event_list = EventList.read(
            "$GAMMAPY_DATA/cta-1dc/data/baseline/gps/gps_baseline_111140.fits"
        )

    def test_check_all(self):
        records = list(self.event_list.check())
        assert len(records) == 3


class TestEventSelection:
    def setup_class(self):
        table = Table()
        table["RA"] = [0.0, 0.0, 0.0, 10.0] * u.deg
        table["DEC"] = [0.0, 0.9, 10.0, 10.0] * u.deg
        table["ENERGY"] = [1.0, 1.5, 1.5, 10.0] * u.TeV

        self.events = EventListBase(table)

        center1 = SkyCoord(0.0, 0.0, frame="icrs", unit="deg")
        on_region1 = CircleSkyRegion(center1, radius=1.0 * u.deg)
        center2 = SkyCoord(0.0, 10.0, frame="icrs", unit="deg")
        on_region2 = RectangleSkyRegion(center2, width=0.5 * u.deg, height=0.3 * u.deg)
        self.on_regions = [on_region1, on_region2]

    def test_region_select(self):
        geom = WcsGeom.create(skydir=(0, 0), binsz=0.2, width=4.0 * u.deg, proj="TAN")
        new_list = self.events.select_region(self.on_regions[0], geom.wcs)
        assert len(new_list.table) == 2

        union_region = self.on_regions[0].union(self.on_regions[1])
        new_list = self.events.select_region(union_region, geom.wcs)
        assert len(new_list.table) == 3

        region_string = "fk5;box(0,10, 0.25, 0.15)"
        new_list = self.events.select_region(region_string, geom.wcs)
        assert len(new_list.table) == 1

    def test_map_select(self):
        axis = MapAxis.from_edges((0.5, 2.0), unit="TeV", name="ENERGY")
        geom = WcsGeom.create(
            skydir=(0, 0), binsz=0.2, width=4.0 * u.deg, proj="TAN", axes=[axis]
        )

        mask_data = geom.region_mask(regions=[self.on_regions[0]], inside=True)
        mask = Map.from_geom(geom, data=mask_data)
        new_list = self.events.select_map_mask(mask)
        assert len(new_list.table) == 2
