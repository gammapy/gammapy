# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.io import fits
from astropy.units import Quantity
from gammapy.irf import (
    Background3D,
    EffectiveAreaTable2D,
    EnergyDispersion2D,
    RadMax2D,
    load_cta_irfs,
    load_irf_dict_from_file,
)
from gammapy.maps import MapAxis
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import requires_data


@requires_data()
def test_cta_irf():
    """Test that CTA IRFs can be loaded and evaluated."""
    irf = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )

    energy = Quantity(1, "TeV")
    offset = Quantity(3, "deg")

    val = irf["aeff"].evaluate(energy_true=energy, offset=offset)
    assert_allclose(val.value, 545269.4675, rtol=1e-5)
    assert val.unit == "m2"

    val = irf["edisp"].evaluate(offset=offset, energy_true=energy, migra=1)
    assert_allclose(val.value, 3183.6882, rtol=1e-5)
    assert val.unit == ""

    val = irf["psf"].evaluate(
        rad=Quantity(0.1, "deg"), energy_true=energy, offset=offset
    )
    assert_allclose(val, 3.56989 * u.Unit("deg-2"), rtol=1e-5)

    val = irf["bkg"].evaluate(energy=energy, fov_lon=offset, fov_lat="0 deg")
    assert_allclose(val.value, 9.400071e-05, rtol=1e-5)
    assert val.unit == "1 / (MeV s sr)"


@requires_data()
def test_cta_irf_alpha_config_south():
    """Test that CTA IRFs can be loaded and evaluated."""
    irf = load_cta_irfs(
        "$GAMMAPY_DATA/cta-caldb/Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits.gz"
    )

    energy = Quantity(1, "TeV")
    offset = Quantity(3, "deg")

    val = irf["aeff"].evaluate(energy_true=energy, offset=offset)
    assert_allclose(val.value, 493538.4460737773, rtol=1e-5)
    assert val.unit == "m2"

    val = irf["edisp"].evaluate(offset=offset, energy_true=energy, migra=1)
    assert_allclose(val.value, 0.0499099, rtol=1e-5)
    assert val.unit == ""

    val = irf["psf"].evaluate(
        rad=Quantity(0.1, "deg"), energy_true=energy, offset=offset
    )
    assert_allclose(val, 3.31135957 * u.Unit("deg-2"), rtol=1e-5)

    val = irf["bkg"].evaluate(energy=energy, fov_lon=offset, fov_lat="0 deg")
    assert_allclose(val.value, 8.98793486e-05, rtol=1e-5)
    assert val.unit == "1 / (MeV s sr)"


@requires_data()
def test_cta_irf_alpha_config_north():
    """Test that CTA IRFs can be loaded and evaluated."""
    irf = load_cta_irfs(
        "$GAMMAPY_DATA/cta-caldb/Prod5-North-20deg-AverageAz-4LSTs09MSTs.180000s-v0.1.fits.gz"
    )

    energy = Quantity(1, "TeV")
    offset = Quantity(3, "deg")

    val = irf["aeff"].evaluate(energy_true=energy, offset=offset)
    assert_allclose(val.value, 277301.26585409, rtol=1e-5)
    assert val.unit == "m2"

    val = irf["edisp"].evaluate(offset=offset, energy_true=energy, migra=1)
    assert_allclose(val.value, 0.04070749, rtol=1e-5)
    assert val.unit == ""

    val = irf["psf"].evaluate(
        rad=Quantity(0.1, "deg"), energy_true=energy, offset=offset
    )
    assert_allclose(val, 6.20107085 * u.Unit("deg-2"), rtol=1e-5)

    val = irf["bkg"].evaluate(energy=energy, fov_lon=offset, fov_lat="0 deg")
    assert_allclose(val.value, 5.43334659e-05, rtol=1e-5)
    assert val.unit == "1 / (MeV s sr)"


@requires_data()
def test_load_irf_dict_from_file():
    """Test that the IRF components in a dictionary loaded from a DL3 file can
    be loaded in a dictionary and correctly used"""
    irf = load_irf_dict_from_file(
        "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
    )

    energy = Quantity(1, "TeV")
    offset = Quantity(0.5, "deg")

    val = irf["aeff"].evaluate(energy_true=energy, offset=offset)
    assert_allclose(val.value, 273372.44851054, rtol=1e-5)
    assert val.unit == "m2"

    val = irf["edisp"].evaluate(offset=offset, energy_true=energy, migra=1)
    assert_allclose(val.value, 1.84269482, rtol=1e-5)
    assert val.unit == ""

    val = irf["psf"].evaluate(
        rad=Quantity(0.1, "deg"), energy_true=energy, offset=offset
    )
    assert_allclose(val, 6.75981573 * u.Unit("deg-2"), rtol=1e-5)

    val = irf["bkg"].evaluate(energy=energy, fov_lon=offset, fov_lat="0.1 deg")
    assert_allclose(val.value, 0.00031552, rtol=1e-5)
    assert val.unit == "1 / (MeV s sr)"


@requires_data()
def test_irf_dict_from_file_duplicate_irfs(caplog, tmp_path):
    """catch the warning message about two type of IRF with the same hdu class
    encountered in the same file"""
    original_file = make_path(
        "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
    )
    dummy_file = tmp_path / "020136_duplicated_psf.fits"

    # create a dummy file with the PSF HDU repeated twice
    f = fits.open(original_file)
    f.append(f[5].copy())
    f[7].name = "PSF2"
    f.writeto(dummy_file)

    load_irf_dict_from_file(dummy_file)

    assert "more than one HDU" in caplog.text
    assert "loaded the PSF HDU in the dictionary" in caplog.text


@requires_data()
def test_irf_dict_from_file_fixed_rad_max():
    """test that for point-like IRF without RAD_MAX_2D HDU a RadMax2D with a
    single value is generated from the RAD_MAX header keyword"""
    irf = load_irf_dict_from_file(
        "$GAMMAPY_DATA/joint-crab/dl3/magic/run_05029748_DL3.fits"
    )

    assert "RAD_MAX" in irf["aeff"].meta
    assert "rad_max" in irf
    assert isinstance(irf["rad_max"], RadMax2D)

    # check that has a single-bin in energy and offset
    assert irf["rad_max"].axes["energy"].nbin == 1
    assert irf["rad_max"].axes["offset"].nbin == 1
    assert irf["rad_max"].quantity.to_value("deg") == irf["aeff"].meta["RAD_MAX"]


class TestIRFWrite:
    def setup(self):
        self.energy_lo = np.logspace(0, 1, 10)[:-1] * u.TeV
        self.energy_hi = np.logspace(0, 1, 10)[1:] * u.TeV
        self.energy_axis_true = MapAxis.from_energy_bounds(
            "1 TeV", "10 TeV", nbin=9, name="energy_true"
        )

        self.offset_lo = np.linspace(0, 1, 4)[:-1] * u.deg
        self.offset_hi = np.linspace(0, 1, 4)[1:] * u.deg

        self.offset_axis = MapAxis.from_bounds(
            0, 1, nbin=3, unit="deg", name="offset", node_type="edges"
        )
        self.migra_lo = np.linspace(0, 3, 4)[:-1]
        self.migra_hi = np.linspace(0, 3, 4)[1:]
        self.migra_axis = MapAxis.from_bounds(
            0, 3, nbin=3, name="migra", node_type="edges"
        )
        self.fov_lon_lo = np.linspace(-6, 6, 11)[:-1] * u.deg
        self.fov_lon_hi = np.linspace(-6, 6, 11)[1:] * u.deg
        self.fov_lon_axis = MapAxis.from_bounds(-6, 6, nbin=10, name="fov_lon")

        self.fov_lat_lo = np.linspace(-6, 6, 11)[:-1] * u.deg
        self.fov_lat_hi = np.linspace(-6, 6, 11)[1:] * u.deg
        self.fov_lat_axis = MapAxis.from_bounds(-6, 6, nbin=10, name="fov_lat")

        self.aeff_data = np.random.rand(9, 3) * u.cm * u.cm
        self.edisp_data = np.random.rand(9, 3, 3)
        self.bkg_data = np.random.rand(9, 10, 10) / u.MeV / u.s / u.sr

        self.aeff = EffectiveAreaTable2D(
            axes=[self.energy_axis_true, self.offset_axis],
            data=self.aeff_data.value,
            unit=self.aeff_data.unit,
        )
        self.edisp = EnergyDispersion2D(
            axes=[
                self.energy_axis_true,
                self.migra_axis,
                self.offset_axis,
            ],
            data=self.edisp_data,
        )
        axes = [
            self.energy_axis_true.copy(name="energy"),
            self.fov_lon_axis,
            self.fov_lat_axis,
        ]
        self.bkg = Background3D(
            axes=axes, data=self.bkg_data.value, unit=self.bkg_data.unit
        )

    def test_array_to_container(self):
        assert_allclose(self.aeff.quantity, self.aeff_data)
        assert_allclose(self.edisp.quantity, self.edisp_data)
        assert_allclose(self.bkg.quantity, self.bkg_data)

    def test_container_to_table(self):
        assert_allclose(self.aeff.to_table()["ENERG_LO"].quantity[0], self.energy_lo)
        assert_allclose(self.edisp.to_table()["ENERG_LO"].quantity[0], self.energy_lo)
        assert_allclose(self.bkg.to_table()["ENERG_LO"].quantity[0], self.energy_lo)

        assert_allclose(self.aeff.to_table()["EFFAREA"].quantity[0].T, self.aeff_data)
        assert_allclose(self.edisp.to_table()["MATRIX"].quantity[0].T, self.edisp_data)
        assert_allclose(self.bkg.to_table()["BKG"].quantity[0].T, self.bkg_data)

        assert self.aeff.to_table()["EFFAREA"].quantity[0].unit == self.aeff_data.unit
        assert self.bkg.to_table()["BKG"].quantity[0].unit == self.bkg_data.unit

    def test_container_to_fits(self):
        assert_allclose(self.aeff.to_table()["ENERG_LO"].quantity[0], self.energy_lo)

        assert self.aeff.to_table_hdu().header["EXTNAME"] == "EFFECTIVE AREA"
        assert self.edisp.to_table_hdu().header["EXTNAME"] == "ENERGY DISPERSION"
        assert self.bkg.to_table_hdu().header["EXTNAME"] == "BACKGROUND"

        hdu = self.aeff.to_table_hdu()
        assert_allclose(
            hdu.data[hdu.header["TTYPE1"]][0], self.aeff.axes[0].edges[:-1].value
        )
        hdu = self.aeff.to_table_hdu()
        assert_allclose(hdu.data[hdu.header["TTYPE5"]][0].T, self.aeff.data)

        hdu = self.edisp.to_table_hdu()
        assert_allclose(
            hdu.data[hdu.header["TTYPE1"]][0], self.edisp.axes[0].edges[:-1].value
        )
        hdu = self.edisp.to_table_hdu()
        assert_allclose(hdu.data[hdu.header["TTYPE7"]][0].T, self.edisp.data)

        hdu = self.bkg.to_table_hdu()
        assert_allclose(
            hdu.data[hdu.header["TTYPE1"]][0], self.bkg.axes[0].edges[:-1].value
        )
        hdu = self.bkg.to_table_hdu()
        assert_allclose(hdu.data[hdu.header["TTYPE7"]][0].T, self.bkg.data)

    def test_writeread(self, tmp_path):
        path = tmp_path / "tmp.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                self.aeff.to_table_hdu(),
                self.edisp.to_table_hdu(),
                self.bkg.to_table_hdu(),
            ]
        ).writeto(path)

        read_aeff = EffectiveAreaTable2D.read(path, hdu="EFFECTIVE AREA")
        assert_allclose(read_aeff.quantity, self.aeff_data)

        read_edisp = EnergyDispersion2D.read(path, hdu="ENERGY DISPERSION")
        assert_allclose(read_edisp.quantity, self.edisp_data)

        read_bkg = Background3D.read(path, hdu="BACKGROUND")
        assert_allclose(read_bkg.quantity, self.bkg_data)
