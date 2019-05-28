# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.io import fits
from ..effective_area import EffectiveAreaTable2D
from ..energy_dispersion import EnergyDispersion2D
from ..background import Background3D


class TestIRFWrite:
    def setup(self):
        self.energy_lo = np.logspace(0, 1, 11)[:-1] * u.TeV
        self.energy_hi = np.logspace(0, 1, 11)[1:] * u.TeV
        self.offset_lo = np.linspace(0, 1, 4)[:-1] * u.deg
        self.offset_hi = np.linspace(0, 1, 4)[1:] * u.deg
        self.migra_lo = np.linspace(0, 3, 4)[:-1]
        self.migra_hi = np.linspace(0, 3, 4)[1:]
        self.fov_lon_lo = np.linspace(-6, 6, 11)[:-1] * u.deg
        self.fov_lon_hi = np.linspace(-6, 6, 11)[1:] * u.deg
        self.fov_lat_lo = np.linspace(-6, 6, 11)[:-1] * u.deg
        self.fov_lat_hi = np.linspace(-6, 6, 11)[1:] * u.deg
        self.aeff_data = np.random.rand(10, 3) * u.cm * u.cm
        self.edisp_data = np.random.rand(10, 3, 3)
        self.bkg_data = np.random.rand(10, 10, 10) / u.MeV / u.s / u.sr
        self.aeff = EffectiveAreaTable2D(
            energy_lo=self.energy_lo,
            energy_hi=self.energy_hi,
            offset_lo=self.offset_lo,
            offset_hi=self.offset_hi,
            data=self.aeff_data,
        )
        self.edisp = EnergyDispersion2D(
            e_true_lo=self.energy_lo,
            e_true_hi=self.energy_hi,
            migra_lo=self.migra_lo,
            migra_hi=self.migra_hi,
            offset_lo=self.offset_lo,
            offset_hi=self.offset_hi,
            data=self.edisp_data,
        )
        self.bkg = Background3D(
            energy_lo=self.energy_lo,
            energy_hi=self.energy_hi,
            fov_lon_lo=self.fov_lon_lo,
            fov_lon_hi=self.fov_lon_hi,
            fov_lat_lo=self.fov_lat_lo,
            fov_lat_hi=self.fov_lat_hi,
            data=self.bkg_data,
        )

    def test_array_to_container(self):
        assert_allclose(self.aeff.data.data, self.aeff_data)
        assert_allclose(self.edisp.data.data, self.edisp_data)
        assert_allclose(self.bkg.data.data, self.bkg_data)

    def test_container_to_table(self):
        assert_allclose(self.aeff.to_table()["ENERG_LO"].quantity[0], self.energy_lo)
        assert_allclose(self.edisp.to_table()["ENERG_LO"].quantity[0], self.energy_lo)
        assert_allclose(self.bkg.to_table()["ENERG_LO"].quantity[0], self.energy_lo)

        assert_allclose(self.aeff.to_table()["EFFAREA"].quantity[0].T, self.aeff_data)
        assert_allclose(self.edisp.to_table()["MATRIX"].quantity[0].T, self.edisp_data)
        assert_allclose(self.bkg.to_table()["BKG"].quantity[0], self.bkg_data)

        assert self.aeff.to_table()["EFFAREA"].quantity[0].unit == self.aeff_data.unit
        assert self.bkg.to_table()["BKG"].quantity[0].unit == self.bkg_data.unit

    def test_container_to_fits(self):
        assert_allclose(self.aeff.to_table()["ENERG_LO"].quantity[0], self.energy_lo)

        assert self.aeff.to_fits().header["EXTNAME"] == "EFFECTIVE AREA"
        assert self.edisp.to_fits().header["EXTNAME"] == "ENERGY DISPERSION"
        assert self.bkg.to_fits().header["EXTNAME"] == "BACKGROUND"

        assert self.aeff.to_fits(name="TEST").header["EXTNAME"] == "TEST"
        assert self.edisp.to_fits(name="TEST").header["EXTNAME"] == "TEST"
        assert self.bkg.to_fits(name="TEST").header["EXTNAME"] == "TEST"

        hdu = self.aeff.to_fits()
        assert_allclose(
            hdu.data[hdu.header["TTYPE1"]][0], self.aeff.data.axes[0].edges[:-1].value
        )
        hdu = self.aeff.to_fits()
        assert_allclose(hdu.data[hdu.header["TTYPE5"]][0].T, self.aeff.data.data.value)

        hdu = self.edisp.to_fits()
        assert_allclose(
            hdu.data[hdu.header["TTYPE1"]][0], self.edisp.data.axes[0].edges[:-1].value
        )
        hdu = self.edisp.to_fits()
        assert_allclose(hdu.data[hdu.header["TTYPE7"]][0].T, self.edisp.data.data.value)

        hdu = self.bkg.to_fits()
        assert_allclose(
            hdu.data[hdu.header["TTYPE1"]][0], self.bkg.data.axes[1].edges[:-1].value
        )
        hdu = self.bkg.to_fits()
        assert_allclose(hdu.data[hdu.header["TTYPE7"]][0], self.bkg.data.data.value)

    def test_writeread(self, tmpdir):
        filename = str(tmpdir / "testirf.fits")
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                self.aeff.to_fits(),
                self.edisp.to_fits(),
                self.bkg.to_fits(),
            ]
        ).writeto(filename)

        read_aeff = EffectiveAreaTable2D.read(filename=filename, hdu="EFFECTIVE AREA")
        assert_allclose(read_aeff.data.data, self.aeff_data)

        read_edisp = EnergyDispersion2D.read(filename=filename, hdu="ENERGY DISPERSION")
        assert_allclose(read_edisp.data.data, self.edisp_data)

        read_bkg = Background3D.read(filename=filename, hdu="BACKGROUND")
        assert_allclose(read_bkg.data.data, self.bkg_data)
