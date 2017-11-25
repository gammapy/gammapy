from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
import astropy.units as u
from numpy.testing import assert_allclose, assert_equal
from astropy.tests.helper import assert_quantity_allclose
from ...utils.testing import requires_dependency, requires_data
from ...irf.effective_area import EffectiveAreaTable2D, EffectiveAreaTable
from ...utils.fits import table_to_fits_table
from ...irf import EnergyDispersion, EnergyDispersion2D
from ..background import Background3D
from ...utils.fits import table_to_fits_table
from astropy.io import fits



class TestIRFWrite:
    def setup(self):
        self.energy_lo = np.logspace(0, 1, 11)[:-1] * u.TeV
        self.energy_hi = np.logspace(0, 1, 11)[1:] * u.TeV
        self.offset_lo = np.linspace(0, 1, 4)[:-1] * u.deg
        self.offset_hi = np.linspace(0, 1, 4)[1:] * u.deg
        self.migra_lo = np.linspace(0, 3, 4)[:-1]
        self.migra_hi = np.linspace(0, 3, 4)[1:]
        self.detx_lo = np.linspace(-6, 6, 11)[:-1] * u.deg
        self.detx_hi = np.linspace(-6, 6, 11)[1:] * u.deg
        self.dety_lo = np.linspace(-6, 6, 11)[:-1] * u.deg
        self.dety_hi = np.linspace(-6, 6, 11)[1:] * u.deg
        self.aeff_data = np.random.rand(10, 3) * u.cm * u.cm
        self.edisp_data = np.random.rand(10, 3, 3)
        self.bkg_data = np.random.rand(10, 10, 10) / u.MeV / u.s / u.sr
        self.aeff = EffectiveAreaTable2D(energy_lo=self.energy_lo, energy_hi=self.energy_hi,
                                         offset_lo=self.offset_lo, offset_hi=self.offset_hi,
                                         data=self.aeff_data)
        self.edisp = EnergyDispersion2D(e_true_lo=self.energy_lo, e_true_hi=self.energy_hi,
                                        migra_lo=self.migra_lo, migra_hi=self.migra_hi,
                                        offset_lo=self.offset_lo, offset_hi=self.offset_hi,
                                        data=self.edisp_data)
        self.bkg = Background3D(energy_lo=self.energy_lo, energy_hi=self.energy_hi,
                                detx_lo=self.detx_lo, detx_hi=self.detx_hi,
                                dety_lo=self.dety_lo, dety_hi=self.dety_hi,
                                data=self.bkg_data)

    def test_array_to_container(self):
        assert_equal(self.aeff.data.data, self.aeff_data)
        assert_equal(self.edisp.data.data, self.edisp_data)
        assert_equal(self.bkg.data.data, self.bkg_data)

    def test_container_to_table(self):
        assert_equal(self.aeff.to_table()['ENERG_LO'].quantity[0], self.energy_lo)
        assert_equal(self.edisp.to_table()['ENERG_LO'].quantity[0], self.energy_lo)
        assert_equal(self.bkg.to_table()['ENERG_LO'].quantity[0], self.energy_lo)

        assert_equal(self.aeff.to_table()['EFFAREA'].quantity[0].T, self.aeff_data)
        assert_equal(self.edisp.to_table()['MATRIX'].quantity[0].T, self.edisp_data)
        assert_equal(self.bkg.to_table()['BKG'].quantity[0], self.bkg_data)

        assert self.aeff.to_table()['EFFAREA'].quantity[0].unit == self.aeff_data.unit
        assert self.bkg.to_table()['BKG'].quantity[0].unit == self.bkg_data.unit

    def test_container_to_fits(self):
        assert_equal(self.aeff.to_table()['ENERG_LO'].quantity[0], self.energy_lo)

        assert self.aeff.to_fits().header['EXTNAME'] == 'EFFECTIVE AREA'
        assert self.edisp.to_fits().header['EXTNAME'] == 'ENERGY DISPERSION'
        assert self.bkg.to_fits().header['EXTNAME'] == 'BACKGROUND'

        assert self.aeff.to_fits(name='TEST').header['EXTNAME'] == 'TEST'
        assert self.edisp.to_fits(name='TEST').header['EXTNAME'] == 'TEST'
        assert self.bkg.to_fits(name='TEST').header['EXTNAME'] == 'TEST'
        assert_equal(self.aeff.to_fits().data[self.aeff.to_fits().header['TTYPE1']][0]
                     * u.Unit(self.aeff.to_fits().header['TUNIT1']),
                     self.aeff.data.axes[0].lo)
        assert_equal(self.aeff.to_fits().data[self.aeff.to_fits().header['TTYPE5']][0].T
                     * u.Unit(self.aeff.to_fits().header['TUNIT5']),
                     self.aeff.data.data)

        assert_equal(self.edisp.to_fits().data[self.edisp.to_fits().header['TTYPE1']][0]
                     * u.Unit(self.edisp.to_fits().header['TUNIT1']),
                     self.edisp.data.axes[0].lo)
        assert_equal(self.edisp.to_fits().data[self.edisp.to_fits().header['TTYPE7']][0].T
                     * u.Unit(self.edisp.to_fits().header['TUNIT7']),
                     self.edisp.data.data)

        assert_equal(self.bkg.to_fits().data[self.bkg.to_fits().header['TTYPE1']][0]
                     * u.Unit(self.bkg.to_fits().header['TUNIT1']),
                     self.bkg.data.axes[1].lo)
        assert_equal(self.bkg.to_fits().data[self.bkg.to_fits().header['TTYPE7']][0]
                     * u.Unit(self.bkg.to_fits().header['TUNIT7']),
                     self.bkg.data.data)

    def test_writeread(self, tmpdir):
        filename = str(tmpdir / 'testirf.fits')
        prim_hdu = fits.PrimaryHDU()
        hdu_aeff = self.aeff.to_fits()
        hdu_edisp = self.edisp.to_fits()
        hdu_bkg = self.bkg.to_fits()
        fits.HDUList([prim_hdu, hdu_aeff, hdu_edisp, hdu_bkg]).writeto(filename)
        read_aeff = EffectiveAreaTable2D.read(filename=filename, hdu='EFFECTIVE AREA')
        read_edisp = EnergyDispersion2D.read(filename=filename, hdu='ENERGY DISPERSION')
        read_bkg = Background3D.read(filename=filename, hdu='BACKGROUND')
        assert_equal(read_aeff.data.data, self.aeff_data)
        assert_equal(read_edisp.data.data, self.edisp_data)
        assert_equal(read_bkg.data.data, self.bkg_data)