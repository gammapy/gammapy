# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import assert_quantity_allclose
from astropy.table import Table
from astropy.units import Quantity
from astropy.coordinates import Angle
from ...utils.energy import Energy
from ...utils.testing import requires_dependency, requires_data
from ...datasets import gammapy_extra
from ...irf import PSF3D


@requires_data('gammapy-extra')
@requires_dependency('scipy')
class TestPSF3D:
    def setup(self):
        filename = str(gammapy_extra.dir) + '/test_datasets/psf_table_023523.fits.gz'
        self.psf = PSF3D.read(filename)
        self.table = Table.read(filename, 'PSF_2D_TABLE')

    def test_read(self):
        table = self.table
        elo = Energy(table["ENERG_LO"].squeeze(), unit=table["ENERG_LO"].unit)
        ehi = Energy(table["ENERG_HI"].squeeze(), unit=table["ENERG_HI"].unit)
        offlo = Angle(table["THETA_LO"].squeeze(), unit=table["THETA_LO"].unit)
        offhi = Angle(table["THETA_HI"].squeeze(), unit=table["THETA_HI"].unit)
        radlo = Angle(table["RAD_LO"].squeeze(), unit=table["RAD_LO"].unit)
        radhi = Angle(table["RAD_HI"].squeeze(), unit=table["RAD_HI"].unit)
        rpsf = Quantity(table["RPSF"].squeeze(), unit=table["RPSF"].unit)

        assert_quantity_allclose(self.psf.energy_lo, elo)
        assert_quantity_allclose(self.psf.energy_hi, ehi)
        assert_quantity_allclose(self.psf.offset, offlo)
        assert_quantity_allclose(self.psf.offset, offhi)
        assert_quantity_allclose(self.psf.rad_lo, radlo)
        assert_quantity_allclose(self.psf.rad_hi, radhi)
        assert_quantity_allclose(self.psf.psf_value, rpsf)

    def test_write(self, tmpdir):
        filename = str(tmpdir / 'psf.fits')
        self.psf.write(filename)
        psf2 = PSF3D.read(filename)
        assert_quantity_allclose(self.psf.energy_lo, psf2.energy_lo)
        assert_quantity_allclose(self.psf.energy_hi, psf2.energy_hi)
        assert_quantity_allclose(self.psf.offset, psf2.offset)
        assert_quantity_allclose(self.psf.rad_lo, psf2.rad_lo)
        assert_quantity_allclose(self.psf.rad_hi, psf2.rad_hi)
        assert_quantity_allclose(self.psf.psf_value, psf2.psf_value)

    @requires_dependency('matplotlib')
    def test_peek(self):
        self.psf.peek()
