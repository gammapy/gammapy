# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import assert_quantity_allclose
from astropy.table import Table
from ...utils.testing import requires_dependency, requires_data, mpl_savefig_check
from ...utils.scripts import make_path
from ...irf import PSF3D


@requires_data('gammapy-extra')
@requires_dependency('scipy')
class TestPSF3D:
    def setup(self):
        filename = '$GAMMAPY_EXTRA/test_datasets/psf_table_023523.fits.gz'
        self.psf = PSF3D.read(filename)
        filename = str(make_path(filename))
        self.table = Table.read(filename, 'PSF_2D_TABLE')

    def test_read(self):
        table = self.table
        energy_lo = table["ENERG_LO"].quantity[0]
        energy_hi = table["ENERG_HI"].quantity[0]
        offset_lo = table["THETA_LO"].quantity[0]
        offset_hi = table["THETA_HI"].quantity[0]
        rad_lo = table["RAD_LO"].quantity[0]
        rad_hi = table["RAD_HI"].quantity[0]
        psf_value = table["RPSF"].quantity[0]

        assert_quantity_allclose(self.psf.energy_lo, energy_lo)
        assert_quantity_allclose(self.psf.energy_hi, energy_hi)
        assert_quantity_allclose(self.psf.offset, offset_lo)
        assert_quantity_allclose(self.psf.offset, offset_hi)
        assert_quantity_allclose(self.psf.rad_lo, rad_lo)
        assert_quantity_allclose(self.psf.rad_hi, rad_hi)
        assert_quantity_allclose(self.psf.psf_value, psf_value)

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
        mpl_savefig_check()

