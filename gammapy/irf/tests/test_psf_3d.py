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


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_PSF3D_read():
    filename = str(gammapy_extra.dir) + '/test_datasets/psf_table_023523.fits.gz'
    psf_table = PSF3D.read(filename)
    table = Table.read(filename, 'PSF_2D_TABLE')
    elo = Energy(table["ENERG_LO"].squeeze(), unit=table["ENERG_LO"].unit)
    ehi = Energy(table["ENERG_HI"].squeeze(), unit=table["ENERG_HI"].unit)
    offlo = Angle(table["THETA_LO"].squeeze(), unit=table["THETA_LO"].unit)
    offhi = Angle(table["THETA_HI"].squeeze(), unit=table["THETA_HI"].unit)
    radlo = Angle(table["RAD_LO"].squeeze(), unit=table["RAD_LO"].unit)
    radhi = Angle(table["RAD_HI"].squeeze(), unit=table["RAD_HI"].unit)
    rpsf = Quantity(table["RPSF"].squeeze(), unit=table["RPSF"].unit)

    assert_quantity_allclose(elo, psf_table.energy_lo)
    assert_quantity_allclose(ehi, psf_table.energy_hi)
    assert_quantity_allclose(offlo, psf_table.offset)
    assert_quantity_allclose(offhi, psf_table.offset)
    assert_quantity_allclose(radlo, psf_table.rad_lo)
    assert_quantity_allclose(radhi, psf_table.rad_hi)
    assert_quantity_allclose(rpsf, psf_table.psf_value)


@requires_data('gammapy-extra')
def test_PSF3D_write(tmpdir):
    filename = str(gammapy_extra.dir) + '/test_datasets/psf_table_023523.fits.gz'
    psf = PSF3D.read(filename)

    filename = str(tmpdir / 'psf.fits')
    psf.write(filename)
    psf2 = PSF3D.read(filename)
    assert_quantity_allclose(psf.energy_lo, psf2.energy_lo)
    assert_quantity_allclose(psf.energy_hi, psf2.energy_hi)
    assert_quantity_allclose(psf.offset, psf2.offset)
    assert_quantity_allclose(psf.rad_lo, psf2.rad_lo)
    assert_quantity_allclose(psf.rad_hi, psf2.rad_hi)
    assert_quantity_allclose(psf.psf_value, psf2.psf_value)
