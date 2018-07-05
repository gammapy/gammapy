# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.units import Unit
from astropy.coordinates import SkyCoord
from ...irf import PSF3D
from ...maps import WcsNDMap, MapAxis, WcsGeom
from ...cube import PSFMap, make_psf_map
from ...utils.testing import requires_dependency, requires_data


def fake_psf3d(sigma=0.15 * u.deg):
    offsets = np.array((0., 1., 2., 3.)) * u.deg
    energy = np.logspace(-1, 1, 5) * u.TeV
    energy_lo = energy[:-1]
    energy_hi = energy[1:]
    energy = np.sqrt(energy_lo * energy_hi)
    rad = np.linspace(0, 1., 101) * u.deg
    rad_lo = rad[:-1]
    rad_hi = rad[1:]
    #    rad = 0.5*(rad_lo + rad_hi)

    O, R, E = np.meshgrid(offsets, rad, energy)

    Rmid = 0.5 * (R[:-1] + R[1:])
    gaus = np.exp(-0.5 * Rmid ** 2 / sigma ** 2)
    drad = 2 * np.pi * (np.cos(R[:-1]) - np.cos(R[1:])) * u.Unit('sr')
    psf_values = gaus / ((gaus * drad).sum(0)[0])

    return PSF3D(energy_lo, energy_hi, offsets, rad_lo, rad_hi, psf_values)


def test_make_psf_map():
    psf = fake_psf3d(0.3 * u.deg)

    pointing = SkyCoord(0, 0, unit='deg')
    energy_axis = MapAxis(nodes=[0.2, 0.7, 1.5, 2., 10.], unit='TeV')
    rad_axis = MapAxis(nodes=np.linspace(0., 1., 51), unit='deg')

    geom = WcsGeom.create(skydir=pointing, binsz=0.2, width=5, axes=[rad_axis, energy_axis])

    psf_map = make_psf_map(psf, pointing, geom, 3 * u.deg)

    # check axes ordering
    assert psf_map.geom.axes[0] == rad_axis
    assert psf_map.geom.axes[1] == energy_axis

    # Check unit
    assert psf_map.unit == Unit('sr-1')

    # check size
    assert psf_map.data.shape == (4,50,25,25)

@requires_dependency('scipy')
def test_psfmap(tmpdir):
    psf = fake_psf3d(0.15 * u.deg)

    pointing = SkyCoord(0, 0, unit='deg')
    energy_axis = MapAxis(nodes=[0.2, 0.7, 1.5, 2., 10.], unit='TeV')
    rad_axis = MapAxis(nodes=np.linspace(0., 0.6, 50), unit='deg')

    geom = WcsGeom.create(skydir=pointing, binsz=0.2, width=5, axes=[rad_axis, energy_axis])

    psfmap = PSFMap(make_psf_map(psf, pointing, geom, 3 * u.deg))

    # Extract EnergyDependentTablePSF
    table_psf = psfmap.get_energy_dependent_table_psf(SkyCoord(1, 1, unit='deg'))

    # Check that containment radius is consistent between psf_table and psf3d
    assert_allclose(table_psf.containment_radius(1 * u.TeV, 0.9)[0],
                    psf.containment_radius(1 * u.TeV, 0 * u.deg, 0.9), rtol=1e-3)
    assert_allclose(table_psf.containment_radius(1 * u.TeV, 0.5)[0],
                    psf.containment_radius(1 * u.TeV, 0 * u.deg, 0.5), rtol=1e-3)

    # test read/write
    filename = str(tmpdir / "psfmap.fits")
    psfmap.write(filename, overwrite=True)
    new_psfmap = PSFMap.read(filename)

    assert_allclose(psfmap.psf_map.quantity, new_psfmap.psf_map.quantity)