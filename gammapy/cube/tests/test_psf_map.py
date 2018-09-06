# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.units import Unit
from astropy.coordinates import SkyCoord
from ...irf import PSF3D
from ...maps import MapAxis, WcsGeom
from ...cube import PSFMap, make_psf_map
from ...utils.testing import requires_dependency


def fake_psf3d(sigma=0.15 * u.deg):
    offsets = np.array((0., 1., 2., 3.)) * u.deg
    energy = np.logspace(-1, 1, 5) * u.TeV
    energy_lo = energy[:-1]
    energy_hi = energy[1:]
    energy = np.sqrt(energy_lo * energy_hi)
    rad = np.linspace(0, 1., 101) * u.deg
    rad_lo = rad[:-1]
    rad_hi = rad[1:]

    O, R, E = np.meshgrid(offsets, rad, energy)

    Rmid = 0.5 * (R[:-1] + R[1:])
    gaus = np.exp(-0.5 * Rmid ** 2 / sigma ** 2)
    drad = 2 * np.pi * (np.cos(R[:-1]) - np.cos(R[1:])) * u.Unit("sr")
    psf_values = gaus / ((gaus * drad).sum(0)[0])

    return PSF3D(energy_lo, energy_hi, offsets, rad_lo, rad_hi, psf_values)


@requires_dependency("scipy")
def test_make_psf_map():
    psf = fake_psf3d(0.3 * u.deg)

    pointing = SkyCoord(0, 0, unit="deg")
    energy_axis = MapAxis(nodes=[0.2, 0.7, 1.5, 2., 10.], unit="TeV", name="energy")
    rad_axis = MapAxis(nodes=np.linspace(0., 1., 51), unit="deg", name="theta")

    geom = WcsGeom.create(
        skydir=pointing, binsz=0.2, width=5, axes=[rad_axis, energy_axis]
    )

    psfmap = make_psf_map(psf, pointing, geom, 3 * u.deg)

    assert psfmap.psf_map.geom.axes[0] == rad_axis
    assert psfmap.psf_map.geom.axes[1] == energy_axis
    assert psfmap.psf_map.unit == Unit("sr-1")
    assert psfmap.data.shape == (4, 50, 25, 25)


@requires_dependency("scipy")
def test_psfmap(tmpdir):
    psf = fake_psf3d(0.15 * u.deg)

    pointing = SkyCoord(0, 0, unit="deg")
    energy_axis = MapAxis(nodes=[0.2, 0.7, 1.5, 2., 10.], unit="TeV", name="energy")
    rad_axis = MapAxis(nodes=np.linspace(0., 0.6, 50), unit="deg", name="theta")

    geom = WcsGeom.create(
        skydir=pointing, binsz=0.2, width=5, axes=[rad_axis, energy_axis]
    )

    psfmap = make_psf_map(psf, pointing, geom, 3 * u.deg)

    # Extract EnergyDependentTablePSF
    table_psf = psfmap.get_energy_dependent_table_psf(SkyCoord(1, 1, unit="deg"))

    # Check that containment radius is consistent between psf_table and psf3d
    assert_allclose(
        table_psf.containment_radius(1 * u.TeV, 0.9)[0],
        psf.containment_radius(1 * u.TeV, 0 * u.deg, 0.9),
        rtol=1e-3,
    )
    assert_allclose(
        table_psf.containment_radius(1 * u.TeV, 0.5)[0],
        psf.containment_radius(1 * u.TeV, 0 * u.deg, 0.5),
        rtol=1e-3,
    )

    # create PSFKernel
    kern_geom = WcsGeom.create(binsz=0.02, width=5., axes=[energy_axis])
    psfkernel = psfmap.get_psf_kernel(
        SkyCoord(1, 1, unit="deg"), kern_geom, max_radius=1 * u.deg
    )
    assert_allclose(psfkernel.psf_kernel_map.data.sum(axis=(1, 2)), 1.0)

    # test read/write
    filename = str(tmpdir / "psfmap.fits")
    psfmap.write(filename, overwrite=True)
    new_psfmap = PSFMap.read(filename)

    assert_allclose(psfmap.psf_map.quantity, new_psfmap.psf_map.quantity)


@requires_dependency("scipy")
def test_containment_radius_map(tmpdir):
    psf = fake_psf3d(0.15 * u.deg)
    pointing = SkyCoord(0, 0, unit="deg")
    energy_axis = MapAxis(nodes=[0.2, 1, 2], unit="TeV", name="energy")
    psf_theta_axis = MapAxis(nodes=np.linspace(0., 0.6, 30), unit="deg", name="theta")
    geom = WcsGeom.create(
        skydir=pointing, binsz=0.5, width=(4, 3), axes=[psf_theta_axis, energy_axis]
    )

    psfmap = make_psf_map(psf, pointing, geom, 3 * u.deg)
    m = psfmap.containment_radius_map(2 * u.TeV)
    coord = SkyCoord(0.3, 0, unit="deg")
    val = m.interp_by_coord(coord)

    assert_allclose(val, 0.227463, rtol=1e-3)
