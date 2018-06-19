# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from ...irf import PSF3D
from ...maps import WcsNDMap, MapAxis, WcsGeom
from ...cube import PSFMap, make_psf_map


def fake_psf3d(sigma = 0.15 * u.deg):
    offsets = np.array((0.,1.))*u.deg
    energy = np.logspace(-1,1,5)*u.TeV
    energy_lo = energy[:-1]
    energy_hi = energy[1:]
    energy = np.sqrt(energy_lo*energy_hi)
    rad = np.linspace(0,1.,101)*u.deg
    rad_lo = rad[:-1]
    rad_hi = rad[1:]
    rad = 0.5*(rad_lo + rad_hi)

    O, R, E = np.meshgrid(offsets, rad, energy)

    gaus = np.exp(-0.5*R**2/sigma**2)
    psf_values = gaus/(gaus.sum(0)[0])*u.Unit('sr-1')

    return PSF3D(energy_lo, energy_hi, offsets, rad_lo, rad_hi, psf_values)



def test_make_psf_map():
    psf = fake_psf3d()

    pointing = SkyCoord(0,0,unit='deg')
    axis = MapAxis(nodes=[0.2, 0.7, 1.5, 2., 10.],unit='TeV')
    rad_axis = MapAxis(nodes=np.linspace(0.,1.,50),unit='deg')

    geom = WcsGeom.create(skydir=pointing, binsz=0.2, width=5, axes=[rad_axis,axis])

    psfmap = make_psf_map(psf, pointing, geom, 3*u.deg)

    pmap = PSFMap(psfmap)
    # need to assert something...