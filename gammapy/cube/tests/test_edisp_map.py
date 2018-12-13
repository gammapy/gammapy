# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.units import Unit
from astropy.coordinates import SkyCoord
from ...irf import EnergyDispersion2D
from ...maps import MapAxis, WcsGeom
from ...cube import EDispMap, make_edisp_map


def test_make_edisp_map():

    etrue = [0.2, 0.7, 1.5, 2.0, 10.0]*u.TeV
    migra = np.linspace(0.0, 3.0, 51)

    edisp2d = EnergyDispersion2D.from_gauss(e_true, migra, 0.0, 0.1, offset)

    pointing = SkyCoord(0, 0, unit="deg")
    energy_axis = MapAxis(nodes=[0.2, 0.7, 1.5, 2.0, 10.0], unit="TeV", name="energy")
    migra_axis = MapAxis(nodes=np.linspace(0.0, 3.0, 51), unit="", name="migra")

    geom = WcsGeom.create(
        skydir=pointing, binsz=0.2, width=5, axes=[migra_axis, energy_axis]
    )

    edmap = make_edisp_map(edisp2d, pointing, geom, 3 * u.deg)

    assert edmap.psf_map.geom.axes[0] == migra_axis
    assert edmap.psf_map.geom.axes[1] == energy_axis
    assert edmap.psf_map.unit == Unit("sr-1")
    assert edmap.data.shape == (4, 50, 25, 25)



#def test_edisp_map(tmpdir):
#    migra = np.linspace(0.,3.0,100.)
#    edisp2d = EnergyDispersion2D.from_gauss(e_true, migra, 0.0, 0.1, offset)

