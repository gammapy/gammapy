# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.units import Unit
from astropy.coordinates import SkyCoord
from ...irf import EnergyDispersion2D, EffectiveAreaTable2D
from ...maps import MapAxis, WcsGeom
from ...cube import EDispMap, make_edisp_map, make_map_exposure_true_energy


def fake_aeff2d(area=1e6 * u.m ** 2):
    offsets = np.array((0.0, 1.0, 2.0, 3.0)) * u.deg
    energy = np.logspace(-1, 1, 5) * u.TeV
    energy_lo = energy[:-1]
    energy_hi = energy[1:]

    aeff_values = np.ones((4, 3)) * area

    return EffectiveAreaTable2D(
        energy_lo,
        energy_hi,
        offset_lo=offsets[:-1],
        offset_hi=offsets[1:],
        data=aeff_values,
    )


def make_edisp_map_test():
    etrue = [0.2, 0.7, 1.5, 2.0, 10.0] * u.TeV
    migra = np.linspace(0.0, 3.0, 51)
    offsets = np.array((0.0, 1.0, 2.0, 3.0)) * u.deg

    pointing = SkyCoord(0, 0, unit="deg")
    energy_axis = MapAxis(
        nodes=[0.2, 0.7, 1.5, 2.0, 10.0],
        unit="TeV",
        name="energy",
        node_type="edges",
        interp="log",
    )
    migra_axis = MapAxis(nodes=np.linspace(0.0, 3.0, 51), unit="", name="migra")

    edisp2d = EnergyDispersion2D.from_gauss(etrue, migra, 0.0, 0.2, offsets)

    geom = WcsGeom.create(
        skydir=pointing, binsz=1.0, width=5.0, axes=[migra_axis, energy_axis]
    )

    aeff2d = fake_aeff2d()
    exposure_geom = WcsGeom.create(
        skydir=pointing, binsz=1.0, width=5.0, axes=[energy_axis]
    )

    exposure_map = make_map_exposure_true_energy(pointing, "1 h", aeff2d, exposure_geom)

    return make_edisp_map(edisp2d, pointing, geom, 3 * u.deg, exposure_map)


def test_make_edisp_map():
    energy_axis = MapAxis(
        nodes=[0.2, 0.7, 1.5, 2.0, 10.0],
        unit="TeV",
        name="energy",
        node_type="edges",
        interp="log",
    )
    migra_axis = MapAxis(nodes=np.linspace(0.0, 3.0, 51), unit="", name="migra")

    edmap = make_edisp_map_test()

    assert edmap.edisp_map.geom.axes[0] == migra_axis
    assert edmap.edisp_map.geom.axes[1] == energy_axis
    assert edmap.edisp_map.unit == Unit("")
    assert edmap.edisp_map.data.shape == (4, 50, 5, 5)


def test_edisp_map_to_from_hdulist():
    edmap = make_edisp_map_test()
    hdulist = edmap.to_hdulist(edisp_hdu="EDISP", edisp_hdubands="BANDSEDISP")
    assert "EDISP" in hdulist
    assert "BANDSEDISP" in hdulist
    assert "EXPMAP" in hdulist
    assert "BANDSEXP" in hdulist

    new_edmap = EDispMap.from_hdulist(
        hdulist, edisp_hdu="EDISP", edisp_hdubands="BANDSEDISP"
    )
    assert_allclose(edmap.edisp_map.data, new_edmap.edisp_map.data)
    assert new_edmap.edisp_map.geom == edmap.edisp_map.geom
    assert new_edmap.exposure_map.geom == edmap.exposure_map.geom


def test_edisp_map_read_write(tmpdir):
    edmap = make_edisp_map_test()

    filename = str(tmpdir / "edispmap.fits")
    edmap.write(filename, overwrite=True)
    new_edmap = EDispMap.read(filename)

    assert_allclose(edmap.edisp_map.quantity, new_edmap.edisp_map.quantity)


def test_edisp_map_to_energydispersion():
    edmap = make_edisp_map_test()

    position = SkyCoord(0, 0, unit="deg")
    e_reco = np.logspace(-0.3, 0.2, 200) * u.TeV

    edisp = edmap.get_energy_dispersion(position, e_reco)
    # Note that the bias and resolution are rather poorly evaluated on an EnergyDisperion object
    assert_allclose(edisp.get_bias(e_true=1.0 * u.TeV), 0.0, atol=3e-2)
    assert_allclose(edisp.get_resolution(e_true=1.0 * u.TeV), 0.2, atol=3e-2)


def test_edisp_map_stacking():
    edmap1 = make_edisp_map_test()
    edmap2 = make_edisp_map_test()
    edmap2.exposure_map.quantity *= 2

    edmap_stack = edmap1.stack(edmap2)
    assert_allclose(edmap_stack.edisp_map.data, edmap1.edisp_map.data)
    assert_allclose(edmap_stack.exposure_map.data, edmap1.exposure_map.data * 3)
