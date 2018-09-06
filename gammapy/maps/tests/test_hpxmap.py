# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from ..utils import fill_poisson
from ..geom import MapAxis, coordsys_to_frame
from ..base import Map
from ..hpx import HpxGeom
from ..hpxmap import HpxMap
from ..hpxnd import HpxNDMap
from ..hpxsparse import HpxSparseMap
from ...utils.testing import mpl_plot_check, requires_dependency

pytest.importorskip("scipy")
pytest.importorskip("healpy")
pytest.importorskip("numpy", "1.12.0")

axes1 = [MapAxis(np.logspace(0., 3., 3), interp="log")]

hpx_test_allsky_geoms = [
    (8, False, "GAL", None, None),
    (8, False, "GAL", None, axes1),
    ([4, 8], False, "GAL", None, axes1),
]

hpx_test_partialsky_geoms = [
    ([4, 8], False, "GAL", "DISK(110.,75.,30.)", axes1),
    (8, False, "GAL", "DISK(110.,75.,10.)", [MapAxis(np.logspace(0., 3., 4))]),
    (
        8,
        False,
        "GAL",
        "DISK(110.,75.,10.)",
        [
            MapAxis(np.logspace(0., 3., 4), name="axis0"),
            MapAxis(np.logspace(0., 2., 3), name="axis1"),
        ],
    ),
]

hpx_test_geoms = hpx_test_allsky_geoms + hpx_test_partialsky_geoms

hpx_test_geoms_sparse = [tuple(list(t) + [True]) for t in hpx_test_geoms]
hpx_test_geoms_sparse += [tuple(list(t) + [False]) for t in hpx_test_geoms]


def create_map(nside, nested, coordsys, region, axes, sparse):
    if sparse:
        m = HpxSparseMap(
            HpxGeom(
                nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes
            )
        )
    else:
        m = HpxNDMap(
            HpxGeom(
                nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes
            )
        )

    return m


@pytest.mark.parametrize(
    ("nside", "nested", "coordsys", "region", "axes"), hpx_test_geoms
)
def test_hpxmap_init(nside, nested, coordsys, region, axes):
    geom = HpxGeom(
        nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes
    )
    shape = [int(np.max(geom.npix))]
    if axes:
        shape += [ax.nbin for ax in axes]
    shape = shape[::-1]
    data = np.random.uniform(0, 1, shape)
    m = HpxNDMap(geom)
    assert m.data.shape == data.shape
    m = HpxNDMap(geom, data)
    assert_allclose(m.data, data)


@pytest.mark.parametrize(
    ("nside", "nested", "coordsys", "region", "axes", "sparse"), hpx_test_geoms_sparse
)
def test_hpxmap_create(nside, nested, coordsys, region, axes, sparse):
    create_map(nside, nested, coordsys, region, axes, sparse)


@pytest.mark.parametrize(
    ("nside", "nested", "coordsys", "region", "axes", "sparse"), hpx_test_geoms_sparse
)
def test_hpxmap_read_write(tmpdir, nside, nested, coordsys, region, axes, sparse):
    filename = str(tmpdir / "map.fits")

    m = create_map(nside, nested, coordsys, region, axes, sparse)
    fill_poisson(m, mu=0.5, random_state=0)
    m.write(filename, sparse=sparse, overwrite=True)

    m2 = HpxNDMap.read(filename)
    m3 = HpxSparseMap.read(filename)
    m4 = Map.read(filename, map_type="hpx")
    if sparse:
        msk = np.isfinite(m2.data[...])
    else:
        msk = np.ones_like(m2.data[...], dtype=bool)

    assert_allclose(m.data[...][msk], m2.data[...][msk])
    assert_allclose(m.data[...][msk], m3.data[...][msk])
    assert_allclose(m.data[...][msk], m4.data[...][msk])

    m.write(filename, sparse=True, overwrite=True)
    m2 = HpxNDMap.read(filename)
    m3 = HpxMap.read(filename, map_type="hpx")
    m4 = Map.read(filename, map_type="hpx")
    assert_allclose(m.data[...][msk], m2.data[...][msk])
    assert_allclose(m.data[...][msk], m3.data[...][msk])
    assert_allclose(m.data[...][msk], m4.data[...][msk])

    # Specify alternate HDU name for IMAGE and BANDS table
    m.write(filename, hdu="IMAGE", hdu_bands="TEST", overwrite=True)
    m2 = HpxNDMap.read(filename)
    m3 = Map.read(filename)
    m4 = Map.read(filename, map_type="hpx")


def test_hpxmap_read_write_fgst(tmpdir):
    filename = str(tmpdir / "map.fits")

    axis = MapAxis.from_bounds(100., 1000., 4, name="energy", unit="MeV")

    # Test Counts Cube
    m = create_map(8, False, "GAL", None, [axis], False)
    m.write(filename, conv="fgst-ccube", overwrite=True)
    with fits.open(filename) as h:
        assert "SKYMAP" in h
        assert "EBOUNDS" in h
        assert h["SKYMAP"].header["HPX_CONV"] == "FGST-CCUBE"
        assert h["SKYMAP"].header["TTYPE1"] == "CHANNEL1"

    m2 = Map.read(filename)
    assert m2.geom.conv == "fgst-ccube"

    # Test Model Cube
    m.write(filename, conv="fgst-template", overwrite=True)
    with fits.open(filename) as h:
        assert "SKYMAP" in h
        assert "ENERGIES" in h
        assert h["SKYMAP"].header["HPX_CONV"] == "FGST-TEMPLATE"
        assert h["SKYMAP"].header["TTYPE1"] == "ENERGY1"

    m2 = Map.read(filename)
    assert m2.geom.conv == "fgst-template"


@pytest.mark.parametrize(
    ("nside", "nested", "coordsys", "region", "axes", "sparse"), hpx_test_geoms_sparse
)
def test_hpxmap_set_get_by_pix(nside, nested, coordsys, region, axes, sparse):
    m = create_map(nside, nested, coordsys, region, axes, sparse)
    coords = m.geom.get_coord(flat=True)
    idx = m.geom.get_idx(flat=True)
    m.set_by_pix(idx, coords[0])
    assert_allclose(coords[0], m.get_by_pix(idx))


@pytest.mark.parametrize(
    ("nside", "nested", "coordsys", "region", "axes", "sparse"), hpx_test_geoms_sparse
)
def test_hpxmap_set_get_by_coord(nside, nested, coordsys, region, axes, sparse):
    m = create_map(nside, nested, coordsys, region, axes, sparse)
    coords = m.geom.get_coord(flat=True)
    m.set_by_coord(coords, coords[0])
    assert_allclose(coords[0], m.get_by_coord(coords))

    # Test with SkyCoords
    m = create_map(nside, nested, coordsys, region, axes, sparse)
    coords = m.geom.get_coord(flat=True)
    skydir = SkyCoord(
        coords[0], coords[1], unit="deg", frame=coordsys_to_frame(m.geom.coordsys)
    )
    skydir_cel = skydir.transform_to("icrs")
    skydir_gal = skydir.transform_to("galactic")
    m.set_by_coord((skydir_gal,) + tuple(coords[2:]), coords[0])
    assert_allclose(coords[0], m.get_by_coord(coords))
    assert_allclose(
        m.get_by_coord((skydir_cel,) + tuple(coords[2:])),
        m.get_by_coord((skydir_gal,) + tuple(coords[2:])),
    )


@pytest.mark.parametrize(
    ("nside", "nested", "coordsys", "region", "axes"), hpx_test_geoms
)
def test_hpxmap_interp_by_coord(nside, nested, coordsys, region, axes):
    m = HpxNDMap(
        HpxGeom(nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes)
    )
    coords = m.geom.get_coord(flat=True)
    m.set_by_coord(coords, coords[1])
    assert_allclose(m.get_by_coord(coords), m.interp_by_coord(coords, interp="linear"))


def test_hpxmap_interp_by_coord_quantities():
    ax = MapAxis(np.logspace(0., 3., 3), interp="log", name="energy", unit="TeV")
    geom = HpxGeom(nside=1, axes=[ax])
    m = HpxNDMap(geom=geom)

    coords_dict = {"lon": 99, "lat": 42, "energy": 1000 * u.GeV}

    coords = m.geom.get_coord(flat=True)
    m.set_by_coord(coords, coords["lat"])

    coords_dict["energy"] = 1 * u.TeV
    val = m.interp_by_coord(coords_dict)
    assert_allclose(val, 42, rtol=1e-2)


@pytest.mark.parametrize(
    ("nside", "nested", "coordsys", "region", "axes", "sparse"), hpx_test_geoms_sparse
)
def test_hpxmap_fill_by_coord(nside, nested, coordsys, region, axes, sparse):
    m = create_map(nside, nested, coordsys, region, axes, sparse)
    coords = m.geom.get_coord(flat=True)
    m.fill_by_coord(coords, coords[1])
    m.fill_by_coord(coords, coords[1])
    assert_allclose(m.get_by_coord(coords), 2.0 * coords[1])


@pytest.mark.parametrize(
    ("nside", "nested", "coordsys", "region", "axes"), hpx_test_geoms
)
def test_hpxmap_iter(nside, nested, coordsys, region, axes):
    m = HpxNDMap(
        HpxGeom(nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes)
    )
    coords = m.geom.get_coord(flat=True)
    m.fill_by_coord(coords, coords[0])
    for vals, pix in m.iter_by_pix(buffersize=100):
        assert_allclose(vals, m.get_by_pix(pix))
    for vals, coords in m.iter_by_coord(buffersize=100):
        assert_allclose(vals, m.get_by_coord(coords))


@pytest.mark.parametrize(
    ("nside", "nested", "coordsys", "region", "axes"), hpx_test_geoms
)
def test_hpxmap_to_wcs(nside, nested, coordsys, region, axes):
    m = HpxNDMap(
        HpxGeom(nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes)
    )
    m.to_wcs(sum_bands=False, oversample=2, normalize=False)
    m.to_wcs(sum_bands=True, oversample=2, normalize=False)


@pytest.mark.parametrize(
    ("nside", "nested", "coordsys", "region", "axes"), hpx_test_geoms
)
def test_hpxmap_swap_scheme(nside, nested, coordsys, region, axes):
    m = HpxNDMap(
        HpxGeom(nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes)
    )
    fill_poisson(m, mu=1.0, random_state=0)
    m2 = m.to_swapped()
    coords = m.geom.get_coord(flat=True)
    assert_allclose(m.get_by_coord(coords), m2.get_by_coord(coords))


@pytest.mark.parametrize(
    ("nside", "nested", "coordsys", "region", "axes"), hpx_test_geoms
)
def test_hpxmap_ud_grade(nside, nested, coordsys, region, axes):
    m = HpxNDMap(
        HpxGeom(nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes)
    )
    m.to_ud_graded(4)


@pytest.mark.parametrize(
    ("nside", "nested", "coordsys", "region", "axes"), hpx_test_partialsky_geoms
)
def test_hpxmap_pad(nside, nested, coordsys, region, axes):
    m = HpxNDMap(
        HpxGeom(nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes)
    )
    m.set_by_pix(m.geom.get_idx(flat=True), 1.0)
    cval = 2.2
    m_pad = m.pad(1, mode="constant", cval=cval)
    coords_pad = m_pad.geom.get_coord(flat=True)
    msk = m.geom.contains(coords_pad)
    coords_out = tuple([c[~msk] for c in coords_pad])
    assert_allclose(m_pad.get_by_coord(coords_out), cval * np.ones_like(coords_out[0]))
    coords_in = tuple([c[msk] for c in coords_pad])
    assert_allclose(m_pad.get_by_coord(coords_in), np.ones_like(coords_in[0]))


@pytest.mark.parametrize(
    ("nside", "nested", "coordsys", "region", "axes"), hpx_test_partialsky_geoms
)
def test_hpxmap_crop(nside, nested, coordsys, region, axes):
    m = HpxNDMap(
        HpxGeom(nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes)
    )
    m.crop(1)


@pytest.mark.parametrize(
    ("nside", "nested", "coordsys", "region", "axes"), hpx_test_geoms
)
def test_hpxmap_upsample(nside, nested, coordsys, region, axes):
    m = HpxNDMap(
        HpxGeom(nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes),
        unit="m2",
    )
    m.set_by_pix(m.geom.get_idx(flat=True), 1.0)
    m_up = m.upsample(2, preserve_counts=True)
    assert_allclose(np.nansum(m.data), np.nansum(m_up.data))
    m_up = m.upsample(2, preserve_counts=False)
    assert_allclose(4.0 * np.nansum(m.data), np.nansum(m_up.data))
    assert m.unit == m_up.unit


@pytest.mark.parametrize(
    ("nside", "nested", "coordsys", "region", "axes"), hpx_test_geoms
)
def test_hpxmap_downsample(nside, nested, coordsys, region, axes):
    m = HpxNDMap(
        HpxGeom(nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes),
        unit="m2",
    )
    m.set_by_pix(m.geom.get_idx(flat=True), 1.0)
    m_down = m.downsample(2, preserve_counts=True)
    assert_allclose(np.nansum(m.data), np.nansum(m_down.data))
    assert m.unit == m_down.unit


@pytest.mark.parametrize(
    ("nside", "nested", "coordsys", "region", "axes"), hpx_test_geoms
)
def test_hpxmap_sum_over_axes(nside, nested, coordsys, region, axes):
    m = HpxNDMap(
        HpxGeom(nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes)
    )
    coords = m.geom.get_coord(flat=True)
    m.fill_by_coord(coords, coords[0])
    msum = m.sum_over_axes()

    if m.geom.is_regular:
        assert_allclose(np.nansum(m.data), np.nansum(msum.data))


def test_coadd_unit():
    geom = HpxGeom.create(nside=128)
    m1 = HpxNDMap(geom, unit="m2")
    m2 = HpxNDMap(geom, unit="cm2")

    idx = geom.get_idx()

    weights = u.Quantity(np.ones_like(idx[0]), unit="cm2")
    m1.fill_by_idx(idx, weights=weights)
    assert_allclose(m1.data, 0.0001)

    weights = u.Quantity(np.ones_like(idx[0]), unit="m2")
    m1.fill_by_idx(idx, weights=weights)
    m1.coadd(m2)

    assert_allclose(m1.data, 1.0001)


@requires_dependency("matplotlib")
def test_plot():
    m = HpxNDMap.create(binsz=10)
    with mpl_plot_check():
        m.plot()


@requires_dependency("matplotlib")
def test_plot_poly():
    m = HpxNDMap.create(binsz=10)
    with mpl_plot_check():
        m.plot(method="poly")
