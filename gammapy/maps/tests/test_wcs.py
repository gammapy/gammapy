# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from ..wcs import WcsGeom, _check_width
from ..geom import MapAxis

pytest.importorskip("scipy")

axes1 = [MapAxis(np.logspace(0., 3., 3), interp="log", name="energy")]
axes2 = [
    MapAxis(np.logspace(0., 3., 3), interp="log", name="energy"),
    MapAxis(np.logspace(1., 3., 4), interp="lin"),
]
skydir = SkyCoord(110., 75.0, unit="deg", frame="icrs")

wcs_allsky_test_geoms = [
    (None, 10.0, "GAL", "AIT", skydir, None),
    (None, 10.0, "GAL", "AIT", skydir, axes1),
    (None, [10.0, 20.0], "GAL", "AIT", skydir, axes1),
    (None, 10.0, "GAL", "AIT", skydir, axes2),
    (None, [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]], "GAL", "AIT", skydir, axes2),
]

wcs_partialsky_test_geoms = [
    (10, 0.1, "GAL", "AIT", skydir, None),
    (10, 0.1, "GAL", "AIT", skydir, axes1),
    (10, [0.1, 0.2], "GAL", "AIT", skydir, axes1),
]

wcs_test_geoms = wcs_allsky_test_geoms + wcs_partialsky_test_geoms


@pytest.mark.parametrize(
    ("npix", "binsz", "coordsys", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsgeom_init(npix, binsz, coordsys, proj, skydir, axes):
    WcsGeom.create(
        npix=npix, binsz=binsz, skydir=skydir, proj=proj, coordsys=coordsys, axes=axes
    )


@pytest.mark.parametrize(
    ("npix", "binsz", "coordsys", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsgeom_get_pix(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, skydir=skydir, proj=proj, coordsys=coordsys, axes=axes
    )
    pix = geom.get_idx()
    if axes is not None:
        idx = tuple([1] * len(axes))
        pix_img = geom.get_idx(idx=idx)
        m = np.all(np.stack([x == y for x, y in zip(idx, pix[2:])]), axis=0)
        m2 = pix_img[0] != -1
        assert_allclose(pix[0][m], np.ravel(pix_img[0][m2]))
        assert_allclose(pix[1][m], np.ravel(pix_img[1][m2]))


@pytest.mark.parametrize(
    ("npix", "binsz", "coordsys", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsgeom_test_pix_to_coord(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, skydir=skydir, proj=proj, coordsys=coordsys, axes=axes
    )
    assert_allclose(geom.get_coord()[0], geom.pix_to_coord(geom.get_idx())[0])


@pytest.mark.parametrize(
    ("npix", "binsz", "coordsys", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsgeom_test_coord_to_idx(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, proj=proj, coordsys=coordsys, axes=axes
    )
    assert_allclose(geom.get_idx()[0], geom.coord_to_idx(geom.get_coord())[0])

    if not geom.is_allsky:
        coords = geom.center_coord[:2] + tuple([ax.center[0] for ax in geom.axes])
        coords[0][...] += 2.0 * np.max(geom.width[0])
        idx = geom.coord_to_idx(coords)
        assert_allclose(np.full_like(coords[0], -1, dtype=int), idx[0])
        idx = geom.coord_to_idx(coords, clip=True)
        assert np.all(np.not_equal(np.full_like(coords[0], -1, dtype=int), idx[0]))


@pytest.mark.parametrize(
    ("npix", "binsz", "coordsys", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsgeom_read_write(tmpdir, npix, binsz, coordsys, proj, skydir, axes):
    geom0 = WcsGeom.create(
        npix=npix, binsz=binsz, proj=proj, coordsys=coordsys, axes=axes
    )

    hdu_bands = geom0.make_bands_hdu(hdu="BANDS")
    hdu_prim = fits.PrimaryHDU()
    hdu_prim.header.update(geom0.make_header())

    filename = str(tmpdir / "wcsgeom.fits")
    hdulist = fits.HDUList([hdu_prim, hdu_bands])
    hdulist.writeto(filename, overwrite=True)

    with fits.open(filename) as hdulist:
        geom1 = WcsGeom.from_header(hdulist[0].header, hdulist["BANDS"])

    assert_allclose(geom0.npix, geom1.npix)
    assert geom0.coordsys == geom1.coordsys


def test_wcsgeom_to_hdulist():
    npix, binsz, coordsys, proj, skydir, axes = wcs_test_geoms[3]
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, proj=proj, coordsys=coordsys, axes=axes
    )

    hdu = geom.make_bands_hdu(hdu="TEST")
    assert hdu.header["AXCOLS1"] == "E_MIN,E_MAX"
    assert hdu.header["AXCOLS2"] == "AXIS1_MIN,AXIS1_MAX"


@pytest.mark.parametrize(
    ("npix", "binsz", "coordsys", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsgeom_contains(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, skydir=skydir, proj=proj, coordsys=coordsys, axes=axes
    )
    coords = geom.get_coord()
    coords = [c[np.isfinite(c)] for c in coords]
    assert_allclose(geom.contains(coords), np.ones(coords[0].shape, dtype=bool))

    if axes is not None:
        coords = [c[0] for c in coords[:2]] + [ax.edges[-1] + 1.0 for ax in axes]
        assert_allclose(geom.contains(coords), np.zeros((1,), dtype=bool))

    if not geom.is_allsky:
        coords = [0.0, 0.0] + [ax.center[0] for ax in geom.axes]
        assert_allclose(geom.contains(coords), np.zeros((1,), dtype=bool))


def test_wcsgeom_solid_angle():
    # Test using a CAR projection map with an extra axis
    binsz = 1.0 * u.deg
    npix = 10
    geom = WcsGeom.create(
        skydir=(0, 0),
        npix=(npix, npix),
        binsz=binsz,
        coordsys="GAL",
        proj="CAR",
        axes=[MapAxis.from_edges([0, 2, 3])],
    )

    solid_angle = geom.solid_angle()

    # Check array size
    assert solid_angle.shape == (2, npix, npix)

    # Test at b = 0 deg
    assert solid_angle.unit == "sr"
    assert_allclose(solid_angle.value[0, 5, 5], 0.0003046, rtol=1e-3)

    # Test at b = 5 deg
    assert_allclose(solid_angle.value[0, 9, 5], 0.0003038, rtol=1e-3)


def test_wcsgeom_solid_angle_ait():
    # Pixels that don't correspond to locations on ths sky
    # should have solid angles set to NaN
    ait_geom = WcsGeom.create(
        skydir=(0, 0), npix=(10, 4), binsz=50, coordsys="GAL", proj="AIT"
    )
    solid_angle = ait_geom.solid_angle()
    assert np.isnan(solid_angle[0, 0])


def test_wcsgeom_separation():
    geom = WcsGeom.create(
        skydir=(0, 0),
        npix=10,
        binsz=0.1,
        coordsys="GAL",
        proj="CAR",
        axes=[MapAxis.from_edges([0, 2, 3])],
    )
    position = SkyCoord(1, 0, unit="deg", frame="galactic").icrs
    separation = geom.separation(position)

    assert separation.unit == "deg"
    assert separation.shape == (10, 10)
    assert_allclose(separation.value[0, 0], 0.7106291438079875)

    # Make sure it also works for 2D maps as input
    separation = geom.to_image().separation(position)
    assert separation.unit == "deg"
    assert separation.shape == (10, 10)
    assert_allclose(separation.value[0, 0], 0.7106291438079875)


def test_wcsgeom_get_coord():
    geom = WcsGeom.create(
        skydir=(0, 0), npix=(4, 3), binsz=1, coordsys="GAL", proj="CAR"
    )
    coord = geom.get_coord(mode="edges")
    assert_allclose(coord.lon[0, 0], 2)
    assert_allclose(coord.lat[0, 0], -1.5)


def test_wcsgeom_get_pix_coords():
    geom = WcsGeom.create(
        skydir=(0, 0), npix=(4, 3), binsz=1, coordsys="GAL", proj="CAR", axes=axes1
    )
    idx_center = geom.get_pix(mode="center")

    for idx in idx_center:
        assert idx.shape == (2, 3, 4)
        assert_allclose(idx[0, 0, 0], 0)

    idx_edge = geom.get_pix(mode="edges")
    for idx, desired in zip(idx_edge, [-0.5, -0.5, 0]):
        assert idx.shape == (2, 4, 5)
        assert_allclose(idx[0, 0, 0], desired)


def test_geom_repr():
    geom = WcsGeom.create(
        skydir=(0, 0), npix=(10, 4), binsz=50, coordsys="GAL", proj="AIT"
    )
    assert geom.__class__.__name__ in repr(geom)


def test_geom_refpix():
    refpix = (400, 300)
    geom = WcsGeom.create(
        skydir=(0, 0), npix=(800, 600), refpix=refpix, binsz=0.1, coordsys="GAL"
    )
    assert_allclose(geom.wcs.wcs.crpix, refpix)


def test_region_mask():
    from regions import CircleSkyRegion

    geom = WcsGeom.create(npix=(3, 3), binsz=2, proj="CAR")

    r1 = CircleSkyRegion(SkyCoord(0, 0, unit="deg"), 1 * u.deg)
    r2 = CircleSkyRegion(SkyCoord(20, 20, unit="deg"), 1 * u.deg)
    regions = [r1, r2]

    mask = geom.region_mask(regions)  # default inside=True
    assert mask.dtype == bool
    assert np.sum(mask) == 1

    mask = geom.region_mask(regions, inside=False)
    assert np.sum(mask) == 8


@pytest.mark.parametrize(
    ("width", "out"),
    [
        (10, (10, 10)),
        ((10 * u.deg).to("rad"), (10, 10)),
        ((10, 5), (10, 5)),
        (("10 deg", "5 deg"), (10, 5)),
        (Angle([10, 5], "deg"), (10, 5)),
        ((10 * u.deg, 5 * u.deg), (10, 5)),
        ((10, 5) * u.deg, (10, 5)),
        ([10, 5], (10, 5)),
        (["10 deg", "5 deg"], (10, 5)),
        (np.array([10, 5]), (10, 5)),
    ],
)
def test_check_width(width, out):
    width = _check_width(width)
    assert isinstance(width, tuple)
    assert isinstance(width[0], float)
    assert isinstance(width[1], float)
    assert width == out

    geom = WcsGeom.create(width=width, binsz=1.)
    assert tuple(geom.npix) == out


def test_check_width_bad_input():
    with pytest.raises(IndexError):
        _check_width(width=(10,))
