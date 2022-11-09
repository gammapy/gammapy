# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from regions import CircleSkyRegion
from gammapy.maps import HpxGeom, MapAxis, MapCoord
from gammapy.maps.hpx.utils import (
    HpxToWcsMapping,
    get_pix_size_from_nside,
    get_subpixels,
    get_superpixels,
    nside_to_order,
    ravel_hpx_index,
    unravel_hpx_index,
)

pytest.importorskip("healpy")

hpx_allsky_test_geoms = [
    # 2D All-sky
    (8, False, "galactic", None, None),
    # 3D All-sky
    (8, False, "galactic", None, [MapAxis(np.logspace(0.0, 3.0, 4))]),
    # 3D All-sky w/ variable pixel size
    ([2, 4, 8], False, "galactic", None, [MapAxis(np.logspace(0.0, 3.0, 4))]),
    # 4D All-sky
    (
        8,
        False,
        "galactic",
        None,
        [
            MapAxis(np.logspace(0.0, 3.0, 3), name="axis0"),
            MapAxis(np.logspace(0.0, 2.0, 4), name="axis1"),
        ],
    ),
]

hpx_partialsky_test_geoms = [
    # 2D Partial-sky
    (8, False, "galactic", "DISK(110.,75.,10.)", None),
    # 3D Partial-sky
    (8, False, "galactic", "DISK(110.,75.,10.)", [MapAxis(np.logspace(0.0, 3.0, 4))]),
    # 3D Partial-sky w/ variable pixel size
    (
        [8, 16, 32],
        False,
        "galactic",
        "DISK(110.,75.,10.)",
        [MapAxis(np.logspace(0.0, 3.0, 4))],
    ),
    # 4D Partial-sky w/ variable pixel size
    (
        [[8, 16, 32], [8, 8, 16]],
        False,
        "galactic",
        "DISK(110.,75.,10.)",
        [
            MapAxis(np.logspace(0.0, 3.0, 3), name="axis0"),
            MapAxis(np.logspace(0.0, 2.0, 4), name="axis1"),
        ],
    ),
]

hpx_test_geoms = hpx_allsky_test_geoms + hpx_partialsky_test_geoms


def make_test_coords(geom, lon, lat):
    coords = [lon, lat] + [ax.center for ax in geom.axes]
    coords = np.meshgrid(*coords)
    coords = tuple([np.ravel(t) for t in coords])
    return MapCoord.create(coords)


def test_unravel_hpx_index():
    npix = np.array([2, 7])
    assert_allclose(
        unravel_hpx_index(np.array([0, 4]), npix), (np.array([0, 2]), np.array([0, 1]))
    )
    npix = np.array([[2, 7], [3, 1]])
    assert_allclose(
        unravel_hpx_index(np.array([0, 3, 10]), npix),
        (np.array([0, 1, 1]), np.array([0, 0, 1]), np.array([0, 1, 0])),
    )


def test_ravel_hpx_index():
    npix = np.array([2, 7])
    idx = (np.array([0, 2]), np.array([0, 1]))
    assert_allclose(ravel_hpx_index(idx, npix), np.array([0, 4]))
    npix = np.array([[2, 7], [3, 1]])
    idx = (np.array([0, 1, 1]), np.array([0, 0, 1]), np.array([0, 1, 0]))
    assert_allclose(ravel_hpx_index(idx, npix), np.array([0, 3, 10]))


def make_test_nside(nside, nside0, nside1):
    npix = 12 * nside**2
    nside_test = np.concatenate(
        (nside0 * np.ones(npix // 2, dtype=int), nside1 * np.ones(npix // 2, dtype=int))
    )
    return nside_test


@pytest.mark.parametrize(
    ("nside_subpix", "nside_superpix", "nest"),
    [
        (4, 2, True),
        (8, 2, True),
        (8, make_test_nside(8, 4, 2), True),
        (4, 2, False),
        (8, 2, False),
        (8, make_test_nside(8, 4, 2), False),
    ],
)
def test_get_superpixels(nside_subpix, nside_superpix, nest):
    import healpy as hp

    npix = 12 * nside_subpix**2
    subpix = np.arange(npix)
    ang_subpix = hp.pix2ang(nside_subpix, subpix, nest=nest)
    superpix = get_superpixels(subpix, nside_subpix, nside_superpix, nest=nest)
    pix1 = hp.ang2pix(nside_superpix, *ang_subpix, nest=nest)
    assert_allclose(superpix, pix1)

    subpix = subpix.reshape((12, -1))
    if not np.isscalar(nside_subpix):
        nside_subpix = nside_subpix.reshape((12, -1))
    if not np.isscalar(nside_superpix):
        nside_superpix = nside_superpix.reshape((12, -1))

    ang_subpix = hp.pix2ang(nside_subpix, subpix, nest=nest)
    superpix = get_superpixels(subpix, nside_subpix, nside_superpix, nest=nest)
    pix1 = hp.ang2pix(nside_superpix, *ang_subpix, nest=nest)
    assert_allclose(superpix, pix1)


@pytest.mark.parametrize(
    ("nside_superpix", "nside_subpix", "nest"),
    [(2, 4, True), (2, 8, True), (2, 4, False), (2, 8, False)],
)
def test_get_subpixels(nside_superpix, nside_subpix, nest):
    import healpy as hp

    npix = 12 * nside_superpix**2
    superpix = np.arange(npix)
    subpix = get_subpixels(superpix, nside_superpix, nside_subpix, nest=nest)
    ang1 = hp.pix2ang(nside_subpix, subpix, nest=nest)
    pix1 = hp.ang2pix(nside_superpix, *ang1, nest=nest)
    assert np.all(superpix[..., None] == pix1)

    superpix = superpix.reshape((12, -1))
    subpix = get_subpixels(superpix, nside_superpix, nside_subpix, nest=nest)
    ang1 = hp.pix2ang(nside_subpix, subpix, nest=nest)
    pix1 = hp.ang2pix(nside_superpix, *ang1, nest=nest)
    assert np.all(superpix[..., None] == pix1)

    pix1 = get_superpixels(subpix, nside_subpix, nside_superpix, nest=nest)
    assert np.all(superpix[..., None] == pix1)


def test_hpx_global_to_local():
    ax0 = np.linspace(0.0, 1.0, 3)
    ax1 = np.linspace(0.0, 1.0, 3)

    # 2D All-sky
    hpx = HpxGeom(16, False, "galactic")
    assert_allclose(hpx.global_to_local(0, ravel=True), np.array([0]))
    assert_allclose(hpx.global_to_local(633, ravel=True), np.array([633]))
    assert_allclose(hpx.global_to_local((0, 633), ravel=True), np.array([0, 633]))
    assert_allclose(
        hpx.global_to_local(np.array([0, 633]), ravel=True), np.array([0, 633])
    )

    # 3D All-sky
    hpx = HpxGeom(16, False, "galactic", axes=[ax0])
    assert_allclose(
        hpx.global_to_local((np.array([177, 177]), np.array([0, 1])), ravel=True),
        np.array([177, 177 + 3072]),
    )

    # 2D Partial-sky
    hpx = HpxGeom(64, False, "galactic", region="DISK(110.,75.,2.)")
    assert_allclose(
        hpx.global_to_local((0, 633, 706), ravel=True), np.array([-1, 0, 2])
    )

    # 3D Partial-sky
    hpx = HpxGeom(64, False, "galactic", region="DISK(110.,75.,2.)", axes=[ax0])
    assert_allclose(hpx.global_to_local(633, ravel=True), np.array([0]))
    assert_allclose(hpx.global_to_local(49859, ravel=True), np.array([19]))
    assert_allclose(
        hpx.global_to_local((0, 633, 706, 49859, 49935), ravel=True),
        np.array([-1, 0, 2, 19, 21]),
    )
    assert_allclose(
        hpx.global_to_local(np.array([0, 633, 706, 49859, 49935]), ravel=True),
        np.array([-1, 0, 2, 19, 21]),
    )
    idx_global = (np.array([0, 633, 706, 707, 783]), np.array([0, 0, 0, 1, 1]))
    assert_allclose(hpx.global_to_local(idx_global, ravel=True), [-1, 0, 2, 19, 21])

    # 3D Partial-sky w/ variable bin size
    hpx = HpxGeom([32, 64], False, "galactic", region="DISK(110.,75.,2.)", axes=[ax0])

    assert_allclose(hpx.global_to_local(191, ravel=True), [0])
    assert_allclose(hpx.global_to_local(12995, ravel=True), [6])
    assert_allclose(
        hpx.global_to_local((0, 191, 233, 12995), ravel=True), [-1, 0, 2, 6]
    )

    idx_global = (np.array([0, 191, 233, 707]), np.array([0, 0, 0, 1]))
    assert_allclose(
        hpx.global_to_local(idx_global, ravel=True),
        np.array([-1, 0, 2, 6]),
    )

    # 4D Partial-sky w/ variable bin size
    hpx = HpxGeom(
        [[16, 32], [32, 64]],
        False,
        "galactic",
        region="DISK(110.,75.,2.)",
        axes=[ax0, ax1],
    )
    assert_allclose(hpx.global_to_local(3263, ravel=True), [1])
    assert_allclose(hpx.global_to_local(28356, ravel=True), [11])

    idx_global = (np.array([46]), np.array([0]), np.array([0]))
    assert_allclose(hpx.global_to_local(idx_global, ravel=True), [0])


@pytest.mark.parametrize(
    ("nside", "nested", "frame", "region", "axes"), hpx_allsky_test_geoms
)
def test_hpxgeom_init_with_pix(nside, nested, frame, region, axes):
    geom = HpxGeom(nside, nested, frame, region=region, axes=axes)

    idx0 = geom.get_idx(flat=True)
    idx1 = tuple([t[::10] for t in idx0])
    geom = HpxGeom(nside, nested, frame, region=idx0, axes=axes)
    assert_allclose(idx0, geom.get_idx(flat=True))
    assert_allclose(len(idx0[0]), np.sum(geom.npix))
    geom = HpxGeom(nside, nested, frame, region=idx1, axes=axes)
    assert_allclose(idx1, geom.get_idx(flat=True))
    assert_allclose(len(idx1[0]), np.sum(geom.npix))


@pytest.mark.parametrize(("nside", "nested", "frame", "region", "axes"), hpx_test_geoms)
def test_hpxgeom_get_pix(nside, nested, frame, region, axes):
    geom = HpxGeom(nside, nested, frame, region=region, axes=axes)
    idx = geom.get_idx(local=False, flat=True)
    idx_local = geom.get_idx(local=True, flat=True)
    assert_allclose(idx, geom.local_to_global(idx_local))

    if axes is not None:
        idx_img = geom.get_idx(local=False, idx=tuple([1] * len(axes)), flat=True)
        idx_img_local = geom.get_idx(local=True, idx=tuple([1] * len(axes)), flat=True)
        assert_allclose(idx_img, geom.local_to_global(idx_img_local))


@pytest.mark.parametrize(("nside", "nested", "frame", "region", "axes"), hpx_test_geoms)
def test_hpxgeom_coord_to_idx(nside, nested, frame, region, axes):
    import healpy as hp

    geom = HpxGeom(nside, nested, frame, region=region, axes=axes)
    lon = np.array([112.5, 135.0, 105.0])
    lat = np.array([75.3, 75.3, 74.6])
    coords = make_test_coords(geom, lon, lat)
    zidx = tuple([ax.coord_to_idx(t) for t, ax in zip(coords[2:], geom.axes)])

    if geom.nside.size > 1:
        nside = geom.nside[zidx]
    else:
        nside = geom.nside

    phi, theta = coords.phi, coords.theta
    idx = geom.coord_to_idx(coords)
    assert_allclose(hp.ang2pix(nside, theta, phi), idx[0])
    for i, z in enumerate(zidx):
        assert_allclose(z, idx[i + 1])

    # Test w/ coords outside the geometry
    lon = np.array([0.0, 5.0, 10.0])
    lat = np.array([75.3, 75.3, 74.6])
    coords = make_test_coords(geom, lon, lat)
    zidx = [ax.coord_to_idx(t) for t, ax in zip(coords[2:], geom.axes)]

    idx = geom.coord_to_idx(coords)
    if geom.region is not None:
        assert_allclose(np.full_like(coords[0], -1, dtype=int), idx[0])

    idx = geom.coord_to_idx(coords, clip=True)
    assert np.all(np.not_equal(np.full_like(coords[0], -1, dtype=int), idx[0]))


def test_hpxgeom_coord_to_pix():
    lon = np.array([110.25, 114.0, 105.0])
    lat = np.array([75.3, 75.3, 74.6])
    z0 = np.array([0.5, 1.5, 2.5])
    z1 = np.array([3.5, 4.5, 5.5])
    ax0 = np.linspace(0.0, 3.0, 4)
    ax1 = np.linspace(3.0, 6.0, 4)

    pix64 = np.array([784, 785, 864])

    # 2D all-sky
    coords = (lon, lat)
    hpx = HpxGeom(64, False, "galactic")
    assert_allclose(hpx.coord_to_pix(coords)[0], pix64)

    # 2D partial-sky
    coords = (lon, lat)
    hpx = HpxGeom(64, False, "galactic", region="DISK(110.,75.,2.)")
    assert_allclose(hpx.coord_to_pix(coords)[0], pix64)

    # 3D partial-sky
    coords = (lon, lat, z0)
    hpx = HpxGeom(64, False, "galactic", region="DISK(110.,75.,2.)", axes=[ax0])
    assert_allclose(hpx.coord_to_pix(coords), (pix64, np.array([0, 1, 2])))

    # 3D partial-sky w/ variable bin size
    coords = (lon, lat, z0)
    nside = [16, 32, 64]
    hpx_bins = [
        HpxGeom(n, False, "galactic", region="DISK(110.,75.,2.)") for n in nside
    ]
    hpx = HpxGeom(nside, False, "galactic", region="DISK(110.,75.,2.)", axes=[ax0])
    for i, (x, y, z) in enumerate(np.vstack(coords).T):
        pix0 = hpx.coord_to_pix((np.array([x]), np.array([y]), np.array([z])))
        pix1 = hpx_bins[i].coord_to_pix((np.array([x]), np.array([y])))
        assert_allclose(pix0[0], pix1[0])

    # 4D partial-sky
    coords = (lon, lat, z0, z1)
    hpx = HpxGeom(64, False, "galactic", region="DISK(110.,75.,2.)", axes=[ax0, ax1])
    assert_allclose(
        hpx.coord_to_pix(coords), (pix64, np.array([0, 1, 2]), np.array([0, 1, 2]))
    )


def test_hpx_nside_to_order():
    assert_allclose(nside_to_order(64), np.array([6]))
    assert_allclose(
        nside_to_order(np.array([10, 32, 42, 64, 128, 256])),
        np.array([-1, 5, -1, 6, 7, 8]),
    )

    order = np.linspace(1, 10, 10).astype(int)
    nside = 2**order
    assert_allclose(nside_to_order(nside), order)
    assert_allclose(nside_to_order(nside).reshape((2, 5)), order.reshape((2, 5)))


def test_hpx_get_pix_size_from_nside():
    assert_allclose(
        get_pix_size_from_nside(np.array([1, 2, 4])), np.array([32.0, 16.0, 8.0])
    )


def test_hpx_get_hpxregion_size():
    geom = HpxGeom.create(nside=128, region="DISK(110.,75.,2.)")
    assert_allclose(geom.width, 2.0 * u.deg)


def test_hpxgeom_get_hpxregion_dir():
    geom = HpxGeom.create(nside=128, region="DISK(110.,75.,2.)", frame="galactic")
    refdir = geom.center_skydir
    assert_allclose(refdir.l.deg, 110.0)
    assert_allclose(refdir.b.deg, 75.0)

    geom = HpxGeom.create(nside=128, frame="galactic")
    refdir = geom.center_skydir
    assert_allclose(refdir.l.deg, 0.0)
    assert_allclose(refdir.b.deg, 0.0)


def test_hpxgeom_make_wcs():
    ax0 = np.linspace(0.0, 3.0, 4)

    hpx = HpxGeom(64, False, "galactic", region="DISK(110.,75.,2.)")
    wcs = hpx.to_wcs_geom()
    assert_allclose(wcs.wcs.wcs.crval, np.array([110.0, 75.0]))

    hpx = HpxGeom(64, False, "galactic", region="DISK(110.,75.,2.)", axes=[ax0])
    wcs = hpx.to_wcs_geom()
    assert_allclose(wcs.wcs.wcs.crval, np.array([110.0, 75.0]))


def test_hpxgeom_get_coord():
    ax0 = np.linspace(0.0, 3.0, 4)

    # 2D all-sky
    hpx = HpxGeom(16, False, "galactic")
    c = hpx.get_coord()
    assert_allclose(c[0][:3], np.array([45.0, 135.0, 225.0]))
    assert_allclose(c[1][:3], np.array([87.075819, 87.075819, 87.075819]))

    # 3D all-sky
    hpx = HpxGeom(16, False, "galactic", axes=[ax0])
    c = hpx.get_coord()
    assert_allclose(c[0][0, :3], np.array([45.0, 135.0, 225.0]))
    assert_allclose(c[1][0, :3], np.array([87.075819, 87.075819, 87.075819]))
    assert_allclose(c[2][0, :3], np.array([0.5, 0.5, 0.5]))

    # 2D partial-sky
    hpx = HpxGeom(64, False, "galactic", region="DISK(110.,75.,2.)")
    c = hpx.get_coord()
    assert_allclose(c[0][:3], np.array([107.5, 112.5, 106.57894737]))
    assert_allclose(c[1][:3], np.array([76.813533, 76.813533, 76.07742]))

    # 3D partial-sky
    hpx = HpxGeom(64, False, "galactic", region="DISK(110.,75.,2.)", axes=[ax0])
    c = hpx.get_coord()
    assert_allclose(c[0][0, :3], np.array([107.5, 112.5, 106.57894737]))
    assert_allclose(c[1][0, :3], np.array([76.813533, 76.813533, 76.07742]))
    assert_allclose(c[2][0, :3], np.array([0.5, 0.5, 0.5]))

    # 3D partial-sky w/ variable bin size
    hpx = HpxGeom(
        [16, 32, 64], False, "galactic", region="DISK(110.,75.,2.)", axes=[ax0]
    )
    c = hpx.get_coord(flat=True)
    assert_allclose(c[0][:3], np.array([117.0, 103.5, 112.5]))
    assert_allclose(c[1][:3], np.array([75.340734, 75.340734, 75.340734]))
    assert_allclose(c[2][:3], np.array([0.5, 1.5, 1.5]))


@pytest.mark.parametrize(("nside", "nested", "frame", "region", "axes"), hpx_test_geoms)
def test_hpxgeom_contains(nside, nested, frame, region, axes):
    geom = HpxGeom(nside, nested, frame, region=region, axes=axes)
    coords = geom.get_coord(flat=True)
    assert_allclose(geom.contains(coords), np.ones_like(coords[0], dtype=bool))

    if axes is not None:
        coords = [c[0] for c in coords[:2]] + [ax.edges[-1] + 1.0 for ax in axes]
        assert_allclose(geom.contains(coords), np.zeros((1,), dtype=bool))

    if geom.region is not None:
        coords = [0.0, 0.0] + [ax.center[0] for ax in geom.axes]
        assert_allclose(geom.contains(coords), np.zeros((1,), dtype=bool))


def test_make_hpx_to_wcs_mapping():
    ax0 = np.linspace(0.0, 1.0, 3)
    hpx = HpxGeom(16, False, "galactic", region="DISK(110.,75.,2.)")
    # FIXME construct explicit WCS projection here
    wcs = hpx.to_wcs_geom()
    hpx2wcs = HpxToWcsMapping.create(hpx, wcs)
    assert_allclose(
        hpx2wcs.ipix,
        np.array(
            [
                67,
                46,
                46,
                46,
                46,
                29,
                67,
                67,
                46,
                46,
                46,
                46,
                67,
                67,
                67,
                46,
                46,
                46,
                67,
                67,
                67,
                28,
                28,
                28,
                45,
                45,
                45,
                45,
                28,
                28,
                66,
                45,
                45,
                45,
                45,
                28,
            ]
        ),
    )
    assert_allclose(
        hpx2wcs.mult_val,
        np.array(
            [
                0.11111111,
                0.09090909,
                0.09090909,
                0.09090909,
                0.09090909,
                1.0,
                0.11111111,
                0.11111111,
                0.09090909,
                0.09090909,
                0.09090909,
                0.09090909,
                0.11111111,
                0.11111111,
                0.11111111,
                0.09090909,
                0.09090909,
                0.09090909,
                0.11111111,
                0.11111111,
                0.11111111,
                0.16666667,
                0.16666667,
                0.16666667,
                0.125,
                0.125,
                0.125,
                0.125,
                0.16666667,
                0.16666667,
                1.0,
                0.125,
                0.125,
                0.125,
                0.125,
                0.16666667,
            ]
        ),
    )

    hpx = HpxGeom([8, 16], False, "galactic", region="DISK(110.,75.,2.)", axes=[ax0])
    hpx2wcs = HpxToWcsMapping.create(hpx, wcs)
    assert_allclose(
        hpx2wcs.ipix,
        np.array(
            [
                [
                    15,
                    6,
                    6,
                    6,
                    6,
                    6,
                    15,
                    15,
                    6,
                    6,
                    6,
                    6,
                    15,
                    15,
                    15,
                    6,
                    6,
                    6,
                    15,
                    15,
                    15,
                    6,
                    6,
                    6,
                    15,
                    15,
                    15,
                    15,
                    6,
                    6,
                    15,
                    15,
                    15,
                    15,
                    15,
                    6,
                ],
                [
                    67,
                    46,
                    46,
                    46,
                    46,
                    29,
                    67,
                    67,
                    46,
                    46,
                    46,
                    46,
                    67,
                    67,
                    67,
                    46,
                    46,
                    46,
                    67,
                    67,
                    67,
                    28,
                    28,
                    28,
                    45,
                    45,
                    45,
                    45,
                    28,
                    28,
                    66,
                    45,
                    45,
                    45,
                    45,
                    28,
                ],
            ]
        ),
    )


def test_hpxgeom_from_header():
    pars = {
        "HPX_REG": "DISK(110.,75.,2.)",
        "EXTNAME": "SKYMAP",
        "NSIDE": 2**6,
        "ORDER": 6,
        "PIXTYPE": "HEALPIX",
        "ORDERING": "RING",
        "COORDSYS": "CEL",
        "TTYPE1": "PIX",
        "TFORM1": "K",
        "TTYPE2": "CHANNEL1",
        "TFORM2": "D",
        "INDXSCHM": "EXPLICIT",
    }
    header = fits.Header()
    header.update(pars)
    hpx = HpxGeom.from_header(header)

    assert hpx.frame == "icrs"
    assert not hpx.nest
    assert_allclose(hpx.nside, np.array([64]))


@pytest.mark.parametrize(("nside", "nested", "frame", "region", "axes"), hpx_test_geoms)
def test_hpxgeom_read_write(tmp_path, nside, nested, frame, region, axes):
    geom0 = HpxGeom(nside, nested, frame, region=region, axes=axes)
    hdu_bands = geom0.to_bands_hdu(hdu_bands="TEST_BANDS")
    hdu_prim = fits.PrimaryHDU()
    hdu_prim.header.update(geom0.to_header())

    hdulist = fits.HDUList([hdu_prim, hdu_bands])
    hdulist.writeto(tmp_path / "tmp.fits")

    with fits.open(tmp_path / "tmp.fits", memmap=False) as hdulist:
        geom1 = HpxGeom.from_header(hdulist[0].header, hdulist["TEST_BANDS"])

    assert_allclose(geom0.nside, geom1.nside)
    assert_allclose(geom0.npix, geom1.npix)
    assert_allclose(geom0.nest, geom1.nest)
    assert geom0.frame == geom1.frame


@pytest.mark.parametrize(("nside", "nested", "frame", "region", "axes"), hpx_test_geoms)
def test_hpxgeom_upsample(nside, nested, frame, region, axes):
    # NESTED
    geom = HpxGeom(nside, True, frame, region=region, axes=axes)
    geom_up = geom.upsample(2)
    assert_allclose(2 * geom.nside, geom_up.nside)
    assert_allclose(4 * geom.npix, geom_up.npix)
    coords = geom_up.get_coord(flat=True)
    assert np.all(geom.contains(coords))

    # RING
    geom = HpxGeom(nside, False, frame, region=region, axes=axes)
    geom_up = geom.upsample(2)
    assert_allclose(2 * geom.nside, geom_up.nside)
    assert_allclose(4 * geom.npix, geom_up.npix)
    coords = geom_up.get_coord(flat=True)
    assert np.all(geom.contains(coords))


@pytest.mark.parametrize(("nside", "nested", "frame", "region", "axes"), hpx_test_geoms)
def test_hpxgeom_downsample(nside, nested, frame, region, axes):
    # NESTED
    geom = HpxGeom(nside, True, frame, region=region, axes=axes)
    geom_down = geom.downsample(2)
    assert_allclose(geom.nside, 2 * geom_down.nside)
    coords = geom.get_coord(flat=True)
    assert np.all(geom_down.contains(coords))

    # RING
    geom = HpxGeom(nside, False, frame, region=region, axes=axes)
    geom_down = geom.downsample(2)
    assert_allclose(geom.nside, 2 * geom_down.nside)
    coords = geom.get_coord(flat=True)
    assert np.all(geom_down.contains(coords))


def test_hpxgeom_solid_angle():
    geom = HpxGeom.create(
        nside=8, frame="galactic", axes=[MapAxis.from_edges([0, 2, 3])]
    )

    solid_angle = geom.solid_angle()

    assert solid_angle.shape == (1,)
    assert_allclose(solid_angle.value, 0.016362461737446838)


def test_hpxgeom_pixel_scales():
    geom = HpxGeom.create(
        nside=8, frame="galactic", axes=[MapAxis.from_edges([0, 2, 3])]
    )

    pixel_scales = geom.pixel_scales

    assert_allclose(pixel_scales, [4] * u.deg)


def test_hpx_geom_cutout():
    geom = HpxGeom.create(
        nside=8, frame="galactic", axes=[MapAxis.from_edges([0, 2, 3])]
    )

    cutout = geom.cutout(position=SkyCoord("0d", "0d"), width=30 * u.deg)

    assert cutout.nside == 8
    assert cutout.data_shape == (2, 14)
    assert cutout.data_shape_axes == (2, 1)

    center = cutout.center_skydir.icrs
    assert_allclose(center.ra.deg, 0, atol=1e-8)
    assert_allclose(center.dec.deg, 0, atol=1e-8)


def test_hpx_geom_is_aligned():
    geom = HpxGeom.create(nside=8, frame="galactic")

    assert geom.is_aligned(geom)

    cutout = geom.cutout(position=SkyCoord("0d", "0d"), width=30 * u.deg)
    assert cutout.is_aligned(geom)

    geom_other = HpxGeom.create(nside=4, frame="galactic")
    assert not geom.is_aligned(geom_other)

    geom_other = HpxGeom.create(nside=8, frame="galactic", nest=False)
    assert not geom.is_aligned(geom_other)

    geom_other = HpxGeom.create(nside=8, frame="icrs")
    assert not geom.is_aligned(geom_other)


def test_hpx_geom_to_wcs_tiles():
    geom = HpxGeom.create(
        nside=8, frame="galactic", axes=[MapAxis.from_edges([0, 2, 3])]
    )

    tiles = geom.to_wcs_tiles(nside_tiles=2)
    assert len(tiles) == 48
    assert tiles[0].projection == "TAN"
    assert_allclose(tiles[0].width, [[43.974226], [43.974226]] * u.deg)

    tiles = geom.to_wcs_tiles(nside_tiles=4)
    assert len(tiles) == 192
    assert tiles[0].projection == "TAN"
    assert_allclose(tiles[0].width, [[21.987113], [21.987113]] * u.deg)


def test_geom_repr():
    geom = HpxGeom(nside=8)
    assert geom.__class__.__name__ in repr(geom)
    assert "nside" in repr(geom)


hpx_equality_test_geoms = [
    (16, False, "galactic", None, True),
    (16, True, "galactic", None, False),
    (8, False, "galactic", None, False),
    (16, False, "icrs", None, False),
]


@pytest.mark.parametrize(
    ("nside", "nested", "frame", "region", "result"), hpx_equality_test_geoms
)
def test_hpxgeom_equal(nside, nested, frame, region, result):
    geom0 = HpxGeom(16, False, "galactic", region=None)
    geom1 = HpxGeom(nside, nested, frame, region=region)

    assert (geom0 == geom1) is result
    assert (geom0 != geom1) is not result


def test_hpx_geom_to_binsz():
    geom = HpxGeom.create(nside=32, frame="galactic", nest=True)

    geom_new = geom.to_binsz(1 * u.deg)

    assert geom_new.nside[0] == 64
    assert geom_new.frame == "galactic"
    assert geom_new.nest

    geom = HpxGeom.create(
        nside=32, frame="galactic", nest=True, region="DISK(110.,75.,10.)"
    )

    geom_new = geom.to_binsz(1 * u.deg)
    assert geom_new.nside[0] == 64

    center = geom_new.center_skydir.galactic

    assert_allclose(center.l.deg, 110)
    assert_allclose(center.b.deg, 75)


def test_hpx_geom_region_mask():
    geom = HpxGeom.create(nside=256, region="DISK(0.,0.,5.)")

    circle = CircleSkyRegion(center=SkyCoord("0d", "0d"), radius=3 * u.deg)

    mask = geom.region_mask(circle)

    assert_allclose(mask.data.sum(), 534)
    assert mask.geom.nside == 256

    solid_angle = (mask.data * geom.solid_angle()).sum()
    assert_allclose(solid_angle, 2 * np.pi * (1 - np.cos(3 * u.deg)) * u.sr, rtol=0.01)


def test_hpx_geom_separation():
    geom = HpxGeom.create(binsz=0.1, frame="galactic", nest=True)
    position = SkyCoord(0, 0, unit="deg", frame="galactic")
    separation = geom.separation(position)
    assert separation.unit == "deg"
    assert_allclose(separation.value[0], 45.000049)

    # Make sure it also works for 2D maps as input
    separation = geom.to_image().separation(position)
    assert separation.unit == "deg"
    assert_allclose(separation.value[0], 45.000049)

    # make sure works for partial geometry
    geom = HpxGeom.create(binsz=0.1, frame="galactic", nest=True, region="DISK(0,0,10)")
    separation = geom.separation(position)
    assert separation.unit == "deg"
    assert_allclose(separation.value[0], 9.978725)


def test_check_nside():
    with pytest.raises(ValueError):
        HpxGeom.create(nside=3)
