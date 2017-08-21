# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits
from ..geom import MapAxis
from ..hpx import HpxGeom, get_pixel_size_from_nside, nside_to_order, lonlat_to_colat
from ..hpx import make_hpx_to_wcs_mapping, unravel_hpx_index, ravel_hpx_index

pytest.importorskip('scipy')
pytest.importorskip('healpy')

hpx_allsky_test_geoms = [
    # 2D All-sky
    (8, False, 'GAL', None, None),
    # 3D All-sky
    (8, False, 'GAL', None, [MapAxis(np.logspace(0., 3., 4))]),
    # 3D All-sky w/ variable pixel size
    ([2, 4, 8], False, 'GAL', None, [MapAxis(np.logspace(0., 3., 4))]),
    # 4D All-sky
    (8, False, 'GAL', None, [MapAxis(np.logspace(0., 3., 3), name='axis0'),
                             MapAxis(np.logspace(0., 2., 4), name='axis1')]),
]

hpx_partialsky_test_geoms = [
    # 2D Partial-sky
    (8, False, 'GAL', 'DISK(110.,75.,10.)', None),
    # 3D Partial-sky
    (8, False, 'GAL', 'DISK(110.,75.,10.)', [MapAxis(np.logspace(0., 3., 4))]),
    # 3D Partial-sky w/ variable pixel size
    ([8, 16, 32], False, 'GAL', 'DISK(110.,75.,10.)',
     [MapAxis(np.logspace(0., 3., 4))]),
    # 4D Partial-sky w/ variable pixel size
    ([[8, 16, 32], [8, 8, 16]], False, 'GAL', 'DISK(110.,75.,10.)',
     [MapAxis(np.logspace(0., 3., 3), name='axis0'),
      MapAxis(np.logspace(0., 2., 4), name='axis1')])
]

hpx_test_geoms = hpx_allsky_test_geoms + hpx_partialsky_test_geoms


def make_test_coords(geom, lon, lat):
    coords = [lon, lat] + [ax.center for ax in geom.axes]
    coords = np.meshgrid(*coords)
    coords = tuple([np.ravel(t) for t in coords])
    return coords


def test_unravel_hpx_index():
    npix = np.array([2, 7])
    assert_allclose(unravel_hpx_index(np.array([0, 4]), npix),
                    (np.array([0, 2]), np.array([0, 1])))
    npix = np.array([[2, 7], [3, 1]])
    assert_allclose(unravel_hpx_index(np.array([0, 3, 10]), npix),
                    (np.array([0, 1, 1]), np.array([0, 0, 1]),
                     np.array([0, 1, 0])))


def test_ravel_hpx_index():
    npix = np.array([2, 7])
    idx = (np.array([0, 2]), np.array([0, 1]))
    assert_allclose(ravel_hpx_index(idx, npix), np.array([0, 4]))
    npix = np.array([[2, 7], [3, 1]])
    idx = (np.array([0, 1, 1]), np.array([0, 0, 1]), np.array([0, 1, 0]))
    assert_allclose(ravel_hpx_index(idx, npix), np.array([0, 3, 10]))


def test_hpx_global_to_local():
    ax0 = np.linspace(0., 1., 3)
    ax1 = np.linspace(0., 1., 3)

    # 2D All-sky
    hpx = HpxGeom(16, False, 'GAL')
    assert_allclose(hpx[0], np.array([0]))
    assert_allclose(hpx[633], np.array([633]))
    assert_allclose(hpx[0, 633], np.array([0, 633]))
    assert_allclose(hpx[np.array([0, 633])], np.array([0, 633]))

    # 3D All-sky
    hpx = HpxGeom(16, False, 'GAL', axes=[ax0])
    assert_allclose(hpx[(np.array([177, 177]), np.array([0, 1]))],
                    np.array([177, 177 + 3072]))

    # 2D Partial-sky
    hpx = HpxGeom(64, False, 'GAL', region='DISK(110.,75.,2.)')
    assert_allclose(hpx[0, 633, 706], np.array([-1, 0, 2]))

    # 3D Partial-sky
    hpx = HpxGeom(64, False, 'GAL', region='DISK(110.,75.,2.)', axes=[ax0])
    assert_allclose(hpx[633], np.array([0]))
    assert_allclose(hpx[49859], np.array([19]))
    assert_allclose(hpx[0, 633, 706, 49859, 49935],
                    np.array([-1, 0, 2, 19, 21]))
    assert_allclose(hpx[np.array([0, 633, 706, 49859, 49935])],
                    np.array([-1, 0, 2, 19, 21]))
    assert_allclose(hpx[(np.array([0, 633, 706, 707, 783]),
                         np.array([0, 0, 0, 1, 1]))],
                    np.array([-1, 0, 2, 19, 21]))

    # 3D Partial-sky w/ variable bin size
    hpx = HpxGeom([32, 64], False, 'GAL',
                  region='DISK(110.,75.,2.)', axes=[ax0])

    assert_allclose(hpx[191], np.array([0]))
    assert_allclose(hpx[12995], np.array([6]))
    assert_allclose(hpx[0, 191, 233, 12995], np.array([-1, 0, 2, 6]))
    assert_allclose(hpx[(np.array([0, 191, 233, 707]), np.array([0, 0, 0, 1]))],
                    np.array([-1, 0, 2, 6]))

    # 4D Partial-sky w/ variable bin size
    hpx = HpxGeom([[16, 32], [32, 64]], False, 'GAL',
                  region='DISK(110.,75.,2.)', axes=[ax0, ax1])
    assert_allclose(hpx[3263], np.array([1]))
    assert_allclose(hpx[28356], np.array([11]))
    assert_allclose(hpx[(np.array([46]), np.array([0]), np.array([0]))],
                    np.array([0]))


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_allsky_test_geoms)
def test_hpxgeom_init_with_pix(nside, nested, coordsys, region, axes):
    geom = HpxGeom(nside, nested, coordsys, region=region, axes=axes)

    pix0 = geom.get_pixels()
    pix1 = tuple([t[::10] for t in pix0])
    geom = HpxGeom(nside, nested, coordsys, region=pix0, axes=axes)
    assert_allclose(pix0, geom.get_pixels())
    assert_allclose(len(pix0[0]), np.sum(geom.npix))
    geom = HpxGeom(nside, nested, coordsys, region=pix1, axes=axes)
    assert_allclose(pix1, geom.get_pixels())
    assert_allclose(len(pix1[0]), np.sum(geom.npix))


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxgeom_to_slice(nside, nested, coordsys, region, axes):
    geom = HpxGeom(nside, nested, coordsys, region=region, axes=axes)
    slices = tuple([slice(1, 2) for i in range(2, geom.ndim)])
    geom_slice = geom.to_slice(slices)
    assert_allclose(geom_slice.ndim, 2)
    assert_allclose(geom_slice.npix, np.squeeze(geom.npix[slices]))

    pix = geom.get_pixels()
    pix_slice = geom_slice.get_pixels()
    if geom.ndim > 2:
        m = np.all([np.in1d(t, [1]) for t in pix[1:]], axis=0)
        assert_allclose(pix_slice, (pix[0][m],))
    else:
        assert_allclose(pix_slice, pix)

    # Test slicing with explicit geometry
    geom = HpxGeom(nside, nested, coordsys, region=tuple(
        [t[::10] for t in pix]), axes=axes)
    geom_slice = geom.to_slice(slices)
    assert_allclose(geom_slice.ndim, 2)
    assert_allclose(geom_slice.npix, np.squeeze(geom.npix[slices]))

    pix = geom.get_pixels()
    pix_slice = geom_slice.get_pixels()
    if geom.ndim > 2:
        m = np.all([np.in1d(t, [1]) for t in pix[1:]], axis=0)
        assert_allclose(pix_slice, (pix[0][m],))
    else:
        assert_allclose(pix_slice, pix)


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxgeom_get_pixels(nside, nested, coordsys, region, axes):

    geom = HpxGeom(nside, nested, coordsys, region=region, axes=axes)
    pix = geom.get_pixels(local=False)
    pix_local = geom.get_pixels(local=True)
    assert_allclose(pix, geom.local_to_global(pix_local))

    if axes is not None:
        pix_img = geom.get_pixels(local=False, idx=tuple([1] * len(axes)))
        pix_img_local = geom.get_pixels(local=True, idx=tuple([1] * len(axes)))
        assert_allclose(pix_img, geom.local_to_global(pix_img_local))


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxgeom_coord_to_idx(nside, nested, coordsys, region, axes):
    import healpy as hp

    geom = HpxGeom(nside, nested, coordsys, region=region, axes=axes)
    lon = np.array([112.5, 135., 105.])
    lat = np.array([75.3, 75.3, 74.6])
    coords = make_test_coords(geom, lon, lat)
    zidx = [ax.coord_to_idx(t) for t, ax in zip(coords[2:], geom.axes)]

    if geom.nside.size > 1:
        nside = geom.nside[zidx]
    else:
        nside = geom.nside

    phi, theta = lonlat_to_colat(coords[0], coords[1])
    idx = geom.coord_to_idx(coords)
    assert_allclose(hp.ang2pix(nside, theta, phi), idx[0])
    for i, z in enumerate(zidx):
        assert_allclose(z, idx[i + 1])

    lon = np.array([0.0, 5.0, 10.0])
    lat = np.array([75.3, 75.3, 74.6])
    coords = make_test_coords(geom, lon, lat)
    zidx = [ax.coord_to_idx(t) for t, ax in zip(coords[2:], geom.axes)]

    idx = geom.coord_to_idx(coords)
    if geom.region is not None:
        assert_allclose(-1 * np.ones(len(coords[0]), dtype=int),
                        idx[0])


def test_hpxgeom_coord_to_pix():
    lon = np.array([110.25, 114., 105.])
    lat = np.array([75.3, 75.3, 74.6])
    z0 = np.array([0.5, 1.5, 2.5])
    z1 = np.array([3.5, 4.5, 5.5])
    ax0 = np.linspace(0., 3., 4)
    ax1 = np.linspace(3., 6., 4)

    pix64 = np.array([784, 785, 864])

    # 2D all-sky
    coords = (lon, lat)
    hpx = HpxGeom(64, False, 'GAL')
    assert_allclose(hpx.coord_to_pix(coords)[0], pix64)

    # 2D partial-sky
    coords = (lon, lat)
    hpx = HpxGeom(64, False, 'GAL', region='DISK(110.,75.,2.)')
    assert_allclose(hpx.coord_to_pix(coords)[0], pix64)

    # 3D partial-sky
    coords = (lon, lat, z0)
    hpx = HpxGeom(64, False, 'GAL', region='DISK(110.,75.,2.)', axes=[ax0])
    assert_allclose(hpx.coord_to_pix(coords), (pix64, np.array([0, 1, 2])))

    # 3D partial-sky w/ variable bin size
    coords = (lon, lat, z0)
    nside = [16, 32, 64]
    hpx_bins = [HpxGeom(n, False, 'GAL', region='DISK(110.,75.,2.)')
                for n in nside]
    hpx = HpxGeom(nside, False, 'GAL', region='DISK(110.,75.,2.)', axes=[ax0])
    for i, (x, y, z) in enumerate(np.vstack(coords).T):
        pix0 = hpx.coord_to_pix((np.array([x]), np.array([y]), np.array([z])))
        pix1 = hpx_bins[i].coord_to_pix((np.array([x]), np.array([y])))
        assert_allclose(pix0[0], pix1[0])

    # 4D partial-sky
    coords = (lon, lat, z0, z1)
    hpx = HpxGeom(64, False, 'GAL',
                  region='DISK(110.,75.,2.)', axes=[ax0, ax1])
    assert_allclose(hpx.coord_to_pix(coords),
                    (pix64, np.array([0, 1, 2]), np.array([0, 1, 2])))


def test_hpx_nside_to_order():
    assert_allclose(nside_to_order(64), np.array([6]))
    assert_allclose(nside_to_order(np.array([10, 32, 42, 64, 128, 256])),
                    np.array([-1, 5, -1, 6, 7, 8]))

    order = np.linspace(1, 10, 10).astype(int)
    nside = 2 ** order
    assert_allclose(nside_to_order(nside), order)
    assert_allclose(nside_to_order(nside).reshape((2, 5)),
                    order.reshape((2, 5)))


def test_hpx_get_pixel_size_from_nside():
    assert_allclose(get_pixel_size_from_nside(np.array([1, 2, 4])),
                    np.array([32.0, 16.0, 8.0]))


def test_hpx_get_region_size():
    assert_allclose(HpxGeom.get_region_size('DISK(110.,75.,2.)'), 2.0)


def test_hpxgeom_get_ref_dir():
    refdir = HpxGeom.get_ref_dir('DISK(110.,75.,2.)', 'GAL')
    assert_allclose(refdir.l.deg, 110.)
    assert_allclose(refdir.b.deg, 75.)

    refdir = HpxGeom.get_ref_dir(None, 'GAL')
    assert_allclose(refdir.l.deg, 0.)
    assert_allclose(refdir.b.deg, 0.)


def test_hpxgeom_make_wcs():
    ax0 = np.linspace(0., 3., 4)

    hpx = HpxGeom(64, False, 'GAL', region='DISK(110.,75.,2.)')
    wcs = hpx.make_wcs()
    assert_allclose(wcs.wcs.wcs.crval, np.array([110., 75.]))

    hpx = HpxGeom(64, False, 'GAL', region='DISK(110.,75.,2.)', axes=[ax0])
    wcs = hpx.make_wcs()
    assert_allclose(wcs.wcs.wcs.crval, np.array([110., 75.]))


def test_hpxgeom_get_coords():
    ax0 = np.linspace(0., 3., 4)

    # 2D all-sky
    hpx = HpxGeom(16, False, 'GAL')
    c = hpx.get_coords()
    assert_allclose(c[0][:3], np.array([45., 135., 225.]))
    assert_allclose(c[1][:3], np.array([87.075819, 87.075819, 87.075819]))

    # 3D all-sky
    hpx = HpxGeom(16, False, 'GAL', axes=[ax0])
    c = hpx.get_coords()
    assert_allclose(c[0][:3], np.array([45., 135., 225.]))
    assert_allclose(c[1][:3], np.array([87.075819, 87.075819, 87.075819]))
    assert_allclose(c[2][:3], np.array([0.5, 0.5, 0.5]))

    # 2D partial-sky
    hpx = HpxGeom(64, False, 'GAL', region='DISK(110.,75.,2.)')
    c = hpx.get_coords()
    assert_allclose(c[0][:3], np.array([107.5, 112.5, 106.57894737]))
    assert_allclose(c[1][:3], np.array([76.813533, 76.813533, 76.07742]))

    # 3D partial-sky
    hpx = HpxGeom(64, False, 'GAL', region='DISK(110.,75.,2.)', axes=[ax0])
    c = hpx.get_coords()
    assert_allclose(c[0][:3], np.array([107.5, 112.5, 106.57894737]))
    assert_allclose(c[1][:3], np.array([76.813533, 76.813533, 76.07742]))
    assert_allclose(c[2][:3], np.array([0.5, 0.5, 0.5]))

    # 3D partial-sky w/ variable bin size
    hpx = HpxGeom([16, 32, 64], False, 'GAL',
                  region='DISK(110.,75.,2.)', axes=[ax0])
    c = hpx.get_coords()
    assert_allclose(c[0][:3], np.array([117., 103.5, 112.5]))
    assert_allclose(c[1][:3], np.array([75.340734, 75.340734, 75.340734]))
    assert_allclose(c[2][:3], np.array([0.5, 1.5, 1.5]))


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxgeom_contains(nside, nested, coordsys, region, axes):
    geom = HpxGeom(nside, nested, coordsys, region=region, axes=axes)
    coords = geom.get_coords()
    assert_allclose(geom.contains(coords), np.ones(
        coords[0].shape, dtype=bool))

    if axes is not None:

        coords = [c[0] for c in coords[:2]] + \
            [ax.edges[-1] + 1.0 for ax in axes]
        assert_allclose(geom.contains(coords), np.zeros((1,), dtype=bool))

    if geom.region is not None:

        coords = [0.0, 0.0] + [ax.center[0] for ax in geom.axes]
        assert_allclose(geom.contains(coords), np.zeros((1,), dtype=bool))


def test_make_hpx_to_wcs_mapping():
    ax0 = np.linspace(0., 1., 3)
    hpx = HpxGeom(16, False, 'GAL', region='DISK(110.,75.,2.)')
    # FIXME construct explicit WCS projection here
    wcs = hpx.make_wcs()
    hpx2wcs = make_hpx_to_wcs_mapping(hpx, wcs)
    assert_allclose(hpx2wcs[0],
                    np.array([67, 46, 46, 46, 46, 29, 67, 67, 46, 46,
                              46, 46, 67, 67, 67, 46, 46, 46, 67, 67,
                              67, 28, 28, 28, 45, 45, 45, 45, 28, 28,
                              66, 45, 45, 45, 45, 28]))
    assert_allclose(hpx2wcs[1],
                    np.array([0.11111111, 0.09090909, 0.09090909, 0.09090909,
                              0.09090909, 1., 0.11111111, 0.11111111, 0.09090909,
                              0.09090909, 0.09090909, 0.09090909, 0.11111111,
                              0.11111111, 0.11111111, 0.09090909, 0.09090909,
                              0.09090909, 0.11111111, 0.11111111, 0.11111111,
                              0.16666667, 0.16666667, 0.16666667, 0.125,
                              0.125, 0.125, 0.125, 0.16666667, 0.16666667,
                              1., 0.125, 0.125, 0.125, 0.125, 0.16666667]))

    hpx = HpxGeom([8, 16], False, 'GAL',
                  region='DISK(110.,75.,2.)', axes=[ax0])
    hpx2wcs = make_hpx_to_wcs_mapping(hpx, wcs)
    assert_allclose(hpx2wcs[0],
                    np.array([[15, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6,
                               15, 15, 15, 6, 6, 6, 15, 15, 15, 6, 6, 6, 15,
                               15, 15, 15, 6, 6, 15, 15, 15, 15, 15, 6],
                              [67, 46, 46, 46, 46, 29, 67, 67, 46, 46, 46, 46,
                               67, 67, 67, 46, 46, 46, 67, 67, 67, 28, 28, 28,
                               45, 45, 45, 45, 28, 28, 66, 45, 45, 45, 45, 28]]))


def test_hpxgeom_from_header():
    pars = {
        'HPX_REG': 'DISK(110.,75.,2.)',
        'EXTNAME': 'SKYMAP',
        'NSIDE': 2 ** 6,
        'ORDER': 6,
        'PIXTYPE': 'HEALPIX',
        'ORDERING': 'RING',
        'COORDSYS': 'CEL',
        'TTYPE1': 'PIX',
        'TFORM1': 'K',
        'TTYPE2': 'CHANNEL1',
        'TFORM2': 'D',
        'INDXSCHM': 'EXPLICIT',
    }
    header = fits.Header()
    header.update(pars)
    hpx = HpxGeom.from_header(header)

    assert hpx.coordsys == pars['COORDSYS']
    assert hpx.nest is False
    assert_allclose(hpx.nside, np.array([64]))


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxgeom_read_write(tmpdir, nside, nested, coordsys, region, axes):
    geom0 = HpxGeom(nside, nested, coordsys, region=region, axes=axes)
    hdu_bands = geom0.make_bands_hdu(extname='BANDS')
    hdu_prim = fits.PrimaryHDU()
    hdu_prim.header.update(geom0.make_header())

    filename = str(tmpdir / 'hpxgeom.fits')
    hdulist = fits.HDUList([hdu_prim, hdu_bands])
    hdulist.writeto(filename, overwrite=True)

    hdulist = fits.open(filename)
    geom1 = HpxGeom.from_header(hdulist[0].header, hdulist['BANDS'])

    assert_allclose(geom0.nside, geom1.nside)
    assert_allclose(geom0.npix, geom1.npix)
    assert_allclose(geom0.nest, geom1.nest)
    assert(geom0.coordsys == geom1.coordsys)
