# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.io import fits
from ..hpx import HPXGeom, get_pixel_size_from_nside, nside_to_order
from ..hpx import make_hpx_to_wcs_mapping, unravel_hpx_index, ravel_hpx_index

pytest.importorskip('healpy')


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
    hpx = HPXGeom(16, False, 'GAL')
    assert_allclose(hpx[0], np.array([0]))
    assert_allclose(hpx[633], np.array([633]))
    assert_allclose(hpx[0, 633], np.array([0, 633]))
    assert_allclose(hpx[np.array([0, 633])], np.array([0, 633]))

    # 3D All-sky
    hpx = HPXGeom(16, False, 'GAL', axes=[ax0])
    assert_allclose(hpx[(np.array([177, 177]), np.array([0, 1]))],
                    np.array([177, 177 + 3072]))

    # 2D Partial-sky
    hpx = HPXGeom(64, False, 'GAL', region='DISK(110.,75.,2.)')
    assert_allclose(hpx[0, 633, 706], np.array([-1, 0, 2]))

    # 3D Partial-sky
    hpx = HPXGeom(64, False, 'GAL', region='DISK(110.,75.,2.)', axes=[ax0])
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
    hpx = HPXGeom([32, 64], False, 'GAL',
                  region='DISK(110.,75.,2.)', axes=[ax0])

    assert_allclose(hpx[191], np.array([0]))
    assert_allclose(hpx[12995], np.array([6]))
    assert_allclose(hpx[0, 191, 233, 12995], np.array([-1, 0, 2, 6]))
    assert_allclose(hpx[(np.array([0, 191, 233, 707]), np.array([0, 0, 0, 1]))],
                    np.array([-1, 0, 2, 6]))

    # 4D Partial-sky w/ variable bin size
    hpx = HPXGeom([[16, 32], [32, 64]], False, 'GAL',
                  region='DISK(110.,75.,2.)', axes=[ax0, ax1])
    assert_allclose(hpx[3263], np.array([1]))
    assert_allclose(hpx[28356], np.array([11]))
    assert_allclose(hpx[(np.array([46]), np.array([0]), np.array([0]))],
                    np.array([0]))


def test_hpx_coord_to_pix():
    lon = np.array([110.25, 114., 105.])
    lat = np.array([75.3, 75.3, 74.6])
    z0 = np.array([0.5, 1.5, 2.5])
    z1 = np.array([3.5, 4.5, 5.5])
    ax0 = np.linspace(0., 3., 4)
    ax1 = np.linspace(3., 6., 4)

    pix64 = np.array([784, 785, 864])

    # 2D all-sky
    coords = (lon, lat)
    hpx = HPXGeom(64, False, 'GAL')
    assert_allclose(hpx.coord_to_pix(coords)[0], pix64)

    # 2D partial-sky
    coords = (lon, lat)
    hpx = HPXGeom(64, False, 'GAL', region='DISK(110.,75.,2.)')
    assert_allclose(hpx.coord_to_pix(coords)[0], pix64)

    # 3D partial-sky
    coords = (lon, lat, z0)
    hpx = HPXGeom(64, False, 'GAL', region='DISK(110.,75.,2.)', axes=[ax0])
    assert_allclose(hpx.coord_to_pix(coords), (pix64, np.array([0, 1, 2])))

    # 3D partial-sky w/ variable bin size
    coords = (lon, lat, z0)
    nside = [16, 32, 64]
    hpx_bins = [HPXGeom(n, False, 'GAL', region='DISK(110.,75.,2.)')
                for n in nside]
    hpx = HPXGeom(nside, False, 'GAL', region='DISK(110.,75.,2.)', axes=[ax0])
    for i, (x, y, z) in enumerate(np.vstack(coords).T):
        pix0 = hpx.coord_to_pix((np.array([x]), np.array([y]), np.array([z])))
        pix1 = hpx_bins[i].coord_to_pix((np.array([x]), np.array([y])))
        assert_allclose(pix0[0], pix1[0])

    # 4D partial-sky
    coords = (lon, lat, z0, z1)
    hpx = HPXGeom(64, False, 'GAL',
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
    assert_allclose(HPXGeom.get_region_size('DISK(110.,75.,2.)'), 2.0)


def test_hpx_make_wcs():
    hpx = HPXGeom(64, False, 'GAL', region='DISK(110.,75.,2.)')
    wcs = hpx.make_wcs()
    assert_allclose(wcs.wcs.wcs.crval, np.array([110., 75.]))


def test_make_hpx_to_wcs_mapping():
    ax0 = np.linspace(0., 1., 3)
    hpx = HPXGeom(16, False, 'GAL', region='DISK(110.,75.,2.)')
    # FIXME construct explicit WCS projection here
    wcs = hpx.make_wcs()
    hpx2wcs = make_hpx_to_wcs_mapping(hpx, wcs)
    assert_allclose(hpx2wcs[0],
                    np.array([67, 46, 46, 46, 67, 46, 46, 28,
                              45, 45, 28, 28, 45, 45, 45, 28]))
    assert_allclose(hpx2wcs[1],
                    np.array([0.5, 0.2, 0.2, 0.2, 0.5, 0.2, 0.2, 0.25,
                              0.2, 0.2, 0.25, 0.25, 0.2, 0.2, 0.2, 0.25]))

    hpx = HPXGeom([8, 16], False, 'GAL', region='DISK(110.,75.,2.)', axes=[ax0])
    hpx2wcs = make_hpx_to_wcs_mapping(hpx, wcs)
    assert_allclose(hpx2wcs[0],
                    np.array([[15, 6, 6, 6, 15, 6, 6, 6,
                               15, 15, 6, 6, 15, 15, 15, 6],
                              [67, 46, 46, 46, 67, 46, 46, 28,
                               45, 45, 28, 28, 45, 45, 45, 28]]))


def test_hpx_from_header():
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
    hpx = HPXGeom.from_header(header)

    assert hpx.coordsys == pars['COORDSYS']
    assert hpx.nest is False
    assert_allclose(hpx.nside, np.array([64]))


def test_hpx_make_header():
    hpx = HPXGeom(16, False, 'GAL')
    header = hpx.make_header()
    # TODO: assert on something
