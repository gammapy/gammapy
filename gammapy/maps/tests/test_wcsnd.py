# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel
import astropy.units as u
from ...utils.testing import requires_dependency, requires_data, mpl_plot_check
from ...cube import PSFKernel
from ...irf import EnergyDependentMultiGaussPSF
from ..utils import fill_poisson
from ..geom import MapAxis, MapCoord, coordsys_to_frame
from ..base import Map
from ..wcs import WcsGeom
from ..hpx import HpxGeom
from ..wcsnd import WcsNDMap

pytest.importorskip("scipy")
pytest.importorskip("reproject")

axes1 = [MapAxis(np.logspace(0., 3., 3), interp="log", name="spam")]
axes2 = [
    MapAxis(np.logspace(0., 3., 3), interp="log"),
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
    (10, 1.0, "GAL", "AIT", skydir, None),
    (10, 1.0, "GAL", "AIT", skydir, axes1),
    (10, [1.0, 2.0], "GAL", "AIT", skydir, axes1),
    (10, 1.0, "GAL", "AIT", skydir, axes2),
    (10, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], "GAL", "AIT", skydir, axes2),
]

wcs_test_geoms = wcs_allsky_test_geoms + wcs_partialsky_test_geoms


@pytest.mark.parametrize(
    ("npix", "binsz", "coordsys", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_init(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, proj=proj, coordsys=coordsys, axes=axes
    )
    m0 = WcsNDMap(geom)
    coords = m0.geom.get_coord()
    m0.set_by_coord(coords, coords[1])
    m1 = WcsNDMap(geom, m0.data)
    assert_allclose(m0.data, m1.data)


@pytest.mark.parametrize(
    ("npix", "binsz", "coordsys", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_read_write(tmpdir, npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, proj=proj, coordsys=coordsys, axes=axes
    )
    filename = str(tmpdir / "map.fits")

    m0 = WcsNDMap(geom)
    fill_poisson(m0, mu=0.5)
    m0.write(filename, overwrite=True)
    m1 = WcsNDMap.read(filename)
    m2 = Map.read(filename)
    m3 = Map.read(filename, map_type="wcs")
    assert_allclose(m0.data, m1.data)
    assert_allclose(m0.data, m2.data)
    assert_allclose(m0.data, m3.data)

    m0.write(filename, sparse=True, overwrite=True)
    m1 = WcsNDMap.read(filename)
    m2 = Map.read(filename)
    m3 = Map.read(filename, map_type="wcs")
    assert_allclose(m0.data, m1.data)
    assert_allclose(m0.data, m2.data)
    assert_allclose(m0.data, m3.data)

    # Specify alternate HDU name for IMAGE and BANDS table
    m0.write(filename, hdu="IMAGE", hdu_bands="TEST", overwrite=True)
    m1 = WcsNDMap.read(filename)
    m2 = Map.read(filename)
    m3 = Map.read(filename, map_type="wcs")


def test_wcsndmap_read_write_fgst(tmpdir):
    filename = str(tmpdir / "map.fits")

    axis = MapAxis.from_bounds(100., 1000., 4, name="energy", unit="MeV")
    geom = WcsGeom.create(npix=10, binsz=1.0, proj="AIT", coordsys="GAL", axes=[axis])

    # Test Counts Cube
    m = WcsNDMap(geom)
    m.write(filename, conv="fgst-ccube", overwrite=True)
    with fits.open(filename) as h:
        assert "EBOUNDS" in h

    m2 = Map.read(filename)
    assert m2.geom.conv == "fgst-ccube"

    # Test Model Cube
    m.write(filename, conv="fgst-template", overwrite=True)
    with fits.open(filename) as h:
        assert "ENERGIES" in h

    m2 = Map.read(filename)
    assert m2.geom.conv == "fgst-template"


def test_wcs_nd_map_data_transpose_issue(tmpdir):
    # Regression test for https://github.com/gammapy/gammapy/issues/1346

    # Our test case: a little map with WCS shape (3, 2), i.e. numpy array shape (2, 3)
    data = np.array([[0, 1, 2], [np.nan, np.inf, -np.inf]])
    geom = WcsGeom.create(npix=(3, 2))

    # Data should be unmodified after init
    m = WcsNDMap(data=data, geom=geom)
    assert_equal(m.data, data)

    # Data should be unmodified if initialised like this
    m = WcsNDMap(geom=geom)
    # and then filled via an in-place Numpy array operation
    m.data += data
    assert_equal(m.data, data)

    # Data should be unmodified after write / read to normal image format
    filename = str(tmpdir / "normal.fits.gz")
    m.write(filename)
    assert_equal(Map.read(filename).data, data)

    # Data should be unmodified after write / read to sparse image format
    filename = str(tmpdir / "sparse.fits.gz")
    m.write(filename)
    assert_equal(Map.read(filename).data, data)


@pytest.mark.parametrize(
    ("npix", "binsz", "coordsys", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_set_get_by_pix(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, skydir=skydir, proj=proj, coordsys=coordsys, axes=axes
    )
    m = WcsNDMap(geom)
    coords = m.geom.get_coord()
    pix = m.geom.get_idx()
    m.set_by_pix(pix, coords[0])
    assert_allclose(coords[0], m.get_by_pix(pix))


@pytest.mark.parametrize(
    ("npix", "binsz", "coordsys", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_set_get_by_coord(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, skydir=skydir, proj=proj, coordsys=coordsys, axes=axes
    )
    m = WcsNDMap(geom)
    coords = m.geom.get_coord()
    m.set_by_coord(coords, coords[0])
    assert_allclose(coords[0], m.get_by_coord(coords))

    if not geom.is_allsky:
        coords[1][...] = 0.0
        assert_allclose(np.nan * np.ones(coords[0].shape), m.get_by_coord(coords))

    # Test with SkyCoords
    m = WcsNDMap(geom)
    coords = m.geom.get_coord()
    skydir = SkyCoord(
        coords[0], coords[1], unit="deg", frame=coordsys_to_frame(geom.coordsys)
    )
    skydir_cel = skydir.transform_to("icrs")
    skydir_gal = skydir.transform_to("galactic")

    m.set_by_coord((skydir_gal,) + tuple(coords[2:]), coords[0])
    assert_allclose(coords[0], m.get_by_coord(coords))
    assert_allclose(
        m.get_by_coord((skydir_cel,) + tuple(coords[2:])),
        m.get_by_coord((skydir_gal,) + tuple(coords[2:])),
    )

    # Test with MapCoord
    m = WcsNDMap(geom)
    coords = m.geom.get_coord()
    coords_dict = dict(lon=coords[0], lat=coords[1])
    if axes:
        for i, ax in enumerate(axes):
            coords_dict[ax.name] = coords[i + 2]
    map_coords = MapCoord.create(coords_dict, coordsys=coordsys)
    m.set_by_coord(map_coords, coords[0])
    assert_allclose(coords[0], m.get_by_coord(map_coords))


def test_set_get_by_coord_quantities():
    ax = MapAxis(np.logspace(0., 3., 3), interp="log", name="energy", unit="TeV")
    geom = WcsGeom.create(binsz=0.1, npix=(3, 4), axes=[ax])
    m = WcsNDMap(geom)
    coords_dict = {"lon": 0, "lat": 0, "energy": 1000 * u.GeV}

    m.set_by_coord(coords_dict, 42)

    coords_dict["energy"] = 1 * u.TeV
    assert_allclose(42, m.get_by_coord(coords_dict))


@pytest.mark.parametrize(
    ("npix", "binsz", "coordsys", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_fill_by_coord(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, skydir=skydir, proj=proj, coordsys=coordsys, axes=axes
    )
    m = WcsNDMap(geom)
    coords = m.geom.get_coord()
    fill_coords = tuple([np.concatenate((t, t)) for t in coords])
    fill_vals = fill_coords[1]
    m.fill_by_coord(fill_coords, fill_vals)
    assert_allclose(m.get_by_coord(coords), 2.0 * coords[1])

    # Test with SkyCoords
    m = WcsNDMap(geom)
    coords = m.geom.get_coord()
    skydir = SkyCoord(
        coords[0], coords[1], unit="deg", frame=coordsys_to_frame(geom.coordsys)
    )
    skydir_cel = skydir.transform_to("icrs")
    skydir_gal = skydir.transform_to("galactic")
    fill_coords_cel = (skydir_cel,) + tuple(coords[2:])
    fill_coords_gal = (skydir_gal,) + tuple(coords[2:])
    m.fill_by_coord(fill_coords_cel, coords[1])
    m.fill_by_coord(fill_coords_gal, coords[1])
    assert_allclose(m.get_by_coord(coords), 2.0 * coords[1])


@pytest.mark.parametrize(
    ("npix", "binsz", "coordsys", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_coadd(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, skydir=skydir, proj=proj, coordsys=coordsys, axes=axes
    )
    m0 = WcsNDMap(geom)
    m1 = WcsNDMap(geom.upsample(2))
    coords = m0.geom.get_coord()
    m1.fill_by_coord(
        tuple([np.concatenate((t, t)) for t in coords]),
        np.concatenate((coords[1], coords[1])),
    )
    m0.coadd(m1)
    assert_allclose(np.nansum(m0.data), np.nansum(m1.data), rtol=1E-4)


@pytest.mark.parametrize(
    ("npix", "binsz", "coordsys", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_interp_by_coord(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, skydir=skydir, proj=proj, coordsys=coordsys, axes=axes
    )
    m = WcsNDMap(geom)
    coords = m.geom.get_coord(flat=True)
    m.set_by_coord(coords, coords[1])
    assert_allclose(coords[1], m.interp_by_coord(coords, interp="nearest"))
    assert_allclose(coords[1], m.interp_by_coord(coords, interp="linear"))
    assert_allclose(coords[1], m.interp_by_coord(coords, interp=1))
    if geom.is_regular and not geom.is_allsky:
        assert_allclose(coords[1], m.interp_by_coord(coords, interp="cubic"))


def test_interp_by_coord_quantities():
    ax = MapAxis(np.logspace(0., 3., 3), interp="log", name="energy", unit="TeV")
    geom = WcsGeom.create(binsz=0.1, npix=(3, 3), axes=[ax])
    m = WcsNDMap(geom)
    coords_dict = {"lon": 0, "lat": 0, "energy": 1000 * u.GeV}

    m.set_by_coord(coords_dict, 42)

    coords_dict["energy"] = 1 * u.TeV
    assert_allclose(42, m.interp_by_coord(coords_dict, interp="nearest"))


def test_wcsndmap_interp_by_coord_fill_value():
    # Introduced in https://github.com/gammapy/gammapy/pull/1559/files
    m = Map.create(npix=(20, 10))
    m.data += 42
    # With `fill_value` one should be able to control what gets filled
    assert_allclose(m.interp_by_coord((99, 0), fill_value=99), 99)
    # Default is to extrapolate
    assert_allclose(m.interp_by_coord((99, 0)), 42)


@pytest.mark.parametrize(
    ("npix", "binsz", "coordsys", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_iter(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, proj=proj, coordsys=coordsys, axes=axes
    )
    m = WcsNDMap(geom)
    coords = m.geom.get_coord()
    m.fill_by_coord(coords, coords[0])
    for vals, pix in m.iter_by_pix(buffersize=100):
        assert_allclose(vals, m.get_by_pix(pix))
    for vals, coords in m.iter_by_coord(buffersize=100):
        assert_allclose(vals, m.get_by_coord(coords))


@pytest.mark.parametrize(
    ("npix", "binsz", "coordsys", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_sum_over_axes(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, proj=proj, coordsys=coordsys, axes=axes
    )
    m = WcsNDMap(geom)
    coords = m.geom.get_coord()
    m.fill_by_coord(coords, coords[0])
    msum = m.sum_over_axes()

    if m.geom.is_regular:
        assert_allclose(np.nansum(m.data), np.nansum(msum.data))


@pytest.mark.parametrize(
    ("npix", "binsz", "coordsys", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_reproject(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, proj=proj, skydir=skydir, coordsys=coordsys, axes=axes
    )
    m = WcsNDMap(geom, unit="m2")

    if geom.projection == "AIT" and geom.is_allsky:
        pytest.xfail("Bug in reproject version <= 0.3.1")

    if geom.ndim > 3 or geom.npix[0].size > 1:
        pytest.xfail("> 3 dimensions or multi-resolution geometries not supported")

    geom0 = WcsGeom.create(
        npix=npix, binsz=binsz, proj=proj, skydir=skydir, coordsys=coordsys, axes=axes
    )
    m0 = m.reproject(geom0, order=1)

    assert_allclose(m.data, m0.data)
    assert m0.unit == m.unit

    # TODO : Reproject to a different spatial geometry


def test_wcsndmap_reproject_allsky_car():
    geom = WcsGeom.create(binsz=10.0, proj="CAR", coordsys="CEL")
    m = WcsNDMap(geom)
    coords = m.geom.get_coord()
    m.set_by_coord(coords, coords[0])

    geom0 = WcsGeom.create(
        binsz=1.0, proj="CAR", coordsys="CEL", skydir=(180.0, 0.0), width=30.0
    )
    m0 = m.reproject(geom0, order=1)
    coords0 = m0.geom.get_coord()
    assert_allclose(m0.get_by_coord(coords0), coords0[0])

    geom1 = HpxGeom.create(binsz=5.0, coordsys="CEL")
    m1 = m.reproject(geom1, order=1)
    coords1 = m1.geom.get_coord()

    m = (coords1[0] > 10.0) & (coords1[0] < 350.0)
    assert_allclose(m1.get_by_coord((coords1[0][m], coords1[1][m])), coords1[0][m])


@pytest.mark.parametrize(
    ("npix", "binsz", "coordsys", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_pad(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, proj=proj, coordsys=coordsys, axes=axes
    )
    m = WcsNDMap(geom)
    m2 = m.pad(1, mode="constant", cval=2.2)
    if not geom.is_allsky:
        coords = m2.geom.get_coord()
        msk = m2.geom.contains(coords)
        coords = tuple([c[~msk] for c in coords])
        assert_allclose(m2.get_by_coord(coords), 2.2)
    m.pad(1, mode="interp", order=0)
    m.pad(1, mode="interp")


@pytest.mark.parametrize(
    ("npix", "binsz", "coordsys", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_crop(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, proj=proj, coordsys=coordsys, axes=axes
    )
    m = WcsNDMap(geom)
    m.crop(1)


@pytest.mark.parametrize(
    ("npix", "binsz", "coordsys", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_downsample(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, proj=proj, coordsys=coordsys, axes=axes
    )
    m = WcsNDMap(geom, unit="m2")
    # Check whether we can downsample
    if np.all(np.mod(geom.npix[0], 2) == 0) and np.all(np.mod(geom.npix[1], 2) == 0):
        m2 = m.downsample(2, preserve_counts=True)
        assert_allclose(np.nansum(m.data), np.nansum(m2.data))
        assert m.unit == m2.unit


@pytest.mark.parametrize(
    ("npix", "binsz", "coordsys", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_upsample(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, proj=proj, coordsys=coordsys, axes=axes
    )
    m = WcsNDMap(geom, unit="m2")
    m2 = m.upsample(2, order=0, preserve_counts=True)
    assert_allclose(np.nansum(m.data), np.nansum(m2.data))
    assert m.unit == m2.unit


def test_coadd_unit():
    geom = WcsGeom.create(npix=(10, 10), binsz=1, proj="CAR", coordsys="GAL")
    m1 = WcsNDMap(geom, data=np.ones((10, 10)), unit="m2")
    m2 = WcsNDMap(geom, data=np.ones((10, 10)), unit="cm2")

    m1.coadd(m2)

    assert_allclose(m1.data, 1.0001)


@requires_dependency("scipy")
@pytest.mark.parametrize("kernel", ["gauss", "box", "disk"])
def test_smooth(kernel):
    axes = [
        MapAxis(np.logspace(0., 3., 3), interp="log"),
        MapAxis(np.logspace(1., 3., 4), interp="lin"),
    ]
    geom = WcsGeom.create(npix=(10, 10), binsz=1, proj="CAR", coordsys="GAL", axes=axes)
    m = WcsNDMap(geom, data=np.ones(geom.data_shape), unit="m2")

    desired = m.data.sum()
    smoothed = m.smooth(0.2 * u.deg, kernel)
    actual = smoothed.data.sum()
    assert_allclose(actual, desired)


@pytest.mark.parametrize("mode", ["partial", "strict", "trim"])
def test_make_cutout(mode):
    pos = SkyCoord(0, 0, unit="deg", frame="galactic")
    geom = WcsGeom.create(
        npix=(10, 10), binsz=1, skydir=pos, proj="CAR", coordsys="GAL", axes=axes2
    )
    m = WcsNDMap(geom, data=np.ones((3, 2, 10, 10)), unit="m2")
    cutout = m.cutout(position=pos, width=(2.0, 3.0) * u.deg, mode=mode)
    actual = cutout.data.sum()
    assert_allclose(actual, 36.0)
    assert_allclose(cutout.geom.shape, m.geom.shape)
    assert_allclose(cutout.geom.width, [[2.0], [3.0]])


@requires_dependency("scipy")
def test_convolve_vs_smooth():
    axes = [
        MapAxis(np.logspace(0., 3., 3), interp="log"),
        MapAxis(np.logspace(1., 3., 4), interp="lin"),
    ]

    binsz = 0.05 * u.deg
    m = WcsNDMap.create(binsz=binsz, width=1.05 * u.deg, axes=axes)
    m.data[:, :, 10, 10] = 1.

    desired = m.smooth(kernel="gauss", width=0.5 * u.deg, mode="constant")
    gauss = Gaussian2DKernel(10).array
    actual = m.convolve(kernel=gauss)
    assert_allclose(actual.data, desired.data, rtol=1e-3)


@requires_dependency("scipy")
@requires_data("gammapy-extra")
def test_convolve_nd():
    energy_axis = MapAxis.from_edges(np.logspace(-1., 1., 4), unit="TeV", name="energy")
    geom = WcsGeom.create(binsz=0.02 * u.deg, width=4.0 * u.deg, axes=[energy_axis])
    m = Map.from_geom(geom)
    m.fill_by_coord([[0.2, 0.4], [-0.1, 0.6], [0.5, 3.6]])

    # TODO : build EnergyDependentTablePSF programmatically rather than using CTA 1DC IRF
    filename = "$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta//1dc/bcf/South_z20_50h/irf_file.fits"
    psf = EnergyDependentMultiGaussPSF.read(filename, hdu="POINT SPREAD FUNCTION")
    table_psf = psf.to_energy_dependent_table_psf(theta=0.5 * u.deg)

    psf_kernel = PSFKernel.from_table_psf(table_psf, geom, max_radius=1 * u.deg)

    assert psf_kernel.psf_kernel_map.data.shape == (3, 101, 101)

    mc = m.convolve(psf_kernel)

    assert_allclose(mc.data.sum(axis=(1, 2)), [0, 1, 1], atol=1e-5)


@requires_dependency("scipy")
def test_convolve_pixel_scale_error():
    m = WcsNDMap.create(binsz=0.05 * u.deg, width=5 * u.deg)
    kgeom = WcsGeom.create(binsz=0.04 * u.deg, width=0.5 * u.deg)

    kernel = PSFKernel.from_gauss(kgeom, sigma=0.1 * u.deg, max_radius=1.5 * u.deg)

    with pytest.raises(ValueError) as err:
        m.convolve(kernel)
        assert "Kernel shape larger" in str(err.value)


@requires_dependency("matplotlib")
def test_plot():
    m = WcsNDMap.create(binsz=0.1 * u.deg, width=1 * u.deg)
    with mpl_plot_check():
        m.plot(add_cbar=True)


@requires_dependency("matplotlib")
def test_plot_allsky():
    m = WcsNDMap.create(binsz=10 * u.deg)
    with mpl_plot_check():
        m.plot()
