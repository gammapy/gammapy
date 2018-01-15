# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
from astropy import units as u
from astropy.units import Quantity
from astropy.tests.helper import assert_quantity_allclose
import pytest
from astropy.wcs import WcsError
from regions import PixCoord, CirclePixelRegion, CircleSkyRegion
from ...utils.testing import requires_dependency, requires_data
from ...data import DataStore, EventList
from ...datasets import load_poisson_stats_image
from ..core import SkyImage


class _TestImage:
    """A set of small test images.

    This class is used to group test images and reference results.

    The images are organised in an `OrderedDict` with keys that
    are tuples `(proj, coordsys)`, like e.g. `('CAR', 'GAL')`.

    - projections: CAR, TAN, AIT
    - coordinate systems: GAL, CEL
    """

    params = [
        ('CAR', 'CEL'), ('CAR', 'GAL'),
        ('TAN', 'CEL'), ('TAN', 'GAL'),
        ('AIT', 'CEL'), ('AIT', 'GAL'),
    ]

    def __init__(self, proj='CAR', coordsys='CEL'):
        self.proj = proj
        self.coordsys = coordsys
        self.param = (proj, coordsys)
        self.input_image = self._make_input_image(proj, coordsys)
        self.solid_angle = self._make_solid_angle(proj)

    @staticmethod
    def _make_input_image(proj, coordsys):
        """Input test image"""
        image = SkyImage.empty(
            nxpix=6, nypix=3, binsz=60,
            proj=proj, coordsys=coordsys,
        )
        image.data = np.arange(6 * 3).reshape(image.data.shape)
        return image

    @staticmethod
    def _make_solid_angle(proj):
        """Solid angle reference results.

        Note: the first and last row isn't the same, because the solid angle algorithm
        isn't symmetric (uses lower-left pixel), so for CAR the pixel solid angle is
        approximately zero, because the lower-left and lower-right pixel are both
        at the pole.
        A more precise method would be e.g. to call
        http://spacetelescope.github.io/sphere/api/spherical_geometry.polygon.SphericalPolygon.html#spherical_geometry.polygon.SphericalPolygon.area
        and then for CAR the solid angle
        - in the first and last row should be the same
        - sum should be `4 * pi` because the image covers the whole sphere
        """
        if proj == 'CAR':
            return [
                [6.4122356457393e-17, 6.412235645739299e-17, 6.412235645739299e-17,
                 6.4122356457393e-17, 6.412235645739303e-17, 6.412235645739298e-17],
                [0.9379379788209616, 0.9379379788209615, 0.9379379788209615,
                 0.9379379788209616, 0.9379379788209621, 0.9379379788209613],
                [0.9379379788209619, 0.9379379788209617, 0.9379379788209617,
                 0.9379379788209619, 0.9379379788209623, 0.9379379788209615],
            ]
        elif proj == 'TAN':
            return [
                [0.0550422978927337, 0.12377680480476913, 0.24543234890769688,
                 0.26718078554015406, 0.1589357310716131, 0.07152397341869869],
                [0.047056238883475876, 0.1458035992891698, 0.519025419357332,
                 0.7215164186829529, 0.2280005335348649, 0.06629375814418971],
                [0.04289253711963261, 0.1225836131320998, 0.35831703773805357,
                 0.39006849765871243, 0.15740361210066894, 0.05573613025356901]
            ]
        elif proj == 'AIT':
            nan = np.nan
            return [
                [nan, nan, nan, nan, nan, nan],
                [nan, 0.96302079, 1.02533278, 1.06937617, 1.11576917, nan],
                [nan, nan, nan, nan, nan, nan],
            ]


@pytest.fixture(
    scope="session",
    params=_TestImage.params,
    ids=[str(x) for x in _TestImage.params],
)
def test_image(request):
    return _TestImage(*request.param)


@requires_data('gammapy-extra')
class TestSkyImagePoisson:
    def setup(self):
        f = load_poisson_stats_image(return_filenames=True)
        self.image = SkyImage.read(f)

    def test_read_hdu(self):
        f = load_poisson_stats_image(return_filenames=True)
        hdulist = fits.open(f)
        image = SkyImage.from_image_hdu(hdulist[0])
        assert_equal(image.data, self.image.data)

    def test_io(self, tmpdir):
        filename = tmpdir / 'test_image.fits'
        self.image.write(str(filename))
        image = SkyImage.read(str(filename))
        assert self.image.name == image.name

    def test_unit_io(self, tmpdir):
        filename = tmpdir / 'test_image_unit.fits'
        image_ref = SkyImage(data=np.zeros((3, 3)), unit='1 / cm2')
        image_ref.write(str(filename))
        image = SkyImage.read(str(filename))
        assert image.unit == image_ref.unit

    def test_lookup_skycoord(self):
        position = SkyCoord(0, 0, frame='galactic', unit='deg')
        assert self.image.lookup(position) == 5

    def test_coordinates(self):
        coordinates = self.image.coordinates()
        assert_allclose(coordinates.data.lon[100, 100].degree, 0.01)
        assert_allclose(coordinates.data.lat[100, 100].degree, 0.01)

    def test_solid_angle(self):
        solid_angle = self.image.solid_angle()
        assert_quantity_allclose(solid_angle[0, 0], Angle(0.02, "deg") ** 2, rtol=1e-3)

    def test_solid_angle_with_small_images(self, test_image):
        solid_angle = test_image.input_image.solid_angle()
        assert_allclose(solid_angle.value, test_image.solid_angle, rtol=1e-6)

    def test_contains(self):
        position = SkyCoord(0, 0, frame='galactic', unit='deg')
        assert self.image.contains(position)

        position = SkyCoord([42, 0, -42], [11, 0, -11], frame='galactic', unit='deg')
        assert_equal(self.image.contains(position), [False, True, False])

        coordinates = self.image.coordinates()
        assert np.all(self.image.contains(coordinates[2:5, 2:5]))

    def test_info(self):
        refstring = ""
        refstring += "Name: None\n"
        refstring += "Data shape: (200, 200)\n"
        refstring += "Data type: >i4\n"
        refstring += "Data unit: \n"
        refstring += "Data mean: 1.022e+00\n"
        refstring += "WCS type: ['GLON-CAR', 'GLAT-CAR']\n"
        assert str(self.image) == refstring

    def test_to_quantity(self):
        q = self.image.to_quantity()
        assert_equal(q.value, self.image.data)

    @requires_dependency('sherpa')
    def test_to_sherpa_data2d(self):
        from sherpa.data import Data2D
        data = self.image.to_sherpa_data2d()
        assert isinstance(data, Data2D)

    def test_empty(self):
        empty = SkyImage.empty()
        assert empty.data.shape == (200, 200)

    @requires_data('gammapy-extra')
    def test_fill_events(self):
        dirname = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2'
        data_store = DataStore.from_dir(dirname)

        events = data_store.obs(obs_id=23523).events

        counts = SkyImage.empty(nxpix=200, nypix=200,
                                xref=events.table.meta['RA_OBJ'],
                                yref=events.table.meta['DEC_OBJ'],
                                dtype='int', coordsys='CEL')
        counts.fill_events(events)
        assert counts.data.sum() == 1233
        assert counts.data.shape == (200, 200)

    @requires_dependency('reproject')
    def test_reproject(self):
        image_1 = SkyImage.empty(nxpix=200, nypix=200, xref=0, yref=0, coordsys='CEL')
        image_2 = SkyImage.empty(nxpix=100, nypix=100, xref=0, yref=0, binsz=0.04, coordsys='CEL')
        image_1.data.fill(1)
        image_1_repr = image_1.reproject(image_2)
        assert_allclose(image_1_repr.data, np.full((100, 100), 1))

    def test_lookup_max(self):
        pos, value = self.image.lookup_max()
        assert value == 15
        assert_allclose((359.93, -0.01), (pos.galactic.l.deg, pos.galactic.b.deg))

    def test_lookup_max_region(self):
        center = SkyCoord(0, 0, unit='deg', frame='galactic')
        circle = CircleSkyRegion(center, radius=Quantity(1, 'deg'))
        pos, value = self.image.lookup_max(circle)
        assert value == 15
        assert_allclose((359.93, -0.01), (pos.galactic.l.deg, pos.galactic.b.deg))

    def test_cutout_paste(self):
        positions = SkyCoord([0, 0, 0, 0.4, -0.4], [0, 0.4, -0.4, 0, 0],
                             unit='deg', frame='galactic')
        BINSZ = 0.02

        # setup coordinate images
        lon = SkyImage.empty(nxpix=41, nypix=41, binsz=BINSZ)
        lat = SkyImage.empty(nxpix=41, nypix=41, binsz=BINSZ)

        c = lon.coordinates()
        lon.data = c.galactic.l.deg
        lat.data = c.galactic.b.deg

        size = Quantity([0.3, 0.3], 'deg')
        for pos in positions:
            cutout = lon.cutout(pos, size=size)

            # recompute coordinates and paste into coordinate images
            c = cutout.coordinates()
            cutout.data = c.galactic.l.deg
            lon.paste(cutout, method='replace')

            cutout.data = c.galactic.b.deg
            lat.paste(cutout, method='replace')

        c = lon.coordinates()
        assert_allclose(lon.data, c.galactic.l.deg)
        assert_allclose(lat.data, c.galactic.b.deg)

    def test_cutout_paste_wcs_error(self):
        # setup coordinate images
        image = SkyImage.empty(nxpix=7, nypix=7, binsz=0.02)
        cutout = SkyImage.empty(nxpix=4, nypix=4, binsz=0.02)
        with pytest.raises(WcsError):
            image.paste(cutout)

    @requires_dependency('skimage')
    @pytest.mark.parametrize(('shape', 'factor', 'proj'), [((4, 6), 2, 'CAR'),
                                                           ((9, 12), 3, 'CAR')])
    def test_downsample(self, shape, factor, proj):
        nypix, nxpix = shape
        image = SkyImage.empty(nxpix=nxpix, nypix=nypix, binsz=0.02, proj=proj)
        image_downsampled = image.downsample(factor=factor)
        separation = image.center.separation(image_downsampled.center)

        # check WCS
        assert_quantity_allclose(separation, Quantity(0, 'deg'))

        # check data shape
        assert image_downsampled.data.shape == (shape[0] // factor, shape[1] // factor)

    @requires_dependency('scipy')
    @pytest.mark.parametrize(('shape', 'factor', 'proj'), [((2, 3), 2, 'TAN'),
                                                           ((3, 4), 3, 'TAN')])
    def test_upsample(self, shape, factor, proj):
        nypix, nxpix = shape
        image = SkyImage.empty(nxpix=nxpix, nypix=nypix, binsz=0.02, proj=proj)
        image_upsampled = image.upsample(factor=factor)
        separation = image.center.separation(image_upsampled.center)

        # check WCS
        assert_quantity_allclose(separation, Quantity(0, 'deg'), atol=Quantity(1e-17, 'deg'))

        # check data shape
        assert image_upsampled.data.shape == (shape[0] * factor, shape[1] * factor)

    @requires_dependency('scipy')
    @requires_dependency('skimage')
    @pytest.mark.parametrize(('shape', 'factor', 'proj'), [((4, 4), 2, 'AIT'),
                                                           ((9, 9), 3, 'AIT')])
    def test_down_and_upsample(self, shape, factor, proj):
        nypix, nxpix = shape
        image = SkyImage.empty(nxpix=nxpix, nypix=nypix, binsz=10, fill=1.)
        image_downsampled = image.downsample(factor=factor, method=np.nanmean)
        image_upsampled = image_downsampled.upsample(factor=factor)
        assert_allclose(image.data, image_upsampled.data)

    def test_crop(self):
        image = SkyImage.empty(nxpix=9, nypix=9, binsz=0.02, proj='AIT', xref=135,
                               yref=60)
        image_cropped = image.crop(((2, 2), (2, 2)))

        separation = image.center.separation(image_cropped.center)
        assert_quantity_allclose(separation, Quantity(0, 'deg'), atol=Quantity(1e-17, 'deg'))

        # check data shape
        assert image_cropped.data.shape == (5, 5)


class TestSkyImage:
    def setup(self):
        self.center = SkyCoord(83.63, 22.01, unit='deg').galactic
        self.image = SkyImage.empty(
            nxpix=10, nypix=5, binsz=0.2,
            xref=self.center.l.deg, yref=self.center.b.deg,
            proj='TAN', coordsys='GAL', meta={'TEST': 42},
        )
        self.image.data = np.arange(50, dtype=float).reshape((5, 10))

    def test_center_pix(self):
        center = self.image.center_pix
        assert_allclose(center.x, 4.5)
        assert_allclose(center.y, 2.0)

    def test_center_sky(self):
        center = self.image.center
        assert_allclose(center.l.deg, self.center.l.deg, atol=1e-5)
        assert_allclose(center.b.deg, self.center.b.deg, atol=1e-5)

    def test_width(self):
        width = self.image.width
        assert_quantity_allclose(width, 2 * u.deg, rtol=1e-3)

    def test_height(self):
        height = self.image.height
        assert_allclose(height, 1 * u.deg, rtol=1e-3)

    @requires_dependency('scipy')
    @pytest.mark.parametrize('kernel', ['gauss', 'box', 'disk'])
    def test_smooth(self, kernel):
        desired = self.image.data.sum()
        smoothed = self.image.smooth(kernel, 0.2 * u.deg)
        actual = smoothed.data.sum()
        assert_allclose(actual, desired)

    def test_meta(self):
        assert self.image.meta['TEST'] == 42


def test_image_pad():
    image = SkyImage.empty(nxpix=10, nypix=13)
    assert image.data.shape == (13, 10)

    image2 = image.pad(pad_width=4, mode='reflect')
    assert image2.data.shape == (21, 18)


def test_skycoord_pixel_conversion():
    image = SkyImage.empty(nxpix=10, nypix=15)

    x, y = [5, 3.4], [8, 11.2]
    coords = image.wcs_pixel_to_skycoord(xp=x, yp=y)
    assert_allclose(coords.data.lon.deg, [3.5999e+02, 2.2e-02])
    assert_allclose(coords.data.lat.deg, [0.02, 0.084])

    x_new, y_new = image.wcs_skycoord_to_pixel(coords=coords)
    assert_allclose(x, x_new)
    assert_allclose(y, y_new)


def test_wcs_pixel_scale():
    image = SkyImage.empty(nxpix=10, nypix=15, yref=10)
    assert_allclose(image.wcs_pixel_scale(method='cdelt'),
                    Angle([0.02, 0.02], unit='deg'))
    assert_allclose(image.wcs_pixel_scale(method='proj_plane'),
                    Angle([0.02, 0.02], unit='deg'))


def test_footprint():
    image = SkyImage.empty(nxpix=3, nypix=2)

    coord = image.footprint(mode='center')
    assert_allclose(image.wcs_skycoord_to_pixel(coord['lower left']), (0, 0))
    assert_allclose(image.wcs_skycoord_to_pixel(coord['upper left']), (0, 2))
    assert_allclose(image.wcs_skycoord_to_pixel(coord['upper right']), (3, 2))
    assert_allclose(image.wcs_skycoord_to_pixel(coord['lower right']), (3, 0))


def test_region_mask():
    mask = SkyImage.empty(nxpix=5, nypix=4)

    # Test with a pixel region
    region = CirclePixelRegion(center=PixCoord(x=2, y=1), radius=1.1)
    actual = mask.region_mask(region)
    expected = [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    assert_equal(actual.data, expected)

    # Test with a sky region
    # Output is the same in this case, because the sky region
    # represents roughly the same region as the pixel region
    sky_region = region.to_sky(wcs=mask.wcs)
    actual = mask.region_mask(sky_region)
    assert_equal(actual.data, expected)


@requires_data('gammapy-extra')
def test_fits_header_comment_io(tmpdir):
    """
    Test what happens to multi-line comments in the FITS header.

    This is a regression test for
    https://github.com/gammapy/gammapy/issues/701
    """
    filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/fermi/gll_iem_v02_cutout.fits'
    image = SkyImage.read(filename)
    image.write(tmpdir / 'temp.fits')


def test_image_fill_events():
    """A simple test case that can by checked by hand"""

    image = SkyImage.empty(
        nxpix=2, nypix=1, binsz=10,
        xref=0, yref=0, proj='CAR',
    )

    # GLON pixel edges: (+10, 0, -10)
    # GLAT pixel edges: (-5, +5)

    EPS = 0.1
    data = [
        (5, 5, 1),  # in image[0, 0]
        (0, 0 + EPS, 2),  # in image[1, 0]
        (5, -5 + EPS, 3),  # in image[0, 0]
        (5, 5 + EPS, 99),  # outside image
        (10 + EPS, 0, 99),  # outside image
    ]
    lon, lat, weights = np.array(data).T
    coord = SkyCoord(lon, lat, unit='deg', frame='galactic').icrs
    table = Table()
    table['RA'] = coord.ra.deg
    table['DEC'] = coord.dec.deg
    table['WEIGHT'] = weights
    events = EventList(table)

    image.fill_events(events, weights='WEIGHT')

    assert image.data[0, 0] == 1 + 3
    assert image.data[0, 1] == 2


@requires_dependency('scipy')
def test_distance_image():
    mask = SkyImage.empty(nxpix=3, nypix=2)
    distance = mask.distance_image.data
    assert_allclose(distance, -1e10)

    mask = SkyImage.empty(nxpix=3, nypix=2, fill=1.)
    distance = mask.distance_image.data
    assert_allclose(distance, 1e10)

    data = np.array([[0., 0., 1.], [1., 1., 1.]])
    mask = SkyImage(data=data)
    distance = mask.distance_image.data
    expected = [[-1, -1, 1], [1, 1, 1.41421356]]
    assert_allclose(distance, expected)


@requires_dependency('scipy')
def test_conversion_wcs_map_nd():
    """Check conversion SkyCube <-> WCSMapNd"""
    image = SkyImage.empty(nxpix=3, nypix=2, unit='cm', name='axel')

    map = image.to_wcs_nd_map()
    image2 = SkyImage.from_wcs_nd_map(map)

    # Note: WCSMapNd doesn't store name or unit at the moment
    SkyImage.assert_allclose(image, image2, check_name=False, check_unit=False)
