# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy import nan
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
from astropy.units import Quantity
from astropy.tests.helper import pytest, assert_quantity_allclose
from astropy.wcs import WcsError
from astropy.extern.six import string_types
from ...extern.regions import CircleSkyRegion
from ...utils.testing import requires_dependency, requires_data
from ...data import DataStore
from ...datasets import load_poisson_stats_image
from ..maps import SkyMap


class TestImage:
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
        image = SkyMap.empty(
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
            return [
                [nan, nan, nan, nan, nan, nan],
                [nan, 0.96302079, 1.02533278, 1.06937617, 1.11576917, nan],
                [nan, nan, nan, nan, nan, nan],
            ]


@pytest.fixture(
    scope="session",
    params=TestImage.params,
    ids=TestImage.params,
)
def test_image(request):
    return TestImage(*request.param)


@requires_data('gammapy-extra')
class TestSkyMapPoisson:
    """
    Test sky map class.
    """

    def setup(self):
        f = load_poisson_stats_image(return_filenames=True)
        self.skymap = SkyMap.read(f)

    def test_read_hdu(self):
        f = load_poisson_stats_image(return_filenames=True)
        hdulist = fits.open(f)
        skymap = SkyMap.from_image_hdu(hdulist[0])
        assert_equal(skymap.data, self.skymap.data)

    def test_io(self, tmpdir):
        filename = tmpdir / 'test_skymap.fits'
        self.skymap.meta['COMMENT'] = 'Test comment'
        self.skymap.meta['HISTORY'] = 'Test history'
        self.skymap.write(str(filename))
        skymap = SkyMap.read(str(filename))
        assert self.skymap.name == skymap.name
        assert isinstance(skymap.meta['COMMENT'], string_types)
        assert isinstance(skymap.meta['HISTORY'], string_types)

    def test_unit_io(self, tmpdir):
        filename = tmpdir / 'test_skymap_unit.fits'
        skymap_ref = SkyMap(data=np.zeros((3, 3)), unit='1 / cm2')
        skymap_ref.write(str(filename))
        skymap = SkyMap.read(str(filename))
        assert skymap.unit == skymap_ref.unit

    def test_lookup_skycoord(self):
        position = SkyCoord(0, 0, frame='galactic', unit='deg')
        assert self.skymap.lookup(position) == 5

    def test_coordinates(self):
        coordinates = self.skymap.coordinates()
        assert_allclose(coordinates.data.lon[100, 100].degree, 0.01)
        assert_allclose(coordinates.data.lat[100, 100].degree, 0.01)

    def test_solid_angle(self):
        solid_angle = self.skymap.solid_angle()
        assert_quantity_allclose(solid_angle[0, 0], Angle(0.02, "deg") ** 2, rtol=1e-3)

    def test_solid_angle_with_small_images(self, test_image):
        solid_angle = test_image.input_image.solid_angle()
        assert_allclose(solid_angle.value, test_image.solid_angle, rtol=1e-6)

    def test_contains(self):
        position = SkyCoord(0, 0, frame='galactic', unit='deg')
        assert self.skymap.contains(position)

        position = SkyCoord([42, 0, -42], [11, 0, -11], frame='galactic', unit='deg')
        assert_equal(self.skymap.contains(position), [False, True, False])

        coordinates = self.skymap.coordinates()
        assert np.all(self.skymap.contains(coordinates[2:5, 2:5]))

    def test_info(self):
        refstring = ""
        refstring += "Name: None\n"
        refstring += "Data shape: (200, 200)\n"
        refstring += "Data type: >i2\n"
        refstring += "Data unit: None\n"
        refstring += "Data mean: 1.022e+00\n"
        refstring += "WCS type: ['GLON-CAR', 'GLAT-CAR']\n"
        assert str(self.skymap) == refstring

    def test_to_quantity(self):
        q = self.skymap.to_quantity()
        assert_equal(q.value, self.skymap.data)

    @requires_dependency('sherpa')
    def test_to_sherpa_data2d(self):
        from sherpa.data import Data2D
        data = self.skymap.to_sherpa_data2d()
        assert isinstance(data, Data2D)

    def test_empty(self):
        empty = SkyMap.empty()
        assert empty.data.shape == (200, 200)

    def test_center(self):
        center = self.skymap.center()
        assert center.galactic.l == 0
        assert center.galactic.b == 0

    def test_fill_float(self):
        skymap = SkyMap.empty(nxpix=200, nypix=200, xref=0, yref=0, dtype='int',
                              coordsys='CEL')
        skymap.fill(42)
        assert_equal(skymap.data, np.full((200, 200), 42))

    @requires_data('gammapy-extra')
    def test_fill_events(self):
        dirname = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2'
        data_store = DataStore.from_dir(dirname)

        events = data_store.obs(obs_id=23523).events

        counts = SkyMap.empty(nxpix=200, nypix=200, xref=events.meta['RA_OBJ'],
                              yref=events.meta['DEC_OBJ'], dtype='int',
                              coordsys='CEL')
        counts.fill(events)
        assert counts.data.sum() == 1233
        assert counts.data.shape == (200, 200)

    @requires_dependency('reproject')
    def test_reproject(self):
        skymap_1 = SkyMap.empty(nxpix=200, nypix=200, xref=0, yref=0, coordsys='CEL')
        skymap_2 = SkyMap.empty(nxpix=100, nypix=100, xref=0, yref=0, binsz=0.04,
                                coordsys='CEL')
        skymap_1.fill(1)
        skymap_1_repr = skymap_1.reproject(skymap_2)
        assert_allclose(skymap_1_repr.data, np.full((100, 100), 1))

    def test_lookup_max(self):
        pos, value = self.skymap.lookup_max()
        assert value == 15
        assert_allclose((359.93, -0.01), (pos.galactic.l.deg, pos.galactic.b.deg))

    def test_lookup_max_region(self):
        center = SkyCoord(0, 0, unit='deg', frame='galactic')
        circle = CircleSkyRegion(center, radius=Quantity(1, 'deg'))
        pos, value = self.skymap.lookup_max(circle)
        assert value == 15
        assert_allclose((359.93, -0.01), (pos.galactic.l.deg, pos.galactic.b.deg))

    def test_cutout_paste(self):
        positions = SkyCoord([0, 0, 0, 0.4, -0.4], [0, 0.4, -0.4, 0, 0],
                             unit='deg', frame='galactic')
        BINSZ = 0.02

        # setup coordinate images
        lon = SkyMap.empty(nxpix=41, nypix=41, binsz=BINSZ)
        lat = SkyMap.empty(nxpix=41, nypix=41, binsz=BINSZ)

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
        skymap = SkyMap.empty(nxpix=7, nypix=7, binsz=0.02)
        cutout = SkyMap.empty(nxpix=4, nypix=4, binsz=0.02)
        with pytest.raises(WcsError):
            skymap.paste(cutout)


class TestSkyMapCrab:
    """
    Test sky map class.
    """

    def crab_coord(self):
        coord = SkyCoord(83.63, 22.01, unit='deg').galactic
        return coord

    def setup(self):
        center = self.crab_coord()
        self.skymap = SkyMap.empty(nxpix=250, nypix=250, binsz=0.02, xref=center.l.deg,
                                   yref=center.b.deg, proj='TAN', coordsys='GAL')

    def test_center(self):
        crab_coord = self.crab_coord()
        center = self.skymap.center()
        assert_allclose(center.galactic.l, crab_coord.l, rtol=1e-2)
        assert_allclose(center.galactic.b, crab_coord.b, rtol=1e-2)
