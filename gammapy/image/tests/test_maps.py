# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.coordinates import SkyCoord, Angle
from astropy.modeling.models import Gaussian2D
from astropy.io import fits
import astropy.units as u
from ..maps import SkyMap
from ...data import DataStore
from ...datasets import load_poisson_stats_image
from ...utils.testing import requires_dependency, requires_data


@requires_data('gammapy-extra')
class TestSkyMapPoisson():
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
        self.skymap.write(str(filename))
        skymap = SkyMap.read(str(filename))
        assert self.skymap.name == skymap.name

    def test_lookup(self):
        assert self.skymap.lookup((0, 0)) == 5

    def test_lookup_skycoord(self):
        position = SkyCoord(0, 0, frame='galactic', unit='deg')
        assert self.skymap.lookup(position) == self.skymap.lookup((0, 0))

    def test_coordinates(self):
        coordinates = self.skymap.coordinates()
        assert_allclose(coordinates.data.lon[100, 100].degree, 0.01)
        assert_allclose(coordinates.data.lat[100, 100].degree, 0.01)

    def test_solid_angle(self):
        solid_angle = self.skymap.solid_angle()
        assert_allclose(solid_angle.to("deg2")[0, 0], Angle(0.02, "deg") ** 2)

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
        from ...extern.regions.shapes import CircleSkyRegion
        center = SkyCoord(0, 0, unit='deg', frame='galactic')
        circle = CircleSkyRegion(center, radius=1 * u.deg)
        pos, value = self.skymap.lookup_max(circle) 
        assert value == 15
        assert_allclose((359.93, -0.01), (pos.galactic.l.deg, pos.galactic.b.deg))

    def test_cutout_paste(self):
        BINSZ = 0.02
        sigma = 0.2
        ampl = 1. / (2 * np.pi * (sigma / BINSZ) ** 2)
        sources = [Gaussian2D(ampl, 0, 0, sigma, sigma),
                   Gaussian2D(ampl, 1.9, 0, sigma, sigma),
                   Gaussian2D(ampl, 0, -1.9, sigma, sigma)]

        skymap_all = SkyMap.empty(nxpix=201, nypix=201, binsz=BINSZ)
        skymap_cutout = SkyMap.empty(nxpix=201, nypix=201, binsz=BINSZ)
        for source in sources:
            # Evaluate on whole image
            l, b = skymap_all.coordinates('galactic')
            skymap_all.data += source(l.deg, b.deg)

            # Evaluate on cut out
            pos = SkyCoord(source.x_mean, source.y_mean,
                           unit='deg', frame='galactic')
            cutout = skymap_cutout.cutout(pos, size=(2 * u.deg, 2 * u.deg))
            l, b = cutout.coordinates('galactic')
            cutout.data = source(l.deg, b.deg)
            skymap_cutout.paste(cutout)

        assert_allclose(skymap_all, skymap_cutout, atol=1E-8, rtol=0)

class TestSkyMapCrab():
    """
    Test sky map class.
    """

    def Crab_coord(self):
        coord = SkyCoord(83.63, 22.01, unit='deg').galactic
        return coord

    def setup(self):
        center = self.Crab_coord()
        self.skymap = SkyMap.empty(nxpix=250, nypix=250, binsz=0.02, xref=center.l.deg,
                                   yref=center.b.deg, proj='TAN', coordsys='GAL')

    def test_center(self):
        crab_coord = self.Crab_coord()
        center = self.skymap.center()
        assert_allclose(center.galactic.l, crab_coord.l, rtol=1e-2)
        assert_allclose(center.galactic.b, crab_coord.b, rtol=1e-2)
