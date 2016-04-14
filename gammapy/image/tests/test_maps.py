# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose, assert_equal

from astropy.coordinates import SkyCoord
from astropy.io import fits

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
        skymap = SkyMap.read(hdulist[0])
        assert_equal(skymap.data, self.skymap.data) 
        
    def test_io(self, tmpdir):
        filename = tmpdir / 'test_skymap.fits'
        self.skymap.write(str(filename))
        skymap = SkyMap.read(str(filename))
        assert self.skymap.name == skymap.name

    def test_lookup(self):
        assert self.skymap.lookup((0, 0)) == 8

    def test_lookup_skycoord(self):
        position = SkyCoord(0, 0, frame='galactic', unit='deg')
        assert self.skymap.lookup(position) == self.skymap.lookup((0, 0))

    def test_coordinates(self):
        coordinates = self.skymap.coordinates('galactic')
        assert coordinates[0][98, 98] == 0
        assert coordinates[1][98, 98] == 0

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


    @requires_data('gammapy-extra')
    def test_fill(self):
        dirname = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2'
        data_store = DataStore.from_dir(dirname)

        events = data_store.obs(obs_id=23523).events

        counts = SkyMap.empty(nxpix=200, nypix=200, xref=events.meta['RA_OBJ'],
                              yref=events.meta['DEC_OBJ'], dtype='int', 
                              coordsys='CEL')
        counts.fill(events)
        assert counts.data.sum().value == 1233
        assert counts.data.shape == (200, 200)
