# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.modeling.models import Gaussian1D
from ...background.models import GaussianBand2D, CubeBackgroundModel


try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
class TestGaussianBand2D():

    def setup(self):
        table = Table()
        table['x'] = [-30, -10, 10, 20]
        table['amplitude'] = [0, 1, 10, 0]
        table['mean'] = [-1, 0, 1, 0]
        table['stddev'] = [0.4, 0.5, 0.3, 1.0]
        self.table = table
        self.model = GaussianBand2D(table)

    def test_evaluate(self):
        x = np.linspace(-100, 20, 5)
        y = np.linspace(-2, 2, 7)
        x, y = np.meshgrid(x, y)
        image = self.model.evaluate(x, y)
        assert_allclose(image.sum(), 1.223962643740966)

    def test_parvals(self):
        par = self.model.parvals(-30)
        assert_allclose(par['amplitude'], 0)
        assert_allclose(par['mean'], -1)
        assert_allclose(par['stddev'], 0.4)

    def test_y_model(self):
        model = self.model.y_model(-30)
        assert isinstance(model, Gaussian1D)
        assert_allclose(model.parameters, [0, -1, 0.4])

class TestCubeBackgroundModel():

##    def test_read(self):
##        # test shape of bg cube when reading a file
##        filename = 'data/bg_test.fits'
##        #DIR = '/home/mapaz/astropy/development_code/gammapy/gammapy/background/tests/'
##        #filename = DIR + filename
##        filename = get_pkg_data_filename(filename)
##        # TODO: this is failing!!!
##        # maybe because the class is not part of the module?!!!
##        # Here it works:
##        # gammapy/irf/tests/test_effective_area.py test_EffectiveAreaTable
##        # TODO: the test file doesn't have units for the det x,y axes!!!
##        #        produce correct file (test_write), then come back here!!!
##        bg_cube_file = CubeBackgroundModel.read(filename)
##        assert len(bg_cube_file.background.shape) == 3


    def test_image_plot(self):

        DIR = '/home/mapaz/astropy/testing_cube_bg_michael_mayer/background/'
        filename = 'hist_alt3_az0.fits.gz'
        filename = DIR + filename
        #TODO: change this, when test_read is fixed!!!
        #      - use data/bg_test.fits
        #      - use get_pkg_data_filename
        bg_cube_model = CubeBackgroundModel.read(filename)

        # test bg rate values plotted for image plot of energy bin conaining E = 2 TeV
        energy = Quantity(2., 'TeV')
        fig_image, ax_im, image_im = bg_cube_model.plot_images(energy)
        plot_data = image_im.get_array()

        # get data from bg model object to compare
        energy_bin, energy_bin_edges = bg_cube_model.find_energy_bin(energy)
        model_data = bg_cube_model.background[energy_bin]

        # test if both arrays are equal
        decimal = 4
        np.testing.assert_almost_equal(plot_data, model_data.value, decimal)
        # TODO: clean up after test (remove created files)


    def test_spectrum_plot(self):

        DIR = '/home/mapaz/astropy/testing_cube_bg_michael_mayer/background/'
        filename = 'hist_alt3_az0.fits.gz'
        filename = DIR + filename
        #TODO: change this, when test_read is fixed!!!
        #      - use data/bg_test.fits
        #      - use get_pkg_data_filename
        bg_cube_model = CubeBackgroundModel.read(filename)

        # test bg rate values plotted for spectrum plot of detector bin conaining det (0, 0) deg (center)
        det = Angle([0., 0.], 'degree')
        fig_spec, ax_spec, image_spec = bg_cube_model.plot_spectra(det)
        plot_data = ax_spec.get_lines()[0].get_xydata()

        # get data from bg model object to compare
        det_bin, det_bin_edges = bg_cube_model.find_det_bin(det)
        model_data = bg_cube_model.background[:, det_bin[0], det_bin[1]]

        # test if both arrays are equal
        decimal = 4
        np.testing.assert_almost_equal(plot_data[:,1], model_data.value, decimal)
        # TODO: clean up after test (remove created files)


    def test_write(self):

        DIR = '/home/mapaz/astropy/testing_cube_bg_michael_mayer/background/'
        filename = 'hist_alt3_az0.fits.gz'
        filename = DIR + filename
        #TODO: change this, when test_read is fixed!!!
        #      - use data/bg_test.fits
        #      - use get_pkg_data_filename
        bg_model_1 = CubeBackgroundModel.read(filename)

        outfile = 'test_write_bg_cube_model.fits'
        bg_model_1.write(outfile)

        # test if values are correct in the saved file: compare both files
        bg_model_2 = CubeBackgroundModel.read(outfile)
        decimal = 4
        np.testing.assert_almost_equal(bg_model_2.background.value,
                                       bg_model_1.background.value, decimal)
        np.testing.assert_almost_equal(bg_model_2.det_bins.value,
                                       bg_model_1.det_bins.value, decimal)
        np.testing.assert_almost_equal(bg_model_2.energy_bins.value,
                                       bg_model_1.energy_bins.value, decimal)
        # TODO: clean up after test (remove created files)
        # TODO: test also write_image
