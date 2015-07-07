# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from tempfile import NamedTemporaryFile
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest, remote_data
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.modeling.models import Gaussian1D
from ...background.models import GaussianBand2D, CubeBackgroundModel
from ... import datasets


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

    @remote_data
    def test_read_bin_table(self):
        
        # test shape of bg cube when reading a file
        filename = '../test_datasets/background/bg_cube_model_test.fits'
        filename = datasets.get_path(filename, location='remote')
        bg_cube_file = CubeBackgroundModel.read_bin_table(filename)
        assert len(bg_cube_file.background.shape) == 3


    @remote_data
    def test_image_plot(self):

        filename = '../test_datasets/background/bg_cube_model_test.fits'
        filename = datasets.get_path(filename, location='remote')
        bg_cube_model = CubeBackgroundModel.read_bin_table(filename)

        # test bg rate values plotted for image plot of energy bin conaining E = 2 TeV
        energy = Quantity(2., 'TeV')
        fig_image, ax_im, image_im = bg_cube_model.plot_images(energy)
        plot_data = image_im.get_array()

        # get data from bg model object to compare
        energy_bin, energy_bin_edges = bg_cube_model.find_energy_bin(energy)
        model_data = bg_cube_model.background[energy_bin]

        # test if both arrays are equal
        assert_allclose(plot_data, model_data.value)
        # TODO: clean up after test (remove created files)


    @remote_data
    def test_spectrum_plot(self):

        filename = '../test_datasets/background/bg_cube_model_test.fits'
        filename = datasets.get_path(filename, location='remote')
        bg_cube_model = CubeBackgroundModel.read_bin_table(filename)

        # test bg rate values plotted for spectrum plot of detector bin conaining det (0, 0) deg (center)
        det = Angle([0., 0.], 'degree')
        fig_spec, ax_spec, image_spec = bg_cube_model.plot_spectra(det)
        plot_data = ax_spec.get_lines()[0].get_xydata()

        # get data from bg model object to compare
        det_bin, det_bin_edges = bg_cube_model.find_det_bin(det)
        model_data = bg_cube_model.background[:, det_bin[0], det_bin[1]]

        # test if both arrays are equal
        assert_allclose(plot_data[:,1], model_data.value)
        # TODO: clean up after test (remove created files)


    @remote_data
    def test_write_bin_table(self):

        filename = '../test_datasets/background/bg_cube_model_test.fits'
        filename = datasets.get_path(filename, location='remote')
        bg_model_1= CubeBackgroundModel.read_bin_table(filename)

        outfile = NamedTemporaryFile(suffix='.fits').name
        bg_model_1.write_bin_table(outfile)

        # test if values are correct in the saved file: compare both files
        bg_model_2 = CubeBackgroundModel.read_bin_table(outfile)
        assert_allclose(bg_model_2.background.value,
                                       bg_model_1.background.value)
        assert_allclose(bg_model_2.detx_bins.value,
                                       bg_model_1.detx_bins.value)
        assert_allclose(bg_model_2.dety_bins.value,
                                       bg_model_1.dety_bins.value)
        assert_allclose(bg_model_2.energy_bins.value,
                                       bg_model_1.energy_bins.value)

        # TODO: test also read_image and write_image
