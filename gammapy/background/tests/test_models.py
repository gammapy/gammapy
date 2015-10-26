# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import assert_quantity_allclose
from astropy.table import Table
from astropy.modeling.models import Gaussian1D
from ...utils.testing import requires_dependency, requires_data
from ...datasets import gammapy_extra
from ...background import GaussianBand2D, CubeBackgroundModel
from ...obs import ObservationTable


@requires_dependency('scipy')
class TestGaussianBand2D:
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


@requires_data('gammapy-extra')
class TestCubeBackgroundModel:
    def test_read(self):

        # test shape and scheme of cubes when reading a file
        filename = gammapy_extra.filename('test_datasets/background/bg_cube_model_test2.fits.gz')
        bg_cube_model = CubeBackgroundModel.read(filename, format='table')
        cubes = [bg_cube_model.counts_cube,
                 bg_cube_model.livetime_cube,
                 bg_cube_model.background_cube]
        schemes = ['bg_counts_cube', 'bg_livetime_cube', 'bg_cube']
        for cube, scheme in zip(cubes, schemes):
            assert len(cube.data.shape) == 3
            assert cube.data.shape == (len(cube.energy_edges) - 1,
                                       len(cube.coordy_edges) - 1,
                                       len(cube.coordx_edges) - 1)
            assert cube.scheme == scheme

    def test_write(self, tmpdir):

        filename = gammapy_extra.filename('test_datasets/background/bg_cube_model_test2.fits.gz')
        bg_cube_model_1 = CubeBackgroundModel.read(filename, format='table')

        outfile = str(tmpdir / 'cubebackground_table_test.fits')
        bg_cube_model_1.write(outfile, format='table')

        # test if values are correct in the saved file: compare both files
        bg_cube_model_2 = CubeBackgroundModel.read(outfile, format='table')
        cubes1 = [bg_cube_model_1.counts_cube,
                  bg_cube_model_1.livetime_cube,
                  bg_cube_model_1.background_cube]
        cubes2 = [bg_cube_model_2.counts_cube,
                  bg_cube_model_2.livetime_cube,
                  bg_cube_model_2.background_cube]
        for cube1, cube2 in zip(cubes1, cubes2):
            assert_quantity_allclose(cube2.data,
                                     cube1.data)
            assert_quantity_allclose(cube2.coordx_edges,
                                     cube1.coordx_edges)
            assert_quantity_allclose(cube2.coordy_edges,
                                     cube1.coordy_edges)
            assert_quantity_allclose(cube2.energy_edges,
                                     cube1.energy_edges)

    def test_define_binning(self):

        obs_table = ObservationTable()
        obs_table['OBS_ID'] = np.arange(100)
        bg_cube_model = CubeBackgroundModel.define_cube_binning(
            observation_table=obs_table, data_dir='dummy', method='default',
        )

        assert bg_cube_model.background_cube.data.shape == (20, 60, 60)

    # TODO: implement this test of remove this
    # def test_fill_events(self):
    # fill_events is tested (high-level) by
    # gammapy/scripts/tests/test_make_bg_cube_models.py

    def test_smooth(self):

        filename = gammapy_extra.filename('test_datasets/background/bg_cube_model_test2.fits.gz')
        bg_cube_model1 = CubeBackgroundModel.read(filename, format='table')

        bg_cube_model2 = bg_cube_model1

        # reset background and fill again
        bg_cube_model2.background_cube.data = np.array([0])
        bg_cube_model2.background_cube.data = bg_cube_model2.counts_cube.data.copy()
        bg_cube_model2.smooth()
        bg_cube_model2.background_cube.data /= bg_cube_model2.livetime_cube.data
        bg_cube_model2.background_cube.divide_bin_volume()
        bg_cube_model2.background_cube.set_zero_level()

        # test: the bg should be the same as at the beginning
        assert (bg_cube_model2.background_cube.data == bg_cube_model1.background_cube.data).all()
