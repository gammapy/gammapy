# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_equal
from astropy.tests.helper import assert_quantity_allclose
from astropy.table import Table
from astropy.coordinates import Angle
from astropy.modeling.models import Gaussian1D
from ...utils.testing import requires_dependency, requires_data
from ...datasets import gammapy_extra
from ...background import GaussianBand2D, CubeBackgroundModel, EnergyOffsetBackgroundModel
from ...utils.energy import EnergyBounds
from ...data import ObservationTable
from ...data import DataStore



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
            observation_table=obs_table, method='default',
        )

        assert bg_cube_model.background_cube.data.shape == (20, 60, 60)

    @requires_dependency('scipy')
    def test_smooth(self):

        filename = gammapy_extra.filename('test_datasets/background/bg_cube_model_test2.fits.gz')
        bg_cube_model1 = CubeBackgroundModel.read(filename, format='table')

        bg_cube_model2 = bg_cube_model1

        # reset background and fill again
        bg_cube_model2.background_cube.data = np.array([0])
        bg_cube_model2.background_cube.data = bg_cube_model2.counts_cube.data.copy()
        bg_cube_model2.smooth()
        bg_cube_model2.background_cube.data /= bg_cube_model2.livetime_cube.data
        bg_cube_model2.background_cube.data /= bg_cube_model2.background_cube.bin_volume

        # test: the bg should be the same as at the beginning
        assert (bg_cube_model2.background_cube.data == bg_cube_model1.background_cube.data).all()

def make_test_array_empty():
        ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 100, 'TeV')
        offset = Angle(np.linspace(0, 2.5, 100),"deg")
        multi_array = EnergyOffsetBackgroundModel(ebounds, offset)
        return multi_array

@requires_data('gammapy-extra')
class TestEnergyOffsetBackgroundModel:

    def test_read_write(self):
        multi_array=make_test_array_empty()
        multi_array.livetime.data.value[:,:]=1
        multi_array.bg_rate.data.value[:,:]=2
        print(multi_array.livetime.data)
        filename = 'multidata.fits'
        multi_array.write(filename)
        multi_array2 = multi_array.read(filename)

        assert_equal(multi_array.counts.data, multi_array2.counts.data)
        assert_equal(multi_array.livetime.data, multi_array2.livetime.data)
        assert_equal(multi_array.bg_rate.data, multi_array2.bg_rate.data)
        assert_equal(multi_array.energy, multi_array2.energy)
        assert_equal(multi_array.offset, multi_array2.offset)

    def test_fill_obs(self):
        dir = str(gammapy_extra.dir) + '/datasets/hess-crab4-hd-hap-prod2'
        data_store = DataStore.from_dir(dir)
        obs_table=data_store.obs_table
        multi_array=make_test_array_empty()
        multi_array.fill_obs(obs_table, data_store)
        assert_equal(multi_array.counts.data.value.sum(), 5403)
        assert_equal(multi_array.livetime.data.value.sum(), 62506736.499239981)
        assert_equal(multi_array.bg_rate.data.value.sum(), 0)

    def test_compute_rate(self):
        dir = str(gammapy_extra.dir) + '/datasets/hess-crab4-hd-hap-prod2'
        data_store = DataStore.from_dir(dir)
        obs_table=data_store.obs_table
        multi_array=make_test_array_empty()
        multi_array.fill_obs(obs_table, data_store)
        multi_array.compute_rate()
        assert_equal(multi_array.bg_rate.data.value.sum(), 0.27537506238749021)

