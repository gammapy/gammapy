# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest, remote_data, assert_quantity_allclose
from astropy.units import Quantity
from astropy.coordinates import Angle
from ...background import Cube
from ... import datasets
from ...datasets import make_test_bg_cube_model

try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TestCube():
    @remote_data
    def test_read_fits_table(self):
        # test shape and scheme of cube when reading a file
        filename = datasets.get_path('../test_datasets/background/bg_cube_model_test1.fits',
                                     location='remote')
        scheme = 'bg_cube'
        cube = Cube.read(filename, format='table', scheme=scheme)
        assert len(cube.data.shape) == 3
        assert cube.data.shape == (len(cube.energy_edges) - 1,
                                   len(cube.coordy_edges) - 1,
                                   len(cube.coordx_edges) - 1)
        assert cube.scheme == scheme

    @pytest.mark.skipif('not HAS_MATPLOTLIB')
    def test_image_plot(self):
        cube = make_test_bg_cube_model().background_cube

        # test bg rate values plotted for image plot of energy bin
        # conaining E = 2 TeV
        energy = Quantity(2., 'TeV')
        ax_im = cube.plot_image(energy)
        # get plot data (stored in the image)
        image_im = ax_im.get_images()[0]
        plot_data = image_im.get_array()

        # get data from bg model object to compare
        energy_bin = cube.find_energy_bin(energy)
        model_data = cube.data[energy_bin]

        # test if both arrays are equal
        assert_allclose(plot_data, model_data.value)

    @pytest.mark.skipif('not HAS_MATPLOTLIB')
    def test_spectrum_plot(self):
        cube = make_test_bg_cube_model().background_cube

        # test bg rate values plotted for spectrum plot of coordinate bin
        # conaining coord (0, 0) deg (center)
        coord = Angle([0., 0.], 'degree')
        ax_spec = cube.plot_spectrum(coord)
        # get plot data (stored in the line)
        plot_data = ax_spec.get_lines()[0].get_xydata()

        # get data from bg model object to compare
        coord_bin = cube.find_coord_bin(coord)
        model_data = cube.data[:, coord_bin[1], coord_bin[0]]

        # test if both arrays are equal
        assert_allclose(plot_data[:, 1], model_data.value)

    @remote_data
    def test_write_fits_table(self, tmpdir):
        filename = datasets.get_path('../test_datasets/background/bg_cube_model_test1.fits',
                                     location='remote')
        cube1 = Cube.read(filename, format='table', scheme='bg_cube')

        outfile = str(tmpdir.join('cube_table_test.fits'))
        cube1.write(outfile, format='table')

        # test if values are correct in the saved file: compare both files
        cube2 = Cube.read(outfile, format='table', scheme='bg_cube')
        assert_quantity_allclose(cube2.data,
                                 cube1.data)
        assert_quantity_allclose(cube2.coordx_edges,
                                 cube1.coordx_edges)
        assert_quantity_allclose(cube2.coordy_edges,
                                 cube1.coordy_edges)
        assert_quantity_allclose(cube2.energy_edges,
                                 cube1.energy_edges)

    @remote_data
    def test_read_write_fits_image(self, tmpdir):
        filename = datasets.get_path('../test_datasets/background/bg_cube_model_test1.fits',
                                     location='remote')
        cube1 = Cube.read(filename, format='table', scheme='bg_cube')

        outfile = str(tmpdir.join('cube_image_test.fits'))
        cube1.write(outfile, format='image')

        # test if values are correct in the saved file: compare both files
        cube2 = Cube.read(outfile, format='image', scheme='bg_cube')
        assert_quantity_allclose(cube2.data,
                                 cube1.data)
        assert_quantity_allclose(cube2.coordx_edges,
                                 cube1.coordx_edges)
        assert_quantity_allclose(cube2.coordy_edges,
                                 cube1.coordy_edges)
        assert_quantity_allclose(cube2.energy_edges,
                                 cube1.energy_edges)
