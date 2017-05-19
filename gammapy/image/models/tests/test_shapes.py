# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.utils.data import get_pkg_data_filename
from astropy.modeling.tests.test_models import Fittable2DModelTester
from ....extern import xmltodict
from ....utils.testing import requires_dependency, requires_data
from ..shapes import Delta2D, Gaussian2D, Sphere2D, Shell2D, Template2D

models_2D = [

    (Sphere2D, {
        'parameters': [1, 0, 0, 10, False],
        'x_values': [0, 10, 5],
        'y_values': [0, 10, 0],
        'z_values': [1, 0, np.sqrt(75) / 10],
        'x_lim': [-11, 11],
        'y_lim': [-11, 11],
        'integral': 4. / 3 * np.pi * 10 ** 3 / (2 * 10),
    }),

    (Shell2D, {
        'parameters': [1, 0, 0, 9, 1, 10, False],
        'x_values': [0],
        'y_values': [9],
        'z_values': [1],
        'x_lim': [-11, 11],
        'y_lim': [-11, 11],
        'integral': 2 * np.pi / 3 * (10 ** 3 - 9 ** 3) / np.sqrt(10 ** 2 - 9 ** 2),
    }),

    (Sphere2D, {
        'parameters': [(4. / 3 * np.pi * 10 ** 3 / (2 * 10)), 0, 0, 10, True],
        'constraints': {'fixed': {'amplitude': True, 'x_0': True, 'y_0': True}},
        'x_values': [0, 10, 5],
        'y_values': [0, 10, 0],
        'z_values': [1, 0, np.sqrt(75) / 10],
        'x_lim': [-11, 11],
        'y_lim': [-11, 11],
        'integral': (4. / 3 * np.pi * 10 ** 3 / (2 * 10)),
    }),

    (Shell2D, {
        'parameters': [(2 * np.pi / 3 * (10 ** 3 - 8 ** 3) /
                        np.sqrt(10 ** 2 - 8 ** 2)), 0, 0, 8, 2, 10, True],
        'constraints': {'fixed': {'amplitude': True, 'x_0': True, 'y_0': True, 'width': True}},
        'x_values': [0],
        'y_values': [8],
        'z_values': [1.],
        'x_lim': [-11, 11],
        'y_lim': [-11, 11],
        'integral': (2 * np.pi / 3 * (10 ** 3 - 8 ** 3) /
                     np.sqrt(10 ** 2 - 8 ** 2)),
    })
]

X_0 = [0, 1, -0.5, 0.012, -0.1245]
Y_0 = [0, 1, -0.5, -0.0345, 0.35345]


@pytest.mark.parametrize(('x_0', 'y_0'), list(zip(X_0, Y_0)))
def test_delta2d(x_0, y_0):
    y, x = np.mgrid[-3:4, -3:4]
    delta = Delta2D(1, x_0, y_0)
    values = delta(x, y)
    assert_allclose(values.sum(), 1)


@requires_dependency('scipy')
@pytest.mark.parametrize(('x_0', 'y_0'), list(zip(X_0, Y_0)))
def test_delta2d_against_gauss(x_0, y_0):
    from scipy.ndimage.filters import gaussian_filter
    width = 5
    y, x = np.mgrid[-21:22, -21:22]
    delta = Delta2D(1, x_0, y_0)
    amplitude = 1 / (2 * np.pi * width ** 2)
    gauss = Gaussian2D(amplitude=amplitude, x_mean=x_0, y_mean=y_0,
                       x_stddev=width, y_stddev=width)

    values = delta(x, y)
    values_convolved = gaussian_filter(values, width, mode='constant')
    values_gauss = gauss(x, y)
    assert_allclose(values_convolved, values_gauss, atol=1e-4)


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_template2d():
    filename = ('$GAMMAPY_EXTRA/datasets/catalogs/fermi/Extended_archive_v18'
                '/Templates/HESSJ1841-055.fits')
    template = Template2D.read(filename)
    assert_allclose(template(26.7, 0), 1.1553735159851262)


@pytest.mark.parametrize(('model_class', 'test_parameters'), models_2D)
class TestMorphologyModels(Fittable2DModelTester):
    pass


def test_model_xml_read_write():
    filename = get_pkg_data_filename('data/fermi_model.xml')
    sources = xmltodict.parse(open(filename).read())
    sources = sources['source_library']['source']
    assert sources[0]['@name'] == '3C 273'
    assert sources[0]['spectrum']['parameter'][1]['@name'] == 'Index'
    assert sources[0]['spectrum']['parameter'][1]['@value'] == '-2.1'
