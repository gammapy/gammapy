"""Sherpa 3D model example.

References:
- http://pysherpa.blogspot.de/2010/06/user-defined-sherpa-model-types-using.html
- http://cxc.harvard.edu/sherpa/threads/user_model/
- http://python4astronomers.github.io/fitting/low-level.html
"""
import numpy as np
from numpy.testing import assert_allclose
from sherpa.models import ArithmeticModel, Parameter
from sherpa.models import Const1D, Const2D
from sherpa.models import Polynom1D, Polynom2D
from sherpa.models import PowLaw1D, NormGauss2D
from sherpa.data import Data1D, Data2D, DataND
from sherpa.stats import Cash
from sherpa.optmethods import LevMar
from sherpa.estmethods import Covariance
from sherpa.fit import Fit


class Const3D(ArithmeticModel):
    """Const 3D model.
    """

    def __init__(self, name='model'):
        self.c = Parameter(name, 'c', 2)

        pars = (self.c,)
        ArithmeticModel.__init__(self, name, pars)

    @staticmethod
    def calc(pars, *args, **kwargs):
        print('args: ', args)
        print('kwargs: ', kwargs)
        c = pars
        return c


class FromScratchModel3D(ArithmeticModel):
    """Toy example of a 3D model.

    f(x, y, e) = (c_x * x) + (c_y * y) + (c_e * e)
    """

    def __init__(self, name='model'):
        self.c_x = Parameter(name, 'c_x', 2)
        self.c_y = Parameter(name, 'c_y', 3)
        self.c_e = Parameter(name, 'c_e', 4)

        pars = (self.c_x, self.c_y, self.c_e)
        ArithmeticModel.__init__(self, name, pars)

    @staticmethod
    def calc(pars, x, y, e):
        c_x, c_y, c_e = pars
        return (c_x * x) + (c_y * y) + (c_e * e)


class CombinedModel3D(ArithmeticModel):
    """Combine a spatial and spectral model into a 3D model.
    """

    def __init__(self, spatial_model, spectral_model, name='model'):
        # Store references to the parts (not really needed?)
        self.spatial_model = spatial_model
        self.spectral_model = spectral_model

        # Concatenate parameters
        pars = spatial_model.pars + spectral_model.pars

        ArithmeticModel.__init__(self, name, pars)

    def calc(self, pars, *args, **kwargs):
        # c_x, c_y, c_e = pars
        # import IPython; IPython.embed()
        print('pars: ', pars)
        print('args: ', args)
        print('kwargs: ', kwargs)

        # TODO: really we should compute the integral flux over the energy bin,
        # and then set the norm of the spatial model so that it corresponds to that flux.

        # spectral_term = self.spatial_model(x, y)
        # spatial_term = self.spectral_model(energy)
        # value = spectral_term * spatial_term
        value = 290

        return value


def test_FromScratchModel3D():
    print('Running test_FromScratchModel3D')
    model = FromScratchModel3D()
    print(model)
    actual = model(20, 30, 40)
    expected = (2 * 20) + (3 * 30) + (4 * 40)
    assert_allclose(actual, expected)


def test_CombinedModel3D():
    print('Running test_CombinedModel3D')
    spatial_model = Polynom2D()
    spatial_model.c = 0
    spatial_model.cx1 = 2
    spatial_model.cy1 = 3
    spectral_model = Polynom1D()
    spectral_model.c0 = 0
    spectral_model.c1 = 4

    model = CombinedModel3D(spatial_model, spectral_model)
    print(model)
    actual = model(20, 30, 40)
    expected = (2 * 20) + (3 * 30) + (4 * 40)
    assert_allclose(actual, expected)


def run_cash_fit(data, model):
    fit = Fit(data=data, model=model, stat=Cash(), method=LevMar(), estmethod=Covariance())
    result = fit.fit()
    result_err = fit.est_errors()

    print('=> data:')
    print(data)
    print('=> model:')
    print(model)
    print('=> fit:')
    print(fit)
    print('=> result:')
    print(result)
    print('=> result_err:')
    print(result_err)


def test_fit_1D():
    x = np.array([1, 2, 10])
    y = np.array([10, 30, 20])
    data = Data1D(name='data', x=x, y=y)
    model = Const1D()
    run_cash_fit(data, model)


def test_fit_2D():
    x0 = np.array([[1, 2, 10], [1, 2, 10]])
    x1 = np.array([[1, 2, 10], [1, 2, 10]])
    y = np.array([[10, 30, 20], [10, 30, 20]])
    data = Data2D(name='data', x0=x0.flatten(), x1=x1.flatten(), y=y.flatten())
    model = Const2D()
    run_cash_fit(data, model)


def test_fit_1D_ND():
    """Same as with Data1D, but using DataND class.
    """
    x = np.array([1, 2, 10])
    y = np.array([10, 30, 20])
    # x0=x0.flatten(), x1=x1.flatten(), y=y.flatten()
    indep = (x, )
    dep = (y, )
    import IPython; IPython.embed()
    data = DataND(name='data', indep=indep, dep=dep)
    model = Const2D()
    run_cash_fit(data, model)


if __name__ == '__main__':
    # test_FromScratchModel3D()
    # test_CombinedModel3D()
    # test_fit_1D()
    # test_fit_2D()
    test_fit_1D_ND()
