# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from sherpa.models import ArithmeticModel, Parameter, modelCacher1d
from sherpa.data import DataND, BaseData
from sherpa.utils.err import DataErr, NotImplementedErr
from sherpa.utils import SherpaFloat, NoNewAttributesAfterInit, \
    print_fields, create_expr, calc_total_error, bool_cast, \
    filter_bins, interpolate, linear_interp


# This class was copy pasted from sherpa.data.Data2D and modified to account
# for the third dimension
# it is set up to integrate over the energy (first) axis, but not all class
# methods are adapted to that yet (TODO)

class Data3D(DataND):
    """Sherpa 3-D data set.
    """

    def _set_mask(self, val):
        DataND._set_mask(self, val)
        try:
            self._lo = self.apply_filter(self.xlo)
            self._hi = self.apply_filter(self.xhi)
            self._x1 = self.apply_filter(self.x1)
            self._x2 = self.apply_filter(self.x2)
        except DataErr:
            self._hi = self.xhi
            self._lo = self.xlo
            self._x1 = self.x1
            self._x2 = self.x2

    mask = property(DataND._get_mask, _set_mask,
                    doc='Mask array for dependent variable')

    def __init__(self, name, xlo, xhi, x1, x2, y, shape=None, staterror=None,
                 syserror=None):
        self._lo = xlo
        self._hi = xhi
        self._x1 = x1
        self._x2 = x2
        BaseData.__init__(self)

    def get_indep(self, filter=False):
        filter = bool_cast(filter)
        if filter:
            return (self._lo, self._hi, self._x1, self._x2)
        return (self.xlo, self.xhi, self.x1, self.x2)

    def get_x0(self, filter=False):
        return self.get_indep(filter)[0]

    def get_x1(self, filter=False):
        return self.get_indep(filter)[1]

    def get_x2(self, filter=False):
        return self.get_indep(filter)[2]

    def get_axes(self):
        self._check_shape()
        # FIXME: how to filter an axis when self.mask is size of self.y?
        return (np.arange(self.shape[1]) + 1, np.arange(self.shape[0]) + 1, np.arange(self.shape[0]) + 1)

    def get_dims(self, filter=False):
        # self._check_shape()
        if self.shape is not None:
            return self.shape[::-1]
        return (len(self.get_x0(filter)), len(self.get_x1(filter)), len(self.get_x2(filter)))

    def get_filter_expr(self):
        return ''

    get_filter = get_filter_expr

    def _check_shape(self):
        if self.shape is None:
            raise DataErr('shape', self.name)

    def get_max_pos(self, dep=None):
        if dep is None:
            dep = self.get_dep(True)
        x0 = self.get_x0(True)
        x1 = self.get_x1(True)
        x2 = self.get_x2(True)

        pos = np.asarray(np.where(dep == dep.max())).squeeze()
        if pos.ndim == 0:
            pos = int(pos)
            return (x0[pos], x1[pos], x2[pos])

        return [(x0[index], x1[index], x2[index]) for index in pos]

    def get_img(self, yfunc=None):
        self._check_shape()
        y_img = self.get_y(False, yfunc)
        if yfunc is not None:
            y_img = (y_img[0].reshape(*self.shape),
                     y_img[1].reshape(*self.shape))
        else:
            y_img = y_img.reshape(*self.shape)
        return y_img

    def get_imgerr(self):
        self._check_shape()
        err = self.get_error()
        if err is not None:
            err = err.reshape(*self.shape)
        return err

    def notice(self, x0lo=None, x0hi=None, x1lo=None, x1hi=None, x2lo=None, x2hi=None, ignore=False):
        BaseData.notice(self, (x0lo, x1lo, x2lo), (x0hi, x1hi, x2hi), self.get_indep(),
                        ignore)


class CombinedModel3D(ArithmeticModel):
    """
    Combined spatial and spectral 3D model.
    """

    def __init__(self, name='cube-model', spatial_model=None, spectral_model=None):
        self.spatial_model = spatial_model
        self.spectral_model = spectral_model

        # Fix spectral ampl parameter
        spectral_model.ampl = 1
        spectral_model.ampl.freeze()

        pars = []
        for _ in spatial_model.pars + spectral_model.pars:
            setattr(self, _.name, _)
            pars.append(_)

        self._spatial_pars = slice(0, len(spatial_model.pars))
        self._spectral_pars = slice(len(spatial_model.pars), len(pars))
        ArithmeticModel.__init__(self, name, pars)

    def calc(self, pars, elo, ehi, x, y):
        _spatial = self.spatial_model.calc(pars[self._spatial_pars], x, y)
        _spectral = self.spectral_model.calc(pars[self._spectral_pars], elo, ehi)
        return _spatial * _spectral
