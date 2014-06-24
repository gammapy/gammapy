# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy import sqrt
from .models import PowerLaw

__all__ = ['FluxPoints', 'FluxPointCalculator']


class FluxPoints(object):
    """List of flux points.

    TODO: Implement. Document.
    """
    def from_dict(self, data):
        pass

    def from_ascii(self, data):
        pass


class FluxPointCalculator(object):
    """Compute differential flux points for given integral flux points.

    TODO: I think it would be better to change this into functions.
    """
    def __init__(self, model=None,
                 xmethod="LogCenter",
                 ymethod='ExcessEqualsExpected'):
        self.xmethods = dict(LogCenter=self._XLogCenter,
                             ModelWeightedMean=self._XModelWeightedMean)
        self.ymethods = dict(ExcessEqualsExpected=self._YExcessEqualsExpected,
                             PowerLaw=self._YPowerLaw)
        if model is None:
            raise NotImplementedError
            model = PowerLaw()
        self.model = model
        if not xmethod in self.xmethods:
            raise ValueError('Unknown xmethod: {0}'.format(xmethod))
        self.xmethod = xmethod
        if not ymethod in self.ymethods:
            raise ValueError('Unknown ymethod: {0}'.format(ymethod))
        self.ymethod = ymethod

    def calc_xy(self, yint, xmin, xmax):
        """Compute differential flux points (x,y)"""
        self.x = self.calc_x(yint, xmin, xmax)
        self.y = self.calc_y(self.x, yint, xmin, xmax)
        return self.x, self.y

    def calc_x(self, yint, xmin, xmax):
        """Compute x position of differential flux point"""
        self.yint = np.asarray(yint)
        self.xmin = np.asarray(xmin)
        self.xmax = np.asarray(xmax)
        return self.xmethods[self.xmethod]()

    def calc_y(self, x, yint, xmin, xmax):
        """Compute y position of differential flux point"""
        self.x = np.asarray(x)
        self.yint = np.asarray(yint)
        self.xmin = np.asarray(xmin)
        self.xmax = np.asarray(xmax)
        return self.ymethods[self.ymethod]()

    def _XLogCenter(self):
        """The LogCenter method to compute X"""
        self.x = sqrt(self.xmin * self.xmax)
        return self.x

    def _XModelWeightedMean(self):
        """The ModelWeightedMean method to compute X"""
        raise NotImplementedError

    def _YExcessEqualsExpected(self):
        """The ExcessEqualsExpected method to compute Y
        y / yint = y_model / yint_model"""
        yint_model = self.model.integral(self.xmin, self.xmax)
        y_model = self.model(self.x)
        return y_model * (self.yint / yint_model)

    def _YPowerLaw(self):
        """The PowerLaw method to compute Y"""
        # Formula taken from
        # http://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/python_tutorial.html
        e1 = self.xmin / self.x
        e2 = self.xmax / self.x
        g = self.model.pars[1].value
        temp = e1 ** (1 - g) - e2 ** (1 - g)
        self.y = self.yint / self.x / temp / (g - 1)
        return self.y
