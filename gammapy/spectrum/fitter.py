# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np

__all__ = ['Fitter']


class Fitter(object):
    """Chi^2 fitter for spectral models and flux points.
    - can use asymmetric Gaussian errors
    - allows free and fixed parameters
    - can find pivot energy

    TODO Change model from function to class!
    """

    def __init__(self, data, model, sym=False):
        """@param symm: if true, symmetrized errors (arithmetic mean) are used,
        if false asymmetric errors are used"""
        self.data = data
        self.model = model
        self.sym = sym

        """
        result = (None, None, None, None, None)
        self.popt, self.pcov, self.infodict, self.mesg, self.status = result
        """

    def error(self, par_vals=None):
        """Compute the error for each data point.
        data = data points with asymmetric y errors
        model = function with parameters and constants

        @return: error = vector of errors for each point
        """
        if par_vals != None:
            self.model.pars.set_free_vals(par_vals)
        self.ymodel = self.model(self.data.e)
        self.diff = self.data.f - self.ymodel
        if self.sym:
            self.err = self.data.f_err
        else:
            # where point above model use low error
            # where point below model, use high error
            self.err = np.where(self.diff > 0,
                                self.data.f_low,
                                self.data.f_high)
        self.chi = self.diff / self.err
        return self.chi

    def fit(self):
        """Fit model to data using the asymmetric chi ** 2 fit statistic"""
        from scipy.optimize import leastsq
        x0 = self.model.pars.get_free_vals()
        result = leastsq(func=self.error, x0=x0, full_output=1, ftol=1e-10, xtol=1e-10)
        self.popt, self.pcov, self.infodict, self.mesg, self.status = result

    def chi2(self):
        """Compute chi**2"""
        return (self.chi ** 2).sum()

    def chi2_lin(self):
        """Compute chi2 in the linear approximation using the
        optimized parameter values and covariance matrix from the fit."""
        from scipy import matrix
        from scipy.linalg import inv
        # Make matrices to do linear algebra easily
        p = matrix(self.p)
        popt = matrix(self.popt)
        pcov = matrix(self.pcov)

        pcovinv = inv(pcov)
        self.chi2lin = ((p - popt) * pcovinv * (p - popt).T)[0, 0]
        return self.chi2lin

    def __str__(self):
        try:
            npoints = len(self.chi)
            npar = len(self.popt)
            ndof = npoints - npar
            s = 'sym_fit: {0}\n'.format(self.sym)
            s += 'npar: {0}\n'.format(npar)
            s += 'npoints: {0}\n'.format(npoints)
            s += 'ndof: {0}\n'.format(ndof)
            s += 'chi2: {0}\n'.format(self.chi2())
            s += 'chi2/ndof: {0}\n'.format(self.chi2() / ndof)
            s += 'popt:\n{0}\n'.format(self.popt)
            s += 'pcov:\n{0}\n'.format(self.pcov)
            for i in range(len(self.popt)):
                fmt = 'Parameter {0}: {1:15g} +/- {2:15g}\n'
                s += (fmt.format(i, self.popt[i], np.sqrt(self.pcov[i, i])))
            s += 'status: {0}\n'.format(self.status)
            s += 'nfev: {0}\n'.format(self.infodict['nfev'])
            s += 'chi:\n{0}\n'.format(self.chi)

            # s += 'infodict:\n{0}\n'.format(self.infodict)
            # s += 'mesg:    \n{0}'.format(self.mesg)
        except AttributeError:
            s = 'Not fitted.'
        return s

    def plot(self):
        self.model.plot()
        self.data.plot()
