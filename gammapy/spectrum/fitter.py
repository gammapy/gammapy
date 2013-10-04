# Licensed under a 3-clause BSD style license - see LICENSE.rst
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
            s = 'sym_fit: %s\n' % self.sym
            s += 'npar: %s\n' % npar
            s += 'npoints: %s\n' % npoints
            s += 'ndof: %s\n' % ndof
            s += 'chi2: %s\n' % self.chi2()
            s += 'chi2/ndof: %s\n' % (self.chi2() / ndof)
            s += 'popt:\n%s\n' % self.popt
            s += 'pcov:\n%s\n' % self.pcov
            for i in range(len(self.popt)):
                s += ('Parameter %d: %15g +/- %15g\n'
                      % (i, self.popt[i], np.sqrt(self.pcov[i, i])))
            s += 'status: %s\n' % self.status
            s += 'nfev: %s\n' % self.infodict['nfev']
            s += 'chi:\n%s\n' % self.chi

            # s += 'infodict:\n%s\n' % self.infodict
            # s += 'mesg:    \n%s' % self.mesg
        except AttributeError:
            s = 'Not fitted.'
        return s

    def plot(self):
        self.model.plot()
        self.data.plot()
