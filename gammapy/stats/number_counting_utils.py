# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

__all__ = ['NumberCountingUtils']

# TODO: If useful, rewrite as functions and with proper tests and docstrings.


class NumberCountingUtils(object):
    """Formulas for computing tail probabilities of on/off
    observations in RooStats::NumberCountingUtils in ROOT.

    @see: The ROOT documentation mentions the relevant papers:
    http://root.cern.ch/root/html/RooStats__NumberCountingUtils.html

    @note To me it is not quite clear how these formulae relate
    to Li & Ma."""

    @staticmethod
    def _f(self, main, aux, tau):
        """Little wrapper for incomplete beta function computation
        because the argument order is different from ROOT and
        there are problems with certain sets of arguments."""
        from scipy.special import betainc
        a = main
        b = aux + 1
        x = 1. / (1. + tau)
        result = betainc(a, b, x)
        return result

    @staticmethod
    def BinomialExpP(self, signalExp, backgroundExp, relativeBkgUncert):
        """Expected P-value for s=0 in a ratio of Poisson means.
        Here the background and its uncertainty are provided directly and
        assumed to be from the double Poisson counting setup described in the
        BinomialWithTau functions.
        Normally one would know tau directly, but here it is determiend from
        the background uncertainty.  This is not strictly correct, but a useful
        approximation."""
        # SIDE BAND EXAMPLE
        # See Eqn. (19) of Cranmer and pp. 36-37 of Linnemann.
        # 150 total events in signalExp region, 100 in sideband of equal size
        mainInf = signalExp + backgroundExp
        tau = 1. / backgroundExp / (relativeBkgUncert * relativeBkgUncert)
        auxiliaryInf = backgroundExp * tau
        return self._f(mainInf, auxiliaryInf, tau)

    @staticmethod
    def BinomialWithTauExpP(self, signalExp, backgroundExp, tau):
        """Expected P-value for s=0 in a ratio of Poisson means.
        Based on two expectations, a main measurement that might have signal
        and an auxiliarly measurement for the background that is signal free.
        The expected background in the auxiliary measurement is a factor
        tau larger than in the main measurement."""
        mainInf = signalExp + backgroundExp
        auxiliaryInf = backgroundExp * tau
        return self._f(mainInf, auxiliaryInf, tau)

    @staticmethod
    def BinomialObsP(self, mainObs, backgroundObs, relativeBkgUncert):
        """"P-value for s=0 in a ratio of Poisson means.
        Here the background and its uncertainty are provided directly and
        assumed to be from the double Poisson counting setup.
        Normally one would know tau directly, but here it is determiend from
        the background uncertainty.  This is not strictly correct, but a useful
        approximation."""
        tau = 1. / backgroundObs / (relativeBkgUncert * relativeBkgUncert)
        auxiliaryInf = backgroundObs * tau
        # SIDE BAND EXAMPLE
        # See Eqn. (19) of Cranmer and pp. 36-37 of Linnemann.
        return self._f(mainObs, auxiliaryInf, tau)

    @staticmethod
    def BinomialWithTauObsP(self, mainObs, auxiliaryObs, tau):
        """"P-value for s=0 in a ratio of Poisson means.
        Based on two observations, a main measurement that might have signal
        and an auxiliarly measurement for the background that is signal free.
        The expected background in the auxiliary measurement is a factor
        tau larger than in the main measurement."""
        # SIDE BAND EXAMPLE
        # See Eqn. (19) of Cranmer and pp. 36-37 of Linnemann.
        return self._f(mainObs, auxiliaryObs, tau)
