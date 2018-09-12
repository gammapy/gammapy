from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import abc

from astropy.utils.misc import InheritDocstrings

from ...extern import six
from .iminuit import fit_iminuit

__all__ = ["Fit"]

log = logging.getLogger(__name__)


class FitMeta(InheritDocstrings, abc.ABCMeta):
    pass


@six.add_metaclass(FitMeta)
class Fit(object):
    """Abstract Fit base class.
    """
    @abc.abstractmethod
    def total_stat(self, parameters):
        """Total likelihood given the current model parameters"""
        pass

    def fit(self, opts_minuit=None):
        """Run the fit

        Parameters
        ----------
        opts_minuit : dict (optional)
            Options passed to `iminuit.Minuit` constructor

        Returns
        -------
        fit_result : dict
            Dictionary with the fit result.
        """
        minuit = fit_iminuit(
            parameters=self._model.parameters,
            function=self.total_stat,
            opts_minuit=opts_minuit,
        )
        self._minuit = minuit

        return {
            'best-fit-model': self._model.copy(),
            'statval': self.total_stat(self._model.parameters),
        }

    def run(self, steps='all', opts_minuit=None):
        """
        Run all fitting steps.

        Parameters
        ----------
        opts_minuit : dict (optional)
            Options passed to `iminuit.Minuit` constructor

        """
        if steps == 'all':
            steps = ['fit']

        result = {}
        if 'fit' in steps:
            result.update(self.fit(opts_minuit=opts_minuit))
        return result

