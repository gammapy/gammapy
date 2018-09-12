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
    def _total_stat(self, parameters):
        """Total likelihood given the current model parameters"""
        pass

    @property
    def total_stat(self):
        """Total likelihood given the current model parameters"""
        return self._total_stat(self._model.parameters)

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
            function=self._total_stat,
            opts_minuit=opts_minuit,
        )
        self._minuit = minuit

        return {
            'best_fit_model': self._model.copy(),
            'statval': self.total_stat,
        }
