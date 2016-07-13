# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spectral models for Gammapy.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from ..extern.bunch import Bunch
import astropy.units as u
from . import CountsSpectrum
import numpy as np


__all__ = [
    'SpectralModel',
    'PowerLaw',
    'ExponentialCutoffPowerLaw',
]

# Note: Consider to move stuff from _models_old.py here

class SpectralModel(object):
    """Spectral model base class.

    Derived classes should store their parameters as ``Bunch`` in an instance
    attribute called ``parameters``, see for example
    `~gammapy.spectrum.models.PowerLaw`.
    """
    def __call__(self, energy):
        """Call evaluate method of derived classes"""
        return self.evaluate(energy, **self.parameters)

    def __str__(self):
        """String representation"""
        ss = self.__class__.__name__
        for parname, parval in self.parameters.items():
            ss += '\n{parname} : {parval:.3g}'.format(**locals())
        return ss
    
    def with_uncertainties(self, covariance, axis):
        """Connect model to uncertainties module
        
        This uses the uncertainties packages as explained here
        https://pythonhosted.org/uncertainties/user_guide.html#use-of-a-covariance-matrix

        Examples
        --------
        TODO
        """
        import IPython; IPython.embed()


    def to_sherpa(self, name='default'):
        """Return `~sherpa.models.ArithmeticModel`

        Parameters
        ----------
        name : str, optional
            Name of the sherpa model instance
        """
        import sherpa.models as m
        if isinstance(self, PowerLaw):
            model = m.PowLaw1D('powlaw1d.' + name)
            model.gamma = self.parameters.index.value
        else:
            raise NotImplementedError

        model.ref = self.parameters.reference.to('keV').value
        model.ampl = self.parameters.amplitude.to('cm-2 s-1 keV-1').value

        return model

    @classmethod
    def from_sherpa(cls, model):
        """Create `~gammapy.spectrum.models.SpectrumModel` from
        `~sherpa.models.ArithmeticModel`

        Parameters
        ----------
        model : `~sherpa.models.ArithmeticModel`
            Sherpa model
        """
        from . import SpectrumFit
        pardict = dict(gamma = ['index', u.Unit('')],
                       ref = ['reference', u.keV],
                       ampl = ['amplitude', SpectrumFit.FLUX_FACTOR * u.Unit('cm-2 s-1 keV-1')])
        kwargs = dict()

        for par in model.pars:
            name = par.name
            kwargs[pardict[name][0]] =  par.val * pardict[name][1]

        return cls(**kwargs)

    def to_dict(self):
        """Serialize to dict"""
        retval = dict()

        retval['name'] = self.__class__.__name__
        retval['parameters'] = list()
        for parname, parval in self.parameters.items():
            retval['parameters'].append(dict(name=parname,
                                             val=parval.value,
                                             unit=str(parval.unit)))
        return retval

    @classmethod
    def from_dict(cls, val):
        """Serialize from dict"""
        kwargs = dict()
        for _ in val['parameters']:
            kwargs[_['name']] = _['val'] * u.Unit(_['unit'])
        return cls(**kwargs)

    def plot(self, ax=None, energy_range=[0.1, 10] * u.TeV,
             energy_unit='TeV', flux_unit='cm-2 s-1 TeV-1',
             energy_power=0, n_points=100, **kwargs):
        """Plot `~gammapy.spectrum.SpectralModel` 

        kwargs are forwarded to :func:`~matplotlib.pyplot.errorbar`

        Parameters
        ----------
        ax : `~matplolib.axes`, optional
            Axis
        energy_range : `~astropy.units.Quantity`
            Plot range
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
        flux_unit : str, `~astropy.units.Unit`, optional
            Unit of the flux axis
        energy_power : int
            Power of energy to multiply flux axis with
        n_points : int
            Number of evaluation nodes

        Returns
        -------
        ax : `~matplolib.axes`, optional
            Axis
        """

        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        x_min = np.log10(energy_range[0].to('TeV').value)
        x_max = np.log10(energy_range[1].to('TeV').value)
        xx = np.logspace(x_min, x_max, n_points) * u.Unit('TeV')
        yy = self(xx)
        x = xx.to(energy_unit).value
        y = yy.to(flux_unit).value
        y = y * np.power(x, energy_power)
        flux_unit = u.Unit(flux_unit) * np.power(u.Unit(energy_unit), energy_power)
        ax.plot(x, y, **kwargs)
        ax.set_xlabel('Energy [{}]'.format(energy_unit))
        ax.set_ylabel('Flux [{}]'.format(flux_unit))
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')
        return ax


class PowerLaw(SpectralModel):
    r"""Spectral power-law model.
    
    .. math:: 

        F(E) = F_0 \cdot \left( \frac{E}{E_0} \right)^{-\Gamma}

    Parameters
    ----------
    index : float, `~astropy.units.Quantity`
        :math:`\Gamma`
    amplitude : float, `~astropy.units.Quantity` 
        :math:`F_0`
    reference : float, `~astropy.units.Quantity` 
        :math:`E_0`
    """
    def __init__(self, index, amplitude, reference):
        self.parameters = Bunch(index = index,
                                amplitude = amplitude.to('cm-2 s-1 TeV-1'),
                                reference = reference.to('TeV'))
        
    @staticmethod
    def evaluate(energy, index, amplitude, reference):
        return amplitude * ( energy / reference ) ** (-1 * index)

    def integral(self, emin, emax):
        """Integrate using analytic formula"""
        pars = self.parameters 
        
        val = -1 * pars.index + 1
        prefactor = pars.amplitude * pars.reference / val
        upper = (emax / pars.reference) ** val
        lower = (emin / pars.reference) ** val

        return prefactor * (upper - lower)


class ExponentialCutoffPowerLaw(SpectralModel):
    """Spectral exponential cutoff power-law model.
    """

