"""Example of how to use the Sherpa low-level API for a MWL fit.

This is a super-simple toy example.

But it can be extended for a full MWL analysis similar to 3ML,
where e.g. radio / X-ray flux points are used,
Fermi ScienceTools are called for Fermi
and gammapy.spectrum for IACT analysis.
"""
import numpy as np
from astropy.units import Quantity
import astropy.units as u

from sherpa.data import Data1D, DataSimulFit, Data, BaseData
from sherpa.models import PowLaw1D, LogParabola, SimulFitModel
from sherpa.stats import Chi2DataVar, Likelihood
from sherpa.optmethods import LevMar
from sherpa.fit import Fit

from gammapy.datasets import load_crab_flux_points
from gammapy.utils.energy import Energy


class FermiStat(Likelihood):
    def __init__(self, name='fermi'):
        Likelihood.__init__(self, name)

    @staticmethod
    def calc_stat(data, model, staterror=None, syserror=None, weight=None, bkg=None):
        # IPython.embed(); 1/0
        # print(len(data))
        fvec = np.power((data - model) / staterror, 2)
        stat = 2.0 * np.sum(fvec)
        # print(fvec, stat, data, model)
        return stat, fvec


class FermiData(object):
    def __init__(self):
        table = load_crab_flux_points(component='nebula')
        table = table[table['paper'] == 'fermi_33months']

        self.name = 'fermi'
        self.x = Quantity(table['energy']).to('TeV').value
        self.y = Quantity(table['energy_flux']).to('erg cm-2 s-1').value
        self.staterror = Quantity(table['energy_flux_err']).to('erg cm-2 s-1').value

    @property
    def sherpa_data(self):
        return Data1D(name=self.name, x=self.x, y=self.y, staterror=self.staterror)


class FermiDataShim(Data):
    """A shim to interface Sherpa to the Fermi science tools.
    """

    def __init__(self, name='fermi_data_shim'):
        # TODO: need dummy? simplify?
        self.dep = np.array([42])  # Dummy
        self.indep = np.array([42])  # Dummy
        BaseData.__init__(self)

    def eval_model(self, modelfunc):
        print('hello from FermiDataShim.eval_model')
        print(modelfunc)
        raise NotImplementedError()

    def eval_model_to_fit(self, modelfunc):
        pars = [(par.name, par.val) for par in modelfunc.pars]
        # IPython.embed();
        return np.atleast_1d(FermiDataShim.fermi_loglike(pars))

    @staticmethod
    def fermi_loglike(pars):
        # TODO: import fermipy here
        # TODO: compute loglike for fermi using the Fermi science tools via fermipy
        print('hello from FermiDataShim.eval_model_to_fit')
        print(pars)
        loglike = 42.2
        return loglike


class FermiStatShim(Likelihood):
    """A shim to interface Sherpa to the Fermi science tools
    """

    def __init__(self, name='fermi'):
        Likelihood.__init__(self, name)

    @staticmethod
    def calc_stat(data, model, staterror=None, syserror=None, weight=None, bkg=None):
        """We pass the logL value in `model` here.
        """
        print('hello from FermiStatShim.calc_stat')
        print(model, model)
        return float(model), model


class IACTData(object):
    def __init__(self):
        table = load_crab_flux_points(component='nebula')
        table = table[table['paper'] == 'hess']

        self.name = 'iact'
        self.x = Quantity(table['energy']).to('TeV').value
        self.y = Quantity(table['energy_flux']).to('erg cm-2 s-1').value
        self.staterror = Quantity(table['energy_flux_err']).to('erg cm-2 s-1').value

    @property
    def sherpa_data(self):
        return Data1D(name=self.name, x=self.x, y=self.y, staterror=self.staterror)


def mwl_fit_low_level_calling_fermi():
    """Example how to do a Sherpa model fit,
    but use the Fermi ScienceTools to evaluate
    the likelihood for the Fermi dataset.
    """

    spec_model = LogParabola('spec_model')
    spec_model.c1 = 0.5
    spec_model.c2 = 0.2
    spec_model.ampl = 5e-11

    model = spec_model

    data = FermiDataShim()
    stat = FermiStatShim()
    method = LevMar()
    fit = Fit(data=data, model=model, stat=stat, method=method)
    result = fit.fit()

    return dict(results=result, model=spec_model)


def mwl_fit_low_level():
    """Use high-level Sherpa API.

    Low-level = no session, classes.

    Example: http://python4astronomers.github.io/fitting/low-level.html
    """
    fermi_data = FermiData().sherpa_data
    hess_data = IACTData().sherpa_data

    # spec_model = PowLaw1D('spec_model')
    spec_model = LogParabola('spec_model')
    spec_model.c1 = 0.5
    spec_model.c2 = 0.2
    spec_model.ampl = 5e-11

    data = DataSimulFit(name='global_data', datasets=[fermi_data, hess_data])
    # TODO: Figure out how to notice using the low-level API
    # data.notice(mins=1e-3, maxes=None, axislist=None)
    model = SimulFitModel(name='global_model', parts=[spec_model, spec_model])
    stat = FermiStat()
    method = LevMar()
    fit = Fit(data=data, model=model, stat=stat, method=method)
    result = fit.fit()

    # IPython.embed()
    return dict(results=result, model=spec_model)


def mwl_fit_high_level():
    """Use high-level Sherpa API.

    High-level = session and convenience functions

    Example: http://cxc.harvard.edu/sherpa/threads/simultaneous/
    Example: http://python4astronomers.github.io/fitting/spectrum.html
    """
    import sherpa.ui as ui

    fermi_data = FermiData()
    ui.load_arrays(fermi_data.name, fermi_data.x, fermi_data.y, fermi_data.staterror)

    ui.load_user_stat('fermi_stat', FermiStat.calc_stat, FermiStat.calc_staterror)
    # TODO: is there a good way to get the stat??
    # ui.get_stat('fermi_stat')
    # fermi_stat = ui._session._get_stat_by_name('fermi_stat')
    ui.set_stat(fermi_stat)
    # IPython.embed()


    iact_data = IACTData()
    ui.load_arrays(iact_data.name, iact_data.x, iact_data.y, iact_data.staterror)

    spec_model = ui.logparabola.spec_model
    spec_model.c1 = 0.5
    spec_model.c2 = 0.2
    spec_model.ampl = 5e-11

    ui.set_source(fermi_data.name, spec_model)
    ui.set_source(iact_data.name, spec_model)

    ui.notice(lo=1e-3, hi=None)

    # IPython.embed()
    ui.fit()

    return dict(results=ui.get_fit_results(), model=spec_model)


def print_results(results):
    print(results)
    print(results.model)
    print(results.results)


def plot_result(results):
    import matplotlib.pyplot as plt

    # Plot model
    energy = Energy.equal_log_spacing(1e-3, 1e2, nbins=100, unit='TeV')
    flux = results['model'](energy.to('TeV').value)
    flux = Quantity(flux, 'erg cm-2 s-1')
    plt.plot(energy, flux, label='model')

    # Plot data points
    data = FermiData()
    plt.errorbar(x=data.x, y=data.y, yerr=data.staterror, fmt='ro', label='data')

    data = IACTData()
    plt.errorbar(x=data.x, y=data.y, yerr=data.staterror, fmt='ro', label='data')

    # Make it pretty
    plt.loglog()
    plt.xlabel('Energy (TeV)')
    plt.ylabel('E^2 * dN / dE (TeV cm^-2 s^-1)')
    plt.legend()
    filename = 'example_sherpa_mwl.png'
    print('Writing {}'.format(filename))
    plt.savefig(filename)


if __name__ == '__main__':
    # results = mwl_fit_high_level()
    # results = mwl_fit_low_level()
    results = mwl_fit_low_level_calling_fermi()

    print_results(results)
    plot_result(results)
