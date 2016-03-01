"""Example of how to use the Sherpa low-level API for a MWL fit.

This is a super-simple toy example.

But it can be extended for a full MWL analysis similar to 3ML,
where e.g. radio / X-ray flux points are used,
Fermi ScienceTools are called for Fermi
and gammapy.spectrum for IACT analysis.
"""
import IPython

from astropy.units import Quantity
import astropy.units as u

from sherpa.data import Data1D, DataSimulFit
from sherpa.models import PowLaw1D, LogParabola, SimulFitModel
from sherpa.stats import Chi2DataVar
from sherpa.optmethods import LevMar
from sherpa.fit import Fit

from gammapy.datasets import load_crab_flux_points
from gammapy.utils.energy import Energy
from gammapy.extern.bunch import Bunch


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


def mwl_fit_low_level():
    """Use high-level Sherpa API.

    Low-level = no session, classes.

    Example: http://python4astronomers.github.io/fitting/low-level.html
    """
    fermi_data = FermiData().sherpa_data
    hess_data = IACTData().sherpa_data

    spec_model = PowLaw1D('spec_model')
    # spec_model = LogParabola('spec_model')
    spec_model.gamma = 2
    spec_model.ampl = 1e-11

    data = DataSimulFit(name='global_data', datasets=[fermi_data, hess_data])
    model = SimulFitModel(name='global_model', parts=[model, model])
    stat = Chi2DataVar()
    method = LevMar()
    fit = Fit(data=data, model=model, stat=stat, method=method)
    fit.fit()

    return Bunch(results=fit.result, model=spec_model)


def mwl_fit_high_level():
    """Use high-level Sherpa API.

    High-level = session and convenience functions

    Example: http://cxc.harvard.edu/sherpa/threads/simultaneous/
    Example: http://python4astronomers.github.io/fitting/spectrum.html
    """
    import sherpa.ui as ui

    fermi_data = FermiData()
    ui.load_arrays(fermi_data.name, fermi_data.x, fermi_data.y, fermi_data.staterror)

    iact_data = IACTData()
    ui.load_arrays(iact_data.name, iact_data.x, iact_data.y, iact_data.staterror)

    spec_model = ui.logparabola.spec_model

    ui.set_source(fermi_data.name, spec_model)
    ui.set_source(iact_data.name, spec_model)

    ui.notice(lo=1e-3, hi=None)
    ui.fit()

    return Bunch(results=ui.get_fit_results(), model=spec_model)


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
    # results = mwl_fit_low_level()
    results = mwl_fit_high_level()

    print_results(results)
    plot_result(results)
