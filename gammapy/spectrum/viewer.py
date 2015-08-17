# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Implement an SED viewer.
"""
from __future__ import print_function, division
import numpy as np
import astropy.units as u
from astropy.units import Quantity
from .utils import energy_bounds_equal_log_spacing

__all__ = ['SEDViewer']


def plot_photon_sed(spectrum,
                    distance=Quantity(1, 'kpc'),
                    energy_range=Quantity([1e6, 1e12], 'eV'),
                    **kwargs):
    """Plot photon spectral energy distribution (SED).

    Parameters
    ----------
    spectrum : callable
        Photon spectrum function
    distance : `~astropy.units.Quantity`
        Distance to the source
    energy_range : `~astropy.units.Quantity`
        Energy range (Quantity of length 2)
    kwargs :
        Extra keyword arguments are passed to `matplotlib.pyplot.plot`.
    """
    import matplotlib.pyplot as plt

    energy_unit = 'MeV'
    sed_unit = 'erg cm^-2 s^-1 kpc^-2'

    energy = energy_bounds_equal_log_spacing(energy_range, bins=100)

    sed = spectrum.sed(energy, distance=distance)

    x = energy.to(energy_unit).value
    y = sed.to(sed_unit).value
    plt.plot(x, y, **kwargs)
    plt.xlabel('Energy ({0})'.format(str(energy_unit)))
    plt.ylabel('Brightness (E^2 * F) ({1})'.format(str(sed_unit)))
    plt.loglog()


def plot_data(source='crab'):
    import matplotlib.pyplot as plt
    from ..datasets import load_crab_flux_points
    table = load_crab_flux_points(component='nebula')
    x = table['energy'].data
    y = table['energy_flux'].data
    yerr_lo = table['energy_flux_err_lo'].data
    yerr_hi = table['energy_flux_err_hi'].data
    plt.errorbar(x, y, yerr=(yerr_lo, yerr_hi), fmt='o', label=source)
    plt.loglog()


def get_spectrum():
    from gammafit.models import ExponentialCutoffPowerLaw
    amplitude = Quantity(1e-12, 'eV^-1')
    e_0 = 1 * u.TeV
    alpha = 1.5
    e_cutoff = 100 * u.GeV
    proton_spectrum = ExponentialCutoffPowerLaw(amplitude, e_0, alpha, e_cutoff)

    amplitude = Quantity(1e-12, 'eV^-1')
    e_0 = 1 * u.TeV
    alpha = 1.8
    e_cutoff = 150 * u.GeV
    electron_spectrum = ExponentialCutoffPowerLaw(amplitude, e_0, alpha, e_cutoff)

    return electron_spectrum


def run():
    from gammafit.models import Synchrotron, InverseCompton, PionDecay
    proton_spectrum = 'TODO'
    distance = 1 * u.kpc # Distance to source
    # Assumed nH = 1 cm^-3
    pion_spectrum = PionDecay(proton_spectrum)
    plot_photon_sed(pion_spectrum, distance=distance, label='pion')


class SEDViewer(object):
    """Spectral energy distribution (SED) interactive viewer.

    This creates a simple interactive viewer for SEDs using IPython widgets.
    It will only run in the IPython notebook via::

    >>> %matplotlib inline
    >>> from gammapy.spectrum import SEDViewer
    >>> SEDViewer()

    Parameters
    ----------
    debug : bool
        Debug this viewer?
    """
    SOURCES = ['Crab Nebula', 'RX J1713.7-3946', 'HESS J1640-465']
    DEFAULT_B = Quantity(1, 'mG')
    DEFAULT_n_H = Quantity(1, 'cm^-3')

    def __init__(self, debug=False):
        self.debug = debug

    def widget(self, display='plot'):
        """Create a widget representing the SEDViewer.

        Parameters
        ----------
        display : {'plot', 'args'}
            What to display. 'args' is only useful for debugging.
        """
        from IPython.html.widgets import interactive
        from IPython.html import widgets
        if display == 'plot':
            function = self.show_plot
        elif display == 'args':
            function = self.show_args
        else:
            raise ValueError('Invalid value for `display`: {0}'.format(display))

        # This defines the SED viewer controls.
        # The IPython `interactive` function inspects the arguments and
        # creates a `widgets.ContainerWidget` for us.
        widget = interactive(function,
                             source=self.SOURCES,
                             distance=(1, 10),
                             )
        return widget

    def show_plot(self, **kwargs):
        distance = kwargs['distance']
        import matplotlib.pyplot as plt
        x = np.linspace(0, 10, 100)
        y = np.sin(a * x)
        plt.plot(x, y)
        plt.show()

    def show_args(self, **kwargs):
        from IPython.display import display, HTML
        s = '<h3>Arguments:</h3><table>\n'
        for k, v in kwargs.items():
            s += '<tr><td>{0}</td><td>{1}</td></tr>\n'.format(k, v)
        s += '</table>'
        display(HTML(s))
