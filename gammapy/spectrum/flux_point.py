# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Differential and integral flux point computations."""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.table import Table
from astropy.units import Unit, Quantity
from gammapy.utils.energy import EnergyBounds
from ..spectrum.powerlaw import power_law_flux

__all__ = [
    'DifferentialFluxPoints',
    'IntegralFluxPoints',
]


class DifferentialFluxPoints(Table):
    """Differential flux points table

    Column names: ENERGY, ENERGY_ERR_HI, ENERGY_ERR_LO,
    DIFF_FLUX, DIFF_FLUX_ERR_HI, DIFF_FLUX_ERR_LO

    For a complete documentation see :ref:`gadf:flux-points`
    """
    @classmethod
    def from_fitspectrum_json(cls, filename):
        import json
        with open(filename) as fh:
            data = json.load(fh)

        # TODO : Adjust column names

        flux_points = Table(data=data['flux_graph']['bin_values'], masked=True)
        flux_points['energy'].unit = 'TeV'
        flux_points['energy'].name = 'ENERGY'
        flux_points['energy_err_hi'].unit = 'TeV'
        flux_points['energy_err_lo'].unit = 'TeV'
        flux_points['flux'].unit = 'cm-2 s-1 TeV-1'
        flux_points['flux_err_hi'].unit = 'cm-2 s-1 TeV-1'
        flux_points['flux_err_lo'].unit = 'cm-2 s-1 TeV-1'

        return cls(flux_points)

    @classmethod
    def from_3fgl(cls, sourcename):
        """Get differential fluxpoints for a 3FGL source

        Parameters
        ----------
        sourcename : str
            3FGL source name
        """
        from gammapy.catalog import source_catalogs
        cat_3fgl = source_catalogs['3fgl']
        source = cat_3fgl[sourcename]
        ebounds = EnergyBounds([100, 300, 1000, 3000, 10000, 100000], 'MeV')
        fluxkeys = ['nuFnu100_300', 'nuFnu300_1000', 'nuFnu1000_3000', 'nuFnu3000_10000', 'nuFnu10000_100000']
        temp_fluxes = [source.data[_] for _ in fluxkeys]

        #fluxerrkeys = ['Unc_Flux100_300', 'Unc_Flux300_1000', 'Unc_Flux1000_3000', 'Unc_Flux3000_10000', 'Unc_Flux10000_100000']
        # For now take upper error as symmetric error
        #temp_fluxes_err = [source.data[_][1] for _ in fluxerrkeys]

        diff_fluxes = Quantity(temp_fluxes, 'erg cm-2 s-1')
        #int_fluxes_err = Quantity(temp_fluxes_err, 'cm-2 s-1')

        return cls


    def plot(self, ax=None, energy_unit='TeV',
             flux_unit='cm-2 s-1 TeV-1', energy_power=0, **kwargs):
        """Plot spectral points

        kwargs are forwarded to :func:`~matplotlib.pyplot.errorbar`

        Parameters
        ----------
        ax : `~matplolib.axes`, optional
            Axis
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
        flux_unit : str, `~astropy.units.Unit`, optional
            Unit of the flux axis
        energy_power : int
            Power of energy to multiply flux axis with

        Returns
        -------
        ax : `~matplolib.axes`, optional
            Axis
        """
        import matplotlib.pyplot as plt

        kwargs.setdefault('fmt', 'o')
        ax = plt.gca() if ax is None else ax
        x = self['energy'].quantity.to(energy_unit).value
        y = self['flux'].quantity.to(flux_unit).value
        yh = self['flux_err_hi'].quantity.to(flux_unit).value
        yl = self['flux_err_lo'].quantity.to(flux_unit).value
        y, yh, yl = np.asarray([y, yh, yl]) * np.power(x, energy_power)
        flux_unit = Unit(flux_unit) * np.power(Unit(energy_unit), energy_power)
        ax.errorbar(x, y, yerr=(yl, yh), **kwargs)
        ax.set_xlabel('Energy [{}]'.format(energy_unit))
        ax.set_ylabel('Flux [{}]'.format(flux_unit))
        return ax


class IntegralFluxPoints(Table):
    """Integral flux points table

    Column names: ENERGY_MIN, ENERGY_MAX, INT_FLUX, INT_FLUX_ERR_HI, INT_FLUX_ERR_LO

    For a complete documentation see :ref:`gadf:flux-points`
    """

    @classmethod
    def from_arrays(cls, ebounds, int_flux, int_flux_err = None):
        """Create flux points table given some numpy arrays"""
        t = Table()
        ebounds = EnergyBounds(ebounds)
        int_flux = Quantity(int_flux)
        if not int_flux.unit.is_equivalent('cm-2 s-1'):
            raise ValueError('Flux (unit {}) not an integrated flux'.format(int_flux.unit))
        t['ENERGY_MIN'] = ebounds.lower_bounds
        t['ENERGY_MAX'] = ebounds.upper_bounds
        t['INT_FLUX'] = int_flux
        t['INT_FLUX_ERR_HI'] = int_flux_err
        t['INT_FLUX_ERR_LO'] = int_flux_err
        return cls(t)

    @classmethod
    def from_3fgl(cls, sourcename):
        """Get integral fluxpoints for a 3FGL source

        Parameters
        ----------
        sourcename : str
            3FGL source name
        """
        from gammapy.catalog import source_catalogs
        cat_3fgl = source_catalogs['3fgl']
        source = cat_3fgl[sourcename]
        ebounds = EnergyBounds([100, 300, 1000, 3000, 10000, 100000], 'MeV')
        fluxkeys = ['Flux100_300', 'Flux300_1000', 'Flux1000_3000', 'Flux3000_10000', 'Flux10000_100000']
        temp_fluxes = [source.data[_] for _ in fluxkeys]

        fluxerrkeys = ['Unc_Flux100_300', 'Unc_Flux300_1000', 'Unc_Flux1000_3000', 'Unc_Flux3000_10000', 'Unc_Flux10000_100000']
        # For now take upper error as symmetric error
        temp_fluxes_err = [source.data[_][1] for _ in fluxerrkeys]

        int_fluxes = Quantity(temp_fluxes, 'cm-2 s-1')
        int_fluxes_err = Quantity(temp_fluxes_err, 'cm-2 s-1')

        return cls.from_arrays(ebounds, int_fluxes, int_fluxes_err)

    @property
    def ebounds(self):
        """Energy bounds"""
        return EnergyBounds.from_lower_and_upper_bounds(
            self['ENERGY_MIN'], self['ENERGY_MAX'])

    def compute_differential_flux_points(self, x_method='lafferty',
                                         y_method='power_law', model=None,
                                         spectral_index=None, energy_table=None):
        """Creates differential flux points table from integral flux points table.

        TODO : Put this into the docs
        - Flux point energy computation method either Lafferty & Wyatt
        model-based positioning, log bin center positioning or user-defined
        `~astropy.table.Table` positioning using column heading ['ENERGY'].
        - Flux computation method assuming PowerLaw or user defined model function.
        - Spectral index if default power law model is used. Either a float
        or array_like (in which case, energy_min, energy_max and int_flux
        must be floats to avoid ambiguity)

        Parameters
        ----------
        x_method : {'lafferty', 'log_center', 'table'}
            Flux point energy computation method
        y_method : {'power_law', 'model'}
            Flux computation method
        model : callable
            User-defined model function
        spectral_index : float, array_like
            Spectral index
        energy_table : `astropy.table.Table`
            Flux point energy table

        Returns
        -------
        differential_flux_points : `~gammapy.spectrum.DifferentialFluxPoints`

        Notes
        -----
        For usage, see this tutorial: :ref:`tutorials-flux_point`.
        """

        # Work with fixed units internally
        energy_min = self['ENERGY_MIN'].to('TeV').value
        energy_max = self['ENERGY_MAX'].to('TeV').value
        int_flux = self['INT_FLUX'].to('cm-2 s-1').value

        # Compute x point
        if x_method == 'table':
            # This is only called if the provided table includes energies
            energy = energy_table['ENERGY'].quantity.to('TeV').value
        elif x_method == 'log_center':
            energy = self.ebounds.log_centers.to('TeV').value
        elif x_method == 'lafferty':
            if y_method == 'power_law':
                # Uses analytical implementation available for the power law case
                if spectral_index is None:
                    raise ValueError('Need spectral index for x method {}'.format(y_method))
                energy = _energy_lafferty_power_law(energy_min, energy_max,
                                                    spectral_index)
            else:
                if model is None:
                    raise ValueError('Need model for x method {}'.format(y_method))
                energy = np.array(_x_lafferty(energy_min,
                                              energy_max, model))
        else:
            raise ValueError('Invalid x method: {0}'.format(x_method))

        # Compute y point
        if y_method == 'power_law':
            if spectral_index is None:
                    raise ValueError('Need spectral index for y method {}'.format(y_method))

            g = -1 * np.abs(spectral_index)
            diff_flux = power_law_flux(int_flux, g, energy, energy_min, energy_max)
        elif y_method == 'model':
            if model is None:
                    raise ValueError('Need model for y method {}'.format(y_method))
            diff_flux = _ydiff_excess_equals_expected(int_flux, energy_min,
            energy_max, energy, model)
        else:
            raise ValueError('Invalid y method: {0}'.format(y_method))

        # Output table
        table = Table()
        energy = Quantity(energy, 'TeV')
        table['ENERGY'] = energy
        table['ENERGY_ERR_HI'] = self.ebounds.upper_bounds.to('TeV') - energy
        table['ENERGY_ERR_LO'] = energy - self.ebounds.lower_bounds.to('TeV')
        table['DIFF_FLUX'] = diff_flux

        # Error processing if required
        try:
            # TODO: more rigorous implementation of error propagation should be implemented
            # I.e. based on MC simulation rather than gaussian error assumption
            int_flux_err = self['INT_FLUX_ERR'].to('cm-2 s-1').value
            err = int_flux_err / int_flux
            diff_flux_err = err * diff_flux
            table['DIFF_FLUX_ERR_HI'] = diff_flux_err
            table['DIFF_FLUX_ERR_LO'] = diff_flux_err
        except:
            pass

        table.meta['spectral_index'] = spectral_index
        table.meta['spectral_index_description'] = "Spectral index assumed in the DIFF_FLUX computation"
        return DifferentialFluxPoints(table)


def _x_lafferty(xmin, xmax, function):
    """The Lafferty & Wyatt method to compute X.

    Pass in a function and bin bounds x_min and x_max i.e. for energy
    See: Lafferty & Wyatt, Nucl. Instr. and Meth. in Phys. Res. A 355(1995) 541-547
    See: http://nbviewer.ipython.org/gist/cdeil/bdab5f236640ef52f736
    """
    from scipy.optimize import brentq
    from scipy import integrate

    indices = np.arange(len(xmin))

    x_points = []
    for index in indices:
        deltax = xmax[index] - xmin[index]
        I = integrate.quad(function, xmin[index], xmax[index], args=())
        F = (I[0] / deltax)

        def g(x):
            return function(x) - F

        x_point = brentq(g, xmin[index], xmax[index])
        x_points.append(x_point)
    return x_points


def _ydiff_excess_equals_expected(yint, xmin, xmax, x, model):
    """The ExcessEqualsExpected method to compute Y (differential).

    y / yint = y_model / yint_model"""
    yint_model = _integrate(xmin, xmax, model)
    y_model = model(x)
    return y_model * (yint / yint_model)


def _integrate(xmin, xmax, function, segments=1e3):
    """Integrates method function using the trapezium rule between xmin and xmax.
    """
    indices = np.arange(len(xmin))
    y_values = []
    for index in indices:
        x_vals = np.arange(xmin[index], xmax[index], 1.0 / segments)
        y_vals = function(x_vals)
        # Division by number of segments required for correct normalization
        y_values.append(np.trapz(y_vals) / segments)

    # Todo : Remove assumption on unit
    # In fact all of this can probably go ocne astropy.models or sherpa is ready
    return y_values


def _energy_lafferty_power_law(energy_min, energy_max, spectral_index):
    """Analytical case for determining lafferty x-position for power law case.
    """
    # Cannot call into gammapy.powerlaw as implementation is different
    # due to different reference energies
    term0 = 1. - spectral_index
    term1 = energy_max - energy_min
    term2 = 1. / term0
    flux_lw = term2 / term1 * (energy_max ** term0 - energy_min ** term0)
    return np.exp(-np.log(flux_lw) / np.abs(spectral_index))
