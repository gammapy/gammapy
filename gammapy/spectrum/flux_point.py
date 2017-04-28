# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Differential and integral flux point computations."""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from collections import OrderedDict
import numpy as np
from astropy.table import Table, vstack
from astropy import units as u
from astropy.io.registry import IORegistryError
from ..utils.scripts import make_path
from ..utils.fits import table_from_row_data

__all__ = [
    'compute_flux_points_dnde',
    'FluxPoints',
    'FluxPointEstimator',
    'FluxPointsFitter',
    'SEDLikelihoodProfile',
]

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = {'dnde': ['e_ref', 'dnde'],
                    'e2dnde': ['e_ref', 'e2dnde'],
                    'flux': ['e_min', 'e_max', 'flux'],
                    'eflux': ['e_min', 'e_max', 'eflux']}

OPTIONAL_COLUMNS = {'dnde': ['dnde_err', 'dnde_errp', 'dnde_errn',
                             'dnde_ul', 'is_ul'],
                    'e2dnde': ['e2dnde_err', 'e2dnde_errp', 'e2dnde_errn',
                               'e2dnde_ul', 'is_ul'],
                    'flux': ['flux_err', 'flux_errp', 'flux_errn',
                             'flux_ul', 'is_ul'],
                    'eflux': ['eflux_err', 'eflux_errp', 'eflux_errn',
                              'eflux_ul', 'is_ul']}

DEFAULT_UNIT = {'dnde': u.Unit('cm-2 s-1 TeV-1'),
                'e2dnde': u.Unit('erg cm-2 s-1'),
                'flux': u.Unit('cm-2 s-1'),
                'eflux': u.Unit('erg cm-2 s-1')}


class FluxPoints(object):
    """
    Flux point object.

    For a complete documentation see :ref:`gadf:flux-points`, for an usage
    example see :ref:`flux-point-computation`.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Input data table, with the following minimal required columns:

        * Format `'dnde'`: `'dnde'` and `'e_ref'`
        * Format `'flux'`: `'flux'`, `'e_min'`, `'e_max'`
        * Format `'eflux'`: `'eflux'`, `'e_min'`, `'e_max'`

    Examples
    --------

    >>> from gammapy.spectrum import FluxPoints
    >>> filename = '$GAMMAPY_EXTRA/test_datasets/spectrum/flux_points/flux_points.fits'
    >>> flux_points = FluxPoints.read(filename)
    >>> flux_points.show()

    """

    def __init__(self, table):
        # validate that the table is a valid representation of the given
        # flux point sed type

        # TODO: this is a temp solution
        # Make sure we don't have "ph" in units
        # Should we use a unit equivalency?
        unit_changes = [
            ('ph cm-2 s-1', 'cm-2 s-1'),
            ('ph cm-2 s-1 TeV-1', 'cm-2 s-1 TeV-1'),
            ('ph cm-2 s-1 MeV-1', 'cm-2 s-1 MeV-1'),
        ]
        for colname in table.colnames:
            for unit_old, unit_new in unit_changes:
                if (table[colname].unit is not None) and (u.Unit(table[colname].unit) == u.Unit(unit_old)):
                    table[colname].unit = u.Unit(unit_new)

        self.table = self._validate_table(table)

    @property
    def sed_type(self):
        """
        Flux points sed type.

        Returns
        -------
        sed_type : str
            Can be either 'dnde', 'e2dnde', 'flux' or 'eflux'.
        """
        return self.table.meta['SED_TYPE']

    @staticmethod
    def _guess_sed_type(table):
        """
        Guess sed type from table content.
        """
        valid_sed_types = list(REQUIRED_COLUMNS.keys())
        for sed_type in valid_sed_types:
            required = set(REQUIRED_COLUMNS[sed_type])
            if required.issubset(table.colnames):
                return sed_type

    @staticmethod
    def _guess_sed_type_from_unit(unit):
        """
        Guess sed type from unit.
        """
        for sed_type, default_unit in DEFAULT_UNIT.items():
            if unit.is_equivalent(default_unit):
                return sed_type

    def _validate_table(self, table):
        """
        Validate input flux point table.
        """
        table = Table(table)
        sed_type = table.meta['SED_TYPE']
        required = set(REQUIRED_COLUMNS[sed_type])

        if not required.issubset(table.colnames):
            missing = required.difference(table.colnames)
            raise ValueError("Missing columns for sed type '{0}':"
                             " {1}".format(sed_type, missing))
        return table

    def _get_y_energy_unit(self, y_unit):
        """
        Get energy part of the given y unit.
        """
        try:
            return [_ for _ in y_unit.bases if _.physical_type == 'energy'][0]
        except IndexError:
            return u.Unit('TeV')

    def plot(self, ax=None, sed_type=None, energy_unit='TeV', flux_unit=None,
             energy_power=0, **kwargs):
        """
        Plot flux points

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axis object to plot on.
        sed_type : ['dnde', 'flux', 'eflux']
            Which sed type to plot.
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
        flux_unit : str, `~astropy.units.Unit`, optional
            Unit of the flux axis
        energy_power : int
            Power of energy to multiply y axis with
        kwargs : dict
            Keyword arguments passed to :func:`~matplotlib.pyplot.errorbar`

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis object
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        sed_type = sed_type or self.sed_type
        y_unit = u.Unit(flux_unit or DEFAULT_UNIT[sed_type])

        y = self.table[sed_type].quantity.to(y_unit)
        x = self.e_ref.to(energy_unit)

        # get errors and ul
        is_ul = self._is_ul
        x_err_all = self.get_energy_err(sed_type)
        y_err_all = self.get_flux_err(sed_type)

        # handle energy power
        e_unit = self._get_y_energy_unit(y_unit)
        y_unit = y.unit * e_unit ** energy_power
        y = (y * np.power(x, energy_power)).to(y_unit)

        y_err, x_err = None, None

        if y_err_all:
            y_errn = (y_err_all[0] * np.power(x, energy_power)).to(y_unit)
            y_errp = (y_err_all[1] * np.power(x, energy_power)).to(y_unit)
            y_err = (y_errn[~is_ul].to(y_unit).value,
                     y_errp[~is_ul].to(y_unit).value)

        if x_err_all:
            x_errn, x_errp = x_err_all
            x_err = (x_errn[~is_ul].to(energy_unit).value,
                     x_errp[~is_ul].to(energy_unit).value)

        # set flux points plotting defaults
        kwargs.setdefault('marker', '+')
        kwargs.setdefault('ls', 'None')

        ebar = ax.errorbar(x[~is_ul].value, y[~is_ul].value, yerr=y_err,
                           xerr=x_err, **kwargs)

        if is_ul.any():
            if x_err_all:
                x_errn, x_errp = x_err_all
                x_err = (x_errn[is_ul].to(energy_unit).value,
                         x_errp[is_ul].to(energy_unit).value)

            y_ul = self.table[sed_type + '_ul'].quantity
            y_ul = (y_ul * np.power(x, energy_power)).to(y_unit)

            y_err = (0.5 * y_ul[is_ul].value, np.zeros_like(y_ul[is_ul].value))

            kwargs.setdefault('c', ebar[0].get_color())
            ax.errorbar(x[is_ul].value, y_ul[is_ul].value, xerr=x_err, yerr=y_err,
                        uplims=True, **kwargs)

        ax.set_xscale('log', nonposx='clip')
        ax.set_yscale('log', nonposy='clip')
        ax.set_xlabel('Energy ({})'.format(energy_unit))
        ax.set_ylabel('{0} ({1})'.format(self.sed_type, y_unit))
        return ax

    def get_energy_err(self, sed_type=None):
        """Compute energy error for given sed type"""
        # TODO: sed_type is not used
        if sed_type is None:
            sed_type = self.sed_type
        try:
            e_min = self.table['e_min'].quantity
            e_max = self.table['e_max'].quantity
            e_ref = self.e_ref
            x_err = ((e_ref - e_min), (e_max - e_ref))
        except KeyError:
            x_err = None
        return x_err

    def get_flux_err(self, sed_type=None):
        """Compute flux error for given sed type"""
        if sed_type is None:
            sed_type = self.sed_type
        try:
            # assymmetric error
            y_errn = self.table[sed_type + '_errn'].quantity
            y_errp = self.table[sed_type + '_errp'].quantity
            y_err = (y_errn, y_errp)
        except KeyError:
            try:
                # symmetric error
                y_err = self.table[sed_type + '_err'].quantity
                y_err = (y_err, y_err)
            except KeyError:
                # no error at all
                y_err = None
        return y_err

    @property
    def _is_ul(self):
        try:
            return self.table['is_ul'].data.astype('bool')
        except KeyError:
            return np.isnan(self.table[self.sed_type])

    def peek(self, figsize=(8, 5), **kwargs):
        """
        Show flux points.

        Parameters
        ----------
        figsize : tuple
            Figure size
        kwargs : dict
            Keyword arguments passed to `FluxPoints.plot()`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Plotting axes object.
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        self.plot(ax=ax, **kwargs)
        return ax

    def __str__(self):
        """
        String representation of the flux points class.
        """
        info = ''
        info += "Flux points of type '{}'".format(self.sed_type)
        return info

    def info(self):
        """
        Print flux points info.
        """
        print(self)

    @classmethod
    def read(cls, filename, **kwargs):
        """
        Read flux points.

        Parameters
        ----------
        filename : str
            Filename
        kwargs : dict
            Keyword arguments passed to `~astropy.table.Table.read`.

        """
        filename = make_path(filename)
        try:
            table = Table.read(str(filename), **kwargs)
        except IORegistryError:
            kwargs.setdefault('format', 'ascii.ecsv')
            table = Table.read(str(filename), **kwargs)

        if 'SED_TYPE' not in table.meta.keys():
            sed_type = cls._guess_sed_type(table)
            table.meta['SED_TYPE'] = sed_type

        return cls(table=table)

    def write(self, filename, **kwargs):
        """
        Write flux points.

        Parameters
        ----------
        filename : str
            Filename
        kwargs : dict
            Keyword arguments passed to `~astropy.table.Table.write`.
        """
        filename = make_path(filename)
        try:
            self.table.write(str(filename), **kwargs)
        except IORegistryError:
            kwargs.setdefault('format', 'ascii.ecsv')
            self.table.write(str(filename), **kwargs)

    # TODO: handle with Energy or EnergyBounds classes?
    @property
    def e_ref(self):
        """
        Reference energy.

        Defined by `e_ref` column in `FluxPoints.table` or computed as log
        center, if `e_min` and `e_max` columns are present in `FluxPoints.table`.

        Returns
        -------
        e_ref : `~astropy.units.Quantity`
            Reference energy.
        """
        try:
            return self.table['e_ref'].quantity
        except KeyError:
            e_ref = np.sqrt(self.e_min * self.e_max)
            return e_ref

    # TODO: handle with Energy or EnergyBounds classes?
    @property
    def e_min(self):
        """
        Lower bound of energy bin.

        Defined by `e_min` column in `FluxPoints.table`.

        Returns
        -------
        e_min : `~astropy.units.Quantity`
            Lower bound of energy bin.
        """
        return self.table['e_min'].quantity

    # TODO: handle with Energy or EnergyBounds classes?
    @property
    def e_max(self):
        """
        Upper bound of energy bin.

        Defined by `e_max` column in `FluxPoints.table`.

        Returns
        -------
        e_max : `~astropy.units.Quantity`
            Upper bound of energy bin.
        """
        return self.table['e_max'].quantity

    def drop_ul(self):
        """
        Drop upper limit flux points.

        Examples
        --------

        from gammapy.spectrum import FluxPoints

        filename = '$GAMMAPY_EXTRA/test_datasets/spectrum/flux_points/flux_points.fits'
        flux_points = FluxPoints.read(filename)

        print(flux_points)
        print(flux_points.drop_ul())

        Returns
        -------
        flux_points : `FluxPoints`
            Flux points with upper limit points removed.
        """
        table = self.table.copy()
        table_drop_ul = table[~self._is_ul]
        return self.__class__(table_drop_ul)

    @classmethod
    def stack(cls, flux_points):
        """
        Create a new `FluxPoints` object by stacking a list of existing
        flux points.

        The first `FluxPoints` object in the list is taken as a reference to infer
        column names and units for the stacked object.

        Parameters
        ----------
        flux_points : list of `FluxPoints` objects
            List of flux points to stack.

        Returns
        -------
        flux_points : `FluxPoints`
            Flux points without upper limit points.
        """
        tables = []
        reference = flux_points[0].table

        for _ in flux_points:
            table = _.table
            for colname in reference.colnames:
                column = reference[colname]
                if column.unit:
                    table[colname] = table[colname].quantity.to(column.unit)
            tables.append(table[reference.colnames])
        table_stacked = vstack(tables)
        table_stacked.meta['SED_TYPE'] = reference.meta['SED_TYPE']
        return cls(table_stacked)


def compute_flux_points_dnde(flux_points, model, method='lafferty'):
    """
    Compute differential flux points quantities.

    See: http://adsabs.harvard.edu/abs/1995NIMPA.355..541L for details
    on the `'lafferty'` method.

    Parameters
    ----------
    flux_points : `FluxPoints`
         Input integral flux points.
    model : `~gammapy.spectrum.SpectralModel`
        Spectral model assumption.  Note that the value of the amplitude parameter
        does not matter. Still it is recommended to use something with the right
        scale and units. E.g. `amplitude = 1e-12 * u.Unit('cm-2 s-1 TeV-1')`
    method : {'lafferty', 'log_center', 'table'}
        Flux points `e_ref` estimation method:

            * `'laferty'` Lafferty & Wyatt model-based e_ref
            * `'log_center'` log bin center e_ref
            * `'table'` using column 'e_ref' from input flux_points

    Returns
    -------
    flux_points : `FluxPoints`
        Flux points including differential quantity columns `dnde`
        and `dnde_err` (optional), `dnde_ul` (optional).

    Examples
    --------
    >>> from astropy import units as u
    >>> from gammapy.spectrum import FluxPoints, compute_flux_points_dnde
    >>> from gammapy.spectrum.models import PowerLaw
    >>> filename = '$GAMMAPY_EXTRA/test_datasets/spectrum/flux_points/flux_points.fits'
    >>> flux_points = FluxPoints.read(filename)
    >>> model = PowerLaw(2.2 * u.Unit(''), 1e-12 * u.Unit('cm-2 s-1 TeV-1'), 1 * u.TeV)
    >>> flux_point_dnde = compute_flux_points_dnde(flux_points, model=model)
    """
    input_table = flux_points.table
    flux = input_table['flux'].quantity
    e_min = flux_points.e_min
    e_max = flux_points.e_max

    # Compute e_ref
    if method == 'table':
        e_ref = input_table['e_ref'].quantity
    elif method == 'log_center':
        e_ref = np.sqrt(e_min * e_max)
    elif method == 'lafferty':
        # set e_ref that it represents the mean dnde in the given energy bin
        e_ref = _e_ref_lafferty(model, e_min, e_max)
    else:
        raise ValueError('Invalid method: {0}'.format(method))

    dnde = _dnde_from_flux(flux, model, e_ref, e_min, e_max)

    # Add to result table
    table = input_table.copy()
    table['e_ref'] = e_ref
    table['dnde'] = dnde

    if 'flux_err' in table.colnames:
        # TODO: implement better error handling, e.g. MC based method
        table['dnde_err'] = dnde * table['flux_err'].quantity / flux

    if 'flux_errn' in table.colnames:
        table['dnde_errn'] = dnde * table['flux_errn'].quantity / flux
        table['dnde_errp'] = dnde * table['flux_errp'].quantity / flux

    if 'flux_ul' in table.colnames:
        flux_ul = table['flux_ul'].quantity
        dnde_ul = _dnde_from_flux(flux_ul, model, e_ref, e_min, e_max)
        table['dnde_ul'] = dnde_ul

    table.meta['SED_TYPE'] = 'dnde'
    return FluxPoints(table)


def _e_ref_lafferty(model, e_min, e_max):
    # compute e_ref that the value at e_ref corresponds to the mean value
    # between e_min and e_max
    flux = model.integral(e_min, e_max)
    dnde_mean = flux / (e_max - e_min)
    return model.inverse(dnde_mean)


def _dnde_from_flux(flux, model, e_ref, e_min, e_max):
    # Compute dnde under the assumption that flux equals expected
    # flux from model
    flux_model = model.integral(e_min, e_max)
    dnde_model = model(e_ref)
    return dnde_model * (flux / flux_model)


class FluxPointEstimator(object):
    """
    Flux point estimator.

    Parameters
    ----------
    obs : `~gammapy.spectrum.SpectrumObservation`
        Spectrum observation
    groups : `~gammapy.spectrum.SpectrumEnergyGroups`
        Energy groups (usually output of `~gammapy.spectrum.SpectrumEnergyGroupsMaker`)
    model : `~gammapy.spectrum.models.SpectralModel`
        Global model (usually output of `~gammapy.spectrum.SpectrumFit`)
    """

    def __init__(self, obs, groups, model):
        self.obs = obs
        self.groups = groups
        self.model = model
        self.flux_points = None

    def __str__(self):
        s = 'FluxPointEstimator:\n'
        s += str(self.obs) + '\n'
        s += str(self.groups) + '\n'
        s += str(self.model) + '\n'
        return s

    def compute_points(self):
        meta = OrderedDict(
            METHOD='TODO',
            SED_TYPE='dnde'
        )
        rows = []
        for group in self.groups:
            if group.bin_type != 'normal':
                log.debug('Skipping energy group:\n{}'.format(group))
                continue

            row = self.compute_flux_point(group)
            rows.append(row)

        self.flux_points = FluxPoints(table_from_row_data(rows=rows,
                                                          meta=meta))

    def compute_flux_point(self, energy_group):
        log.debug('Computing flux point for energy group:\n{}'.format(energy_group))
        model = self.compute_approx_model(
            global_model=self.model,
            energy_range=energy_group.energy_range,
        )

        energy_ref = self.compute_energy_ref(energy_group)

        return self.fit_point(
            model=model, energy_group=energy_group, energy_ref=energy_ref,
        )

    def compute_energy_ref(self, energy_group):
        return energy_group.energy_range.log_center

    @staticmethod
    def compute_approx_model(global_model, energy_range):
        """
        Compute approximate model, to be used in the energy bin.
        TODO: At the moment just the global model with fixed parameters is
        returned
        """
        # binning = EnergyBounds(binning)
        # low_bins = binning.lower_bounds
        # high_bins = binning.upper_bounds
        #
        # from sherpa.models import PowLaw1D
        #
        # if isinstance(model, models.PowerLaw):
        #     temp = model.to_sherpa()
        #     temp.gamma.freeze()
        #     sherpa_models = [temp] * binning.nbins
        # else:
        #     sherpa_models = [None] * binning.nbins
        #
        # for low, high, sherpa_model in zip(low_bins, high_bins, sherpa_models):
        #     log.info('Computing flux points in bin [{}, {}]'.format(low, high))
        #
        #     # Make PowerLaw approximation for higher order models
        #     if sherpa_model is None:
        #         flux_low = model(low)
        #         flux_high = model(high)
        #         index = powerlaw.power_law_g_from_points(e1=low, e2=high,
        #                                                  f1=flux_low,
        #                                                  f2=flux_high)
        #
        #         log.debug('Approximated power law index: {}'.format(index))
        #         sherpa_model = PowLaw1D('powlaw1d.default')
        #         sherpa_model.gamma = index
        #         sherpa_model.gamma.freeze()
        #         sherpa_model.ref = model.parameters.reference.to('keV')
        #         sherpa_model.ampl = 1e-20
        # return PowerLaw(
        #    index=u.Quantity(2, ''),
        #    amplitude=u.Quantity(1, 'm-2 s-1 TeV-1'),
        #    reference=u.Quantity(1, 'TeV'),
        # )
        approx_model = global_model.copy()
        for par in approx_model.parameters.parameters:
            if par.name != 'amplitude':
                par.frozen = True
        return approx_model

    def fit_point(self, model, energy_group, energy_ref):
        from .fit import SpectrumFit

        fit = SpectrumFit(self.obs, model)
        erange = energy_group.energy_range

        # TODO: Notice channels contained in energy_group
        fit.fit_range = erange.min, erange.max

        log.debug(
            'Calling Sherpa fit for flux point '
            ' in energy range:\n{}'.format(fit)
        )

        fit.fit()
        fit.est_errors()

        # First result contain correct model
        res = fit.result[0]

        e_max = energy_group.energy_range.max
        e_min = energy_group.energy_range.min
        diff_flux, diff_flux_err = res.model.evaluate_error(energy_ref)
        return OrderedDict(
            e_ref=energy_ref,
            e_min=e_min,
            e_max=e_max,
            dnde=diff_flux.to('m-2 s-1 TeV-1'),
            dnde_err=diff_flux_err.to('m-2 s-1 TeV-1'),
        )


class SEDLikelihoodProfile(object):
    """SED likelihood profile.

    See :ref:`gadf:likelihood_sed`.

    TODO: merge this class with the classes in ``fermipy/castro.py``,
    which are much more advanced / feature complete.
    This is just a temp solution because we don't have time for that.
    """

    def __init__(self, table):
        self.table = table

    @classmethod
    def read(cls, filename, **kwargs):
        filename = make_path(filename)
        table = Table.read(str(filename), **kwargs)
        return cls(table=table)

    def __str__(self):
        s = self.__class__.__name__ + '\n'
        s += str(self.table)
        return s

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
            # TODO


def chi2_flux_points(flux_points, gp_model):
    """
    Chi2 statistics for a list of flux points and model.
    """
    model = gp_model(flux_points.e_ref)
    data = flux_points.table['dnde'].quantity
    data_err = flux_points.table['dnde_err'].quantity
    stat_per_bin = ((data - model) / data_err).value ** 2
    return np.nansum(stat_per_bin), stat_per_bin


def chi2_flux_points_assym(flux_points, gp_model):
    """
    Assymetric chi2 statistics for a list of flux points and model.
    """
    model = gp_model(flux_points.e_ref)
    data = flux_points.table['dnde'].quantity

    data_errp = flux_points.table['dnde_errp'].quantity
    data_errn = flux_points.table['dnde_errn'].quantity

    data_err = np.where(model > data, data_errp, data_errn)
    stat_per_bin = ((data - model) / data_err).value ** 2

    return np.nansum(stat_per_bin), stat_per_bin


class FluxPointsFitter(object):
    """
    Fit a set of flux points with a parametric model.

    Parameters
    ----------
    optimizer : {'simplex', 'moncar', 'gridsearch'}
        Select optimizer
    error_estimator : {'covar'}
        Select error estimator
    ul_handling : {'ignore'}
        How to handle flux point upper limits in the fit

    Examples
    --------

    Load flux points from file and fit with a power-law model::

        from astropy import units as u
        from gammapy.spectrum import FluxPoints, FluxPointsFitter
        from gammapy.spectrum.models import PowerLaw

        filename = '$GAMMAPY_EXTRA/test_datasets/spectrum/flux_points/diff_flux_points.fits'
        flux_points = FluxPoints.read(filename)

        model = PowerLaw(
            index=2. * u.Unit(''),
            amplitude=1e-12 * u.Unit('cm-2 s-1 TeV-1'),
            reference=1. * u.TeV,
        )

        fitter = FluxPointsFitter()
        result = fitter.run(flux_points, model)
        print(result['best_fit_model'])
    """

    def __init__(self, stat='chi2', optimizer='simplex', error_estimator='covar',
                 ul_handling='ignore'):
        if stat == 'chi2':
            self.stat = chi2_flux_points
        elif stat == 'chi2assym':
            self.stat = chi2_flux_points_assym
        else:
            raise ValueError("'{stat}' is not a valid fit statistic, please choose"
                             " either 'chi2' or 'chi2assym'")

        if not ul_handling == 'ignore':
            raise NotImplementedError('No handling of upper limits implemented.')

        self.parameters = OrderedDict(optimizer=optimizer,
                                      error_estimator=error_estimator,
                                      ul_handling=ul_handling)

    def _setup_sherpa_fit(self, data, model):
        """Fit flux point using sherpa"""
        from sherpa.fit import Fit
        from sherpa.data import DataSimulFit
        from ..utils.sherpa import (SherpaDataWrapper, SherpaStatWrapper,
                                    SherpaModelWrapper, SHERPA_OPTMETHODS)

        optimizer = self.parameters['optimizer']

        if data.sed_type == 'dnde':
            data = SherpaDataWrapper(data)
        else:
            raise NotImplementedError('Only fitting of differential flux points data '
                                      'is supported.')

        stat = SherpaStatWrapper(self.stat)
        data = DataSimulFit(name='GPFluxPoints', datasets=[data])
        method = SHERPA_OPTMETHODS[optimizer]
        models = SherpaModelWrapper(model)
        return Fit(data=data, model=models, stat=stat, method=method)

    def fit(self, data, model):
        """
        Fit given model to data.

        Parameters
        ----------
        model : `~gammapy.spectrum.models.SpectralModel`
            Spectral model (with fit start parameters)

        Returns
        -------
        best_fit_model : `~gammapy.spectrum.models.SpectralModel`
            Best fit model
        """
        p = self.parameters

        # TODO: make copy of model?
        if p['optimizer'] in ['simplex', 'moncar', 'gridsearch']:
            sherpa_fitter = self._setup_sherpa_fit(data, model)
            sherpa_fitter.fit()
        else:
            raise ValueError('Not a valid optimizer')

        return model

    def statval(self, data, model):
        """
        Compute statval for given model and data.

        Parameters
        ----------
        model : `~gammapy.spectrum.models.SpectralModel`
            Spectral model
        """
        return self.stat(data, model)

    def dof(self, data, model):
        """
        Degrees of freedom.

        Parameters
        ----------
        model : `~gammapy.spectrum.models.SpectralModel`
            Spectral model
        """
        m = len(model.parameters.free)
        n = len(data.table)
        return n - m

    def estimate_errors(self, data, model):
        """
        Estimate errors on best fit parameters.
        """
        sherpa_fitter = self._setup_sherpa_fit(data, model)
        result = sherpa_fitter.est_errors()
        covariance = result.extra_output
        covar_axis = model.parameters.free
        model.parameters.set_parameter_covariance(covariance, covar_axis)
        return model

    def run(self, data, model):
        """
        Run all fitting adn extra information steps.

        Parameters
        ----------
        data : list of `~gammapy.spectrum.FluxPoints`
            Flux points.
        model : `~gammapy.spectrum.models.SpectralModel`
            Spectral model

        Returns
        -------
        result : `~collections.OrderedDict`
            Dictionary with fit results and debug output.
        """
        result = OrderedDict()

        best_fit_model = self.fit(data, model)
        best_fit_model = self.estimate_errors(data, best_fit_model)
        result['best_fit_model'] = best_fit_model
        result['dof'] = self.dof(data, model)
        result['statval'] = self.statval(data, model)[0]
        result['statval/dof'] = result['statval'] / result['dof']
        return result
