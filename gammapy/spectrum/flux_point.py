# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from collections import OrderedDict
import numpy as np
from astropy.table import Table, vstack
from astropy import units as u
from astropy.io.registry import IORegistryError
from ..utils.scripts import make_path
from ..utils.fits import table_from_row_data
from ..utils.table import table_standardise_units_copy
from .models import PowerLaw
from .powerlaw import power_law_integral_flux

__all__ = [
    'FluxPoints',
    # 'FluxPointProfiles',
    'FluxPointEstimator',
    'FluxPointFitter',
]

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = OrderedDict([
    ('dnde', ['e_ref', 'dnde']),
    ('e2dnde', ['e_ref', 'e2dnde']),
    ('flux', ['e_min', 'e_max', 'flux']),
    ('eflux', ['e_min', 'e_max', 'eflux']),
])

OPTIONAL_COLUMNS = OrderedDict([
    ('dnde', ['dnde_err', 'dnde_errp', 'dnde_errn', 'dnde_ul', 'is_ul']),
    ('e2dnde', ['e2dnde_err', 'e2dnde_errp', 'e2dnde_errn', 'e2dnde_ul', 'is_ul']),
    ('flux', ['flux_err', 'flux_errp', 'flux_errn', 'flux_ul', 'is_ul']),
    ('eflux', ['eflux_err', 'eflux_errp', 'eflux_errn', 'eflux_ul', 'is_ul']),
])

DEFAULT_UNIT = OrderedDict([
    ('dnde', u.Unit('cm-2 s-1 TeV-1')),
    ('e2dnde', u.Unit('erg cm-2 s-1')),
    ('flux', u.Unit('cm-2 s-1')),
    ('eflux', u.Unit('erg cm-2 s-1')),
])


class FluxPoints(object):
    """Flux points container.

    The supported formats are described here: :ref:`gadf:flux-points`

    In summary, the following formats and minimum required columns are:

    * Format ``dnde``: columns ``e_ref`` and ``dnde``
    * Format ``e2dnde``: columns ``e_ref``, ``e2dnde``
    * Format ``flux``: columns ``e_min``, ``e_max``, ``flux``
    * Format ``eflux``: columns ``e_min``, ``e_max``, ``eflux``

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table with flux point data

    Attributes
    ----------
    table : `~astropy.table.Table`
        Table with flux point data

    Examples
    --------
    The `FluxPoints` object is most easily created by reading a file with
    flux points given in one of the formats documented above:

    .. code::

        from gammapy.spectrum import FluxPoints
        filename = '$GAMMAPY_EXTRA/test_datasets/spectrum/flux_points/flux_points.fits'
        flux_points = FluxPoints.read(filename)
        flux_points.plot()

    An instance of `FluxPoints` can also be created by passing an instance of
    `astropy.table.Table`, which contains the required columns, such as `'e_ref'`
    and `'dnde'`:

    .. code::

        from astropy import units as u
        from astropy.table import Table
        from gammapy.spectrum import FluxPoints
        from gammapy.spectrum.models import PowerLaw

        table = Table()
        pwl = PowerLaw()
        e_ref = np.logspace(0, 2, 7) * u.TeV
        table['e_ref'] = e_ref
        table['dnde'] = pwl(e_ref)
        table.meta['SED_TYPE'] = 'dnde'

        flux_points = FluxPoints(table)
        flux_points.plot()

    If you have flux points in a different data format, the format can be changed
    by renamimg the table columns and adding meta data:

    .. code::

        from astropy import units as u
        from astropy.table import Table
        from gammapy.spectrum import FluxPoints

        table = Table.read('$GAMMAPY_EXTRA/test_datasets/spectrum/flux_points/flux_points_ctb_37b.txt',
                           format='ascii.csv', delimiter=' ', comment='#')
        table.meta['SED_TYPE'] = 'dnde'
        table.rename_column('Differential_Flux', 'dnde')
        table['dnde'].unit = 'cm-2 s-1 TeV-1'

        table.rename_column('lower_error', 'dnde_errn')
        table['dnde_errn'].unit = 'cm-2 s-1 TeV-1'

        table.rename_column('upper_error', 'dnde_errp')
        table['dnde_errp'].unit = 'cm-2 s-1 TeV-1'

        table.rename_column('E', 'e_ref')
        table['e_ref'].unit = 'TeV'

        flux_points = FluxPoints(table)
        flux_points.plot()

    """

    def __init__(self, table):
        self.table = table_standardise_units_copy(table)
        # validate that the table is a valid representation
        # of the given flux point sed type
        self._validate_table(self.table)

    def __repr__(self):
        fmt = '{}(sed_type="{}", n_points={})'
        return fmt.format(self.__class__.__name__, self.sed_type, len(self.table))

    @classmethod
    def read(cls, filename, **kwargs):
        """Read flux points.

        Parameters
        ----------
        filename : str
            Filename
        kwargs : dict
            Keyword arguments passed to `astropy.table.Table.read`.
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
        """Write flux points.

        Parameters
        ----------
        filename : str
            Filename
        kwargs : dict
            Keyword arguments passed to `astropy.table.Table.write`.
        """
        filename = make_path(filename)
        try:
            self.table.write(str(filename), **kwargs)
        except IORegistryError:
            kwargs.setdefault('format', 'ascii.ecsv')
            self.table.write(str(filename), **kwargs)

    @classmethod
    def stack(cls, flux_points):
        """Create flux points by stacking list of flux points.

        The first `FluxPoints` object in the list is taken as a reference to infer
        column names and units for the stacked object.

        Parameters
        ----------
        flux_points : list of `FluxPoints`
            List of flux points to stack.

        Returns
        -------
        flux_points : `FluxPoints`
            Flux points without upper limit points.
        """
        reference = flux_points[0].table

        tables = []
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

    def drop_ul(self):
        """Drop upper limit flux points.

        Returns
        -------
        flux_points : `FluxPoints`
            Flux points with upper limit points removed.

        Examples
        --------

        >>> from gammapy.spectrum import FluxPoints
        >>> filename = '$GAMMAPY_EXTRA/test_datasets/spectrum/flux_points/flux_points.fits'
        >>> flux_points = FluxPoints.read(filename)
        >>> print(flux_points)
        FluxPoints(sed_type="flux", n_points=24)
        >>> print(flux_points.drop_ul())
        FluxPoints(sed_type="flux", n_points=19)
        """
        table_drop_ul = self.table[~self._is_ul]
        return self.__class__(table_drop_ul)

    def to_sed_type(self, sed_type, method='log_center', model=None, pwl_approx=False):
        """Convert to a different SED type (return new `FluxPoints`).

        See: http://adsabs.harvard.edu/abs/1995NIMPA.355..541L for details
        on the `'lafferty'` method.

        Parameters
        ----------
        sed_type : {'dnde'}
             SED type to convert to.
        model : `~gammapy.spectrum.SpectralModel`
            Spectral model assumption.  Note that the value of the amplitude parameter
            does not matter. Still it is recommended to use something with the right
            scale and units. E.g. `amplitude = 1e-12 * u.Unit('cm-2 s-1 TeV-1')`
        method : {'lafferty', 'log_center', 'table'}
            Flux points `e_ref` estimation method:

                * `'laferty'` Lafferty & Wyatt model-based e_ref
                * `'log_center'` log bin center e_ref
                * `'table'` using column 'e_ref' from input flux_points
        pwl_approx : bool
            Use local power law appoximation at e_ref to compute differential flux
            from the integral flux. This method is used by the Fermi-LAT catalogs.

        Returns
        -------
        flux_points : `FluxPoints`
            Flux points including differential quantity columns `dnde`
            and `dnde_err` (optional), `dnde_ul` (optional).

        Examples
        --------
        >>> from astropy import units as u
        >>> from gammapy.spectrum import FluxPoints
        >>> from gammapy.spectrum.models import PowerLaw
        >>> filename = '$GAMMAPY_EXTRA/test_datasets/spectrum/flux_points/flux_points.fits'
        >>> flux_points = FluxPoints.read(filename)
        >>> model = PowerLaw(2.2 * u.Unit(''), 1e-12 * u.Unit('cm-2 s-1 TeV-1'), 1 * u.TeV)
        >>> flux_points_dnde = flux_points.to_sed_type('dnde', model=model)
        """
        # TODO: implement other directions. Refactor!
        if sed_type != 'dnde':
            raise NotImplementedError

        if model is None:
            model = PowerLaw(
                index=2 * u.Unit(''),
                amplitude=1 * u.Unit('cm-2 s-1 TeV-1'),
                reference=1 * u.TeV,
            )

        input_table = self.table.copy()

        e_min, e_max = self.e_min, self.e_max

        # Compute e_ref
        if method == 'table':
            e_ref = input_table['e_ref'].quantity
        elif method == 'log_center':
            e_ref = np.sqrt(e_min * e_max)
        elif method == 'lafferty':
            # set e_ref that it represents the mean dnde in the given energy bin
            e_ref = self._e_ref_lafferty(model, e_min, e_max)
        else:
            raise ValueError('Invalid method: {}'.format(method))

        flux = input_table['flux'].quantity
        dnde = self._dnde_from_flux(flux, model, e_ref, e_min, e_max, pwl_approx)

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
            dnde_ul = self._dnde_from_flux(flux_ul, model, e_ref, e_min, e_max, pwl_approx)
            table['dnde_ul'] = dnde_ul

        table.meta['SED_TYPE'] = 'dnde'
        return FluxPoints(table)

    @staticmethod
    def _e_ref_lafferty(model, e_min, e_max):
        """Helper for `to_sed_type`.

        Compute e_ref that the value at e_ref corresponds
        to the mean value between e_min and e_max.
        """
        flux = model.integral(e_min, e_max)
        dnde_mean = flux / (e_max - e_min)
        return model.inverse(dnde_mean)

    @staticmethod
    def _dnde_from_flux(flux, model, e_ref, e_min, e_max, pwl_approx):
        """Helper for `to_sed_type`.

        Compute dnde under the assumption that flux equals expected
        flux from model.
        """
        dnde_model = model(e_ref)

        if pwl_approx:
            index = model.spectral_index(e_ref)
            flux_model = power_law_integral_flux(f=dnde_model, g=index, e=e_ref,
                                                 e1=e_min, e2=e_max)
        else:
            flux_model = model.integral(e_min, e_max, intervals=True)

        return dnde_model * (flux / flux_model)

    @property
    def sed_type(self):
        """SED type (str).

        One of: {'dnde', 'e2dnde', 'flux', 'eflux'}
        """
        return self.table.meta['SED_TYPE']

    @staticmethod
    def _guess_sed_type(table):
        """Guess SED type from table content."""
        valid_sed_types = list(REQUIRED_COLUMNS.keys())
        for sed_type in valid_sed_types:
            required = set(REQUIRED_COLUMNS[sed_type])
            if required.issubset(table.colnames):
                return sed_type

    @staticmethod
    def _guess_sed_type_from_unit(unit):
        """Guess SED type from unit."""
        for sed_type, default_unit in DEFAULT_UNIT.items():
            if unit.is_equivalent(default_unit):
                return sed_type

    @staticmethod
    def _validate_table(table):
        """Validate input table."""
        # TODO: do we really want to error out on tables that don't have `SED_TYPE` in meta?
        # If yes, then this needs to be documented in the docstring,
        # and the workaround pointed out (to add the meta key before creating FluxPoints).
        sed_type = table.meta['SED_TYPE']
        required = set(REQUIRED_COLUMNS[sed_type])

        if not required.issubset(table.colnames):
            missing = required.difference(table.colnames)
            raise ValueError("Missing columns for sed type '{}':"
                             " {}".format(sed_type, missing))

    def _get_y_energy_unit(self, y_unit):
        """Get energy part of the given y unit."""
        try:
            return [_ for _ in y_unit.bases if _.physical_type == 'energy'][0]
        except IndexError:
            return u.Unit('TeV')

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
            # asymmetric error
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

    @property
    def e_ref(self):
        """Reference energy.

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
            return np.sqrt(self.e_min * self.e_max)

    @property
    def e_min(self):
        """Lower bound of energy bin.

        Defined by `e_min` column in `FluxPoints.table`.

        Returns
        -------
        e_min : `~astropy.units.Quantity`
            Lower bound of energy bin.
        """
        return self.table['e_min'].quantity

    @property
    def e_max(self):
        """Upper bound of energy bin.

        Defined by ``e_max`` column in ``table``.

        Returns
        -------
        e_max : `~astropy.units.Quantity`
            Upper bound of energy bin.
        """
        return self.table['e_max'].quantity

    def plot(self, ax=None, sed_type=None, energy_unit='TeV', flux_unit=None,
             energy_power=0, **kwargs):
        """Plot flux points.

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
            Keyword arguments passed to :func:`matplotlib.pyplot.errorbar`

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

            # pop label keyword to avoid that it appears twice in the legend
            kwargs.pop('label', None)
            ax.errorbar(x[is_ul].value, y_ul[is_ul].value, xerr=x_err, yerr=y_err,
                        uplims=True, **kwargs)

        ax.set_xscale('log', nonposx='clip')
        ax.set_yscale('log', nonposy='clip')
        ax.set_xlabel('Energy ({})'.format(energy_unit))
        ax.set_ylabel('{} ({})'.format(self.sed_type, y_unit))
        return ax


class FluxPointEstimator(object):
    """Flux point estimator.

    Computes flux points for a given spectrum observation dataset
    (a 1-dim on/off observation), energy binning and spectral model.

    Parameters
    ----------
    obs : `~gammapy.spectrum.SpectrumObservation`
        Spectrum observation
    groups : `~gammapy.spectrum.SpectrumEnergyGroups`
        Energy groups (usually output of `~gammapy.spectrum.SpectrumEnergyGroupMaker`)
    model : `~gammapy.spectrum.models.SpectralModel`
        Global model (usually output of `~gammapy.spectrum.SpectrumFit`)
    """

    def __init__(self, obs, groups, model):
        self.obs = obs
        self.groups = groups
        self.model = model
        self.flux_points = None

    def __str__(self):
        s = '{}:\n'.format(self.__class__.__name__)
        s += str(self.obs) + '\n'
        s += str(self.groups) + '\n'
        s += str(self.model) + '\n'
        return s

    def compute_points(self):
        rows = []
        for group in self.groups:
            if group.bin_type != 'normal':
                log.debug('Skipping energy group:\n{}'.format(group))
                continue

            row = self.compute_flux_point(group)
            rows.append(row)

        meta = OrderedDict([
            ('method', 'TODO'),
            ('SED_TYPE', 'dnde'),
        ])
        table = table_from_row_data(rows=rows, meta=meta)
        self.flux_points = FluxPoints(table)

    def compute_flux_point(self, energy_group):
        log.debug('Computing flux point for energy group:\n{}'.format(energy_group))
        model = self.compute_approx_model(
            global_model=self.model,
            energy_range=energy_group.energy_range,
        )
        energy_ref = self.compute_energy_ref(energy_group)

        return self.fit_point(
            model=model,
            energy_group=energy_group,
            energy_ref=energy_ref,
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

    def compute_flux_point_ul(self, fit, best_fit, delta_ts=4, negative=False):
        """
        Compute upper limits for flux point values.


        Parameters
        ----------
        fit : `SpectrumFit`
            Instance of spectrum fit.
        best_fit : `SpectrumFitResult`
            Best fit result.
        delta_ts : float (4)
            Difference in log-likelihood for given confidence interval.
            See Example below.
        negative : bool
            Compute limit in negative direction.


        Examples
        --------
        To compute ~95% confidence upper limits (or 2 sigma) you can use:

            from scipy.stats import chi2, norm

            sigma = 2
            cl = 1 - 2 * norm.sf(sigma) # using two sided p-value
            delta_ts = chi2.isf(1 - cl, df=1)


        Returns
        -------
        dnde_ul : `~astropy.units.Quantity`
            Flux point upper limit.

        """

        from scipy.optimize import brentq

        # this is a prototype for fast flux point upper limit
        # calculation using brentq
        stat_best_fit = best_fit.statval
        amplitude = best_fit.model.parameters['amplitude'].value / 1E-12
        amplitude_err = best_fit.model.parameters.error('amplitude') / 1E-12

        if negative:
            amplitude_max = amplitude
            amplitude_min = amplitude_max - 1E3 * amplitude_err
        else:
            amplitude_max = amplitude + 1E3 * amplitude_err
            amplitude_min = amplitude

        def ts_diff(x):
            fit.model.parameters['amplitude'].value = x * 1E-12
            fit.predict_counts()
            fit.calc_statval()
            return (stat_best_fit + delta_ts) - fit.total_stat

        try:
            result = brentq(ts_diff, amplitude_min, amplitude_max,
                            maxiter=100, rtol=1e-2)
            return 1E-12 * result * fit.model.parameters['amplitude'].unit
        except (RuntimeError, ValueError):
            # Where the root finding fails NaN is set as amplitude
            log.debug('Flux point upper limit computation failed.')
            return np.nan * fit.model.parameters['amplitude'].unit

    def compute_flux_point_sqrt_ts(self, fit, best_fit):
        """
        Compute sqrt(TS) for flux point.


        Parameters
        ----------
        fit : `SpectrumFit`
            Instance of spectrum fit.
        best_fit : `SpectrumFitResult`
            Best fit result.


        Returns
        -------
        sqrt_ts : float
            Sqrt(TS) for flux point.

        """
        amplitude = best_fit.model.parameters['amplitude'].value
        stat_best_fit = best_fit.statval

        fit.model.parameters['amplitude'].value = 0
        fit.predict_counts()
        fit.calc_statval()
        stat_null = fit.total_stat

        fit.model.parameters['amplitude'].value = amplitude
        ts = np.abs(stat_null - stat_best_fit)
        return np.sign(amplitude) * np.sqrt(ts)

    def fit_point(self, model, energy_group, energy_ref, sqrt_ts_threshold=1):
        from .fit import SpectrumFit

        # Set reference and remove min amplitude
        model.parameters['reference'].value = energy_ref.to('TeV').value

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
        dnde, dnde_err = res.model.evaluate_error(energy_ref)
        sqrt_ts = self.compute_flux_point_sqrt_ts(fit, best_fit=res)

        dnde_ul = self.compute_flux_point_ul(fit, best_fit=res)
        dnde_errp = self.compute_flux_point_ul(fit, best_fit=res, delta_ts=1.) - dnde
        dnde_errn = dnde - self.compute_flux_point_ul(fit, best_fit=res, delta_ts=1., negative=True)

        return OrderedDict([
            ('e_ref', energy_ref),
            ('e_min', e_min),
            ('e_max', e_max),
            ('dnde', dnde.to(DEFAULT_UNIT['dnde'])),
            ('dnde_err', dnde_err.to(DEFAULT_UNIT['dnde'])),
            ('dnde_ul', dnde_ul.to(DEFAULT_UNIT['dnde'])),
            ('is_ul', sqrt_ts < sqrt_ts_threshold),
            ('sqrt_ts', sqrt_ts),
            ('dnde_errp', dnde_errp),
            ('dnde_errn', dnde_errn)
        ])


class FluxPointProfiles(object):
    """Flux point likelihood profiles.

    See :ref:`gadf:likelihood_sed`.

    TODO: merge this class with the classes in ``fermipy/castro.py``,
    which are much more advanced / feature complete.
    This is just a temp solution because we don't have time for that.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table holding the data
    """

    def __init__(self, table):
        self.table = table

    @classmethod
    def read(cls, filename, **kwargs):
        """Read from file."""
        filename = make_path(filename)
        table = Table.read(str(filename), **kwargs)
        return cls(table=table)

    def plot(self, ax):
        # TODO: copy code from fermipy, don't start from scratch!
        raise NotImplementedError


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


class FluxPointFitter(object):
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
        from gammapy.spectrum import FluxPoints, FluxPointFitter
        from gammapy.spectrum.models import PowerLaw

        filename = '$GAMMAPY_EXTRA/test_datasets/spectrum/flux_points/diff_flux_points.fits'
        flux_points = FluxPoints.read(filename)

        model = PowerLaw(
            index=2. * u.Unit(''),
            amplitude=1e-12 * u.Unit('cm-2 s-1 TeV-1'),
            reference=1. * u.TeV,
        )

        fitter = FluxPointFitter()
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

        if ul_handling != 'ignore':
            raise NotImplementedError('No handling of upper limits implemented.')

        self.parameters = OrderedDict([
            ('optimizer', optimizer),
            ('error_estimator', error_estimator),
            ('ul_handling', ul_handling),
        ])

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
        model = model.copy()

        if p['optimizer'] in ['simplex', 'moncar', 'gridsearch', 'levmar']:
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
        model = model.copy()
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
        best_fit_model = self.fit(data, model)
        best_fit_model = self.estimate_errors(data, best_fit_model)
        dof = self.dof(data, best_fit_model)
        statval = self.statval(data, best_fit_model)[0]

        return OrderedDict([
            ('best-fit-model', best_fit_model),
            ('dof', int(dof)),
            ('statval', float(statval)),
            ('statval/dof', float(statval / dof)),
        ])
