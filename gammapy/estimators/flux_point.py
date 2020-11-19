# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from astropy.io.registry import IORegistryError
from astropy.table import Table, vstack
from gammapy.datasets import Datasets
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.utils.interpolation import interpolate_profile
from gammapy.utils.scripts import make_path
from gammapy.utils.table import table_from_row_data, table_standardise_units_copy
from .core import Estimator
from .flux import FluxEstimator

__all__ = ["FluxPoints", "FluxPointsEstimator"]

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "dnde": ["e_ref", "dnde"],
    "e2dnde": ["e_ref", "e2dnde"],
    "flux": ["e_min", "e_max", "flux"],
    "eflux": ["e_min", "e_max", "eflux"],
    # TODO: extend required columns
    "likelihood": ["e_min", "e_max", "e_ref", "ref_dnde", "norm",],
}

OPTIONAL_COLUMNS = {
    "dnde": ["dnde_err", "dnde_errp", "dnde_errn", "dnde_ul", "is_ul"],
    "e2dnde": ["e2dnde_err", "e2dnde_errp", "e2dnde_errn", "e2dnde_ul", "is_ul"],
    "flux": ["flux_err", "flux_errp", "flux_errn", "flux_ul", "is_ul"],
    "eflux": ["eflux_err", "eflux_errp", "eflux_errn", "eflux_ul", "is_ul"],
    "likelihood": ["norm_scan", "stat_scan"],
}

DEFAULT_UNIT = {
    "dnde": u.Unit("cm-2 s-1 TeV-1"),
    "e2dnde": u.Unit("erg cm-2 s-1"),
    "flux": u.Unit("cm-2 s-1"),
    "eflux": u.Unit("erg cm-2 s-1"),
}


class FluxPoints:
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
    flux points given in one of the formats documented above::

        from gammapy.estimators import FluxPoints
        filename = '$GAMMAPY_DATA/hawc_crab/HAWC19_flux_points.fits'
        flux_points = FluxPoints.read(filename)
        flux_points.plot()

    An instance of `FluxPoints` can also be created by passing an instance of
    `astropy.table.Table`, which contains the required columns, such as `'e_ref'`
    and `'dnde'`. The corresponding `sed_type` has to be defined in the meta data
    of the table::

        from astropy import units as u
        from astropy.table import Table
        from gammapy.estimators import FluxPoints
        from gammapy.modeling.models import PowerLawSpectralModel

        table = Table()
        pwl = PowerLawSpectralModel()
        e_ref = np.logspace(0, 2, 7) * u.TeV
        table['e_ref'] = e_ref
        table['dnde'] = pwl(e_ref)
        table.meta['SED_TYPE'] = 'dnde'

        flux_points = FluxPoints(table)
        flux_points.plot()

    If you have flux points in a different data format, the format can be changed
    by renaming the table columns and adding meta data::


        from astropy import units as u
        from astropy.table import Table
        from gammapy.estimators import FluxPoints

        table = Table.read('$GAMMAPY_DATA/tests/spectrum/flux_points/flux_points_ctb_37b.txt',
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

    Note: In order to reproduce the example you need the tests datasets folder.
    You may download it with the command
    ``gammapy download datasets --tests --out $GAMMAPY_DATA``
    """

    def __init__(self, table):
        self.table = table_standardise_units_copy(table)
        # validate that the table is a valid representation
        # of the given flux point sed type
        self._validate_table(self.table, table.meta["SED_TYPE"])

    def __repr__(self):
        return f"{self.__class__.__name__}(sed_type={self.sed_type!r}, n_points={len(self.table)})"

    @property
    def table_formatted(self):
        """Return formatted version of the flux points table. Used for pretty printing"""
        table = self.table.copy()

        for column in table.colnames:
            if column.startswith(("dnde", "eflux", "flux", "e2dnde", "ref")):
                table[column].format = ".3e"
            elif column.startswith(
                ("e_min", "e_max", "e_ref", "sqrt_ts", "norm", "ts", "stat")
            ):
                table[column].format = ".3f"

        return table

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
            table = Table.read(filename, **kwargs)
        except IORegistryError:
            kwargs.setdefault("format", "ascii.ecsv")
            table = Table.read(filename, **kwargs)

        if "SED_TYPE" not in table.meta.keys():
            sed_type = cls._guess_sed_type(table)
            table.meta["SED_TYPE"] = sed_type

        # TODO: check sign and factor 2 here
        # https://github.com/gammapy/gammapy/pull/2546#issuecomment-554274318
        # The idea below is to support the format here:
        # https://gamma-astro-data-formats.readthedocs.io/en/latest/spectra/flux_points/index.html#likelihood-columns
        # but internally to go to the uniform "stat"

        if "loglike" in table.colnames and "stat" not in table.colnames:
            table["stat"] = 2 * table["loglike"]

        if "loglike_null" in table.colnames and "stat_null" not in table.colnames:
            table["stat_null"] = 2 * table["loglike_null"]

        if "dloglike_scan" in table.colnames and "stat_scan" not in table.colnames:
            table["stat_scan"] = 2 * table["dloglike_scan"]

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
            self.table.write(filename, **kwargs)
        except IORegistryError:
            kwargs.setdefault("format", "ascii.ecsv")
            self.table.write(filename, **kwargs)

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
        table_stacked.meta["SED_TYPE"] = reference.meta["SED_TYPE"]

        return cls(table_stacked)

    def drop_ul(self):
        """Drop upper limit flux points.

        Returns
        -------
        flux_points : `FluxPoints`
            Flux points with upper limit points removed.

        Examples
        --------
        >>> from gammapy.estimators import FluxPoints
        >>> filename = '$GAMMAPY_DATA/tests/spectrum/flux_points/flux_points.fits'
        >>> flux_points = FluxPoints.read(filename)
        >>> print(flux_points)
        FluxPoints(sed_type="flux", n_points=24)
        >>> print(flux_points.drop_ul())
        FluxPoints(sed_type="flux", n_points=19)

        Note: In order to reproduce the example you need the tests datasets folder.
        You may download it with the command
        ``gammapy download datasets --tests --out $GAMMAPY_DATA``
        """
        table_drop_ul = self.table[~self.is_ul]
        return self.__class__(table_drop_ul)

    def _flux_to_dnde(self, energy_ref, table, model, pwl_approx):
        if model is None:
            model = PowerLawSpectralModel()

        energy_min, energy_max = self.energy_min, self.energy_max

        flux = table["flux"].quantity
        dnde = self._dnde_from_flux(
            flux, model, energy_ref, energy_min, energy_max, pwl_approx
        )

        # Add to result table
        table["e_ref"] = energy_ref
        table["dnde"] = dnde

        if "flux_err" in table.colnames:
            table["dnde_err"] = dnde * table["flux_err"].quantity / flux

        if "flux_errn" in table.colnames:
            table["dnde_errn"] = dnde * table["flux_errn"].quantity / flux
            table["dnde_errp"] = dnde * table["flux_errp"].quantity / flux

        if "flux_ul" in table.colnames:
            flux_ul = table["flux_ul"].quantity
            dnde_ul = self._dnde_from_flux(
                flux_ul, model, energy_ref, energy_min, energy_max, pwl_approx
            )
            table["dnde_ul"] = dnde_ul

        return table

    @staticmethod
    def _dnde_to_e2dnde(energy_ref, table):
        for suffix in ["", "_ul", "_err", "_errp", "_errn"]:
            try:
                data = table["dnde" + suffix].quantity
                table["e2dnde" + suffix] = (energy_ref ** 2 * data).to(
                    DEFAULT_UNIT["e2dnde"]
                )
            except KeyError:
                continue

        return table

    @staticmethod
    def _e2dnde_to_dnde(energy_ref, table):
        for suffix in ["", "_ul", "_err", "_errp", "_errn"]:
            try:
                data = table["e2dnde" + suffix].quantity
                table["dnde" + suffix] = (data / energy_ref ** 2).to(
                    DEFAULT_UNIT["dnde"]
                )
            except KeyError:
                continue

        return table

    def to_sed_type(self, sed_type, method="log_center", model=None, pwl_approx=False):
        """Convert to a different SED type (return new `FluxPoints`).

        See: https://ui.adsabs.harvard.edu/abs/1995NIMPA.355..541L for details
        on the `'lafferty'` method.

        Parameters
        ----------
        sed_type : {'dnde'}
             SED type to convert to.
        model : `~gammapy.modeling.models.SpectralModel`
            Spectral model assumption.  Note that the value of the amplitude parameter
            does not matter. Still it is recommended to use something with the right
            scale and units. E.g. `amplitude = 1e-12 * u.Unit('cm-2 s-1 TeV-1')`
        method : {'lafferty', 'log_center', 'table'}
            Flux points `energy_ref` estimation method:

                * `'laferty'` Lafferty & Wyatt model-based energy_ref
                * `'log_center'` log bin center energy_ref
                * `'table'` using column 'energy_ref' from input flux_points
        pwl_approx : bool
            Use local power law appoximation at energy_ref to compute differential flux
            from the integral flux. This method is used by the Fermi-LAT catalogs.

        Returns
        -------
        flux_points : `FluxPoints`
            Flux points including differential quantity columns `dnde`
            and `dnde_err` (optional), `dnde_ul` (optional).

        Examples
        --------
        >>> from gammapy.estimators import FluxPoints
        >>> from gammapy.modeling.models import PowerLawSpectralModel
        >>> filename = '$GAMMAPY_DATA/tests/spectrum/flux_points/flux_points.fits'
        >>> flux_points = FluxPoints.read(filename)
        >>> model = PowerLawSpectralModel(index=2.2)
        >>> flux_points_dnde = flux_points.to_sed_type('dnde', model=model)

        Note: In order to reproduce the example you need the tests datasets folder.
        You may download it with the command
        ``gammapy download datasets --tests --out $GAMMAPY_DATA``
        """
        # TODO: implement other directions.
        table = self.table.copy()

        if self.sed_type == "flux" and sed_type == "dnde":
            # Compute energy_ref
            if method == "table":
                energy_ref = table["e_ref"].quantity
            elif method == "log_center":
                energy_ref = np.sqrt(self.energy_min * self.energy_max)
            elif method == "lafferty":
                # set energy_ref that it represents the mean dnde in the given energy bin
                energy_ref = self._energy_ref_lafferty(
                    model, self.energy_min, self.energy_max
                )
            else:
                raise ValueError(f"Invalid method: {method}")
            table = self._flux_to_dnde(energy_ref, table, model, pwl_approx)

        elif self.sed_type == "dnde" and sed_type == "e2dnde":
            table = self._dnde_to_e2dnde(self.energy_ref, table)

        elif self.sed_type == "e2dnde" and sed_type == "dnde":
            table = self._e2dnde_to_dnde(self.energy_ref, table)

        elif self.sed_type == "likelihood" and sed_type in ["dnde", "flux", "eflux"]:
            for suffix in ["", "_ul", "_err", "_errp", "_errn"]:
                try:
                    table[sed_type + suffix] = (
                        table["ref_" + sed_type] * table["norm" + suffix]
                    )
                except KeyError:
                    continue
        elif self.sed_type == sed_type:
            # do nothing if the sed type is the same
            pass

        else:
            raise NotImplementedError

        table.meta["SED_TYPE"] = sed_type
        return FluxPoints(table)

    @staticmethod
    def _energy_ref_lafferty(model, energy_min, energy_max):
        """Helper for `to_sed_type`.

        Compute energy_ref that the value at energy_ref corresponds
        to the mean value between energy_min and energy_max.
        """
        flux = model.integral(energy_min, energy_max)
        dnde_mean = flux / (energy_max - energy_min)
        return model.inverse(dnde_mean)

    @staticmethod
    def _dnde_from_flux(flux, model, energy_ref, energy_min, energy_max, pwl_approx):
        """Helper for `to_sed_type`.

        Compute dnde under the assumption that flux equals expected
        flux from model.
        """
        dnde_model = model(energy_ref)

        if pwl_approx:
            index = model.spectral_index(energy_ref)
            flux_model = PowerLawSpectralModel.evaluate_integral(
                energy_min=energy_min,
                energy_max=energy_max,
                index=index,
                reference=energy_ref,
                amplitude=dnde_model,
            )
        else:
            flux_model = model.integral(energy_min, energy_max)

        return dnde_model * (flux / flux_model)

    @property
    def sed_type(self):
        """SED type (str).

        One of: {'dnde', 'e2dnde', 'flux', 'eflux'}
        """
        return self.table.meta["SED_TYPE"]

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
    def _validate_table(table, sed_type):
        """Validate input table."""
        required = set(REQUIRED_COLUMNS[sed_type])

        if not required.issubset(table.colnames):
            missing = required.difference(table.colnames)
            raise ValueError(
                "Missing columns for sed type '{}':" " {}".format(sed_type, missing)
            )

    @staticmethod
    def _get_y_energy_unit(y_unit):
        """Get energy part of the given y unit."""
        try:
            return [_ for _ in y_unit.bases if _.physical_type == "energy"][0]
        except IndexError:
            return u.Unit("TeV")

    def _plot_get_energy_err(self):
        """Compute energy error for given sed type"""
        try:
            energy_min = self.energy_min
            energy_max = self.energy_max
            energy_ref = self.energy_ref
            x_err = ((energy_ref - energy_min), (energy_max - energy_ref))
        except KeyError:
            x_err = None
        return x_err

    def _plot_get_flux_err(self, sed_type=None):
        """Compute flux error for given sed type"""
        try:
            # asymmetric error
            y_errn = self.table[sed_type + "_errn"].quantity
            y_errp = self.table[sed_type + "_errp"].quantity
            y_err = (y_errn, y_errp)
        except KeyError:
            try:
                # symmetric error
                y_err = self.table[sed_type + "_err"].quantity
                y_err = (y_err, y_err)
            except KeyError:
                # no error at all
                y_err = None
        return y_err

    @property
    def is_ul(self):
        try:
            return self.table["is_ul"].data.astype("bool")
        except KeyError:
            return np.isnan(self.table[self.sed_type])

    @property
    def energy_ref(self):
        """Reference energy.

        Defined by `energy_ref` column in `FluxPoints.table` or computed as log
        center, if `energy_min` and `energy_max` columns are present in `FluxPoints.table`.

        Returns
        -------
        energy_ref : `~astropy.units.Quantity`
            Reference energy.
        """
        try:
            return self.table["e_ref"].quantity
        except KeyError:
            return np.sqrt(self.energy_min * self.energy_max)

    @property
    def energy_edges(self):
        """Edges of the energy bin.

        Returns
        -------
        energy_edges : `~astropy.units.Quantity`
            Energy edges.
        """
        energy_edges = list(self.energy_min)
        energy_edges += [self.energy_max[-1]]
        return u.Quantity(energy_edges, self.energy_min.unit, copy=False)

    @property
    def energy_min(self):
        """Lower bound of energy bin.

        Defined by `energy_min` column in `FluxPoints.table`.

        Returns
        -------
        energy_min : `~astropy.units.Quantity`
            Lower bound of energy bin.
        """
        return self.table["e_min"].quantity

    @property
    def energy_max(self):
        """Upper bound of energy bin.

        Defined by ``energy_max`` column in ``table``.

        Returns
        -------
        energy_max : `~astropy.units.Quantity`
            Upper bound of energy bin.
        """
        return self.table["e_max"].quantity

    def plot(
        self, ax=None, energy_unit="TeV", flux_unit=None, energy_power=0, **kwargs
    ):
        """Plot flux points.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axis object to plot on.
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

        sed_type = self.sed_type
        y_unit = u.Unit(flux_unit or DEFAULT_UNIT[sed_type])

        y = self.table[sed_type].quantity.to(y_unit)
        x = self.energy_ref.to(energy_unit)

        # get errors and ul
        is_ul = self.is_ul
        x_err_all = self._plot_get_energy_err()
        y_err_all = self._plot_get_flux_err(sed_type)

        # handle energy power
        energy_unit_y = self._get_y_energy_unit(y_unit)
        y_unit = y.unit * energy_unit_y ** energy_power
        y = (y * np.power(x, energy_power)).to(y_unit)

        y_err, x_err = None, None

        if y_err_all:
            y_errn = (y_err_all[0] * np.power(x, energy_power)).to(y_unit)
            y_errp = (y_err_all[1] * np.power(x, energy_power)).to(y_unit)
            y_err = (y_errn[~is_ul].to_value(y_unit), y_errp[~is_ul].to_value(y_unit))

        if x_err_all:
            x_errn, x_errp = x_err_all
            x_err = (
                x_errn[~is_ul].to_value(energy_unit),
                x_errp[~is_ul].to_value(energy_unit),
            )

        # set flux points plotting defaults
        kwargs.setdefault("marker", "+")
        kwargs.setdefault("ls", "None")

        ebar = ax.errorbar(
            x[~is_ul].value, y[~is_ul].value, yerr=y_err, xerr=x_err, **kwargs
        )

        if is_ul.any():
            if x_err_all:
                x_errn, x_errp = x_err_all
                x_err = (
                    x_errn[is_ul].to_value(energy_unit),
                    x_errp[is_ul].to_value(energy_unit),
                )

            y_ul = self.table[sed_type + "_ul"].quantity
            y_ul = (y_ul * np.power(x, energy_power)).to(y_unit)

            y_err = (0.5 * y_ul[is_ul].value, np.zeros_like(y_ul[is_ul].value))

            kwargs.setdefault("color", ebar[0].get_color())

            # pop label keyword to avoid that it appears twice in the legend
            kwargs.pop("label", None)
            ax.errorbar(
                x[is_ul].value,
                y_ul[is_ul].value,
                xerr=x_err,
                yerr=y_err,
                uplims=True,
                **kwargs,
            )

        ax.set_xscale("log", nonposx="clip")
        ax.set_yscale("log", nonposy="clip")
        ax.set_xlabel(f"Energy ({energy_unit})")
        ax.set_ylabel(f"{self.sed_type} ({y_unit})")
        return ax

    def plot_ts_profiles(
        self,
        ax=None,
        energy_unit="TeV",
        add_cbar=True,
        y_values=None,
        y_unit=None,
        **kwargs,
    ):
        """Plot fit statistic SED profiles as a density plot.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axis object to plot on.
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
        y_values : `astropy.units.Quantity`
            Array of y-values to use for the fit statistic profile evaluation.
        y_unit : str or `astropy.units.Unit`
            Unit to use for the y-axis.
        add_cbar : bool
            Whether to add a colorbar to the plot.
        kwargs : dict
            Keyword arguments passed to :func:`matplotlib.pyplot.pcolormesh`

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis object
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        self._validate_table(self.table, "likelihood")
        y_unit = u.Unit(y_unit or DEFAULT_UNIT[self.sed_type])

        if y_values is None:
            ref_values = self.table["ref_" + self.sed_type].quantity
            y_values = np.logspace(
                np.log10(0.2 * ref_values.value.min()),
                np.log10(5 * ref_values.value.max()),
                500,
            )
            y_values = u.Quantity(y_values, y_unit, copy=False)

        x = self.energy_edges.to(energy_unit)

        # Compute fit statistic "image" one energy bin at a time
        # by interpolating e2dnde at the log bin centers
        z = np.empty((len(self.table), len(y_values)))
        for idx, row in enumerate(self.table):
            y_ref = self.table["ref_" + self.sed_type].quantity[idx]
            norm = (y_values / y_ref).to_value("")
            norm_scan = row["norm_scan"]
            ts_scan = row["stat_scan"] - row["stat"]
            interp = interpolate_profile(norm_scan, ts_scan)
            z[idx] = interp((norm,))

        kwargs.setdefault("vmax", 0)
        kwargs.setdefault("vmin", -4)
        kwargs.setdefault("zorder", 0)
        kwargs.setdefault("cmap", "Blues")
        kwargs.setdefault("linewidths", 0)

        # clipped values are set to NaN so that they appear white on the plot
        z[-z < kwargs["vmin"]] = np.nan
        caxes = ax.pcolormesh(x.value, y_values.value, -z.T, **kwargs)
        ax.set_xscale("log", nonposx="clip")
        ax.set_yscale("log", nonposy="clip")
        ax.set_xlabel(f"Energy ({energy_unit})")
        ax.set_ylabel(f"{self.sed_type} ({y_values.unit})")

        if add_cbar:
            label = "fit statistic difference"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        return ax


class FluxPointsEstimator(Estimator):
    """Flux points estimator.

    Estimates flux points for a given list of datasets, energies and spectral model.

    To estimate the flux point the amplitude of the reference spectral model is
    fitted within the energy range defined by the energy group. This is done for
    each group independently. The amplitude is re-normalized using the "norm" parameter,
    which specifies the deviation of the flux from the reference model in this
    energy group. See https://gamma-astro-data-formats.readthedocs.io/en/latest/spectra/binned_likelihoods/index.html
    for details.

    The method is also described in the Fermi-LAT catalog paper
    https://ui.adsabs.harvard.edu/#abs/2015ApJS..218...23A
    or the HESS Galactic Plane Survey paper
    https://ui.adsabs.harvard.edu/#abs/2018A%26A...612A...1H

    Parameters
    ----------
    energy_edges : `~astropy.units.Quantity`
        Energy edges of the flux point bins.
    source : str or int
        For which source in the model to compute the flux points.
    norm_min : float
        Minimum value for the norm used for the fit statistic profile evaluation.
    norm_max : float
        Maximum value for the norm used for the fit statistic profile evaluation.
    norm_n_values : int
        Number of norm values used for the fit statistic profile.
    norm_values : `numpy.ndarray`
        Array of norm values to be used for the fit statistic profile.
    n_sigma : int
        Number of sigma to use for asymmetric error computation. Default is 1.
    n_sigma_ul : int
        Number of sigma to use for upper limit computation. Default is 2.
    reoptimize : bool
        Re-optimize other free model parameters.
    selection_optional : list of str
        Which additional quantities to estimate. Available options are:

            * "errn-errp": estimate asymmetric errors on flux.
            * "ul": estimate upper limits.
            * "norm-scan": estimate fit statistic profiles.

        By default all steps are executed.
    """

    tag = "FluxPointsEstimator"
    _available_selection_optional = ["errn-errp", "ul", "scan"]

    def __init__(
        self,
        energy_edges=[1, 10] * u.TeV,
        source=0,
        norm_min=0.2,
        norm_max=5,
        norm_n_values=11,
        norm_values=None,
        n_sigma=1,
        n_sigma_ul=2,
        reoptimize=False,
        selection_optional="all",
    ):
        self.energy_edges = energy_edges
        self.source = source
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.norm_n_values = norm_n_values
        self.norm_values = norm_values
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.reoptimize = reoptimize
        self.selection_optional = selection_optional

    def _flux_estimator(self, energy_min, energy_max):
        return FluxEstimator(
            source=self.source,
            energy_min=energy_min,
            energy_max=energy_max,
            norm_min=self.norm_min,
            norm_max=self.norm_max,
            norm_n_values=self.norm_n_values,
            norm_values=self.norm_values,
            n_sigma=self.n_sigma,
            n_sigma_ul=self.n_sigma_ul,
            reoptimize=self.reoptimize,
            selection_optional=self.selection_optional,
        )

    def run(self, datasets):
        """Run the flux point estimator for all energy groups.

        Parameters
        ----------
        datasets : list of `~gammapy.datasets.Dataset`
            Datasets

        Returns
        -------
        flux_points : `FluxPoints`
            Estimated flux points.
        """
        datasets = Datasets(datasets).copy()

        rows = []

        for energy_min, energy_max in zip(
            self.energy_edges[:-1], self.energy_edges[1:]
        ):
            row = self.estimate_flux_point(
                datasets, energy_min=energy_min, energy_max=energy_max
            )
            rows.append(row)

        table = table_from_row_data(rows=rows, meta={"SED_TYPE": "likelihood"})

        # TODO: this should be changed once likelihood is fully supported
        return FluxPoints(table).to_sed_type("dnde")

    def estimate_flux_point(self, datasets, energy_min, energy_max):
        """Estimate flux point for a single energy group.

        Parameters
        ----------
        datasets : Datasets
            Datasets
        energy_min, energy_max : `~astropy.units.Quantity`
            Energy bounds to compute the flux point for.

        Returns
        -------
        result : dict
            Dict with results for the flux point.
        """
        result = self.estimate_counts(
            datasets, energy_min=energy_min, energy_max=energy_max
        )
        fe = self._flux_estimator(energy_min=energy_min, energy_max=energy_max)

        result.update(fe.run(datasets=datasets))

        return result

    @staticmethod
    def estimate_counts(datasets, energy_min, energy_max):
        """Estimate counts for the flux point.

        Parameters
        ----------
        datasets : Datasets
            Datasets
        energy_min, energy_max : `~astropy.units.Quantity`
            Energy bounds to compute the flux point for.

        Returns
        -------
        result : dict
            Dict with an array with one entry per dataset with counts for the flux point.
        """
        counts = []

        for dataset in datasets:
            energy_mask = dataset.counts.geom.energy_mask(
                energy_min=energy_min, energy_max=energy_max, round_to_edge=True
            )
            mask = dataset.mask & energy_mask
            counts.append(dataset.counts.data[mask].sum())

        return {"counts": np.array(counts, dtype=int)}
