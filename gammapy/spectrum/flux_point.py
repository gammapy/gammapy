# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from collections import OrderedDict
import numpy as np
from astropy.table import Table, vstack
from astropy import units as u
from astropy.io.registry import IORegistryError
from ..utils.scripts import make_path
from ..utils.table import table_standardise_units_copy, table_from_row_data
from ..utils.fitting import Fit
from ..utils.interpolation import ScaledRegularGridInterpolator
from .models import PowerLaw, ScaleModel
from .powerlaw import power_law_integral_flux
from .observation import SpectrumObservationList, SpectrumObservation

__all__ = ["FluxPoints", "FluxPointEstimator", "FluxPointsDataset"]

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = OrderedDict(
    [
        ("dnde", ["e_ref", "dnde"]),
        ("e2dnde", ["e_ref", "e2dnde"]),
        ("flux", ["e_min", "e_max", "flux"]),
        ("eflux", ["e_min", "e_max", "eflux"]),
        # TODO: extend required columns
        (
            "likelihood",
            [
                "e_min",
                "e_max",
                "e_ref",
                "ref_dnde",
                "norm",
                "norm_scan",
                "dloglike_scan",
            ],
        ),
    ]
)

OPTIONAL_COLUMNS = OrderedDict(
    [
        ("dnde", ["dnde_err", "dnde_errp", "dnde_errn", "dnde_ul", "is_ul"]),
        ("e2dnde", ["e2dnde_err", "e2dnde_errp", "e2dnde_errn", "e2dnde_ul", "is_ul"]),
        ("flux", ["flux_err", "flux_errp", "flux_errn", "flux_ul", "is_ul"]),
        ("eflux", ["eflux_err", "eflux_errp", "eflux_errn", "eflux_ul", "is_ul"]),
    ]
)

DEFAULT_UNIT = OrderedDict(
    [
        ("dnde", u.Unit("cm-2 s-1 TeV-1")),
        ("e2dnde", u.Unit("erg cm-2 s-1")),
        ("flux", u.Unit("cm-2 s-1")),
        ("eflux", u.Unit("erg cm-2 s-1")),
    ]
)


def _interp_likelihood_profile(norm_scan, dloglike_scan, norm):
    """Helper function to interpolate likelihood profiles"""
    # likelihood profiles are typically of parabolic shape, so we use a
    # sqrt scaling of the values and perform linear interpolation on the scaled
    # values
    sign = np.sign(np.gradient(dloglike_scan))
    interp = ScaledRegularGridInterpolator(
        points=(norm_scan,), values=sign * dloglike_scan, values_scale="sqrt"
    )
    return interp((norm,))


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

        from gammapy.spectrum import FluxPoints
        filename = '$GAMMAPY_DATA/tests/spectrum/flux_points/flux_points.fits'
        flux_points = FluxPoints.read(filename)
        flux_points.plot()

    An instance of `FluxPoints` can also be created by passing an instance of
    `astropy.table.Table`, which contains the required columns, such as `'e_ref'`
    and `'dnde'`. The corresponding `sed_type` has to be defined in the meta data
    of the table::

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
    by renaming the table columns and adding meta data::


        from astropy import units as u
        from astropy.table import Table
        from gammapy.spectrum import FluxPoints

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

    """

    def __init__(self, table):
        self.table = table_standardise_units_copy(table)
        # validate that the table is a valid representation
        # of the given flux point sed type
        self._validate_table(self.table, table.meta["SED_TYPE"])

    def __repr__(self):
        fmt = '{}(sed_type="{}", n_points={})'
        return fmt.format(self.__class__.__name__, self.sed_type, len(self.table))

    @property
    def table_formatted(self):
        """Return formatted version of the flux points table. Used for pretty printing"""
        table = self.table.copy()

        for column in table.colnames:
            if column.startswith(("dnde", "eflux", "flux", "e2dnde", "ref")):
                table[column].format = ".3e"
            elif column.startswith(
                ("e_min", "e_max", "e_ref", "sqrt_ts", "norm", "ts", "loglike")
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
            table = Table.read(str(filename), **kwargs)
        except IORegistryError:
            kwargs.setdefault("format", "ascii.ecsv")
            table = Table.read(str(filename), **kwargs)

        if "SED_TYPE" not in table.meta.keys():
            sed_type = cls._guess_sed_type(table)
            table.meta["SED_TYPE"] = sed_type

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
            kwargs.setdefault("format", "ascii.ecsv")
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

        >>> from gammapy.spectrum import FluxPoints
        >>> filename = '$GAMMAPY_DATA/tests/spectrum/flux_points/flux_points.fits'
        >>> flux_points = FluxPoints.read(filename)
        >>> print(flux_points)
        FluxPoints(sed_type="flux", n_points=24)
        >>> print(flux_points.drop_ul())
        FluxPoints(sed_type="flux", n_points=19)
        """
        table_drop_ul = self.table[~self.is_ul]
        return self.__class__(table_drop_ul)

    def _flux_to_dnde(self, e_ref, table, model, pwl_approx):
        if model is None:
            model = PowerLaw()

        e_min, e_max = self.e_min, self.e_max

        flux = table["flux"].quantity
        dnde = self._dnde_from_flux(flux, model, e_ref, e_min, e_max, pwl_approx)

        # Add to result table
        table["e_ref"] = e_ref
        table["dnde"] = dnde

        if "flux_err" in table.colnames:
            table["dnde_err"] = dnde * table["flux_err"].quantity / flux

        if "flux_errn" in table.colnames:
            table["dnde_errn"] = dnde * table["flux_errn"].quantity / flux
            table["dnde_errp"] = dnde * table["flux_errp"].quantity / flux

        if "flux_ul" in table.colnames:
            flux_ul = table["flux_ul"].quantity
            dnde_ul = self._dnde_from_flux(
                flux_ul, model, e_ref, e_min, e_max, pwl_approx
            )
            table["dnde_ul"] = dnde_ul

        return table

    @staticmethod
    def _dnde_to_e2dnde(e_ref, table):
        for suffix in ["", "_ul", "_err", "_errp", "_errn"]:
            try:
                data = table["dnde" + suffix].quantity
                table["e2dnde" + suffix] = (e_ref ** 2 * data).to(
                    DEFAULT_UNIT["e2dnde"]
                )
            except KeyError:
                continue

        return table

    @staticmethod
    def _e2dnde_to_dnde(e_ref, table):
        for suffix in ["", "_ul", "_err", "_errp", "_errn"]:
            try:
                data = table["e2dnde" + suffix].quantity
                table["dnde" + suffix] = (data / e_ref ** 2).to(DEFAULT_UNIT["dnde"])
            except KeyError:
                continue

        return table

    def to_sed_type(self, sed_type, method="log_center", model=None, pwl_approx=False):
        """Convert to a different SED type (return new `FluxPoints`).

        See: http://adsabs.harvard.edu/abs/1995NIMPA.355..541L for details
        on the `'lafferty'` method.

        Parameters
        ----------
        sed_type : {'dnde'}
             SED type to convert to.
        model : `~gammapy.spectrum.models.SpectralModel`
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
        >>> from gammapy.spectrum import FluxPoints
        >>> from gammapy.spectrum.models import PowerLaw
        >>> filename = '$GAMMAPY_DATA/tests/spectrum/flux_points/flux_points.fits'
        >>> flux_points = FluxPoints.read(filename)
        >>> model = PowerLaw(index=2.2)
        >>> flux_points_dnde = flux_points.to_sed_type('dnde', model=model)
        """
        # TODO: implement other directions.
        table = self.table.copy()

        if self.sed_type == "flux" and sed_type == "dnde":
            # Compute e_ref
            if method == "table":
                e_ref = table["e_ref"].quantity
            elif method == "log_center":
                e_ref = np.sqrt(self.e_min * self.e_max)
            elif method == "lafferty":
                # set e_ref that it represents the mean dnde in the given energy bin
                e_ref = self._e_ref_lafferty(model, self.e_min, self.e_max)
            else:
                raise ValueError("Invalid method: {}".format(method))
            table = self._flux_to_dnde(e_ref, table, model, pwl_approx)

        elif self.sed_type == "dnde" and sed_type == "e2dnde":
            table = self._dnde_to_e2dnde(self.e_ref, table)

        elif self.sed_type == "e2dnde" and sed_type == "dnde":
            table = self._e2dnde_to_dnde(self.e_ref, table)

        elif self.sed_type == "likelihood" and sed_type in ["dnde", "flux", "eflux"]:
            for suffix in ["", "_ul", "_err", "_errp", "_errn"]:
                try:
                    table[sed_type + suffix] = (
                        table["ref_" + sed_type] * table["norm" + suffix]
                    )
                except KeyError:
                    continue
        else:
            raise NotImplementedError

        table.meta["SED_TYPE"] = sed_type
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
            flux_model = power_law_integral_flux(
                f=dnde_model, g=index, e=e_ref, e1=e_min, e2=e_max
            )
        else:
            flux_model = model.integral(e_min, e_max, intervals=True)

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

    def _get_y_energy_unit(self, y_unit):
        """Get energy part of the given y unit."""
        try:
            return [_ for _ in y_unit.bases if _.physical_type == "energy"][0]
        except IndexError:
            return u.Unit("TeV")

    def _plot_get_energy_err(self):
        """Compute energy error for given sed type"""
        try:
            e_min = self.table["e_min"].quantity
            e_max = self.table["e_max"].quantity
            e_ref = self.e_ref
            x_err = ((e_ref - e_min), (e_max - e_ref))
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
            return self.table["e_ref"].quantity
        except KeyError:
            return np.sqrt(self.e_min * self.e_max)

    @property
    def e_edges(self):
        """Edges of the energy bin.

        Returns
        -------
        e_edges : `~astropy.units.Quantity`
            Energy edges.
        """
        e_edges = list(self.e_min)
        e_edges += [self.e_max[-1]]
        return u.Quantity(e_edges, self.e_min.unit, copy=False)

    @property
    def e_min(self):
        """Lower bound of energy bin.

        Defined by `e_min` column in `FluxPoints.table`.

        Returns
        -------
        e_min : `~astropy.units.Quantity`
            Lower bound of energy bin.
        """
        return self.table["e_min"].quantity

    @property
    def e_max(self):
        """Upper bound of energy bin.

        Defined by ``e_max`` column in ``table``.

        Returns
        -------
        e_max : `~astropy.units.Quantity`
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
        x = self.e_ref.to(energy_unit)

        # get errors and ul
        is_ul = self.is_ul
        x_err_all = self._plot_get_energy_err()
        y_err_all = self._plot_get_flux_err(sed_type)

        # handle energy power
        e_unit = self._get_y_energy_unit(y_unit)
        y_unit = y.unit * e_unit ** energy_power
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
                **kwargs
            )

        ax.set_xscale("log", nonposx="clip")
        ax.set_yscale("log", nonposy="clip")
        ax.set_xlabel("Energy ({})".format(energy_unit))
        ax.set_ylabel("{} ({})".format(self.sed_type, y_unit))
        return ax

    def plot_likelihood(
        self,
        ax=None,
        energy_unit="TeV",
        add_cbar=True,
        y_values=None,
        y_unit=None,
        **kwargs
    ):
        """Plot likelihood SED profiles as a density plot..

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axis object to plot on.
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
        y_values : `astropy.units.Quantity`
            Array of y-values to use for the likelihood profile evaluation.
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
                np.log10(ref_values.value.min()) - 1,
                np.log10(ref_values.value.max()) + 1,
                500,
            )
            y_values = u.Quantity(y_values, y_unit, copy=False)

        x = self.e_edges.to(energy_unit)

        # Compute likelihood "image" one energy bin at a time
        # by interpolating e2dnde at the log bin centers
        z = np.empty((len(self.table), len(y_values)))
        for idx in range(len(self.table)):
            y_ref = self.table["ref_" + self.sed_type].quantity[idx]
            norm = (y_values / y_ref).to_value("")
            norm_scan = self.table[idx]["norm_scan"]
            dloglike_scan = (
                self.table[idx]["dloglike_scan"] - self.table[idx]["loglike"]
            )
            z[idx] = _interp_likelihood_profile(norm_scan, dloglike_scan, norm)

        kwargs.setdefault("vmax", 0)
        kwargs.setdefault("vmin", -4)
        kwargs.setdefault("zorder", 0)
        kwargs.setdefault("cmap", "Blues")
        kwargs.setdefault("linewidths", 0)

        # clipped values are set to NaN so that they appear white on the plot
        z[-z < kwargs["vmin"]] = np.nan
        caxes = ax.pcolormesh(x, y_values, -z.T, **kwargs)
        ax.set_xscale("log", nonposx="clip")
        ax.set_yscale("log", nonposy="clip")
        ax.set_xlabel("Energy ({})".format(energy_unit))
        ax.set_ylabel("{} ({})".format(self.sed_type, y_values.unit))

        if add_cbar:
            label = "delta log-likelihood"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        return ax


class FluxPointEstimator:
    """Flux point estimator.

    Estimates flux points for a given spectrum observation dataset, energy groups
    and spectral model.

    To estimate the flux point the amplitude of the reference spectral model is
    fitted within the energy range defined by the energy group. This is done for
    each group independently. The amplitude is re-normalized using the "norm" parameter,
    which specifies the deviation of the flux from the reference model in this
    energy group. See https://gamma-astro-data-formats.readthedocs.io/en/latest/spectra/binned_likelihoods/index.html
    for details.

    The method is also described in the FERMI-LAT catalog paper
    https://ui.adsabs.harvard.edu/#abs/2015ApJS..218...23A
    or the HESS Galactic Plane Survey paper
    https://ui.adsabs.harvard.edu/#abs/2018A%26A...612A...1H

    Parameters
    ----------
    obs : `~gammapy.spectrum.SpectrumObservation` or `~gammapy.spectrum.SpectrumObservationList`
        Spectrum observation(s)
    groups : `~gammapy.spectrum.SpectrumEnergyGroups`
        Energy groups (usually output of `~gammapy.spectrum.SpectrumEnergyGroupMaker`)
    model : `~gammapy.spectrum.models.SpectralModel`
        Global model (usually output of `~gammapy.spectrum.SpectrumFit`)
    norm_min : float
        Minimum value for the norm used for the likelihood profile evaluation.
    norm_max : float
        Maximum value for the norm used for the likelihood profile evaluation.
    norm_n_values : int
        Number of norm values used for the likelihood profile.
    norm_values : `numpy.ndarray`
        Array of norm values to be used for the likelihood profile.
    sigma : int
        Sigma to use for asymmetric error computation.
    sigma_ul : int
        Sigma to use for upper limit computation.
    """

    def __init__(
        self,
        obs,
        groups,
        model,
        norm_min=0.2,
        norm_max=5,
        norm_n_values=11,
        norm_values=None,
        sigma=1,
        sigma_ul=2,
    ):
        if isinstance(obs, SpectrumObservation):
            obs = SpectrumObservationList([obs])
        self.obs = SpectrumObservationList(obs)

        self.groups = groups
        self.model = ScaleModel(model)
        self.model.parameters["norm"].min = 0
        if norm_values is None:
            norm_values = np.logspace(
                np.log10(norm_min), np.log10(norm_max), norm_n_values
            )
        self.norm_values = norm_values
        self.sigma = sigma
        self.sigma_ul = sigma_ul

    @property
    def ref_model(self):
        return self.model.model

    def __str__(self):
        s = "{}:\n".format(self.__class__.__name__)
        s += str(self.obs) + "\n"
        s += str(self.groups) + "\n"
        s += str(self.model) + "\n"
        return s

    def run(self, steps="all"):
        """Run the flux point estimator for all energy groups.

        Returns
        -------
        flux_points : `FluxPoints`
            Estimated flux points.
        steps : list of str
            Which steps to execute. See `estimate_flux_point` for details
            and available options.
        """
        rows = []
        for group in self.groups:
            if group.bin_type != "normal":
                log.debug("Skipping energy group:\n{}".format(group))
                continue

            row = self.estimate_flux_point(group, steps=steps)
            rows.append(row)

        meta = OrderedDict([("SED_TYPE", "likelihood")])
        table = table_from_row_data(rows=rows, meta=meta)
        return FluxPoints(table).to_sed_type("dnde")

    def estimate_flux_point(self, e_group, steps="all"):
        """Estimate flux point for a single energy group.

        Parameters
        ----------
        e_group : `SpectrumEnergyGroup`
            Energy group to compute the flux point for.
        steps : list of str
            Which steps to execute. Available options are:

                * "err": estimate symmetric error.
                * "errn-errp": estimate asymmetric errors.
                * "ul": estimate upper limits.
                * "ts": estimate ts and sqrt(ts) values.
                * "norm-scan": estimate likelihood profiles.

            By default all steps are executed.

        Returns
        -------
        result : dict
            Dict with results for the flux point.
        """
        # Put at log center of the bin
        e_min, e_max = e_group.energy_min, e_group.energy_max
        e_ref = np.sqrt(e_min * e_max)

        result = OrderedDict(
            [
                ("e_ref", e_ref),
                ("e_min", e_min),
                ("e_max", e_max),
                ("ref_dnde", self.ref_model(e_ref)),
                ("ref_flux", self.ref_model.integral(e_min, e_max)),
                ("ref_eflux", self.ref_model.energy_flux(e_min, e_max)),
                ("ref_e2dnde", self.ref_model(e_ref) * e_ref ** 2),
            ]
        )

        quality_orig = self._set_quality(e_group)
        result.update(self.estimate_norm())

        if steps == "all":
            steps = ["err", "errp-errn", "ul", "ts", "norm-scan"]

        if "err" in steps:
            result.update(self.estimate_norm_err())

        if "errp-errn" in steps:
            result.update(self.estimate_norm_errn_errp())

        if "ul" in steps:
            result.update(self.estimate_norm_ul())

        if "ts" in steps:
            result.update(self.estimate_norm_ts())

        if "norm-scan" in steps:
            result.update(self.estimate_norm_scan(result))

        self._restore_quality(quality_orig)
        return result

    def estimate_norm_errn_errp(self):
        """Estimate asymmetric errors for a flux point.

        Returns
        -------
        result : dict
            Dict with asymmetric errors for the flux point norm.
        """
        result = self.fit.confidence(parameter="norm", sigma=self.sigma)
        return {"norm_errp": result["errp"], "norm_errn": result["errn"]}

    def estimate_norm_err(self):
        """Estimate covariance errors for a flux point.

        Returns
        -------
        result : dict
            Dict with symmetric error for the flux point norm.
        """
        result = self.fit.covariance()
        norm_err = result.parameters.error("norm")
        return {"norm_err": norm_err}

    def estimate_norm_ul(self):
        """Estimate upper limit for a flux point.

        Returns
        -------
        result : dict
            Dict with upper limit for the flux point norm.
        """
        norm = self.model.parameters["norm"].value
        result = self.fit.confidence(parameter="norm", sigma=self.sigma_ul)
        return {"norm_ul": result["errp"] + norm}

    def estimate_norm_ts(self):
        """Estimate ts and sqrt(ts) for the flux point.

        Returns
        -------
        result : dict
            Dict with ts and srtq_ts for the flux point.
        """
        parameters = self.model.parameters

        loglike = self.fit.total_stat(parameters)
        norm_best_fit = parameters["norm"].value

        # store best fit amplitude, set amplitude of fit model to zero
        parameters["norm"].value = 0
        loglike_null = self.fit.total_stat(parameters)
        parameters["norm"].value = 1.0

        # compute sqrt TS
        ts = np.abs(loglike_null - loglike)
        sqrt_ts = np.sign(norm_best_fit) * np.sqrt(ts)
        return {"sqrt_ts": sqrt_ts, "ts": ts}

    def estimate_norm_scan(self, flux_point):
        """Estimate likelihood profile for the norm parameter

        Returns
        -------
        result : dict
            Dict with norm_scan and dloglike_scan for the flux point.
        """
        # norm = self.model.parameters["norm"]
        result = self.fit.likelihood_profile("norm", values=self.norm_values)
        dloglike_scan = result["likelihood"]
        return {"norm_scan": result["values"], "dloglike_scan": dloglike_scan}

    def estimate_norm(self):
        """Fit norm of the flux point.

        Returns
        -------
        result : dict
            Dict with "norm" and "loglike" for the flux point.
        """
        from .fit import SpectrumFit

        self.fit = SpectrumFit(self.obs, self.model)
        result = self.fit.optimize()

        if result.success:
            norm = result.parameters["norm"].value
            self.model.parameters["norm"].value = norm
        else:
            emin, emax = self.fit.true_fit_range[0]
            log.warning(
                "Fit failed for flux point between {emin:.3f} and {emax:.3f},"
                " setting NaN.".format(emin=emin, emax=emax)
            )
            norm = np.nan

        return {"norm": norm, "loglike": result.total_stat}

    # TODO: clean up this helper function and maybe move it to SpectrumObservations
    def _set_quality(self, energy_group):
        quality_orig = []

        for index in range(len(self.obs)):
            # Set quality bins to only bins in energy_group
            quality_orig_unit = self.obs[index].on_vector.quality
            quality_len = len(quality_orig_unit)
            quality_orig.append(quality_orig_unit)
            quality = np.zeros(quality_len, dtype=int)
            for bin in range(quality_len):
                if (
                    (bin < energy_group.bin_idx_min)
                    or (bin > energy_group.bin_idx_max)
                    or (energy_group.bin_type != "normal")
                ):
                    quality[bin] = 1
            self.obs[index].on_vector.quality = quality

        return quality_orig

    def _restore_quality(self, quality_orig):
        for index in range(len(quality_orig)):
            self.obs[index].on_vector.quality = quality_orig[index]


class FluxPointsDataset:
    """
    Fit a set of flux points with a parametric model.

    Parameters
    ----------
    model : `~gammapy.spectrum.models.SpectralModel`
        Spectral model
    data : `~gammapy.spectrum.FluxPoints`
        Flux points.
    mask : `numpy.ndarray`
        Mask to apply to the likelihood.
    likelihood : {"chi2", "chi2assym"}
        Likelihood function to use for the fit.

    Examples
    --------

    Load flux points from file and fit with a power-law model::

        from astropy import units as u
        from gammapy.spectrum import FluxPoints, FluxPointsDataset
        form gammapy.utils.fitting import Fit
        from gammapy.spectrum.models import PowerLaw

        filename = '$GAMMAPY_DATA/tests/spectrum/flux_points/diff_flux_points.fits'
        flux_points = FluxPoints.read(filename)

        model = PowerLaw()

        dataset = FluxPointsDataset(model, flux_points)
        fit = Fit(dataset)
        result = fit.run()
        print(result)
        print(result.model)
    """

    def __init__(self, model, data, mask=None, likelihood="chi2"):
        self.model = model
        self.data = data
        self.mask = mask
        self.parameters = model.parameters

        if likelihood in ["chi2", "chi2assym"]:
            self._likelihood = likelihood
        else:
            raise ValueError(
                "'{likelihood}' is not a valid fit statistic, please choose"
                " either 'chi2' or 'chi2assym'"
            )

    def data_shape(self):
        """Shape of the flux points data"""
        return self.data.e_ref.shape

    @staticmethod
    def _likelihood_chi2(data, model, sigma):
        return ((data - model) / sigma).to_value("") ** 2

    @staticmethod
    def _likelihood_chi2_assym(data, model, sigma_n, sigma_p):
        """
        Assymetric chi2 statistics for a list of flux points and model.
        """
        is_p = (model > data)
        sigma = sigma_n
        sigma[is_p] = sigma_p[is_p]
        return FluxPointsDataset._likelihood_chi2(data, model, sigma)

    def flux_pred(self):
        """Compute predicted flux.
        """
        return self.model(self.data.e_ref)

    def likelihood_per_bin(self):
        """Likelihood per bin given the current model parameters"""
        model = self.flux_pred()
        data = self.data.table["dnde"].quantity

        if self._likelihood == "chi2":
            sigma = self.data.table["dnde_err"].quantity
            return self._likelihood_chi2(data, model, sigma)
        elif self._likelihood == "chi2assym":
            sigma_n = self.data.table["dnde_errn"].quantity
            sigma_p = self.data.table["dnde_errp"].quantity
            return self._likelihood_chi2_assym(data, model, sigma_n, sigma_p)
        else:
            # TODO: add likelihood profiles
            pass

    def likelihood(self, parameters, mask=None):
        """Total likelihood given the current model parameters.

        Parameters
        ----------
        mask : `~numpy.ndarray`
            Mask to be combined with the dataset mask.
        """
        if self.mask is None and mask is None:
            stat = self.likelihood_per_bin()
        elif self.mask is None:
            stat = self.likelihood_per_bin()[mask]
        elif mask is None:
            stat = self.likelihood_per_bin()[self.mask]
        else:
            stat = self.likelihood_per_bin()[mask & self.mask]
        return np.nansum(stat, dtype=np.float64)

