# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from scipy import stats
from astropy import units as u
from astropy.io.registry import IORegistryError
from astropy.table import Table, vstack
from astropy.visualization import quantity_support
from gammapy.datasets import Datasets
from gammapy.modeling.models import TemplateSpectralModel
from gammapy.modeling.models.spectral import scale_plot_flux
from gammapy.modeling import Fit
from gammapy.maps import RegionNDMap, Maps, TimeMapAxis, MapAxis
from gammapy.utils.scripts import make_path
from gammapy.utils.pbar import progress_bar
from gammapy.utils.table import table_from_row_data, table_standardise_units_copy
from .flux_map import (
    FluxMaps,
    DEFAULT_UNIT,
)
from. flux import FluxEstimator


__all__ = ["FluxPoints", "FluxPointsEstimator"]

log = logging.getLogger(__name__)


class FluxPoints(FluxMaps):
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

        import numpy as np
        from astropy import units as u
        from astropy.table import Table
        from gammapy.estimators import FluxPoints
        from gammapy.modeling.models import PowerLawSpectralModel

        table = Table()
        pwl = PowerLawSpectralModel()
        e_ref = np.geomspace(1, 100, 7) * u.TeV

        table["e_ref"] = e_ref
        table["dnde"] = pwl(e_ref)
        table.meta["SED_TYPE"] = "dnde"

        flux_points = FluxPoints.from_table(table)
        flux_points.plot(sed_type="flux")

    If you have flux points in a different data format, the format can be changed
    by renaming the table columns and adding meta data::


        from astropy import units as u
        from astropy.table import Table
        from gammapy.estimators import FluxPoints
        from gammapy.utils.scripts import make_path

        table = Table.read(make_path('$GAMMAPY_DATA/tests/spectrum/flux_points/flux_points_ctb_37b.txt'),
                           format='ascii.csv', delimiter=' ', comment='#')
        table.rename_column('Differential_Flux', 'dnde')
        table['dnde'].unit = 'cm-2 s-1 TeV-1'

        table.rename_column('lower_error', 'dnde_errn')
        table['dnde_errn'].unit = 'cm-2 s-1 TeV-1'

        table.rename_column('upper_error', 'dnde_errp')
        table['dnde_errp'].unit = 'cm-2 s-1 TeV-1'

        table.rename_column('E', 'e_ref')
        table['e_ref'].unit = 'TeV'

        flux_points = FluxPoints.from_table(table, sed_type="dnde")
        flux_points.plot(sed_type="e2dnde")


    Note: In order to reproduce the example you need the tests datasets folder.
    You may download it with the command
    ``gammapy download datasets --tests --out $GAMMAPY_DATA``
    """

    @classmethod
    def read(cls, filename, sed_type=None, reference_model=None, **kwargs):
        """Read flux points.

        Parameters
        ----------
        filename : str
            Filename
        sed_type : {"dnde", "flux", "eflux", "e2dnde", "likelihood"}
            Sed type
        reference_model : `SpectralModel`
            Reference spectral model
        **kwargs : dict
            Keyword arguments passed to `astropy.table.Table.read`.

        Returns
        -------
        flux_points : `FluxPoints`
            Flux points
        """
        filename = make_path(filename)

        try:
            table = Table.read(filename, **kwargs)
        except IORegistryError:
            kwargs.setdefault("format", "ascii.ecsv")
            table = Table.read(filename, **kwargs)

        return cls.from_table(table=table, sed_type=sed_type, reference_model=reference_model)

    def write(self, filename, sed_type="likelihood", **kwargs):
        """Write flux points.

        Parameters
        ----------
        filename : str
            Filename
        sed_type : {"dnde", "flux", "eflux", "e2dnde", "likelihood"}
            Sed type
        kwargs : dict
            Keyword arguments passed to `astropy.table.Table.write`.
        """
        filename = make_path(filename)
        table = self.to_table(sed_type=sed_type)
        table.write(filename, **kwargs)

    @staticmethod
    def _convert_loglike_columns(table):
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

        return table

    @classmethod
    def from_table(cls, table, sed_type=None, reference_model=None, gti=None):
        """Create flux points from table

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table
        sed_type : {"dnde", "flux", "eflux", "e2dnde", "likelihood"}
            Sed type
        reference_model : `SpectralModel`
            Reference spectral model
        gti : `GTI`
            Good time intervals

        Returns
        -------
        flux_points : `FluxPoints`
            Flux points
        """
        table = table_standardise_units_copy(table)

        if sed_type is None:
            sed_type = table.meta.get("SED_TYPE", None)

        if sed_type is None:
            sed_type = cls._guess_sed_type(table.colnames)

        if sed_type is None:
            raise ValueError("Specifying the sed type is required")

        if sed_type == "likelihood":
            table = cls._convert_loglike_columns(table)
            if reference_model is None:
                reference_model = TemplateSpectralModel(
                    energy=table["e_ref"].quantity,
                    values=table["ref_dnde"].quantity
                )

        maps = Maps()
        table.meta.setdefault("SED_TYPE", sed_type)

        for name in cls.all_quantities(sed_type=sed_type):
            if name in table.colnames:
                maps[name] = RegionNDMap.from_table(
                    table=table, colname=name, format="gadf-sed"
                )

        meta = cls._get_meta_gadf(table)
        return cls.from_maps(
            maps=maps,
            reference_model=reference_model,
            meta=meta,
            sed_type=sed_type,
            gti=gti
        )

    @staticmethod
    def _get_meta_gadf(table):
        meta = {}
        meta.update(table.meta)
        conf_ul = table.meta.get("UL_CONF")
        if conf_ul:
            n_sigma_ul = np.round(stats.norm.isf(0.5 * (1 - conf_ul)), 1)
            meta["n_sigma_ul"] = n_sigma_ul
        meta["sed_type_init"] = table.meta.get("SED_TYPE")
        return meta

    @staticmethod
    def _format_table(table):
        """Format table"""
        for column in table.colnames:
            if column.startswith(("dnde", "eflux", "flux", "e2dnde", "ref")):
                table[column].format = ".3e"
            elif column.startswith(
                ("e_min", "e_max", "e_ref", "sqrt_ts", "norm", "ts", "stat")
            ):
                table[column].format = ".3f"

        return table

    def to_table(self, sed_type="likelihood", format="gadf-sed", formatted=False):
        """Create table for a given SED type.

        Parameters
        ----------
        sed_type : {"likelihood", "dnde", "e2dnde", "flux", "eflux"}
            sed type to convert to. Default is `likelihood`
        format : {"gadf-sed"}
            Format
        formatted : bool
            Formatted version with column formats applied. Numerical columns are
            formatted to .3f and .3e respectively.

        Returns
        -------
        table : `~astropy.table.Table`
            Flux points table
        """
        if format == "gadf-sed":
            # TODO: what to do with GTI info?
            if not self.geom.axes.names == ["energy"]:
                raise ValueError(
                    "Only flux points with a single energy axis "
                    "can be converted to 'gadf-sed'"
                )

            idx = (Ellipsis, 0, 0)
            table = self.energy_axis.to_table(format="gadf-sed")
            table.meta["SED_TYPE"] = sed_type

            if self.n_sigma_ul:
                table.meta["UL_CONF"] = np.round(1 - 2 * stats.norm.sf(2), 2)

            if sed_type == "likelihood":
                table["ref_dnde"] = self.dnde_ref[idx]
                table["ref_flux"] = self.flux_ref[idx]
                table["ref_eflux"] = self.eflux_ref[idx]

            for quantity in self.all_quantities(sed_type=sed_type):
                data = getattr(self, quantity, None)
                if data:
                    table[quantity] = data.quantity[idx]

            if self.has_stat_profiles:
                norm_axis = self.stat_scan.geom.axes["norm"]
                table["norm_scan"] = norm_axis.center.reshape((1, -1))
                table["stat"] = self.stat.data[idx]
                table["stat_scan"] = self.stat_scan.data[idx]
        elif format == "lightcurve":
            time_axis = self.geom.axes["time"]

            tables = []
            for idx, (time_min, time_max) in enumerate(time_axis.iter_by_edges):
                table_flat = Table()
                table_flat["time_min"] = [time_min.mjd]
                table_flat["time_max"] = [time_max.mjd]

                fp = self.slice_by_idx(slices={"time": idx})
                table = fp.to_table(sed_type=sed_type, format="gadf-sed")

                for column in table.columns:
                    table_flat[column] = table[column][np.newaxis]

                tables.append(table_flat)

            table = vstack(tables)
        else:
            raise ValueError(f"Not a supported format {format}")

        if formatted:
            table = self._format_table(table=table)

        return table

    @staticmethod
    def _energy_ref_lafferty(model, energy_min, energy_max):
        """Helper for `to_sed_type`.

        Compute energy_ref that the value at energy_ref corresponds
        to the mean value between energy_min and energy_max.
        """
        flux = model.integral(energy_min, energy_max)
        dnde_mean = flux / (energy_max - energy_min)
        return model.inverse(dnde_mean)

    def _plot_get_flux_err(self, sed_type=None):
        """Compute flux error for given sed type"""
        y_errn, y_errp = None, None

        if "norm_err" in self.available_quantities:
            # symmetric error
            y_errn = getattr(self, sed_type + "_err")
            y_errp = y_errn.copy()

        if "norm_errp" in self.available_quantities:
            y_errn = getattr(self, sed_type + "_errn")
            y_errp = getattr(self, sed_type + "_errp")

        return y_errn, y_errp

    def plot(
        self, ax=None, sed_type="dnde", energy_power=0, **kwargs
    ):
        """Plot flux points.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axis object to plot on.
        sed_type : {"dnde", "flux", "eflux", "e2dnde"}
            Sed type
        energy_power : float
            Power of energy to multiply flux axis with
        **kwargs : dict
            Keyword arguments passed to `~RegionNDMap.plot`

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis object
        """
        import matplotlib.pyplot as plt

        if not self.norm.geom.is_region:
            raise ValueError("Plotting only supported for region based flux points")

        if ax is None:
            ax = plt.gca()

        flux_unit = DEFAULT_UNIT[sed_type]

        flux = getattr(self, sed_type)

        # get errors and ul
        y_errn, y_errp = self._plot_get_flux_err(sed_type=sed_type)

        is_ul = self.is_ul.data
        if y_errn and is_ul.any():
            flux_ul = getattr(self, sed_type + "_ul").quantity
            y_errn.data[is_ul] = 0.5 * flux_ul[is_ul].to_value(y_errn.unit)
            y_errp.data[is_ul] = 0
            flux.data[is_ul] = flux_ul[is_ul].to_value(flux.unit)

        # set flux points plotting defaults
        if y_errp:
            y_errp = scale_plot_flux(y_errp, energy_power=energy_power).quantity

        if y_errn:
            y_errn = scale_plot_flux(y_errn, energy_power=energy_power).quantity

        kwargs.setdefault("yerr", (y_errn, y_errp))
        kwargs.setdefault("uplims", is_ul)

        flux = scale_plot_flux(flux=flux.to_unit(flux_unit), energy_power=energy_power)
        ax = flux.plot(ax=ax, **kwargs)
        ax.set_ylabel(f"{sed_type} ({ax.yaxis.units})")
        ax.set_yscale("log")
        return ax

    def plot_ts_profiles(
        self,
        ax=None,
        sed_type="dnde",
        add_cbar=True,
        **kwargs,
    ):
        """Plot fit statistic SED profiles as a density plot.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axis object to plot on.
        sed_type : {"dnde", "flux", "eflux", "e2dnde"}
            Sed type
        add_cbar : bool
            Whether to add a colorbar to the plot.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.pcolormesh`

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis object
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        if not self.norm.geom.is_region:
            raise ValueError("Plotting only supported for region based flux points")

        if not self.geom.axes.is_unidimensional:
            raise ValueError("Profile plotting is only supported for unidimensional maps")

        axis = self.geom.axes.primary_axis

        if isinstance(axis, TimeMapAxis) and not axis.is_contiguous:
            axis = axis.to_contiguous()

        yunits = kwargs.pop("yunits", DEFAULT_UNIT[sed_type])

        flux_ref = getattr(self, sed_type + "_ref").to(yunits)

        ts = self.ts_scan

        norm_min, norm_max = ts.geom.axes["norm"].bounds.to_value("")

        flux = MapAxis.from_bounds(
            norm_min * flux_ref.value.min(),
            norm_max * flux_ref.value.max(),
            nbin=500,
            interp=axis.interp,
            unit=flux_ref.unit
        )

        norm = flux.center / flux_ref.reshape((-1, 1))

        coords = ts.geom.get_coord()
        coords["norm"] = norm
        coords[axis.name] = axis.center.reshape((-1, 1))

        z = ts.interp_by_coord(coords, values_scale="stat-profile")

        kwargs.setdefault("vmax", 0)
        kwargs.setdefault("vmin", -4)
        kwargs.setdefault("zorder", 0)
        kwargs.setdefault("cmap", "Blues")
        kwargs.setdefault("linewidths", 0)
        kwargs.setdefault("shading", "auto")

        # clipped values are set to NaN so that they appear white on the plot
        z[-z < kwargs["vmin"]] = np.nan

        with quantity_support():
            caxes = ax.pcolormesh(
                axis.as_plot_edges, flux.edges, -z.T, **kwargs
            )

        axis.format_plot_xaxis(ax=ax)

        ax.set_ylabel(f"{sed_type} ({ax.yaxis.units})")
        ax.set_yscale("log")

        if add_cbar:
            label = "Fit statistic difference"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        return ax


class FluxPointsEstimator(FluxEstimator):
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
    selection_optional : list of str
        Which additional quantities to estimate. Available options are:

            * "all": all the optional steps are executed
            * "errn-errp": estimate asymmetric errors on flux.
            * "ul": estimate upper limits.
            * "scan": estimate fit statistic profiles.

        Default is None so the optionnal steps are not executed.
    fit : `Fit`
        Fit instance specifying the backend and fit options.
    reoptimize : bool
        Re-optimize other free model parameters. Default is True.
    """
    tag = "FluxPointsEstimator"

    def __init__(
        self,
        energy_edges=[1, 10] * u.TeV,
        **kwargs
    ):
        self.energy_edges = energy_edges

        fit = Fit(confidence_opts={"backend": "scipy"})
        kwargs.setdefault("fit", fit)
        super().__init__(**kwargs)

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
        # TODO: remove copy here...
        datasets = Datasets(datasets).copy()

        rows = []

        for energy_min, energy_max in progress_bar(
            zip(self.energy_edges[:-1], self.energy_edges[1:]),
            desc="Energy bins"
        ):
            row = self.estimate_flux_point(
                datasets, energy_min=energy_min, energy_max=energy_max,
            )
            rows.append(row)

        meta = {
            "n_sigma": self.n_sigma,
            "n_sigma_ul": self.n_sigma_ul,
            "sed_type_init": "likelihood"
        }

        table = table_from_row_data(rows=rows, meta=meta)
        model = datasets.models[self.source]
        return FluxPoints.from_table(table, reference_model=model.copy(), gti=datasets.gti)

    def estimate_flux_point(self, datasets, energy_min, energy_max):
        """Estimate flux point for a single energy group.

        Parameters
        ----------
        datasets : `Datasets`
            Datasets
        energy_min, energy_max : `~astropy.units.Quantity`
            Energy bounds to compute the flux point for.

        Returns
        -------
        result : dict
            Dict with results for the flux point.
        """
        datasets_sliced = datasets.slice_by_energy(
            energy_min=energy_min, energy_max=energy_max
        )

        datasets_sliced.models = datasets.models.copy()
        return super().run(datasets=datasets_sliced)