# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from copy import deepcopy
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from astropy.io import fits
from astropy.io.registry import IORegistryError
from astropy.table import Table, vstack
from astropy.time import Time
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
from gammapy.data import GTI
from gammapy.maps import Map, MapAxis, Maps, RegionNDMap, TimeMapAxis
from gammapy.maps.axes import UNIT_STRING_FORMAT, flat_if_equal
from gammapy.modeling.models import TemplateSpectralModel
from gammapy.modeling.models.spectral import scale_plot_flux
from gammapy.modeling.scipy import stat_profile_ul_scipy
from gammapy.utils.interpolation import interpolate_profile
from gammapy.utils.scripts import make_path
from gammapy.utils.table import table_standardise_units_copy
from gammapy.utils.time import time_ref_to_dict
from ..map.core import DEFAULT_UNIT, FluxMaps

__all__ = ["FluxPoints"]

log = logging.getLogger(__name__)


def squash_fluxpoints(flux_point, axis):
    """Squash a FluxPoints object into one point.
    The log-likelihoods profiles in each bin are summed
    to compute the resultant quantities. Stat profiles
    must be present on the fluxpoints object for
    this method to work.
    """

    value_scan = flux_point.stat_scan.geom.axes["norm"].center
    stat_scan = np.sum(flux_point.stat_scan.data, axis=0).ravel()
    f = interp1d(value_scan, stat_scan, kind="quadratic", bounds_error=False)
    f = interpolate_profile(value_scan, stat_scan)
    minimizer = minimize(
        f,
        x0=value_scan[int(len(value_scan) / 2)],
        bounds=[(value_scan[0], value_scan[-1])],
        method="L-BFGS-B",
    )

    maps = Maps()
    geom = flux_point.geom.to_image()
    if axis.name != "energy":
        geom = geom.to_cube([flux_point.geom.axes["energy"]])

    maps["norm"] = Map.from_geom(geom, data=minimizer.x)
    maps["norm_err"] = Map.from_geom(
        geom, data=np.sqrt(minimizer.hess_inv.todense()).reshape(geom.data_shape)
    )
    maps["n_dof"] = Map.from_geom(geom, data=flux_point.geom.axes[axis.name].nbin)

    if "norm_ul" in flux_point.available_quantities:
        delta_ts = flux_point.meta.get("n_sigma_ul", 2) ** 2
        ul = stat_profile_ul_scipy(value_scan, stat_scan, delta_ts=delta_ts)
        maps["norm_ul"] = Map.from_geom(geom, data=ul.value)

    maps["stat"] = Map.from_geom(geom, data=f(minimizer.x))

    geom_scan = geom.to_cube([MapAxis.from_nodes(value_scan, name="norm")])
    maps["stat_scan"] = Map.from_geom(
        geom=geom_scan, data=stat_scan.reshape(geom_scan.data_shape)
    )
    try:
        maps["stat_null"] = Map.from_geom(geom, data=np.sum(flux_point.stat_null.data))
        maps["ts"] = maps["stat_null"] - maps["stat"]
    except AttributeError:
        log.info(
            "Stat null info not present on original FluxPoints object. TS not computed"
        )

    maps["success"] = Map.from_geom(geom=geom, data=minimizer.success, dtype=bool)

    combined_fp = FluxPoints.from_maps(
        maps=maps,
        sed_type=flux_point.sed_type_init,
        reference_model=flux_point.reference_model,
        gti=flux_point.gti,
        meta=flux_point.meta,
    )
    return combined_fp


class FluxPoints(FluxMaps):
    """Flux points container.

    The supported formats are described here: :ref:`gadf:flux-points`.

    In summary, the following formats and minimum required columns are:

    * Format ``dnde``: columns ``e_ref`` and ``dnde``
    * Format ``e2dnde``: columns ``e_ref``, ``e2dnde``
    * Format ``flux``: columns ``e_min``, ``e_max``, ``flux``
    * Format ``eflux``: columns ``e_min``, ``e_max``, ``eflux``

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table with flux point data.

    Attributes
    ----------
    table : `~astropy.table.Table`
        Table with flux point data.

    Examples
    --------
    The `FluxPoints` object is most easily created by reading a file with
    flux points given in one of the formats documented above::

    >>> from gammapy.estimators import FluxPoints
    >>> filename = '$GAMMAPY_DATA/hawc_crab/HAWC19_flux_points.fits'
    >>> flux_points = FluxPoints.read(filename)
    >>> flux_points.plot() #doctest: +SKIP

    An instance of `FluxPoints` can also be created by passing an instance of
    `astropy.table.Table`, which contains the required columns, such as `'e_ref'`
    and `'dnde'`. The corresponding `sed_type` has to be defined in the meta data
    of the table::

    >>> import numpy as np
    >>> from astropy import units as u
    >>> from astropy.table import Table
    >>> from gammapy.estimators import FluxPoints
    >>> from gammapy.modeling.models import PowerLawSpectralModel
    >>> table = Table()
    >>> pwl = PowerLawSpectralModel()
    >>> e_ref = np.geomspace(1, 100, 7) * u.TeV
    >>> table["e_ref"] = e_ref
    >>> table["dnde"] = pwl(e_ref)
    >>> table["dnde_err"] = pwl.evaluate_error(e_ref)[0]
    >>> table.meta["SED_TYPE"] = "dnde"
    >>> flux_points = FluxPoints.from_table(table)
    >>> flux_points.plot(sed_type="flux") #doctest: +SKIP

    If you have flux points in a different data format, the format can be changed
    by renaming the table columns and adding meta data::


    >>> from astropy import units as u
    >>> from astropy.table import Table
    >>> from gammapy.estimators import FluxPoints
    >>> from gammapy.utils.scripts import make_path

    >>> filename = make_path('$GAMMAPY_DATA/tests/spectrum/flux_points/flux_points_ctb_37b.txt')
    >>> table = Table.read(filename ,format='ascii.csv', delimiter=' ', comment='#')
    >>> table.rename_column('Differential_Flux', 'dnde')
    >>> table['dnde'].unit = 'cm-2 s-1 TeV-1'

    >>> table.rename_column('lower_error', 'dnde_errn')
    >>> table['dnde_errn'].unit = 'cm-2 s-1 TeV-1'

    >>> table.rename_column('upper_error', 'dnde_errp')
    >>> table['dnde_errp'].unit = 'cm-2 s-1 TeV-1'

    >>> table.rename_column('E', 'e_ref')
    >>> table['e_ref'].unit = 'TeV'

    >>> flux_points = FluxPoints.from_table(table, sed_type="dnde")
    >>> flux_points.plot(sed_type="e2dnde") #doctest: +SKIP


    Note: In order to reproduce the example you need the tests datasets folder.
    You may download it with the command
    ``gammapy download datasets --tests --out $GAMMAPY_DATA``
    """

    @classmethod
    def read(
        cls,
        filename,
        sed_type=None,
        format=None,
        reference_model=None,
        checksum=False,
        **kwargs,
    ):
        """Read precomputed flux points.

        Parameters
        ----------
        filename : str
            Filename.
        sed_type : {"dnde", "flux", "eflux", "e2dnde", "likelihood"}
            SED type.
        format : {"gadf-sed", "lightcurve", "profile"}, optional
            Format string. If None, the format is extracted from the input.
            Default is None.
        reference_model : `SpectralModel`
            Reference spectral model.
        checksum : bool
            If True checks both DATASUM and CHECKSUM cards in the file headers. Default is False.
        **kwargs : dict, optional
            Keyword arguments passed to `astropy.table.Table.read`.

        Returns
        -------
        flux_points : `FluxPoints`
            Flux points.
        """
        filename = make_path(filename)
        gti = None
        kwargs.setdefault("format", "ascii.ecsv")
        try:
            table = Table.read(filename, **kwargs)
        except (IORegistryError, UnicodeDecodeError):
            with fits.open(filename, checksum=checksum) as hdulist:
                if "FLUXPOINTS" in hdulist:
                    fp = hdulist["FLUXPOINTS"]
                else:
                    fp = hdulist[""]  # to handle older files
                table = Table.read(fp)
                if "GTI" in hdulist:
                    gti = GTI.from_table_hdu(hdulist["GTI"])

        return cls.from_table(
            table=table,
            sed_type=sed_type,
            reference_model=reference_model,
            format=format,
            gti=gti,
        )

    def write(
        self, filename, sed_type=None, format=None, overwrite=False, checksum=False
    ):
        """Write flux points.

        Parameters
        ----------
        filename : str
            Filename.
        sed_type : {"dnde", "flux", "eflux", "e2dnde", "likelihood"}, optional
            SED type. Default is None.
        format : {"gadf-sed", "lightcurve", "binned-time-series", "profile"}, optional
            Format specification. The following formats are supported:

                * "gadf-sed": format for SED flux points see :ref:`gadf:flux-points`
                  for details
                * "lightcurve": Gammapy internal format to store energy dependent
                  lightcurves. Basically a generalisation of the "gadf" format, but
                  currently there is no detailed documentation available.
                * "binned-time-series": table format support by Astropy's
                  `~astropy.timeseries.BinnedTimeSeries`.
                * "profile": Gammapy internal format to store energy dependent
                  flux profiles. Basically a generalisation of the "gadf" format, but
                  currently there is no detailed documentation available.

                If None, the format will be guessed by looking at the axes that are present in the object.
                Default is None.

        overwrite : bool, optional
            Overwrite existing file. Default is False.
        checksum : bool, optional
            When True adds both DATASUM and CHECKSUM cards to the headers written to the file.
            Default is False.
        """
        filename = make_path(filename)

        if sed_type is None:
            sed_type = self.sed_type_init
        table = self.to_table(sed_type=sed_type, format=format)

        if ".fits" not in filename.suffixes:
            table.write(filename, overwrite=overwrite)
            return

        primary_hdu = fits.PrimaryHDU()
        hdu_evt = fits.BinTableHDU(table, name="FLUXPOINTS")
        hdu_all = fits.HDUList([primary_hdu, hdu_evt])
        if self.gti:
            hdu_all.append(self.gti.to_table_hdu())

        hdu_all.writeto(filename, overwrite=overwrite, checksum=checksum)

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

    @staticmethod
    def _table_guess_format(table):
        """Format of the table to be transformed to FluxPoints."""
        names = table.colnames
        if "time_min" in names:
            return "lightcurve"
        elif "x_min" in names:
            return "profile"
        else:
            return "gadf-sed"

    @classmethod
    def from_table(
        cls, table, sed_type=None, format=None, reference_model=None, gti=None
    ):
        """Create flux points from a table. The table column names must be consistent with the
        sed_type.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table.
        sed_type : {"dnde", "flux", "eflux", "e2dnde", "likelihood"}, optional
            SED type. Default is None.
        format : {"gadf-sed", "lightcurve", "profile"}, optional
            Table format. If None, it is extracted from the table column content. Default is None.
        reference_model : `SpectralModel`, optional
            Reference spectral model. Default is None.
        gti : `GTI`, optional
            Good time intervals. Default is None.

        Returns
        -------
        flux_points : `FluxPoints`
            Flux points.
        """
        table = table_standardise_units_copy(table)

        if format is None:
            format = cls._table_guess_format(table)
            log.info("Inferred format: " + format)

        if sed_type is None:
            sed_type = table.meta.get("SED_TYPE", None)

        if sed_type is None:
            sed_type = cls._guess_sed_type(table.colnames)

        if sed_type is None:
            raise ValueError("Specifying the SED type is required")

        if sed_type == "likelihood":
            table = cls._convert_loglike_columns(table)
            if reference_model is None:
                reference_model = TemplateSpectralModel(
                    energy=flat_if_equal(table["e_ref"].quantity),
                    values=flat_if_equal(table["ref_dnde"].quantity),
                )

        maps = Maps()
        table.meta.setdefault("SED_TYPE", sed_type)

        for name in cls.all_quantities(sed_type=sed_type):
            if name in table.colnames:
                maps[name] = RegionNDMap.from_table(
                    table=table, colname=name, format=format
                )

        meta = cls._get_meta_gadf(table)
        return cls.from_maps(
            maps=maps,
            reference_model=reference_model,
            meta=meta,
            sed_type=sed_type,
            gti=gti,
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
        """Format table."""
        for column in table.colnames:
            if column.startswith(("dnde", "eflux", "flux", "e2dnde", "ref")):
                table[column].format = ".3e"
            elif column.startswith(
                ("e_min", "e_max", "e_ref", "sqrt_ts", "norm", "ts", "stat")
            ):
                table[column].format = ".3f"

        return table

    def _guess_format(self):
        """Format of the FluxPoints object."""
        names = self.geom.axes.names
        if "time" in names:
            return "lightcurve"
        elif "projected-distance" in names:
            return "profile"
        else:
            return "gadf-sed"

    def to_table(self, sed_type=None, format=None, formatted=False):
        """Create table for a given SED type.

        Parameters
        ----------
        sed_type : {"likelihood", "dnde", "e2dnde", "flux", "eflux"}
            SED type to convert to. Default is `likelihood`.
        format : {"gadf-sed", "lightcurve", "binned-time-series", "profile"}, optional
            Format specification. The following formats are supported:

                * "gadf-sed": format for SED flux points see :ref:`gadf:flux-points`
                  for details
                * "lightcurve": Gammapy internal format to store energy dependent
                  lightcurves. Basically a generalisation of the "gadf" format, but
                  currently there is no detailed documentation available.
                * "binned-time-series": table format support by Astropy's
                  `~astropy.timeseries.BinnedTimeSeries`.
                * "profile": Gammapy internal format to store energy dependent
                  flux profiles. Basically a generalisation of the "gadf" format, but
                  currently there is no detailed documentation available.

                If None, the format will be guessed by looking at the axes that are present in the object.
                Default is None.

        formatted : bool
            Formatted version with column formats applied. Numerical columns are
            formatted to .3f and .3e respectively. Default is False.

        Returns
        -------
        table : `~astropy.table.Table`
            Flux points table.

        Examples
        --------

        This is how to read and plot example flux points:

        >>> from gammapy.estimators import FluxPoints
        >>> fp = FluxPoints.read("$GAMMAPY_DATA/hawc_crab/HAWC19_flux_points.fits")
        >>> table = fp.to_table(sed_type="flux", formatted=True)
        >>> print(table[:2])
        e_ref e_min e_max     flux      flux_err    flux_ul      ts    sqrt_ts is_ul
         TeV   TeV   TeV  1 / (s cm2) 1 / (s cm2) 1 / (s cm2)
        ----- ----- ----- ----------- ----------- ----------- -------- ------- -----
        1.334 1.000 1.780   1.423e-11   3.135e-13         nan 2734.000  52.288 False
        2.372 1.780 3.160   5.780e-12   1.082e-13         nan 4112.000  64.125 False
        """
        if sed_type is None:
            sed_type = self.sed_type_init

        if format is None:
            format = self._guess_format()
            log.info("Inferred format: " + format)

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

            if not self.is_convertible_to_flux_sed_type:
                table.remove_columns(["e_min", "e_max"])

            if self.n_sigma_ul:
                table.meta["UL_CONF"] = np.round(
                    1 - 2 * stats.norm.sf(self.n_sigma_ul), 7
                )

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

            table["is_ul"] = self.is_ul.data[idx]
            if not self.has_ul:
                table.remove_columns("is_ul")

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

            # serialize with reference time set to mjd=0.0
            ref_time = Time(0.0, format="mjd", scale=time_axis.reference_time.scale)
            table.meta.update(time_ref_to_dict(ref_time, scale=ref_time.scale))
            table.meta["TIMEUNIT"] = "d"

        elif format == "binned-time-series":
            message = (
                "Format 'binned-time-series' support a single time axis "
                f"only. Got {self.geom.axes.names}"
            )

            if not self.geom.axes.is_unidimensional:
                raise ValueError(message)

            axis = self.geom.axes.primary_axis

            if not isinstance(axis, TimeMapAxis):
                raise ValueError(message)

            table = Table()
            table["time_bin_start"] = axis.time_min
            table["time_bin_size"] = axis.time_delta

            for quantity in self.all_quantities(sed_type=sed_type):
                data = getattr(self, quantity, None)
                if data:
                    table[quantity] = data.quantity.squeeze()
        elif format == "profile":
            x_axis = self.geom.axes["projected-distance"]

            tables = []
            for idx, (x_min, x_max) in enumerate(x_axis.iter_by_edges):
                table_flat = Table()
                table_flat["x_min"] = [x_min]
                table_flat["x_max"] = [x_max]
                table_flat["x_ref"] = [(x_max + x_min) / 2]

                fp = self.slice_by_idx(slices={"projected-distance": idx})
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
        """Compute flux error for given SED type"""
        y_errn, y_errp = None, None

        if "norm_err" in self.available_quantities:
            # symmetric error
            y_errn = getattr(self, sed_type + "_err")
            y_errp = y_errn.copy()

        if "norm_errp" in self.available_quantities:
            y_errn = getattr(self, sed_type + "_errn")
            y_errp = getattr(self, sed_type + "_errp")

        return y_errn, y_errp

    def plot(self, ax=None, sed_type=None, energy_power=0, time_format="iso", **kwargs):
        """Plot flux points.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis object to plot on. Default is None.
        sed_type : {"dnde", "flux", "eflux", "e2dnde"}, optional
            SED type. Default is None.
        energy_power : float, optional
            Power of energy to multiply flux axis with. Default is 0.
        time_format : {"iso", "mjd"}
            Used time format is a time axis is present. Default is "iso".
        **kwargs : dict, optional
            Keyword arguments passed to `~RegionNDMap.plot`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis object.
        """
        if sed_type is None:
            sed_type = self.sed_type_plot_default

        if not self.norm.geom.is_region:
            raise ValueError("Plotting only supported for region based flux points")

        if ax is None:
            ax = plt.gca()

        flux_unit = DEFAULT_UNIT[sed_type]

        flux = getattr(self, sed_type)

        # get errors and ul
        y_errn, y_errp = self._plot_get_flux_err(sed_type=sed_type)
        is_ul = self.is_ul.data

        if self.has_ul and y_errn and is_ul.any():
            flux_ul = getattr(self, sed_type + "_ul").quantity
            y_errn.data[is_ul] = np.clip(
                0.5 * flux_ul[is_ul].to_value(y_errn.unit), 0, np.inf
            )
            y_errp.data[is_ul] = 0
            flux.data[is_ul] = flux_ul[is_ul].to_value(flux.unit)
            kwargs.setdefault("uplims", is_ul)

        # set flux points plotting defaults
        if y_errp and y_errn:
            y_errp = np.clip(
                scale_plot_flux(y_errp, energy_power=energy_power).quantity, 0, np.inf
            )
            y_errn = np.clip(
                scale_plot_flux(y_errn, energy_power=energy_power).quantity, 0, np.inf
            )
            kwargs.setdefault("yerr", (y_errn, y_errp))
        else:
            kwargs.setdefault("yerr", None)

        flux = scale_plot_flux(flux=flux.to_unit(flux_unit), energy_power=energy_power)
        if "time" in flux.geom.axes_names:
            flux.geom.axes["time"].time_format = time_format
        ax = flux.plot(ax=ax, **kwargs)
        ax.set_ylabel(f"{sed_type} [{ax.yaxis.units.to_string(UNIT_STRING_FORMAT)}]")
        ax.set_yscale("log")
        return ax

    def plot_ts_profiles(
        self,
        ax=None,
        sed_type=None,
        add_cbar=True,
        **kwargs,
    ):
        """Plot fit statistic SED profiles as a density plot.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis object to plot on. Default is None.
        sed_type : {"dnde", "flux", "eflux", "e2dnde"}, optional
            SED type. Default is None.
        add_cbar : bool, optional
            Whether to add a colorbar to the plot. Default is True.
        **kwargs : dict, optional
            Keyword arguments passed to `~matplotlib.pyplot.pcolormesh`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis object.
        """
        if ax is None:
            ax = plt.gca()

        if sed_type is None:
            sed_type = self.sed_type_plot_default

        if not self.norm.geom.is_region:
            raise ValueError("Plotting only supported for region based flux points")

        if not self.geom.axes.is_unidimensional:
            raise ValueError(
                "Profile plotting is only supported for unidimensional maps"
            )

        axis = self.geom.axes.primary_axis

        if isinstance(axis, TimeMapAxis) and not axis.is_contiguous:
            axis = axis.to_contiguous()

        if ax.yaxis.units is None:
            yunits = DEFAULT_UNIT[sed_type]
        else:
            yunits = ax.yaxis.units

        ax.yaxis.set_units(yunits)

        flux_ref = getattr(self, sed_type + "_ref").to(yunits)

        ts = self.ts_scan

        norm_min, norm_max = ts.geom.axes["norm"].bounds.to_value("")

        flux = MapAxis.from_bounds(
            norm_min * flux_ref.value.min(),
            norm_max * flux_ref.value.max(),
            nbin=500,
            interp=axis.interp,
            unit=flux_ref.unit,
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
            caxes = ax.pcolormesh(axis.as_plot_edges, flux.edges, -z.T, **kwargs)

        axis.format_plot_xaxis(ax=ax)

        ax.set_ylabel(f"{sed_type} [{ax.yaxis.units.to_string(UNIT_STRING_FORMAT)}]")
        ax.set_yscale("log")

        if add_cbar:
            label = "Fit statistic difference"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        return ax

    def recompute_ul(self, n_sigma_ul=2, **kwargs):
        """Recompute upper limits corresponding to the given value.
        The pre-computed statistic profiles must exist for the re-computation.

        Parameters
        ----------
        n_sigma_ul : int
            Number of sigma to use for upper limit computation. Default is 2.
        **kwargs : dict, optional
            Keyword arguments passed to `~scipy.optimize.brentq`.

        Returns
        -------
        flux_points : `~gammapy.estimators.FluxPoints`
            A new FluxPoints object with modified upper limits.

        Examples
        --------
        >>> from gammapy.estimators import FluxPoints
        >>> filename = '$GAMMAPY_DATA/tests/spectrum/flux_points/binlike.fits'
        >>> flux_points = FluxPoints.read(filename)
        >>> flux_points_recomputed = flux_points.recompute_ul(n_sigma_ul=3)
        >>> print(flux_points.meta["n_sigma_ul"], flux_points.flux_ul.data[0])
        2.0 [[3.95451985e-09]]
        >>> print(flux_points_recomputed.meta["n_sigma_ul"], flux_points_recomputed.flux_ul.data[0])
        3 [[6.22245374e-09]]
        """

        if not self.has_stat_profiles:
            raise ValueError(
                "Stat profiles not present. Upper limit computation is not possible"
            )

        delta_ts = n_sigma_ul**2

        flux_points = deepcopy(self)

        value_scan = self.stat_scan.geom.axes["norm"].center
        shape_axes = self.stat_scan.geom._shape[slice(3, None)][::-1]
        for idx in np.ndindex(shape_axes):
            stat_scan = np.abs(
                self.stat_scan.data[idx].squeeze() - self.stat.data[idx].squeeze()
            )
            flux_points.norm_ul.data[idx] = stat_profile_ul_scipy(
                value_scan, stat_scan, delta_ts=delta_ts, **kwargs
            )
        flux_points.meta["n_sigma_ul"] = n_sigma_ul
        return flux_points

    def resample_axis(self, axis_new):
        """Rebin the flux point object along the new axis.
        The log-likelihoods profiles in each bin are summed
        to compute the resultant quantities.
        Stat profiles must be present on the fluxpoints object for
        this method to work.

        For now, works only for flat fluxpoints.

        Parameters
        ----------
        axis_new : `MapAxis` or `TimeMapAxis`
            The new axis to resample along

        Returns
        -------
        flux_points : `~gammapy.estimators.FluxPoints`
            A new FluxPoints object with modified axis.
        """

        if not self.has_stat_profiles:
            raise ValueError("Stat profiles not present, rebinning is not possible")

        fluxpoints = []
        for edge_min, edge_max in zip(axis_new.edges_min, axis_new.edges_max):
            if isinstance(axis_new, TimeMapAxis):
                edge_min = edge_min + axis_new.reference_time
                edge_max = edge_max + axis_new.reference_time
            fp = self.slice_by_coord({axis_new.name: slice(edge_min, edge_max)})
            fp_new = squash_fluxpoints(fp, axis_new)
            fluxpoints.append(fp_new)

        return self.__class__.from_stack(fluxpoints, axis=axis_new)
