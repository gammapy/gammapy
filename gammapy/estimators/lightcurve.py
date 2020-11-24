# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from gammapy.data import GTI
from gammapy.datasets import Datasets
from gammapy.utils.scripts import make_path
from gammapy.utils.table import table_from_row_data
from .core import Estimator
from .flux_point import FluxPoints, FluxPointsEstimator

__all__ = ["LightCurve", "LightCurveEstimator"]

log = logging.getLogger(__name__)


class LightCurve:
    """Lightcurve container.

    The lightcurve data is stored in ``table``.

    For now we only support times stored in MJD format!

    TODO: specification of format is work in progress
    See https://github.com/open-gamma-ray-astro/gamma-astro-data-formats/pull/61

    Usage: :ref:`time`

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table with lightcurve data
    """

    def __init__(self, table):
        self.table = table

    def __repr__(self):
        return f"{self.__class__.__name__}(len={len(self.table)})"

    @property
    def time_scale(self):
        """Time scale (str).

        Taken from table "TIMESYS" header.
        Common values: "TT" or "UTC".
        Assumed default is "UTC".
        """
        return self.table.meta.get("TIMESYS", "utc")

    @property
    def time_format(self):
        """Time format (str)."""
        return "mjd"

    # @property
    # def time_ref(self):
    #     """Time reference (`~astropy.time.Time`)."""
    #     return time_ref_from_dict(self.table.meta)

    def _make_time(self, colname):
        val = self.table[colname].data
        scale = self.time_scale
        format = self.time_format
        return Time(val, scale=scale, format=format)

    @property
    def time(self):
        """Time (`~astropy.time.Time`)."""
        return self.time_mid

    @property
    def time_min(self):
        """Time bin start (`~astropy.time.Time`)."""
        return self._make_time("time_min")

    @property
    def time_max(self):
        """Time bin end (`~astropy.time.Time`)."""
        return self._make_time("time_max")

    @property
    def time_mid(self):
        """Time bin center (`~astropy.time.Time`)."""
        return self.time_min + 0.5 * self.time_delta

    @property
    def time_delta(self):
        """Time bin width (`~astropy.time.TimeDelta`)."""
        return self.time_max - self.time_min

    @classmethod
    def read(cls, filename, **kwargs):
        """Read from file.

        Parameters
        ----------
        filename : str
            Filename
        kwargs : dict
            Keyword arguments passed to `astropy.table.Table.read`.
        """
        table = Table.read(make_path(filename), **kwargs)
        return cls(table=table)

    def write(self, filename, **kwargs):
        """Write to file.

        Parameters
        ----------
        filename : str
            Filename
        kwargs : dict
            Keyword arguments passed to `astropy.table.Table.write`.
        """
        self.table.write(make_path(filename), **kwargs)

    def plot(
        self,
        ax=None,
        energy_index=None,
        time_format="mjd",
        flux_unit="cm-2 s-1",
        **kwargs,
    ):
        """Plot flux points.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional.
            The `~matplotlib.axes.Axes` object to be drawn on.
            If None, uses the current `~matplotlib.axes.Axes`.
        energy_index : int
            The index of the energy band to use. If set to None, use the first energy index.
            Default is None.
        time_format : {'mjd', 'iso'}, optional
            If 'iso', the x axis will contain Matplotlib dates.
            For formatting these dates see: https://matplotlib.org/gallery/ticks_and_spines/date_demo_rrule.html
        flux_unit : str, `~astropy.units.Unit`, optional
            Unit of the flux axis
        kwargs : dict
            Keyword arguments passed to :func:`matplotlib.pyplot.errorbar`

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis object
        """
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter

        if ax is None:
            ax = plt.gca()

        x, xerr = self._get_times_and_errors(time_format=time_format)
        y, yerr = self._get_fluxes_and_errors(unit=flux_unit)
        is_ul, yul = self._get_flux_uls(unit=flux_unit)

        if len(y.shape) > 1:
            if energy_index is None:
                energy_index = 0

            y = y[:, energy_index]
            if len(yerr) > 1:
                yerr = [_[:, energy_index] for _ in yerr]
            else:
                yerr = yerr[:, energy_index]
            is_ul = is_ul[:, energy_index]
            yul = yul[:, energy_index]

        # length of the ul arrow
        ul_arr = (
            np.nanmax(np.concatenate((y[~is_ul], yul[is_ul])))
            - np.nanmin(np.concatenate((y[~is_ul], yul[is_ul])))
        ) * 0.1

        # join fluxes and upper limits for the plot
        y[is_ul] = yul[is_ul]
        yerr[0][is_ul] = ul_arr

        # set plotting defaults and plot
        kwargs.setdefault("marker", "+")
        kwargs.setdefault("ls", "None")

        ax.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, uplims=is_ul, **kwargs)
        ax.set_xlabel("Time ({})".format(time_format.upper()))
        ax.set_ylabel("Flux ({:FITS})".format(u.Unit(flux_unit)))
        ax.legend()
        if time_format == "iso":
            ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M:%S"))
            plt.setp(
                ax.xaxis.get_majorticklabels(),
                rotation=30,
                ha="right",
                rotation_mode="anchor",
            )

        return ax

    def _get_fluxes_and_errors(self, unit="cm-2 s-1"):
        """Extract fluxes and corresponding errors

        Helper function for the plot method.

        Parameters
        ----------
        unit : str, `~astropy.units.Unit`, optional
            Unit of the returned flux and errors values

        Returns
        -------
        y : `numpy.ndarray`
            Flux values
        (yn, yp) : tuple of `numpy.ndarray`
            Flux error values
        """
        y = self.table["flux"].quantity.to(unit)

        if all(k in self.table.colnames for k in ["flux_errp", "flux_errn"]):
            yp = self.table["flux_errp"].quantity.to(unit)
            yn = self.table["flux_errn"].quantity.to(unit)
        elif "flux_err" in self.table.colnames:
            yp = self.table["flux_err"].quantity.to(unit)
            yn = self.table["flux_err"].quantity.to(unit)
        else:
            yp, yn = np.zeros_like(y), np.zeros_like(y)

        return y.value, (yn.value, yp.value)

    def _get_flux_uls(self, unit="cm-2 s-1"):
        """Extract flux upper limits

        Helper function for the plot method.

        Parameters
        ----------
        unit : str, `~astropy.units.Unit`, optional
            Unit of the returned flux upper limit values

        Returns
        -------
        is_ul : `numpy.ndarray`
            Is flux point is an upper limit? (boolean array)
        yul : `numpy.ndarray`
            Flux upper limit values
        """
        try:
            is_ul = self.table["is_ul"].data.astype("bool")
        except KeyError:
            is_ul = np.zeros_like(self.table["flux"]).data.astype("bool")

        if is_ul.any():
            yul = self.table["flux_ul"].quantity.to(unit)
        else:
            yul = np.zeros_like(self.table["flux"]).quantity
            yul[:] = np.nan

        return is_ul, yul.value

    def _get_times_and_errors(self, time_format="mjd"):
        """Extract times and corresponding errors

        Helper function for the plot method.

        Parameters
        ----------
        time_format : {'mjd', 'iso'}, optional
            Time format of the times. If 'iso', times and errors will be
            returned as `~datetime.datetime` and `~datetime.timedelta` objects

        Returns
        -------
        x : `~numpy.ndarray` or of `~datetime.datetime`
            Time values or `~datetime.datetime` instances if 'iso' is chosen
            as time format
        (xn, xp) : tuple of `numpy.ndarray` of `~datetime.timedelta`
            Tuple of time error values or `~datetime.timedelta` instances if
            'iso' is chosen as time format
        """
        x = self.time

        try:
            xn, xp = x - self.time_min, self.time_max - x
        except KeyError:
            xn, xp = x - x, x - x

        if time_format == "iso":
            x = x.datetime
            xn = xn.to_datetime()
            xp = xp.to_datetime()
        elif time_format == "mjd":
            x = x.mjd
            xn = xn.to("day").value
            xp = xp.to("day").value
        else:
            raise ValueError(f"Invalid time_format: {time_format}")

        return x, (xn, xp)


class LightCurveEstimator(Estimator):
    """Estimate light curve.

    The estimator will fit the source model component to datasets in each of the
    provided time intervals.

    If no time intervals are provided, the estimator will use the time intervals
    defined by the datasets GTIs.

    To be included in the estimation, the dataset must have their GTI fully
    overlapping a time interval.

    Parameters
    ----------
    time_intervals : list of `astropy.time.Time`
        Start and stop time for each interval to compute the LC
    source : str
        For which source in the model to compute the flux points. Default is 0
    energy_edges : `~astropy.units.Quantity`
        Energy edges of the light curve.
    atol : `~astropy.units.Quantity`
        Tolerance value for time comparison with different scale. Default 1e-6 sec.
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
        reoptimize other parameters during fit statistic scan?
    selection_optional : list of str
        Which steps to execute. Available options are:

            * "errn-errp": estimate asymmetric errors.
            * "ul": estimate upper limits.
            * "scan": estimate fit statistic profiles.

        By default all steps are executed.
    """

    tag = "LightCurveEstimator"
    _available_selection_optional = ["errn-errp", "ul", "scan"]

    def __init__(
        self,
        time_intervals=None,
        source=0,
        energy_edges=None,
        atol="1e-6 s",
        norm_min=0.2,
        norm_max=5,
        norm_n_values=11,
        norm_values=None,
        n_sigma=1,
        n_sigma_ul=2,
        reoptimize=False,
        selection_optional="all",
    ):

        self.source = source
        self.time_intervals = time_intervals

        self.atol = u.Quantity(atol)

        self.energy_edges = energy_edges

        self.norm_min = norm_min
        self.norm_max = norm_max
        self.norm_n_values = norm_n_values
        self.norm_values = norm_values
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.reoptimize = reoptimize
        self.selection_optional = selection_optional

    def run(self, datasets):
        """Run light curve extraction.

        Normalize integral and energy flux between emin and emax.

        Parameters
        ----------
        datasets : list of `~gammapy.datasets.SpectrumDataset` or `~gammapy.datasets.MapDataset`
            Spectrum or Map datasets.

        Returns
        -------
        lightcurve : `~gammapy.estimators.LightCurve`
            the Light Curve object
        """
        datasets = Datasets(datasets)

        if self.time_intervals is None:
            gti = datasets.gti
        else:
            gti = GTI.from_time_intervals(self.time_intervals)

        gti = gti.union(overlap_ok=False, merge_equal=False)

        rows = []

        for t_min, t_max in gti.time_intervals:
            datasets_to_fit = datasets.select_time(
                t_min=t_min, t_max=t_max, atol=self.atol
            )

            if len(datasets_to_fit) == 0:
                log.debug(f"No Dataset for the time interval {t_min} to {t_max}")
                continue

            row = {"time_min": t_min.mjd, "time_max": t_max.mjd}
            row.update(self.estimate_time_bin_flux(datasets_to_fit))
            rows.append(row)
        if len(rows) == 0:
            raise ValueError("LightCurveEstimator: No datasets in time intervals")

        table = table_from_row_data(rows=rows, meta={"SED_TYPE": "likelihood"})
        table = FluxPoints(table).to_sed_type("flux").table
        return LightCurve(table)

    def estimate_time_bin_flux(self, datasets):
        """Estimate flux point for a single energy group.

        Parameters
        ----------
        datasets : `~gammapy.modeling.Datasets`
            the list of dataset object

        Returns
        -------
        result : dict
            Dict with results for the flux point.
        """
        if self.energy_edges is None:
            energy_min, energy_max = datasets.energy_ranges
            energy_edges = energy_min.min(), energy_max.max()
        else:
            energy_edges = self.energy_edges

        fe = FluxPointsEstimator(
            source=self.source,
            energy_edges=energy_edges,
            norm_min=self.norm_min,
            norm_max=self.norm_max,
            norm_n_values=self.norm_n_values,
            norm_values=self.norm_values,
            n_sigma=self.n_sigma,
            n_sigma_ul=self.n_sigma_ul,
            reoptimize=self.reoptimize,
            selection_optional=self.selection_optional,
        )
        fp = fe.run(datasets)

        # TODO: remove once FluxPointsEstimator returns object with all energies in one row
        result = {}
        for colname in fp.table.colnames:
            if colname is not "counts":
                result[colname] = fp.table[colname].quantity
            else:
                result[colname] = np.atleast_1d(fp.table[colname].quantity.sum(axis=1))

        # return fp.to_sed_type("flux")#
        return result
