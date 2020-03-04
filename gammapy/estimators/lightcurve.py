# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from gammapy.datasets import Datasets
from gammapy.modeling import Fit
from gammapy.modeling.models import ScaleSpectralModel
from gammapy.estimators import FluxPoints
from gammapy.utils.table import table_from_row_data
from gammapy.utils.scripts import make_path

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

    def plot(self, ax=None, time_format="mjd", flux_unit="cm-2 s-1", **kwargs):
        """Plot flux points.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional.
            The `~matplotlib.axes.Axes` object to be drawn on.
            If None, uses the current `~matplotlib.axes.Axes`.
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

        x, xerr = self._get_times_and_errors(time_format)
        y, yerr = self._get_fluxes_and_errors(flux_unit)
        is_ul, yul = self._get_flux_uls(flux_unit)

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




def group_datasets_in_time_interval(datasets, time_intervals, atol="1e-6 s"):
    """Compute the table with the info on the group to which belong each dataset.

    The Tstart and Tstop are stored in MJD from a scale in "utc".

    Parameters
    ----------
    datasets : list of `~gammapy.spectrum.SpectrumDataset` or `~gammapy.cube.MapDataset`
        Spectrum or Map datasets.
    time_intervals : list of `astropy.time.Time`
        Start and stop time for each interval to compute the LC
    atol : `~astropy.units.Quantity`
        Tolerance value for time comparison with different scale. Default 1e-6 sec.

    Returns
    -------
    table_info : `~astropy.table.Table`
        Contains the grouping info for each dataset
    """
    dataset_group_ID_table = Table(
        names=("Name", "Tstart", "Tstop", "Bin_type", "Group_ID"),
        meta={"name": "first table"},
        dtype=("S10", "f8", "f8", "S10", "i8"),
    )
    time_intervals_lowedges = Time(
        [time_interval[0] for time_interval in time_intervals]
    )
    time_intervals_upedges = Time(
        [time_interval[1] for time_interval in time_intervals]
    )

    for dataset in datasets:
        tstart = dataset.gti.time_start[0]
        tstop = dataset.gti.time_stop[-1]
        mask1 = tstart >= time_intervals_lowedges - atol
        mask2 = tstop <= time_intervals_upedges + atol
        mask = mask1 & mask2
        if np.any(mask):
            group_index = np.where(mask)[0]
            bin_type = ""
        else:
            group_index = -1
            if np.any(mask1):
                bin_type = "Overflow"
            elif np.any(mask2):
                bin_type = "Underflow"
            else:
                bin_type = "Outflow"
        dataset_group_ID_table.add_row(
            [dataset.name, tstart.utc.mjd, tstop.utc.mjd, bin_type, group_index]
        )

    return dataset_group_ID_table


class LightCurveEstimator:
    """Compute light curve.

    The estimator will fit the source model component to datasets in each of the time intervals
    provided.

    If no time intervals are provided, the estimator will use the time intervals defined by the datasets GTIs.

    To be included in the estimation, the dataset must have their GTI fully overlapping a time interval.

    Parameters
    ----------
    datasets : list of `~gammapy.spectrum.SpectrumDataset` or `~gammapy.cube.MapDataset`
        Spectrum or Map datasets.
    time_intervals : list of `astropy.time.Time`
        Start and stop time for each interval to compute the LC
    source : str
        For which source in the model to compute the flux points. Default is ''
    norm_min : float
        Minimum value for the norm used for the fit statistic profile evaluation.
    norm_max : float
        Maximum value for the norm used for the fit statistic profile evaluation.
    norm_n_values : int
        Number of norm values used for the fit statistic profile.
    norm_values : `numpy.ndarray`
        Array of norm values to be used for the fit statistic profile.
    sigma : int
        Sigma to use for asymmetric error computation.
    sigma_ul : int
        Sigma to use for upper limit computation.
    reoptimize : bool
        reoptimize other parameters during fit statistic scan?
    """

    def __init__(
        self,
        datasets,
        time_intervals=None,
        source=0,
        norm_min=0.2,
        norm_max=5,
        norm_n_values=11,
        norm_values=None,
        sigma=1,
        sigma_ul=2,
        reoptimize=False,
    ):
        self.datasets = Datasets(datasets)

        if not self.datasets.is_all_same_type and self.datasets.is_all_same_shape:
            raise ValueError(
                "Light Curve estimation requires a list of datasets"
                " of the same type and data shape."
            )

        if time_intervals is None:
            time_intervals = [
                Time([d.gti.time_start[0], d.gti.time_stop[-1]]) for d in self.datasets
            ]

        self._check_and_sort_time_intervals(time_intervals)

        dataset = self.datasets[0]
        model = dataset.models[source].spectral_model

        self.model = ScaleSpectralModel(model)
        self.model.norm.min = 0
        self.model.norm.max = 1e5

        if norm_values is None:
            norm_values = np.logspace(
                np.log10(norm_min), np.log10(norm_max), norm_n_values
            )

        self.norm_values = norm_values

        self.sigma = sigma
        self.sigma_ul = sigma_ul
        self.reoptimize = reoptimize
        self.source = source

        self.group_table_info = None

    def _check_and_sort_time_intervals(self, time_intervals):
        """Sort the time_intervals by increasing time if not already ordered correctly.

        Parameters
        ----------
        time_intervals : list of `astropy.time.Time`
            Start and stop time for each interval to compute the LC
        """
        time_start = Time([interval[0] for interval in time_intervals])
        time_stop = Time([interval[1] for interval in time_intervals])
        sorted_indices = time_start.argsort()
        time_start_sorted = time_start[sorted_indices]
        time_stop_sorted = time_stop[sorted_indices]
        diff_time_stop = np.diff(time_stop_sorted)
        diff_time_interval_edges = time_start_sorted[1:] - time_stop_sorted[:-1]
        if np.any(diff_time_stop < 0) or np.any(diff_time_interval_edges < 0):
            raise ValueError("LightCurveEstimator requires non-overlapping time bins.")
        else:
            self.time_intervals = [
                Time([tstart, tstop])
                for tstart, tstop in zip(time_start_sorted, time_stop_sorted)
            ]

    def _set_scale_model(self, datasets):
        for dataset in datasets:
            dataset.models[self.source].spectral_model = self.model

    @property
    def ref_model(self):
        return self.model.model

    def run(self, e_ref, e_min, e_max, steps="all", atol="1e-6 s"):
        """Run light curve extraction.

        Normalize integral and energy flux between emin and emax.

        Parameters
        ----------
        e_ref : `~astropy.units.Quantity`
            reference energy of dnde flux normalization
        e_min : `~astropy.units.Quantity`
            minimum energy of integral and energy flux normalization interval
        e_max : `~astropy.units.Quantity`
            minimum energy of integral and energy flux normalization interval
        steps : list of str
            Which steps to execute. Available options are:

                * "err": estimate symmetric error.
                * "errn-errp": estimate asymmetric errors.
                * "ul": estimate upper limits.
                * "ts": estimate ts and sqrt(ts) values.
                * "norm-scan": estimate fit statistic profiles.

            By default all steps are executed.
        atol : `~astropy.units.Quantity`
            Tolerance value for time comparison with different scale. Default 1e-6 sec.

        Returns
        -------
        lightcurve : `~gammapy.time.LightCurve`
            the Light Curve object
        """
        atol = u.Quantity(atol)
        self.e_ref = e_ref
        self.e_min = e_min
        self.e_max = e_max

        rows = []
        self.group_table_info = group_datasets_in_time_interval(
            datasets=self.datasets, time_intervals=self.time_intervals, atol=atol
        )
        if np.all(self.group_table_info["Group_ID"] == -1):
            raise ValueError("LightCurveEstimator: No datasets in time intervals")
        for igroup, time_interval in enumerate(self.time_intervals):
            index_dataset = np.where(self.group_table_info["Group_ID"] == igroup)[0]
            if len(index_dataset) == 0:
                log.debug("No Dataset for the time interval " + str(igroup))
                continue

            row = {"time_min": time_interval[0].mjd, "time_max": time_interval[1].mjd}
            interval_list_dataset = Datasets(
                [
                    self.datasets[int(_)].copy(name=self.datasets[int(_)].name)
                    for _ in index_dataset
                ]
            )
            self._set_scale_model(interval_list_dataset)
            row.update(
                self.estimate_time_bin_flux(interval_list_dataset, time_interval, steps)
            )
            rows.append(row)
        table = table_from_row_data(rows=rows, meta={"SED_TYPE": "likelihood"})
        table = FluxPoints(table).to_sed_type("flux").table
        return LightCurve(table)

    def estimate_time_bin_flux(self, datasets, time_interval, steps="all"):
        """Estimate flux point for a single energy group.

        Parameters
        ----------
        datasets : `~gammapy.modeling.Datasets`
            the list of dataset object
        time_interval : astropy.time.Time`
            Start and stop time for each interval
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
        self.fit = Fit(datasets)

        result = {
            "e_ref": self.e_ref,
            "e_min": self.e_min,
            "e_max": self.e_max,
            "ref_dnde": self.ref_model(self.e_ref),
            "ref_flux": self.ref_model.integral(self.e_min, self.e_max),
            "ref_eflux": self.ref_model.energy_flux(self.e_min, self.e_max),
            "ref_e2dnde": self.ref_model(self.e_ref) * self.e_ref ** 2,
        }

        result.update(self.estimate_norm())

        if not result.pop("success"):
            log.warning(
                "Fit failed for time bin between {t_min} and {t_max},"
                " setting NaN.".format(
                    t_min=time_interval[0].mjd, t_max=time_interval[1].mjd
                )
            )

        if steps == "all":
            steps = ["err", "counts", "errp-errn", "ul", "ts", "norm-scan"]

        if "err" in steps:
            result.update(self.estimate_norm_err())

        if "counts" in steps:
            result.update(self.estimate_counts(datasets))

        if "ul" in steps:
            result.update(self.estimate_norm_ul(datasets))

        if "errp-errn" in steps:
            result.update(self.estimate_norm_errn_errp())

        if "ts" in steps:
            result.update(self.estimate_norm_ts(datasets))

        if "norm-scan" in steps:
            result.update(self.estimate_norm_scan())

        return result

    # TODO : most of the following code is copied from FluxPointsEstimator, can it be restructured?
    def estimate_norm_errn_errp(self):
        """Estimate asymmetric errors for a flux point.

        Returns
        -------
        result : dict
            Dict with asymmetric errors for the flux point norm.
        """
        result = self.fit.confidence(parameter=self.model.norm, sigma=self.sigma)
        return {"norm_errp": result["errp"], "norm_errn": result["errn"]}

    def estimate_norm_err(self):
        """Estimate covariance errors for a flux point.

        Returns
        -------
        result : dict
            Dict with symmetric error for the flux point norm.
        """
        result = self.fit.covariance()
        norm_err = result.parameters.error(self.model.norm)
        return {"norm_err": norm_err}

    def estimate_counts(self, datasets):
        """Estimate counts for the flux point.

        Parameters
        ----------
        datasets : `~gammapy.modeling.Datasets`
            the list of dataset object

        Returns
        -------
        result : dict
            Dict with an array with one entry per dataset with counts for the flux point.
        """

        counts = []
        for dataset in datasets:
            mask = dataset.mask
            counts.append(dataset.counts.data[mask].sum())

        return {"counts": np.array(counts, dtype=int).sum()}

    def estimate_norm_ul(self, datasets):
        """Estimate upper limit for a flux point.

        Parameters
        ----------
        datasets : `~gammapy.modeling.Datasets`
            the list of dataset object

        Returns
        -------
        result : dict
            Dict with upper limit for the flux point norm.
        """
        norm = self.model.norm

        # TODO: the minuit backend has convergence problems when the fit statistic is not
        #  of parabolic shape, which is the case, when there are zero counts in the
        #  energy bin. For this case we change to the scipy backend.
        counts = self.estimate_counts(datasets)["counts"]

        if np.all(counts == 0):
            result = self.fit.confidence(
                parameter=norm,
                sigma=self.sigma_ul,
                backend="scipy",
                reoptimize=self.reoptimize,
            )
        else:
            result = self.fit.confidence(parameter=norm, sigma=self.sigma_ul)

        return {"norm_ul": result["errp"] + norm.value}

    def estimate_norm_ts(self, datasets):
        """Estimate ts and sqrt(ts) for the flux point.

        Parameters
        ----------
        datasets : `~gammapy.modeling.Datasets`
            the list of dataset object

        Returns
        -------
        result : dict
            Dict with ts and sqrt(ts) for the flux point.
        """
        stat = datasets.stat_sum()

        # store best fit amplitude, set amplitude of fit model to zero
        self.model.norm.value = 0
        self.model.norm.frozen = True

        if self.reoptimize:
            _ = self.fit.optimize()

        stat_null = datasets.stat_sum()

        # compute sqrt TS
        ts = np.abs(stat_null - stat)
        sqrt_ts = np.sqrt(ts)
        return {"sqrt_ts": sqrt_ts, "ts": ts}

    def estimate_norm_scan(self):
        """Estimate fit statistic profile for the norm parameter.

        Returns
        -------
        result : dict
            Keys "norm_scan", "stat_scan"
        """
        result = self.fit.stat_profile(
            self.model.norm, values=self.norm_values, reoptimize=self.reoptimize
        )
        return {"norm_scan": result["values"], "stat_scan": result["stat"]}

    def estimate_norm(self):
        """Fit norm of the flux point.

        Returns
        -------
        result : dict
            Dict with "norm" and "stat" for the flux point.
        """
        # start optimization with norm=1
        self.model.norm.value = 1.0
        self.model.norm.frozen = False

        result = self.fit.optimize()

        if result.success:
            norm = self.model.norm.value
        else:
            norm = np.nan

        return {"norm": norm, "stat": result.total_stat, "success": result.success}
