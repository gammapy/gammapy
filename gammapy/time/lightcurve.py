# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from ..spectrum.utils import CountsPredictor
from ..stats.poisson import excess_error, excess_ul_helene
from ..utils.scripts import make_path
from ..stats.poisson import significance_on_off

__all__ = ["LightCurve", "LightCurveEstimator"]


class LightCurve(object):
    """Lightcurve container.

    The lightcurve data is stored in ``table``.

    For now we only support times stored in MJD format!

    TODO: specification of format is work in progress
    See https://github.com/open-gamma-ray-astro/gamma-astro-data-formats/pull/61

    Usage: :ref:`time-lc`

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table with lightcurve data
    """

    def __init__(self, table):
        self.table = table

    def __repr__(self):
        return "{}(len={})".format(self.__class__.__name__, len(self.table))

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
        return self._make_time("time")

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
        filename = make_path(filename)
        table = Table.read(str(filename), **kwargs)
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
        filename = make_path(filename)
        self.table.write(str(filename), **kwargs)

    def compute_fvar(self):
        r"""Calculate the fractional excess variance.

        This method accesses the the ``FLUX`` and ``FLUX_ERR`` columns
        from the lightcurve data.

        The fractional excess variance :math:`F_{var}`, an intrinsic
        variability estimator, is given by

        .. math::
            F_{var} = \sqrt{\frac{S^{2} - \bar{\sigma^{2}}}{\bar{x}^{2}}}.

        It is the excess variance after accounting for the measurement errors
        on the light curve :math:`\sigma`. :math:`S` is the variance.

        Returns
        -------
        fvar, fvar_err : `~numpy.ndarray`
            Fractional excess variance.

        References
        ----------
        .. [Vaughan2003] "On characterizing the variability properties of X-ray light
           curves from active galaxies", Vaughan et al. (2003)
           http://adsabs.harvard.edu/abs/2003MNRAS.345.1271V
        """
        flux = self.table["flux"].data.astype("float64")
        flux_err = self.table["flux_err"].data.astype("float64")

        flux_mean = np.mean(flux)
        n_points = len(flux)

        s_square = np.sum((flux - flux_mean) ** 2) / (n_points - 1)
        sig_square = np.nansum(flux_err ** 2) / n_points
        fvar = np.sqrt(np.abs(s_square - sig_square)) / flux_mean

        sigxserr_a = np.sqrt(2 / n_points) * (sig_square / flux_mean) ** 2
        sigxserr_b = np.sqrt(sig_square / n_points) * (2 * fvar / flux_mean)
        sigxserr = np.sqrt(sigxserr_a ** 2 + sigxserr_b ** 2)
        fvar_err = sigxserr / (2 * fvar)

        return fvar, fvar_err

    def compute_chisq(self):
        """Calculate the chi-square test for `LightCurve`.

        Chisquare test is a variability estimator. It computes
        deviations from the expected value here mean value

        Returns
        -------
        ChiSq, P-value : tuple of float or `~numpy.ndarray`
            Tuple of Chi-square and P-value
        """
        import scipy.stats as stats

        flux = self.table["flux"]
        yexp = np.mean(flux)
        yobs = flux.data
        chi2, pval = stats.chisquare(yobs, yexp)
        return chi2, pval

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

        y, yerr = self._get_fluxes_and_errors(flux_unit)
        is_ul, yul = self._get_flux_uls(flux_unit)
        x, xerr = self._get_times_and_errors(time_format)

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
        ax.set_ylabel("Flux ({0:FITS})".format(u.Unit(flux_unit)))
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
        from matplotlib.dates import num2timedelta

        try:
            x = self.time
        except KeyError:
            x = self.time_mid

        try:
            xn, xp = x - self.time_min, self.time_max - x
        except KeyError:
            xn, xp = x - x, x - x

        # convert to given time format
        if time_format == "iso":
            x = x.datetime
            # TODO: In astropy version >3.1 the TimeDelta.to_datetime() method
            # should start working, for now i will use num2timedelta.
            xn = np.asarray(num2timedelta(xn.to("d").value))
            xp = np.asarray(num2timedelta(xp.to("d").value))
        elif time_format == "mjd":
            x = x.mjd
            xn, xp = xn.to("d").value, xp.to("d").value
        else:
            raise ValueError(
                "The time format '{}' is not supported.".format(time_format)
            )

        return x, (xn, xp)


class LightCurveEstimator(object):
    """Light curve estimator.

    For a usage example see :gp-extra-notebook:`light_curve`.

    Parameters
    ----------
    spec_extract : `~gammapy.spectrum.SpectrumExtraction`
       Contains statistics, IRF and event lists
    """

    def __init__(self, spec_extract):
        self.obs_list = spec_extract.obs_list
        self.obs_spec = spec_extract.observations
        self.off_evt_list = self._get_off_evt_list(spec_extract)
        self.on_evt_list = self._get_on_evt_list(spec_extract)

    @staticmethod
    def _get_off_evt_list(spec_extract):
        """
        Returns list of OFF events for each observations
        """
        off_evt_list = []
        for bg in spec_extract.bkg_estimate:
            off_evt_list.append(bg.off_events)
        return off_evt_list

    @staticmethod
    def _get_on_evt_list(spec_extract):
        """
        Returns list of ON events for each observations
        """
        on_evt_list = []
        for obs in spec_extract.bkg_estimate:
            on_evt_list.append(obs.on_events)

        return on_evt_list

    @staticmethod
    def make_time_intervals_fixes(time_step, spectrum_extraction):
        """Create time intervals of fixed size.

        Parameters
        ----------
        time_step : float
            Size of the light curve bins in seconds
        spectrum_extraction : `~gammapy.spectrum.SpectrumExtraction`
            Contains statistics, IRF and event lists

        Returns
        -------
        table : `~astropy.table.Table`
            Table of time intervals

        Examples
        --------
        To extract intervals for light curve::

            intervals = list(zip(table['t_start'], table['t_stop']))
        """
        rows = []
        time_start = Time(100000, format="mjd")
        time_end = Time(0, format="mjd")
        time_step = time_step / (24 * 3600)

        for obs in spectrum_extraction.obs_list:
            time_events = obs.events.time
            if time_start > time_events.min():
                time_start = time_events.min()
            if time_end < time_events.max():
                time_end = time_events.max()

        time = time_start.value
        while time < time_end.value:
            time += time_step
            rows.append(
                dict(
                    t_start=Time(time - time_step, format="mjd", scale="tt"),
                    t_stop=Time(time, format="mjd", scale="tt"),
                )
            )
        return Table(rows=rows)

    def _create_and_filter_onofflists(
        self, t_index, energy_range=None, interval=None, extra=False
    ):
        """Extract on and off events list from an observation and apply energy and time filters

        Helper function for compute_flux_point and make_time_intervals_min_significance

        Parameters
        ----------
        t_index : int
            index in self of the observation to use
        energy_range : `~astropy.units.Quantity`
            True energy range to filter the events
        interval : `~astropy.time.Time`
            Time interval (2-element array)
        extra : boolean
            Define if we want spec and e_reco in output

        Returns
        -------
        on : `~gammapy.data.EventList`
            List of on events
        off : `~gammapy.data.EventList`
            List of on events
        """
        spec = self.obs_spec[t_index]
        # get ON and OFF evt list
        off = self.off_evt_list[t_index]
        on = self.on_evt_list[t_index]
        # introduce the e_reco binning here
        e_reco = spec.e_reco
        if energy_range is not None:
            emin = e_reco[e_reco.searchsorted(max(spec.lo_threshold, energy_range[0]))]
            emax = e_reco[
                e_reco.searchsorted(min(spec.hi_threshold, energy_range[1])) - 1
            ]
            # filter the event list with energy
            on = on.select_energy([emin, emax])
            on = on.select_energy(energy_range)
            off = off.select_energy([emin, emax])
            off = off.select_energy(energy_range)
        if interval is not None:
            # filter the event list with time
            tmin = interval[0]
            tmax = interval[1]
            on = on.select_time([tmin, tmax])
            off = off.select_time([tmin, tmax])
        if extra:
            return on, off, spec, e_reco
        return on, off

    @staticmethod
    def _alpha(time_holder, obs_properties, n, istart, i):
        """ Helper function for make_time_intervals_min_significance

        Parameters
        ----------
        time_holder : list of float and flag
            Contains a list of a time and a flag in 2-element arrays
        obs_properties : `~astropy.table.Table`
            Contains the dead time fraction and ratio of the on/off region
        n : int
            First observation to use
        istart : int
            index of the first event of the interval
        i : int
            index of the last event of the interval
        """

        def in_list(item, L):
            o, j = np.where(L == item)
            for index in o:
                yield index

        alpha = 0
        tmp = 0
        time = 0
        xm1 = istart
        # loop over observations
        for x in in_list("end", time_holder[istart : i + 1]):
            if tmp == 0:
                alpha += (
                    (1 - obs_properties["deadtime"][n])
                    * (
                        float(time_holder[x][0])
                        - (float(time_holder[xm1][0]) + float(time_holder[xm1 - 1][0]))
                        / 2
                    )
                    * obs_properties["A_off"][n + tmp]
                )
                time += (1 - obs_properties["deadtime"][n]) * (
                    float(time_holder[x][0])
                    - (float(time_holder[xm1][0]) + float(time_holder[xm1 - 1][0])) / 2
                )
                xm1 = x + 1
                tmp += 1
            else:
                alpha += (
                    (1 - obs_properties["deadtime"][n + tmp])
                    * (float(time_holder[x][0]) - float(time_holder[xm1][0]))
                    * obs_properties["A_off"][n + tmp]
                )
                time += (1 - obs_properties["deadtime"][n + tmp]) * (
                    float(time_holder[x][0]) - float(time_holder[xm1][0])
                )
                xm1 = x + 1
                tmp += 1
        alpha += (
            (1 - obs_properties["deadtime"][n + tmp])
            * (
                (float(time_holder[i][0]) + float(time_holder[i][0])) / 2
                - float(time_holder[xm1][0])
            )
            * obs_properties["A_off"][n + tmp]
        )
        time += (1 - obs_properties["deadtime"][n + tmp]) * (
            (float(time_holder[i][0]) + float(time_holder[i][0])) / 2
            - float(time_holder[xm1][0])
        )
        alpha = time / alpha
        return alpha

    def make_time_intervals_min_significance(
        self,
        significance,
        significance_method,
        energy_range,
        spectrum_extraction,
        separators=None,
    ):
        """
        Create time intervals such that each bin of a light curve reach a given significance

        The function work event by event to create an interval containing enough statistic and then starting a new one

        Parameters
        ----------
        significance : float
            Target significance for each light curve point
        significance_method : {'lima', 'simple'}
            Significance method (see `~gammapy.stats.significance_on_off`)
        energy_range : `~astropy.units.Quantity`
            True energy range to evaluate integrated flux (true energy)
        spectrum_extraction : `~gammapy.spectrum.SpectrumExtraction`
            Contains statistics, IRF and event lists
        separators : list of `~astropy.time.Time`
            Contains a list of time to stop the current point creation (not saved) and start a new one
            Mostly useful between observations separated by a large time gap

        Returns
        -------
        table : `~astropy.table.Table`
            Table of time intervals  and information about their content : on/off events, alpha, significance

        Examples
        --------
        extract intervals for light curve :
            intervals = list(zip(table['t_start'], table['t_stop']))
        """
        # The function create a list of time associated with identifiers called time_holder.
        # The identifiers can be 'on' for the on events, 'off' for the off events, 'start' for the start of an
        # observation, 'end for the end of an observation and 'break' for a separator.
        # The function then loop other all the elements of this list

        time_holder = []
        obs_properties = []
        n_obs = 0

        # extract the separators
        if separators is not None:
            for time in separators:
                time_holder.append([time.tt.mjd, "break"])

        # recovers the starting and ending time of each observations and useful properties
        for obs in spectrum_extraction.obs_list:
            time_holder.append([obs.events.observation_time_start.tt.mjd, "start"])
            time_holder.append([obs.events.observation_time_end.tt.mjd, "end"])
            obs_properties.append(
                dict(
                    deadtime=obs.observation_dead_time_fraction,
                    A_off=spectrum_extraction.bkg_estimate[n_obs].a_off,
                )
            )
            n_obs += 1
        obs_properties = Table(rows=obs_properties)

        # prepare the on and off photon list as in the flux point computation -> should be updated accordingly
        for t_index, obs in enumerate(self.obs_list):
            on, off = self._create_and_filter_onofflists(
                t_index=t_index, energy_range=energy_range
            )

            for time in on.time.tt.mjd:
                time_holder.append([time, "on"])
            for time in off.time.tt.mjd:
                time_holder.append([time, "off"])

        # sort all elements in the table by time
        time_holder = sorted(time_holder, key=lambda item: item[0])
        time_holder = np.asarray(time_holder)

        rows = []
        istart = 1
        i = 1
        n = 0
        while time_holder[i][0] < time_holder[-1][0]:
            i += 1
            if time_holder[i][1] == "break":
                while time_holder[i + 1][1] != "on" and time_holder[i + 1][1] != "off":
                    i += 1
                n += np.sum(time_holder[istart:i] == "end")
                istart = i
                continue
            if time_holder[i][1] != "on" and time_holder[i][1] != "off":
                continue

            # compute alpha
            alpha = self._alpha(time_holder, obs_properties, n, istart, i)

            non = np.sum(time_holder[istart : i + 1] == "on")
            noff = np.sum(time_holder[istart : i + 1] == "off")

            # check the significance
            signif = significance_on_off(non, noff, alpha, method=significance_method)
            if signif > significance:
                rows.append(
                    dict(
                        t_start=(
                            float(time_holder[istart - 1][0])
                            + float(time_holder[istart][0])
                        )
                        / 2,
                        t_stop=(float(time_holder[i][0]) + float(time_holder[i + 1][0]))
                        / 2,
                        n_on=non,
                        n_off=noff,
                        alpha=alpha,
                        significance=signif,
                    )
                )
                # start the next interval
                while (
                    time_holder[i + 1][0] < time_holder[-1][0]
                    and time_holder[i + 1][1] != "on"
                    and time_holder[i + 1][1] != "off"
                ):
                    i += 1
                n += np.sum(time_holder[istart : i + 1] == "end")
                istart = i + 1
                i = istart
        table = Table(rows=rows)
        table["t_start"] = Time(table["t_start"], format="mjd", scale="tt")
        table["t_stop"] = Time(table["t_stop"], format="mjd", scale="tt")
        return table

    def light_curve(
        self, time_intervals, spectral_model, energy_range, ul_significance=3
    ):
        """Compute light curve.

        Implementation follows what is done in:
        http://adsabs.harvard.edu/abs/2010A%26A...520A..83H.

        To be discussed: assumption that threshold energy in the
        same in reco and true energy.

        Parameters
        ----------
        time_intervals : list of `~astropy.time.Time`
            List of time intervals
        spectral_model : `~gammapy.spectrum.models.SpectralModel`
            Spectral model
        energy_range : `~astropy.units.Quantity`
            True energy range to evaluate integrated flux (true energy)
        ul_significance : float
            Upper limit confidence level significance

        Returns
        -------
        lc : `~gammapy.time.LightCurve`
            Light curve
        """
        rows = []
        for time_interval in time_intervals:
            useinterval, row = self.compute_flux_point(
                time_interval, spectral_model, energy_range, ul_significance
            )
            if useinterval:
                rows.append(row)

        return self._make_lc_from_row_data(rows)

    @staticmethod
    def _make_lc_from_row_data(rows):
        table = Table()
        table["time_min"] = [_["time_min"].value for _ in rows]
        table["time_max"] = [_["time_max"].value for _ in rows]

        table["flux"] = [_["flux"].value for _ in rows] * u.Unit("1 / (s cm2)")
        table["flux_err"] = [_["flux_err"].value for _ in rows] * u.Unit("1 / (s cm2)")
        table["flux_ul"] = [_["flux_ul"].value for _ in rows] * u.Unit("1 / (s cm2)")
        table["is_ul"] = [_["is_ul"] for _ in rows]

        table["livetime"] = [_["livetime"].value for _ in rows] * u.s
        table["n_on"] = [_["n_on"] for _ in rows]
        table["n_off"] = [_["n_off"] for _ in rows]
        table["alpha"] = [_["alpha"] for _ in rows]
        table["measured_excess"] = [_["measured_excess"] for _ in rows]
        table["expected_excess"] = [_["expected_excess"] for _ in rows]

        return LightCurve(table)

    def compute_flux_point(
        self, time_interval, spectral_model, energy_range, ul_significance=3
    ):
        """Compute one flux point for one time interval.

        Parameters
        ----------
        time_interval : `~astropy.time.Time`
            Time interval (2-element array, or a tuple of Time objects)
        spectral_model : `~gammapy.spectrum.models.SpectralModel`
            Spectral model
        energy_range : `~astropy.units.Quantity`
            True energy range to evaluate integrated flux (true energy)
        ul_significance : float
            Upper limit confidence level significance

        Returns
        -------
        useinterval : bool
            Is True if the time_interval produce a valid flux point
        measurements : dict
            Dictionary with flux point measurement in the time interval
        """
        tmin, tmax = time_interval[0], time_interval[1]
        livetime = 0
        alpha_mean = 0.
        alpha_mean_backup = 0.
        measured_excess = 0
        predicted_excess = 0
        n_on = 0
        n_off = 0
        useinterval = False

        # Loop on observations
        for t_index, obs in enumerate(self.obs_list):

            # discard observations not matching the time interval
            obs_start = obs.events.observation_time_start
            obs_stop = obs.events.observation_time_end
            if (tmin < obs_start and tmax < obs_start) or (tmin > obs_stop):
                continue
            useinterval = True
            # get ON and OFF evt list
            on, off, spec, e_reco = self._create_and_filter_onofflists(
                t_index=t_index,
                energy_range=energy_range,
                interval=[tmin, tmax],
                extra=True,
            )
            n_on_obs = len(on.table)
            n_off_obs = len(off.table)

            # compute effective livetime (for the interval)
            if tmin >= obs_start and tmax <= obs_stop:
                # interval included in obs
                livetime_to_add = (tmax - tmin).to("s")
            elif tmin >= obs_start and tmax >= obs_stop:
                # interval min above tstart from obs
                livetime_to_add = (obs_stop - tmin).to("s")
            elif tmin <= obs_start and tmax <= obs_stop:
                # interval min below tstart from obs
                livetime_to_add = (tmax - obs_start).to("s")
            elif tmin <= obs_start and tmax >= obs_stop:
                # obs included in interval
                livetime_to_add = (obs_stop - obs_start).to("s")
            else:
                livetime_to_add = 0 * u.s

            # Take into account dead time
            livetime_to_add *= 1. - obs.observation_dead_time_fraction

            # Compute excess
            obs_measured_excess = n_on_obs - spec.alpha * n_off_obs

            # Compute the expected excess in the range given by the user
            # but must respect the energy threshold of the observation
            # (to match the energy range of the measured excess)
            # We use the effective livetime and the right energy threshold
            e_idx = np.where(
                np.logical_and.reduce(
                    (
                        e_reco >= spec.lo_threshold,  # threshold
                        e_reco <= spec.hi_threshold,  # threshold
                        e_reco >= energy_range[0],  # user
                        e_reco <= energy_range[-1],
                    )  # user
                )
            )[0]
            counts_predictor = CountsPredictor(
                livetime=livetime_to_add,
                aeff=spec.aeff,
                edisp=spec.edisp,
                model=spectral_model,
            )
            counts_predictor.run()
            counts_predicted_excess = counts_predictor.npred.data.data[e_idx[:-1]]

            obs_predicted_excess = np.sum(counts_predicted_excess)

            # compute effective normalisation between ON/OFF (for the interval)
            livetime += livetime_to_add
            alpha_mean += spec.alpha * n_off_obs
            alpha_mean_backup += spec.alpha * livetime_to_add
            measured_excess += obs_measured_excess
            predicted_excess += obs_predicted_excess
            n_on += n_on_obs
            n_off += n_off_obs

        # Fill time interval information
        if useinterval:
            int_flux = spectral_model.integral(energy_range[0], energy_range[1])

            if n_off > 0.:
                alpha_mean /= n_off
            if livetime > 0.:
                alpha_mean_backup /= livetime
            if alpha_mean == 0.:  # use backup if necessary
                alpha_mean = alpha_mean_backup

            flux = measured_excess / predicted_excess.value
            flux *= int_flux

            # Gaussian errors, TODO: should be improved
            flux_err = int_flux / predicted_excess.value
            delta_excess = excess_error(n_on=n_on, n_off=n_off, alpha=alpha_mean)
            flux_err *= delta_excess

            sigma = significance_on_off(
                n_on=n_on, n_off=n_off, alpha=alpha_mean, method="lima"
            )
            is_ul = sigma <= 3
            flux_ul = excess_ul_helene(
                n_on - alpha_mean * n_off, delta_excess, ul_significance
            )
            flux_ul *= int_flux / predicted_excess.value
        else:
            flux = 0
            flux_err = 0
            flux_ul = -1

        # Store measurements in a dict and return that
        return (
            useinterval,
            OrderedDict(
                [
                    ("time_min", Time(tmin, format="mjd")),
                    ("time_max", Time(tmax, format="mjd")),
                    ("flux", flux * u.Unit("1 / (s cm2)")),
                    ("flux_err", flux_err * u.Unit("1 / (s cm2)")),
                    ("flux_ul", flux_ul * u.Unit("1 / (s cm2)")),
                    ("is_ul", is_ul),
                    ("livetime", livetime * u.s),
                    ("alpha", alpha_mean),
                    ("n_on", n_on),
                    ("n_off", n_off),
                    ("measured_excess", measured_excess),
                    ("expected_excess", predicted_excess),
                ]
            ),
        )
