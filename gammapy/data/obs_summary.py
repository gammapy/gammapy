# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from .obs_stats import ObservationStats

__all__ = ["ObservationTableSummary", "ObservationSummary"]


class ObservationTableSummary(object):
    """Observation table summary.

    Class allowing to summarize informations contained in
    Observation index table (`~gammapy.data.ObservationTable`)

    Parameters
    ----------
    obs_table : `~gammapy.data.ObservationTable`
        Observation index table
    target_pos : `~astropy.coordinates.SkyCoord`
        Target position
    """

    def __init__(self, obs_table, target_pos=None):
        self.obs_table = obs_table
        self.target_pos = target_pos

    @property
    def offset(self):
        """Observation pointing ot target offset (`~astropy.coordinates.Angle`).
        """
        t = self.obs_table
        pnt_pos = SkyCoord(t["RA_PNT"], t["DEC_PNT"], unit="deg")

        return pnt_pos.separation(self.target_pos)

    def __str__(self):
        ss = "*** Observation summary ***\n"
        ss += "Target position: {}\n".format(self.target_pos)

        ss += "Number of observations: {}\n".format(len(self.obs_table))

        livetime = u.Quantity(sum(self.obs_table["LIVETIME"]), "second")
        ss += "Livetime: {:.2f}\n".format(livetime.to("hour"))
        zenith = self.obs_table["ZEN_PNT"]
        ss += "Zenith angle: (mean={:.2f}, std={:.2f})\n".format(
            zenith.mean(), zenith.std()
        )
        offset = self.offset
        ss += "Offset: (mean={:.2f}, std={:.2f})\n".format(offset.mean(), offset.std())

        return ss

    def plot_zenith_distribution(self, ax=None, bins=None):
        """Construct the zenith distribution of the observations.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes` or None, optional.
            The `~matplotlib.axes.Axes` object to be drawn on.
            If None, uses the current `~matplotlib.axes.Axes`.
        bins : integer or array_like, optional
            Binning specification, passed to `matplotlib.pyplot.hist`.
            By default, 30 bins from 0 deg to max zenith + 5 deg is used.

        Returns
        --------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        zenith = self.obs_table["ZEN_PNT"]

        if bins is None:
            bins = np.linspace(0, zenith.max() + 5, 30)

        ax.hist(zenith, bins=bins)
        ax.set_title("Zenith distribution")
        ax.set_xlabel("Zenith (Deg)")
        ax.set_ylabel("#Entries")

        return ax

    def plot_offset_distribution(self, ax=None, bins=None):
        """Construct the offset distribution of the observations.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes` or None, optional.
            The `~matplotlib.axes.Axes` object to be drawn on.
            If None, uses the current `~matplotlib.axes.Axes`.
        bins : integer or array_like, optional
            Binning specification, passed to `matplotlib.pyplot.hist`.
            By default, 10 bins from 0 deg to max offset + 0.5 deg is used.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        offset = self.offset

        if bins is None:
            bins = np.linspace(0, offset.degree.max() + 0.5, 10)

        ax.hist(offset.degree, bins=bins)
        ax.set_title("Offset distribution")
        ax.set_xlabel("Offset (Deg)")
        ax.set_ylabel("#Entries")

        return ax


class ObservationSummary(object):
    """Summary of observations.

    For a list of observation stats, this class can make a
    table and do summary printout and plots.

    * TODO: Data should be stored in `~astropy.table.Table`!
    * TODO: there should be a per-run version of the stats
      in addition to the cumulative version that's there now.

    Parameters
    ----------
    obs_stats : list
        Python list of `~gammapy.data.ObservationStats`
    """

    def __init__(self, obs_stats):
        self.obs_stats = obs_stats

        self.obs_id = np.zeros(len(self.obs_stats))
        self.livetime = np.zeros(len(self.obs_stats)) * u.s
        self.n_on = np.zeros(len(self.obs_stats))
        self.n_off = np.zeros(len(self.obs_stats))
        self.alpha = np.zeros(len(self.obs_stats))
        self.background = np.zeros(len(self.obs_stats))
        self.excess = np.zeros(len(self.obs_stats))
        self.sigma = np.zeros(len(self.obs_stats))
        self.gamma_rate = np.zeros(len(self.obs_stats)) / u.min
        self.bg_rate = np.zeros(len(self.obs_stats)) / u.min

        self._init_values()

    def _init_values(self):
        """Initialise vector attributes for plotting methods."""
        stats_list = []

        for index, obs in enumerate(self.obs_stats):
            # per observation stat
            self.obs_id[index] = obs.obs_id  # keep track of the observation
            self.gamma_rate[index] = obs.gamma_rate
            self.bg_rate[index] = obs.bg_rate

            # cumulative information
            stats_list.append(obs)
            stack = ObservationStats.stack(stats_list)
            self.livetime[index] = stack.livetime
            self.n_on[index] = stack.n_on
            self.n_off[index] = stack.n_off
            self.alpha[index] = stack.alpha
            self.background[index] = stack.background
            self.excess[index] = stack.excess
            self.sigma[index] = stack.sigma

    def obs_wise_summary(self):
        """Observation wise summary report (str)."""
        ss = "*** Observation Wise summary ***\n"
        for obs in self.obs_stats:
            ss += "{}\n".format(obs)

        return ss

    def __str__(self):
        stack = ObservationStats.stack(self.obs_stats)
        ss = "*** Observation summary ***\n"
        ss += "{}\n".format(stack)
        return ss

    def plot_significance_vs_livetime(self, ax=None, **kwargs):
        """Plot significance as a function of livetime.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes` or None, optional.
            The `~matplotlib.axes.Axes` object to be drawn on.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax
        ax.plot(self.livetime.to(u.h), self.sigma, "o", **kwargs)

        ax.set_xlabel("Livetime ({})".format(u.h))
        ax.set_ylabel("Significance ($\sigma$)")

        xmax = np.amax(self.livetime.to(u.h).value) * 1.2
        ymax = np.amax(self.sigma) * 1.2
        ax.axis([0, xmax, 0, ymax])
        ax.set_title("Significance evolution")
        return ax

    def plot_excess_vs_livetime(self, ax=None, **kwargs):
        """Plot excess as a function of livetime.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes` or None, optional.
            The `~matplotlib.axes.Axes` object to be drawn on.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax
        ax.plot(self.livetime.to(u.h), self.excess, "o", **kwargs)

        ax.set_xlabel("Livetime ({})".format(u.h))
        ax.set_ylabel("Excess")

        xmax = np.amax(self.livetime.to(u.h).value) * 1.2
        ymax = np.amax(self.excess) * 1.2
        ax.axis([0, xmax, 0, ymax])
        ax.set_title("Excess evolution")
        return ax

    def plot_background_vs_livetime(self, ax=None, **kwargs):
        """Plot background as a function of livetime.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes` or None, optional.
            The `~matplotlib.axes.Axes` object to be drawn on.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax
        ax.plot(self.livetime.to(u.h), self.background, "o", **kwargs)

        ax.set_xlabel("Livetime ({})".format(u.h))
        ax.set_ylabel("Background")

        xmax = np.amax(self.livetime.to(u.h).value) * 1.2
        ymax = np.amax(self.background) * 1.2
        ax.axis([0, xmax, 0, ymax])
        ax.set_title("Background evolution")
        return ax

    def plot_gamma_rate(self, ax=None, **kwargs):
        """Plot gamma rate for each observation.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes` or None, optional.
            The `~matplotlib.axes.Axes` object to be drawn on.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        xtick_vals, xtick_labels = self._get_xtick_info()

        ax.plot(xtick_vals, self.gamma_rate, "o", **kwargs)
        ax.set_xlabel("Observation Ids")

        ax.set_xticks(xtick_vals)
        ax.set_xticklabels(xtick_labels, rotation=-22.5)
        ax.set_ylabel("$\gamma$ rate ({})".format(self.gamma_rate.unit))
        ax.axis([0, len(self.gamma_rate), 0., np.amax(self.gamma_rate.value) * 1.2])
        ax.set_title("$\gamma$ rates")
        return ax

    def plot_background_rate(self, ax=None, **kwargs):
        """Plot background rate for each observation.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes` or None, optional.
            The `~matplotlib.axes.Axes` object to be drawn on.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        xtick_vals, xtick_labels = self._get_xtick_info()

        ax.plot(xtick_vals, self.bg_rate, "o", **kwargs)
        ax.set_xlabel("Observation Ids")

        ax.set_xticks(xtick_vals)
        ax.set_xticklabels(xtick_labels, rotation=-22.5)
        ax.set_ylabel("Background rate ({})".format(self.bg_rate.unit))
        ax.axis([0, len(self.bg_rate), 0., np.amax(self.bg_rate.value) * 1.2])
        ax.set_title("Background rates")
        return ax

    def _get_xtick_info(self):
        idxs = list(range(len(self.obs_stats)))
        vals = [idx + 0.5 for idx in idxs]
        labels = [str(int(self.obs_id[idx])) for idx in idxs]
        return vals, labels
