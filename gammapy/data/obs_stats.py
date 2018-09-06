# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from ..stats import Stats, significance_on_off

__all__ = ["ObservationStats"]


class ObservationStats(Stats):
    """Observation statistics.

    Class allowing to summarize observation
    (`~gammapy.data.DataStoreObservation`) statistics

    Parameters
    ----------
    n_on : int
        Number of on events
    n_off : int
        Number of off events
    a_on : float
        Relative background exposure of the on region
    a_off : float
        Relative background exposure of the off region
    obs_id : int
        ID of the observation
    livetime : float
        Livetime of the observation
    alpha : float
        Normalisation between the on and the off exposure
    bg_rate : float
        Background rate (/min)
    gamma_rate : float
        Gamma rate (/min)
    """

    def __init__(
        self,
        n_on=None,
        n_off=None,
        a_on=None,
        a_off=None,
        obs_id=None,
        livetime=None,
        alpha=None,
        gamma_rate=None,
        bg_rate=None,
    ):
        super(ObservationStats, self).__init__(
            n_on=n_on, n_off=n_off, a_on=a_on, a_off=a_off
        )

        self.obs_id = obs_id
        self.livetime = livetime
        if alpha is not None:
            self.a_on = alpha
            self.a_off = 1
            self.alpha_obs = alpha
        elif a_off > 0:
            self.alpha_obs = a_on / a_off
        else:
            self.alpha_obs = 0

        if gamma_rate is None:
            gamma_rate = self.excess / livetime
        self.gamma_rate = gamma_rate

        if bg_rate is None:
            bg_rate = self.alpha_obs * n_off / livetime
        self.bg_rate = bg_rate

    @classmethod
    def from_obs(cls, obs, bg_estimate):
        """Create from `~gammapy.data.DataStoreObservation`.

        Parameters
        ----------
        obs : `~gammapy.data.DataStoreObservation`
            IACT data store observation
        bg_estimate : `~gammapy.background.BackgroundEstimate`
            Background estimate
        """
        n_on = len(bg_estimate.on_events.table)
        n_off = len(bg_estimate.off_events.table)
        a_on = bg_estimate.a_on
        a_off = bg_estimate.a_off

        obs_id = obs.obs_id
        livetime = obs.observation_live_time_duration
        alpha = 0
        if a_off > 0:
            alpha = a_on / a_off

        gamma_rate = n_on / livetime.to(u.min)
        bg_rate = (alpha * n_off) / livetime.to(u.min)
        stats = cls(
            n_on=n_on,
            n_off=n_off,
            a_on=a_on,
            a_off=a_off,
            obs_id=obs_id,
            livetime=livetime,
            alpha=alpha,
            gamma_rate=gamma_rate,
            bg_rate=bg_rate,
        )
        return stats

    @property
    def alpha(self):
        """Alpha (on / off exposure ratio)

        Override member function from `~gammapy.stats.Stats`
        to take into account weighted alpha by number of Off events
        """
        return self.alpha_obs

    @property
    def sigma(self):
        """Li-Ma significance for observation statistics (float)."""
        return significance_on_off(self.n_on, self.n_off, self.alpha, method="lima")

    @classmethod
    def stack(cls, stats_list):
        """Stack (concatenate) list of `~gammapy.data.ObservationStats`.

        Parameters
        ----------
        stats_list : list
            List of `~gammapy.data.ObservationStats`

        Returns
        -------
        total_stats : `~gammapy.data.ObservationStats`
            Statistics for stacked observation
        """
        n_on = 0
        n_off = 0
        a_on = 0
        a_on_backup = 0
        a_off = 0
        a_off_backup = 0
        obs_id = []
        livetime = 0
        alpha = 0
        alpha_backup = 0
        gamma_rate = 0
        bg_rate = 0
        for stats in stats_list:
            if stats.a_off > 0:
                livetime += stats.livetime
                n_on += stats.n_on
            n_off += stats.n_off
            a_on += stats.a_on * stats.n_off
            a_on_backup += stats.a_on * stats.livetime.value
            a_off += stats.a_off * stats.n_off
            a_off_backup += stats.a_off * stats.livetime.value
            alpha += stats.alpha * stats.n_off
            alpha_backup += stats.alpha * stats.livetime.value
            obs_id.append(stats.obs_id)
            gamma_rate += stats.n_on - stats.alpha * stats.n_off
            bg_rate += stats.n_off * stats.alpha

        # if no off events the weighting of alpha is done
        # with the livetime
        if n_off == 0:
            alpha = alpha_backup / livetime.value
            a_on = a_on_backup / livetime.value
            a_off = a_off_backup / livetime.value
        else:
            a_on /= n_off
            a_off /= n_off
            alpha /= n_off

        gamma_rate /= livetime.to(u.min)
        bg_rate /= livetime.to(u.min)

        return cls(
            n_on=n_on,
            n_off=n_off,
            a_on=a_on,
            a_off=a_off,
            obs_id=obs_id,
            livetime=livetime,
            alpha=alpha,
            gamma_rate=gamma_rate,
            bg_rate=bg_rate,
        )

    def to_dict(self):
        """Data as a dict.

        This is useful for serialisation or putting the info in a table.
        """
        return {
            "obs_id": self.obs_id,
            "livetime": self.livetime,
            "n_on": self.n_on,
            "n_off": self.n_off,
            "a_on": self.a_on,
            "a_off": self.a_off,
            "alpha": self.alpha,
            "background": self.background,
            "excess": self.excess,
            "sigma": self.sigma,
            "gamma_rate": self.gamma_rate,
            "bg_rate": self.bg_rate,
        }

    def __str__(self):
        ss = "*** Observation summary report ***\n"
        if type(self.obs_id) is list:
            obs_str = "[{}-{}]".format(self.obs_id[0], self.obs_id[-1])
        else:
            obs_str = "{}".format(self.obs_id)
        ss += "Observation Id: {}\n".format(obs_str)
        ss += "Livetime: {:.3f}\n".format(self.livetime.to(u.h))
        ss += "On events: {}\n".format(self.n_on)
        ss += "Off events: {}\n".format(self.n_off)
        ss += "Alpha: {:.3f}\n".format(self.alpha)
        ss += "Bkg events in On region: {:.2f}\n".format(self.background)
        ss += "Excess: {:.2f}\n".format(self.excess)
        with np.errstate(invalid="ignore", divide="ignore"):
            ss += "Excess / Background: {:.2f}\n".format(self.excess / self.background)
        ss += "Gamma rate: {:.2f}\n".format(self.gamma_rate)
        ss += "Bkg rate: {:.2f}\n".format(self.bg_rate)
        ss += "Sigma: {:.2f}\n".format(self.sigma)

        return ss
