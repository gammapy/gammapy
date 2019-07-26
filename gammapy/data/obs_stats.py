# Licensed under a 3-clause BSD style license - see LICENSE.rst
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
    livetime : `~astropy.units.Quantity`
        Livetime of the observation
    """

    def __init__(
        self, n_on=None, n_off=None, a_on=None, a_off=None, obs_id=None, livetime=None
    ):
        super().__init__(n_on=n_on, n_off=n_off, a_on=a_on, a_off=a_off)

        self.obs_id = obs_id
        self.livetime = livetime
        if a_off > 0:
            self.alpha_obs = a_on / a_off
        else:
            self.alpha_obs = 0

        self.gamma_rate = self.excess / livetime
        self.bg_rate = self.alpha_obs * n_off / livetime

    @classmethod
    def from_observation(cls, observation, bg_estimate):
        """Create from `~gammapy.data.DataStoreObservation`.

        Parameters
        ----------
        observation : `~gammapy.data.DataStoreObservation`
            IACT data store observation
        bg_estimate : `~gammapy.background.BackgroundEstimate`
            Background estimate
        """
        n_on = len(bg_estimate.on_events.table)
        n_off = len(bg_estimate.off_events.table)
        a_on = bg_estimate.a_on
        a_off = bg_estimate.a_off

        obs_id = observation.obs_id
        livetime = observation.observation_live_time_duration

        stats = cls(
            n_on=n_on,
            n_off=n_off,
            a_on=a_on,
            a_off=a_off,
            obs_id=obs_id,
            livetime=livetime,
        )
        return stats

    @property
    def alpha(self):
        """Alpha (on / off exposure ratio).

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
        for stats in stats_list:
            if stats.a_off > 0:
                livetime += stats.livetime
                n_on += stats.n_on
            n_off += stats.n_off
            a_on += stats.a_on * stats.n_off
            a_on_backup += stats.a_on * stats.livetime.value
            a_off += stats.a_off * stats.n_off
            a_off_backup += stats.a_off * stats.livetime.value
            obs_id.append(stats.obs_id)

        # if no off events the weighting of exposure is done
        # with the livetime
        if n_off == 0:
            a_on = a_on_backup / livetime.value
            a_off = a_off_backup / livetime.value
        else:
            a_on /= n_off
            a_off /= n_off

        return cls(
            n_on=n_on,
            n_off=n_off,
            a_on=a_on,
            a_off=a_off,
            obs_id=obs_id,
            livetime=livetime,
        )

    def to_dict(self):
        """Data as a dict.

        This is useful for serialization or putting the info in a table.
        """
        return {
            "obs_id": self.obs_id,
            "livetime": self.livetime.to("min"),
            "n_on": self.n_on,
            "n_off": self.n_off,
            "a_on": self.a_on,
            "a_off": self.a_off,
            "alpha": self.alpha,
            "background": self.background,
            "excess": self.excess,
            "sigma": self.sigma,
            "gamma_rate": self.gamma_rate.to("1/min"),
            "bg_rate": self.bg_rate.to("1/min"),
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
        if self.background > 0:
            ss += "Excess / Background: {:.2f}\n".format(self.excess / self.background)
        ss += "Gamma rate: {:.2f}\n".format(self.gamma_rate.to("1/min"))
        ss += "Bkg rate: {:.2f}\n".format(self.bg_rate.to("1/min"))
        ss += "Sigma: {:.2f}\n".format(self.sigma)

        return ss
