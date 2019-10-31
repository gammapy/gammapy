# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

__all__ = ["ObservationTableSummary"]


class ObservationTableSummary:
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
        ss += f"Target position: {self.target_pos}\n"

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
        -------
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
