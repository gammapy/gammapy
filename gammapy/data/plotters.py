# Licensed under a 3-clause BSD style license - see LICENSE.rst
import matplotlib.pyplot as plt  # This could be done lazily in each method to limit import time when no plot is performed
from itertools import zip_longest
import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support
from gammapy.maps import MapAxis
from gammapy.maps.axes import UNIT_STRING_FORMAT


class EventListPlotter:
    def __init__(self, event_list):
        self.event_list = event_list

    def peek(self, allsky=False):
        """Quick look plots.

        Parameters
        ----------
        allsky : bool, optional
            Whether to look at the events all-sky. Default is False.
        """
        import matplotlib.gridspec as gridspec

        if allsky:
            gs = gridspec.GridSpec(nrows=2, ncols=2)
            fig = plt.figure(figsize=(8, 8))
        else:
            gs = gridspec.GridSpec(nrows=2, ncols=3)
            fig = plt.figure(figsize=(12, 8))

        # energy plot
        ax_energy = fig.add_subplot(gs[1, 0])
        self.event_list.plot_energy(ax=ax_energy)

        # offset plots
        if not allsky:
            ax_offset = fig.add_subplot(gs[0, 1])
            self.plot_offset2_distribution(ax=ax_offset)
            ax_energy_offset = fig.add_subplot(gs[0, 2])
            self.plot_energy_offset(ax=ax_energy_offset)

        # time plot
        ax_time = fig.add_subplot(gs[1, 1])
        self.plot_time(ax=ax_time)

        # image plot
        m = self.event_list._counts_image(allsky=allsky)
        if allsky:
            ax_image = fig.add_subplot(gs[0, :], projection=m.geom.wcs)
        else:
            ax_image = fig.add_subplot(gs[0, 0], projection=m.geom.wcs)
        m.plot(ax=ax_image, stretch="sqrt", vmin=0)
        plt.subplots_adjust(wspace=0.3)

    def plot_energy(self, ax=None, **kwargs):
        """Plot counts as a function of energy.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes. Default is None
        **kwargs : dict, optional
            Keyword arguments passed to `~matplotlib.pyplot.hist`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Matplotlib axes.
        """
        ax = plt.gca() if ax is None else ax

        energy_axis = self.event_list._default_plot_energy_axis

        kwargs.setdefault("log", True)
        kwargs.setdefault("histtype", "step")
        kwargs.setdefault("bins", energy_axis.edges)

        with quantity_support():
            ax.hist(self.event_list.energy, **kwargs)

        energy_axis.format_plot_xaxis(ax=ax)
        ax.set_ylabel("Counts")
        ax.set_yscale("log")
        return ax

    def plot_time(self, ax=None, **kwargs):
        """Plot an event rate time curve.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes. Default is None.
        **kwargs : dict, optional
            Keyword arguments passed to `~matplotlib.pyplot.errorbar`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Matplotlib axes.
        """
        ax = plt.gca() if ax is None else ax

        # Note the events are not necessarily in time order
        time = self.event_list.table["TIME"]
        time = time - np.min(time)

        ax.set_xlabel(f"Time [{u.s.to_string(UNIT_STRING_FORMAT)}]")
        ax.set_ylabel("Counts")
        y, x_edges = np.histogram(time, bins=20)

        xerr = np.diff(x_edges) / 2
        x = x_edges[:-1] + xerr
        yerr = np.sqrt(y)

        kwargs.setdefault("fmt", "none")

        ax.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, **kwargs)

        return ax

    def plot_offset2_distribution(
        self, ax=None, center=None, max_percentile=98, **kwargs
    ):
        """Plot offset^2 distribution of the events.

        The distribution shown in this plot is for this quantity::

            offset = center.separation(events.radec).deg
            offset2 = offset ** 2

        Note that this method is just for a quicklook plot.

        If you want to do computations with the offset or offset^2 values, you can
        use the line above. As an example, here's how to compute the 68% event
        containment radius using `numpy.percentile`::

            import numpy as np
            r68 = np.percentile(offset, q=68)

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes. Default is None.
        center : `astropy.coordinates.SkyCoord`, optional
            Center position for the offset^2 distribution.
            Default is the observation pointing position.
        max_percentile : float, optional
            Define the percentile of the offset^2 distribution used to define the maximum offset^2 value.
            Default is 98.
        **kwargs : dict, optional
            Extra keyword arguments are passed to `~matplotlib.pyplot.hist`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Matplotlib axes.

        Examples
        --------
        Load an example event list:

        >>> from gammapy.data import EventList
        >>> from astropy import units as u
        >>> filename = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_023523.fits.gz"
        >>> events = EventList.read(filename)

        >>> #Plot the offset^2 distribution wrt. the observation pointing position
        >>> #(this is a commonly used plot to check the background spatial distribution):
        >>> events.plot_offset2_distribution() # doctest: +SKIP
        Plot the offset^2 distribution wrt. the Crab pulsar position (this is
        commonly used to check both the gamma-ray signal and the background
        spatial distribution):

        >>> import numpy as np
        >>> from astropy.coordinates import SkyCoord
        >>> center = SkyCoord(83.63307, 22.01449, unit='deg')
        >>> bins = np.linspace(start=0, stop=0.3 ** 2, num=30) * u.deg ** 2
        >>> events.plot_offset2_distribution(center=center, bins=bins) # doctest: +SKIP

        Note how we passed the ``bins`` option of `matplotlib.pyplot.hist` to control
        the histogram binning, in this case 30 bins ranging from 0 to (0.3 deg)^2.
        """
        ax = plt.gca() if ax is None else ax

        if center is None:
            center = self.event_list._plot_center

        offset2 = center.separation(self.event_list.radec) ** 2
        max2 = np.percentile(offset2, q=max_percentile)

        kwargs.setdefault("histtype", "step")
        kwargs.setdefault("bins", 30)
        kwargs.setdefault("range", (0.0, max2.value))

        with quantity_support():
            ax.hist(offset2, **kwargs)

        ax.set_xlabel(rf"Offset$^2$ [{ax.xaxis.units.to_string(UNIT_STRING_FORMAT)}]")
        ax.set_ylabel("Counts")
        return ax

    def plot_energy_offset(self, ax=None, center=None, **kwargs):
        """Plot counts histogram with energy and offset axes.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axis`, optional
            Plot axis. Default is None.
        center : `~astropy.coordinates.SkyCoord`, optional
            Sky coord from which offset is computed. Default is None.
        **kwargs : dict, optional
            Keyword arguments forwarded to `~matplotlib.pyplot.pcolormesh`.

        Returns
        -------
        ax : `~matplotlib.pyplot.Axis`
            Plot axis.
        """
        from matplotlib.colors import LogNorm

        ax = plt.gca() if ax is None else ax

        if center is None:
            center = self.event_list._plot_center

        energy_axis = self.event_list._default_plot_energy_axis

        offset = center.separation(self.event_list.radec)
        offset_axis = MapAxis.from_bounds(
            0 * u.deg, offset.max(), nbin=30, name="offset"
        )

        counts = np.histogram2d(
            x=self.event_list.energy,
            y=offset,
            bins=(energy_axis.edges, offset_axis.edges),
        )[0]

        kwargs.setdefault("norm", LogNorm())

        with quantity_support():
            ax.pcolormesh(energy_axis.edges, offset_axis.edges, counts.T, **kwargs)

        energy_axis.format_plot_xaxis(ax=ax)
        offset_axis.format_plot_yaxis(ax=ax)
        return ax

    def plot_image(self, ax=None, allsky=False):
        """Quick look counts map sky plot.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`, optional
            Matplotlib axes.
        allsky :  bool, optional
            Whether to plot on an all sky geom. Default is False.
        """
        if ax is None:
            ax = plt.gca()
        m = self.event_list._counts_image(allsky=allsky)
        m.plot(ax=ax, stretch="sqrt")


class ObservationPlotter:
    def __init__(self, observation):
        self.observation = observation

    def peek(self, figsize=(15, 10)):
        """Quick-look plots in a few panels.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size. Default is (15, 10).
        """
        plottable_hds = ["events", "aeff", "psf", "edisp", "bkg", "rad_max"]

        plot_hdus = list(set(plottable_hds) & set(self.observation.available_hdus))
        plot_hdus.sort()

        n_irfs = len(plot_hdus)
        nrows = n_irfs // 2
        ncols = 2 + n_irfs % 2

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            gridspec_kw={"wspace": 0.3, "hspace": 0.3},
        )

        for idx, (ax, name) in enumerate(zip_longest(axes.flat, plot_hdus)):
            if name == "aeff":
                self.observation.aeff.plot(ax=ax)
                ax.set_title("Effective area")

            if name == "bkg":
                bkg = self.observation.bkg
                if not bkg.has_offset_axis:
                    bkg = bkg.to_2d()
                bkg.plot(ax=ax)
                ax.set_title("Background rate")

            if name == "psf":
                self.observation.psf.plot_containment_radius_vs_energy(ax=ax)
                ax.set_title("Point spread function")

            if name == "edisp":
                self.observation.edisp.plot_bias(ax=ax, add_cbar=True)
                ax.set_title("Energy dispersion")

            if name == "rad_max":
                self.observation.rad_max.plot_rad_max_vs_energy(ax=ax)
                ax.set_title("Rad max")

            if name == "events":
                m = self.observation.events._counts_image(allsky=False)
                ax.remove()
                ax = fig.add_subplot(nrows, ncols, idx + 1, projection=m.geom.wcs)
                m.plot(ax=ax, stretch="sqrt", vmin=0, add_cbar=True)
                ax.set_title("Events")

            if name is None:
                ax.set_visible(False)
