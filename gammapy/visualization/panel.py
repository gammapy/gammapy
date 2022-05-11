# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Helper functions and functions for plotting gamma-ray images."""
import numpy as np
from astropy.coordinates import Angle
from matplotlib.gridspec import GridSpec

__all__ = ["MapPanelPlotter"]

__doctest_requires__ = {("colormap_hess", "colormap_milagro"): ["matplotlib"]}


class MapPanelPlotter:
    """
    Map panel plotter class.

    Given a `~matplotlib.pyplot.Figure` object this class creates axes objects
    using `~matplotlib.gridspec.GridSpec` and plots a given sky map onto these.

    Parameters
    ----------
    figure : `~matplotlib.pyplot.figure.`
        Figure instance.
    xlim : `~astropy.coordinates.Angle`
        Angle object specifying the longitude limits.
    ylim : `~astropy.coordinates.Angle`
        Angle object specifying the latitude limits.
    npanels : int
        Number of panels.
    **kwargs : dict
        Keyword arguments passed to `~matplotlib.gridspec.GridSpec`.
    """

    def __init__(self, figure, xlim, ylim, npanels=4, **kwargs):

        self.figure = figure
        self.parameters = {"xlim": xlim, "ylim": ylim, "npanels": npanels}
        self.grid_spec = GridSpec(nrows=npanels, ncols=1, **kwargs)

    def _get_ax_extend(self, ax, panel):
        """Get width and height of the axis in world coordinates."""
        p = self.parameters

        # compute aspect ratio of the axis
        aspect = ax.bbox.width / ax.bbox.height

        # compute width and height in world coordinates
        height = np.abs(p["ylim"].diff())
        width = aspect * height

        left, bottom = p["xlim"][0].wrap_at("180d"), p["ylim"][0]

        width_all = np.abs(p["xlim"].wrap_at("180d").diff())
        xoverlap = ((p["npanels"] * width) - width_all) / (p["npanels"] - 1.0)
        if xoverlap < 0:
            raise ValueError(
                "No overlap between panels. Please reduce figure "
                "height or increase vertical space between the panels."
            )

        left = left - panel * (width - xoverlap)
        return left[0], bottom, width, height

    def _set_ax_fov(self, ax, panel):
        left, bottom, width, height = self._get_ax_extend(ax, panel)

        # set fov
        xlim = Angle([left, left - width])
        ylim = Angle([bottom, bottom + height])
        xlim_pix, ylim_pix = ax.wcs.wcs_world2pix(xlim.deg, ylim.deg, 1)

        ax.set_xlim(*xlim_pix)
        ax.set_ylim(*ylim_pix)
        return ax

    def plot_panel(self, map, panel=1, panel_fov=None, **kwargs):
        """
        Plot sky map on one panel.

        Parameters
        ----------
        map : `~gammapy.maps.WcsNDMap`
            Map to plot.
        panel : int
            Which panel to plot on (counted from top).
        """
        if panel_fov is None:
            panel_fov = panel
        spec = self.grid_spec[panel]
        ax = self.figure.add_subplot(spec, projection=map.geom.wcs)
        try:
            ax = map.plot(ax=ax, **kwargs)
        except AttributeError:
            ax = map.plot_rgb(ax=ax, **kwargs)
        ax = self._set_ax_fov(ax, panel_fov)
        return ax

    def plot(self, map, **kwargs):
        """
        Plot sky map on all panels.

        Parameters
        ----------
        map : `~gammapy.maps.WcsNDMap`
            Map to plot.
        """
        p = self.parameters
        axes = []
        for panel in range(p["npanels"]):
            ax = self.plot_panel(map, panel=panel, **kwargs)
            axes.append(ax)
        return axes
