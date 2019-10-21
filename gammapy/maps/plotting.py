# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Helper functions and functions for plotting gamma-ray images."""
import numpy as np
from astropy.coordinates import Angle

__all__ = ["colormap_hess", "colormap_milagro", "MapPanelPlotter"]

__doctest_requires__ = {("colormap_hess", "colormap_milagro"): ["matplotlib"]}


class MapPanelPlotter:
    """
    Map panel plotter class.

    Given a `~matplotlib.pyplot.Figure` object this class creates axes objects
    using `~matplotlib.gridspec.GridSpec` and plots a given sky map onto these.

    For a usage example see `hgps.html <../notebooks/hgps.html>`__ (at the end).

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
        from matplotlib.gridspec import GridSpec

        self.figure = figure
        self.parameters = {"xlim": xlim, "ylim": ylim, "npanels": npanels}
        self.grid_spec = GridSpec(nrows=npanels, ncols=1, **kwargs)

    def _get_ax_extend(self, ax, panel):
        """Get width and height of the axis in world coordinates"""
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
            ax = map.plot(ax=ax, **kwargs)[1]
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


def colormap_hess(transition=0.5, width=0.1):
    """Colormap often used in H.E.S.S. collaboration publications.

    This colormap goes black -> blue -> red -> yellow -> white.

    A sharp blue -> red -> yellow transition is often used for significance images
    with a value of red at ``transition ~ 5`` or ``transition ~ 7``
    so that the following effect is achieved:

    - black, blue: non-significant features, not well visible
    - red: features at the detection threshold ``transition``
    - yellow, white: significant features, very well visible

    The transition parameter is defined between 0 and 1. To calculate the value
    from data units an `~astropy.visualization.mpl_normalize.ImageNormalize`
    instance should be used (see example below).

    Parameters
    ----------
    transition : float (default = 0.5)
        Value of the transition to red (between 0 and 1).
    width : float (default = 0.5)
        Width of the blue-red color transition (between 0 and 1).

    Returns
    -------
    colormap : `matplotlib.colors.LinearSegmentedColormap`
        Colormap

    Examples
    --------
    >>> from gammapy.maps import colormap_hess
    >>> from astropy.visualization.mpl_normalize import ImageNormalize
    >>> from astropy.visualization import LinearStretch
    >>> normalize = ImageNormalize(vmin=-5, vmax=15, stretch=LinearStretch())
    >>> transition = normalize(5)
    >>> cmap = colormap_hess(transition=transition)
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Compute normalised values (range 0 to 1) that
    # correspond to red, blue, yellow.
    red = float(transition)

    if width > red:
        blue = 0.1 * red
    else:
        blue = red - width

    yellow = 2.0 / 3.0 * (1 - red) + red

    black, white = 0, 1

    # Create custom colormap
    # List entries: (value, (R, G, B))
    colors = [
        (black, "k"),
        (blue, (0, 0, 0.8)),
        (red, "r"),
        (yellow, (1.0, 1.0, 0)),
        (white, "w"),
    ]

    return LinearSegmentedColormap.from_list(name="hess", colors=colors)


def colormap_milagro(transition=0.5, width=0.0001, huestart=0.6):
    """Colormap often used in Milagro collaboration publications.

    This colormap is gray below ``transition`` and similar to the jet colormap above.

    A sharp gray -> color transition is often used for significance images
    with a transition value of ``transition ~ 5`` or ``transition ~ 7``,
    so that the following effect is achieved:

    - gray: non-significant features are not well visible
    - color: significant features at the detection threshold ``transition``

    Note that this colormap is often criticised for over-exaggerating small differences
    in significance below and above the gray - color transition threshold.

    The transition parameter is defined between 0 and 1. To calculate the value
    from data units an `~astropy.visualization.mpl_normalize.ImageNormalize` instance should be
    used (see example below).

    Parameters
    ----------
    transition : float (default = 0.5)
        Transition value (below: gray, above: color).
    width : float (default = 0.0001)
        Width of the transition
    huestart : float (default = 0.6)
        Hue of the color at ``transition``

    Returns
    -------
    colormap : `~matplotlib.colors.LinearSegmentedColormap`
        Colormap

    Examples
    --------
    >>> from gammapy.maps import colormap_milagro
    >>> from astropy.visualization.mpl_normalize import ImageNormalize
    >>> from astropy.visualization import LinearStretch
    >>> normalize = ImageNormalize(vmin=-5, vmax=15, stretch=LinearStretch())
    >>> transition = normalize(5)
    >>> cmap = colormap_milagro(transition=transition)
    """
    from colorsys import hls_to_rgb
    from matplotlib.colors import LinearSegmentedColormap

    # Compute normalised red, blue, yellow values
    transition = float(transition)

    # Create custom colormap
    # List entries: (value, (H, L, S))
    colors = [
        (0, (1, 1, 0)),
        (transition - width, (1, 0, 0)),
        (transition, (huestart, 0.4, 0.5)),
        (transition + width, (huestart, 0.4, 1)),
        (0.99, (0, 0.6, 1)),
        (1, (0, 1, 1)),
    ]

    # Convert HLS values to RGB values
    rgb_colors = [(val, hls_to_rgb(*hls)) for (val, hls) in colors]

    return LinearSegmentedColormap.from_list(name="milagro", colors=rgb_colors)
