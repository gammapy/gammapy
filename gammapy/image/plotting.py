# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Helper functions and functions for plotting gamma-ray images.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
from astropy.coordinates import Angle

__all__ = [
    "colormap_hess",
    "colormap_milagro",
    "MapPanelPlotter",
    "illustrate_colormap",
    "grayify_colormap",
]

__doctest_requires__ = {("colormap_hess", "colormap_milagro"): ["matplotlib"]}


class MapPanelPlotter(object):
    """
    Mape panel plotter class.

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
        from matplotlib.gridspec import GridSpec

        self.figure = figure
        self.parameters = OrderedDict(xlim=xlim, ylim=ylim, npanels=npanels)
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
        xoverlap = ((p["npanels"] * width) - width_all) / (p["npanels"] - 1.)
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

    def plot_panel(self, image, panel=1, panel_fov=None, **kwargs):
        """
        Plot sky image on one panel.

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
        ax = self.figure.add_subplot(spec, projection=image.geom.wcs)
        try:
            ax = image.plot(ax=ax, **kwargs)[1]
        except AttributeError:
            ax = image.plot_rgb(ax=ax, **kwargs)
        ax = self._set_ax_fov(ax, panel_fov)
        return ax

    def plot(self, image, **kwargs):
        """
        Plot sky image on all panels.

        Parameters
        ----------
       map : `~gammapy.maps.WcsNDMap`
            Map to plot.
        """
        p = self.parameters
        axes = []
        for panel in range(p["npanels"]):
            ax = self.plot_panel(image, panel=panel, **kwargs)
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
    >>> from gammapy.image import colormap_hess
    >>> from astropy.visualization.mpl_normalize import ImageNormalize
    >>> from astropy.visualization import LinearStretch
    >>> normalize = ImageNormalize(vmin=-5, vmax=15, stretch=LinearStretch())
    >>> transition = normalize(5)
    >>> cmap = colormap_hess(transition=transition)

    .. plot::

        from gammapy.image import colormap_hess, illustrate_colormap
        import matplotlib.pyplot as plt
        cmap = colormap_hess()
        illustrate_colormap(cmap)
        plt.show()
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Compute normalised values (range 0 to 1) that
    # correspond to red, blue, yellow.
    red = float(transition)

    if width > red:
        blue = 0.1 * red
    else:
        blue = red - width

    yellow = 2. / 3. * (1 - red) + red

    black, white = 0, 1

    # Create custom colormap
    # List entries: (value, (R, G, B))
    colors = [
        (black, "k"),
        (blue, (0, 0, 0.8)),
        (red, "r"),
        (yellow, (1., 1., 0)),
        (white, "w"),
    ]
    cmap = LinearSegmentedColormap.from_list(name="hess", colors=colors)

    return cmap


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
    >>> from gammapy.image import colormap_milagro
    >>> from astropy.visualization.mpl_normalize import ImageNormalize
    >>> from astropy.visualization import LinearStretch
    >>> normalize = ImageNormalize(vmin=-5, vmax=15, stretch=LinearStretch())
    >>> transition = normalize(5)
    >>> cmap = colormap_milagro(transition=transition)


    .. plot::

        from gammapy.image import colormap_milagro, illustrate_colormap
        import matplotlib.pyplot as plt
        cmap = colormap_milagro()
        illustrate_colormap(cmap)
        plt.show()
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

    cmap = LinearSegmentedColormap.from_list(name="milagro", colors=rgb_colors)

    return cmap


def grayify_colormap(cmap, mode="hsp"):
    """
    Return a grayscale version a the colormap.

    The grayscale conversion of the colormap is bases on perceived luminance of
    the colors. For the conversion either the `~skimage.color.rgb2gray` or a
    generic method called ``hsp`` [1]_ can be used. The code is loosely based
    on [2]_.


    Parameters
    ----------
    cmap : str or `~matplotlib.colors.Colormap`
        Colormap name or instance.
    mode : {'skimage, 'hsp'}
        Grayscale conversion method. Either ``skimage`` or ``hsp``.

    References
    ----------

    .. [1] Darel Rex Finley, "HSP Color Model - Alternative to HSV (HSB) and HSL"
       http://alienryderflex.com/hsp.html

    .. [2] Jake VanderPlas, "How Bad Is Your Colormap?"
       https://jakevdp.github.io/blog/2014/10/16/how-bad-is-your-colormap/
    """
    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    if mode == "skimage":
        from skimage.color import rgb2gray  # pylint:disable=import-error

        luminance = rgb2gray(np.array([colors]))
        colors[:, :3] = luminance[0][:, np.newaxis]
    elif mode == "hsp":
        rgb_weight = [0.299, 0.587, 0.114]
        luminance = np.sqrt(np.dot(colors[:, :3] ** 2, rgb_weight))
        colors[:, :3] = luminance[:, np.newaxis]
    else:
        raise ValueError("Not a valid grayscale conversion mode.")

    return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)


def illustrate_colormap(cmap, **kwargs):
    """
    Illustrate color distribution and perceived luminance of a colormap.

    Parameters
    ----------
    cmap : str or `~matplotlib.colors.Colormap`
        Colormap name or instance.
    kwargs : dicts
        Keyword arguments passed to `grayify_colormap`.
    """
    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap(cmap)
    cmap_gray = grayify_colormap(cmap, **kwargs)
    figure = plt.figure(figsize=(8, 6))
    v = np.linspace(0, 1, 4 * cmap.N)

    # Show colormap
    show_cmap = figure.add_axes([0.1, 0.8, 0.8, 0.1])
    im = np.outer(np.ones(50), v)
    show_cmap.imshow(im, cmap=cmap, origin="lower")
    show_cmap.set_xticklabels([])
    show_cmap.set_yticklabels([])
    show_cmap.set_yticks([])
    show_cmap.set_title("RGB & Gray Luminance of colormap {}".format(cmap.name))

    # Show colormap gray
    show_cmap_gray = figure.add_axes([0.1, 0.72, 0.8, 0.09])
    show_cmap_gray.imshow(im, cmap=cmap_gray, origin="lower")
    show_cmap_gray.set_xticklabels([])
    show_cmap_gray.set_yticklabels([])
    show_cmap_gray.set_yticks([])

    # Plot RGB profiles
    plot_rgb = figure.add_axes([0.1, 0.1, 0.8, 0.6])
    plot_rgb.plot(v, [cmap(_)[0] for _ in v], color="#A60628")
    plot_rgb.plot(v, [cmap(_)[1] for _ in v], color="#467821")
    plot_rgb.plot(v, [cmap(_)[2] for _ in v], color="#348ABD")
    plot_rgb.plot(v, [cmap_gray(_)[0] for _ in v], color="k", linestyle="--")
    plot_rgb.set_ylabel("Luminance")
    plot_rgb.set_ylim(-0.005, 1.005)
