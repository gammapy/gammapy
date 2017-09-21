# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Helper functions and functions for plotting gamma-ray images.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np

from astropy.coordinates import Angle

__all__ = [
    'colormap_hess',
    'colormap_milagro',
    'fits_to_png',
    'GalacticPlaneSurveyPanelPlot',
    'fitsfigure_add_psf_inset',
    'illustrate_colormap',
    'grayify_colormap',
]

__doctest_requires__ = {('colormap_hess', 'colormap_milagro'): ['matplotlib']}


class SkyImagePanelPlotter(object):
    """
    Sky image panel plotter class

    Given a `~matplotlib.pyplot.Figure` object this class creates axes objects
    using `~matplotlib.gridspec.GridSpec` and plots a given sky image onto these.

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
        height = np.abs(p['ylim'].diff())
        width = aspect * height

        left, bottom = p['xlim'][0].wrap_at('180d'), p['ylim'][0]

        width_all = np.abs(p['xlim'].wrap_at('180d').diff())
        xoverlap = ((p['npanels'] * width) - width_all) / (p['npanels'] - 1.)
        if xoverlap < 0:
            raise ValueError('No overlap between panels. Please reduce figure '
                             'height or increase vertical space between the panels.')

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

    def plot_panel(self, skyimage, panel=1, panel_fov=None, **kwargs):
        """
        Plot sky image on one panel.

        Parameters
        ----------
        skyimage : `~gammapy.image.SkyImage`
            Sky image to plot.
        panel : int
            Which panel to plot on (counted from top).
        """
        if panel_fov is None:
            panel_fov = panel
        spec = self.grid_spec[panel]
        ax = self.figure.add_subplot(spec, projection=skyimage.wcs)
        try:
            ax = skyimage.plot(ax=ax, **kwargs)[1]
        except AttributeError:
            ax = skyimage.plot_rgb(ax=ax, **kwargs)
        ax = self._set_ax_fov(ax, panel_fov)
        return ax

    def plot(self, skyimage, **kwargs):
        """
        Plot sky image on all panels.

        Parameters
        ----------
        skyimage : `~gammapy.image.SkyImage`
            Sky image to plot.
        """
        p = self.parameters
        axes = []
        for panel in range(p['npanels']):
            ax = self.plot_panel(skyimage, panel=panel, **kwargs)
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
    colors = [(black, 'k'),
              (blue, (0, 0, 0.8)),
              (red, 'r'),
              (yellow, (1., 1., 0)),
              (white, 'w'),
              ]
    cmap = LinearSegmentedColormap.from_list(name='hess', colors=colors)

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
    colors = [(0, (1, 1, 0)),
              (transition - width, (1, 0, 0)),
              (transition, (huestart, 0.4, 0.5)),
              (transition + width, (huestart, 0.4, 1)),
              (0.99, (0, 0.6, 1)),
              (1, (0, 1, 1)),
              ]

    # Convert HLS values to RGB values
    rgb_colors = [(val, hls_to_rgb(*hls)) for (val, hls) in colors]

    cmap = LinearSegmentedColormap.from_list(name='milagro', colors=rgb_colors)

    return cmap


def fits_to_png(infile, outfile, draw, dpi=100):
    """Plot FITS image in PNG format.

    For the default ``dpi=100`` a 1:1 copy of the pixels in the FITS image
    and the PNG image is achieved, i.e. they have exactly the same size.

    Parameters
    ----------
    infile : str
        Input FITS file name
    outfile : str
        Output PNG file name
    draw : callable
        Callback function ``draw(figure)``
        where ``figure`` is an `~aplpy.FITSFigure`.
    dpi : int
        Resolution

    Examples
    --------
    >>> def draw(figure):
    ...     x, y, width, height = 42, 0, 3, 2
    ...     figure.recenter(x, y, width, height)
    ...     figure.show_grayscale()
    >>> from gammapy.image import fits_to_png
    >>> fits_to_png('image.fits', 'image.png', draw)
    """
    import matplotlib
    matplotlib.use('Agg')  # Prevents image popup
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from aplpy import FITSFigure

    # Peak ahead just to get the figure size
    NAXIS1 = float(fits.getval(infile, 'NAXIS1'))
    NAXIS2 = float(fits.getval(infile, 'NAXIS2'))

    # Note: For dpi=100 I get exactly the same FITS and PNG image size in pix.
    figsize = np.array((NAXIS1, NAXIS2))
    figure = plt.figure(figsize=figsize / dpi)
    # Also try this:
    # matplotlib.rcParams['figure.figsize'] = NAXIS1, NAXIS2
    # figsize(x,y)

    subplot = [0, 0, 1, 1]
    figure = FITSFigure(infile, figure=figure, subplot=subplot)

    draw(figure)

    figure.axis_labels.hide()
    figure.tick_labels.hide()
    figure.ticks.set_linewidth(0)
    figure.frame.set_linewidth(0)

    figure.save(outfile, max_dpi=dpi, adjust_bbox=False)


class GalacticPlaneSurveyPanelPlot(object):
    """Plot Galactic plane survey images in multiple panels.

    This is useful for very wide, but not so high survey images
    (~100 deg in Galactic longitude and ~10 deg in Galactic latitude).

    TODO: describe how the callbacks work

    References:
    http://aplpy.readthedocs.io/en/latest/howto_subplot.html

    Attributes:

    * ``panel_parameters`` -- dict of panel parameters
    * ``figure`` --- Main matplotlib figure (cantains all panels)
    * ``fits_figure`` --- Current `aplpy.FITSFigure`

    Parameters
    ----------
    fits_figure : `aplpy.FITSFigure`
        FITSFigure to plot on all panels
    npanels : int
        Number of panels

    Examples
    --------
    TODO

    TODO: Link to tutorial example
    """

    def __init__(self, npanels=4, center=(0, 0), fov=(10, 1),
                 xsize=10, ysize=None, xborder=0.5, yborder=0.5,
                 yspacing=0.5, xoverlap=0):
        """Compute panel parameters and make a matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        self.panel_parameters = _panel_parameters(npanels=npanels,
                                                  center=center,
                                                  fov=fov,
                                                  xsize=xsize,
                                                  ysize=ysize,
                                                  xborder=xborder,
                                                  yborder=yborder,
                                                  yspacing=yspacing,
                                                  xoverlap=xoverlap)

        self.figure = plt.figure(figsize=self.panel_parameters['figsize'])

    def bottom(self, colorbar_pars={}, colorbar_label=''):
        """TODO: needed?
        """
        if colorbar_pars:
            self.fits_figure.add_colorbar(**colorbar_pars)
        if colorbar_label != '':
            self.fits_figure.colorbar.set_font(size='small')
            self.fits_figure.colorbar._colorbar.set_label(colorbar_label)

    def top(self):
        """TODO: needed?
        """
        pass

    def draw_panels(self, panels='all', format=True):
        """Draw panels.

        Parameters
        ----------
        panels : list of ints or 'all'
            List of panels to draw.
        """
        if panels == 'all':
            panels = range(self.panel_parameters['npanels'])

        for panel in panels:
            self.draw_panel(panel, format=format)

            # self.figure.canvas.draw()

    def draw_panel(self, panel=0, format=True):
        """Draw panel.

        Parameters
        ----------
        panel : int
            Panel index
        """
        pp = self.panel_parameters
        center = pp['centers'][panel]
        self.subplot = pp['subplots'][panel]

        # Execute user-defined plotting ...
        # This must set self.fits_figure
        self.main(self.figure, self.subplot)

        # self.fits_figure.set_auto_refresh(False)
        self.fits_figure.recenter(center[0], center[1],
                                  width=pp['width'], height=pp['height'])

        if panel == 0:
            self.bottom()
        if panel == (pp['npanels'] - 1):
            self.top()

        # fits_figure.refresh()
        # self.figure.canvas.draw()

        # To ensure compatibility with old code
        if hasattr(self, 'post'):
            self.post()
        if format:
            GalacticPlaneSurveyPanelPlot.format_fits_figure(self.fits_figure)

    @staticmethod
    def format_fits_figure(fits_figure, theme=None):
        """TODO: describe

        Parameters
        ----------
        TODO
        """
        if theme is not None:
            fits_figure.set_theme('publication')
        fits_figure.axis_labels.hide()
        fits_figure.ticks.set_xspacing(5)
        fits_figure.ticks.set_yspacing(1)
        fits_figure.tick_labels.set_xformat('dd')
        fits_figure.tick_labels.set_yformat('dd')
        fits_figure.tick_labels.set_style('colons')
        # fits_figure.tick_labels.set_font(size='small')


def _panel_parameters(npanels, center, fov, xborder, yborder,
                      yspacing, xoverlap=0, xsize=None, ysize=None):
    """Compute panel parameters.

    This function computes all relevant quantities to plot
    a very wide survey map in n slices.

    This is surprisingly complicated because coordinates are
    relative to figsize, which is already not 1:1.

    TODO: document panel parameters.

    Parameters
    ----------
    npanels : int
        Number of slices
    center : pair
        Image center position (lon, lat)
    fov : pair
        Image full-width and full-height
    xsize : float
        Width of the figure in inches
    ysize : float (None)
        Height of the figure in inches
    xborder : float
        Free space to x border in inches
    yborder : float
        Free space to y border in inches
    yspacing : float
        Free space between slices in inches
    xoverlap : float
        Overlap between single panels in deg.

    Returns
    -------
    panel_parameters : dict
        Dictionary of panel parameters
    """
    # Need floats for precise divisions
    center = [float(center[0]), float(center[1])]
    fov = [float(fov[0]), float(fov[1])]
    xborder, yborder = float(xborder), float(yborder)
    yspacing = float(yspacing)

    # Width and height in deg of a slice
    width = fov[0] / npanels
    height = fov[1]
    # Aspect ratio y:x of a slice
    aspectratio = fov[1] / (fov[0] / npanels)
    # Absolute figure dimensions
    if ysize is None and xsize is not None:
        ysize = (2 * yborder + (npanels - 1) * yspacing +
                 npanels * aspectratio * (float(xsize) - 2 * xborder))
    elif xsize is None and ysize is not None:
        xsize = ((float(ysize) - (2 * yborder + (npanels - 1) * yspacing)) /
                 (npanels * aspectratio) + 2 * xborder)
    else:
        raise ValueError('Either xsize or ysize must be specified.')

    figsize = [xsize, ysize]

    # Relative slice subplot dimensions
    dx = 1 - 2 * xborder / xsize
    dy = aspectratio * dx * xsize / ysize
    dyspacing = yspacing / ysize

    # List of y slice offsets
    subplots = []
    subplot_centers = []
    for ii in range(npanels):
        subplot_center = [center[0] - fov[0] / 2 + (ii + 0.5) * width, center[1]]
        subplot = [xborder / xsize, yborder / ysize + ii * (dy + dyspacing), dx, dy]
        subplot_centers.append(subplot_center)
        subplots.append(subplot)

    pp = dict()
    pp['figsize'] = figsize
    pp['npanels'] = npanels
    pp['centers'] = subplot_centers
    pp['subplots'] = subplots
    pp['width'] = width + xoverlap
    pp['height'] = height

    return pp


def fitsfigure_add_psf_inset(ff, psf, box, linewidth=1, color='w',
                             psf_position=(0, 0), **kwargs):
    """
    Add PSF inset to `~aplpy.FITSFigure` instance.

    Parameters
    ----------
    ff : `~aplpy.FITSFigure`
        `~aplpy.FITSFigure` instance.
    psf : `astropy.io.fits.ImageHDU`
        PSF image.
    box : tuple
        (x, y, width, height) of the PSF inset in world coordinates.
    linewidth : float
        Linewidth of the PSF inset frame.
    color : str
        Color of the PSF inset frame.
    psf_position : tuple
        (x, y) position of the psf in in the psf image in pixel coordinates.
    kwargs : dict
        Further arguments passed to `~matplotlib.pyplot.imshow`.

    Returns
    -------
    psf : `~matplotlib.axes.Axes`
        PSF `~matplotlib.axes.Axes` instance, can be used for further plotting.
    """
    rect = _rect_world2fig(ff, box)

    # WCSAxes should be used here
    psf_axes = ff._figure.add_axes(rect)
    for spline in psf_axes.spines.values():
        spline.set_edgecolor(color)
        spline.set_linewidth(linewidth)

    psf_axes.xaxis.set_ticks([])
    psf_axes.yaxis.set_ticks([])

    psf_axes.imshow(psf.data, **kwargs)
    xc, yc = psf_position
    wc = box[2] / abs(psf.header['CDELT1']) / 2.
    hc = box[3] / abs(psf.header['CDELT2']) / 2.
    psf_axes.set_xlim(xc - wc, xc + wc)
    psf_axes.set_ylim(yc - hc, yc + hc)
    return psf_axes


def fitsfigure_add_colorbar_inset(ff, box, linewidth=1, color='w', normalize=None,
                                  label='', label_position='right', label_pad=0,
                                  n_ticks=5, ticklabel_format='.1f', tick_size=5):
    """
    Add colorbar inset to existing `~aplpy.FITSFigure` instance.

    Parameters
    ----------
    ff : `~aplpy.FITSFigure`
        `~aplpy.FITSFigure` instance.
    box : tuple
        (x, y, width, height) of the colorbar inset in world coordinates.
    linewidth : float
        Linewidth of the colorbar inset frame.
    color : str
        Color of the colorbar inset frame.
    normalize : `~astropy.visualization.mpl_normalize.ImageNormalize` (None)
        `~astropy.visualization.mpl_normalize.ImageNormalize` instance.
    label : str
        Colorbar label.
    label_position : {'right', 'bottom'}
        Colorbar label position.
    label_pad : float
        Colorbar label padding.
    n_ticks : int (default = 5)
        Number of ticks and tick labels.
    ticklabel_format : str (default = '.1f')
        Tick label fomating string.
    ticksize : float
        Size of the colorbar ticks.

    Returns
    -------
    psf : `~matplotlib.axes.Axes`
        Colorbar `~matplotlib.axes.Axes` instance, can be used for further plotting.
    """
    rect = _rect_world2fig(ff, box)
    cbar_axes = ff._figure.add_axes(rect)
    cbar = ff._figure.colorbar(ff.image, cax=cbar_axes)
    cbar.solids.set_edgecolor('face')
    cbar.outline.set_edgecolor(color)
    cbar.outline.set_linewidth(linewidth)
    cbar.ax.yaxis.set_tick_params(color=color, size=tick_size)
    ticks_pos = np.linspace(0, 1, n_ticks)
    if normalize is not None:
        ticks_pos = normalize.inverse(ticks_pos)
    tick_labels = [('{0:' + ticklabel_format + '}').format(_) for _ in ticks_pos]
    cbar.set_ticks(np.linspace(0, 1, n_ticks))
    cbar_axes.set_yticklabels(tick_labels, color=color)
    if label_position == 'bottom':
        cbar_axes.set_xlabel(label, color=color, labelpad=label_pad)
    elif label_position == 'right':
        cbar_axes.set_ylabel(label, color=color, labelpad=label_pad)
    else:
        raise ValueError("Position of the label must be either 'right' or 'bottom'")
    return cbar_axes


def _rect_world2fig(ff, rect):
    """
    Transform rectangle from world to figure coordinates.

    Paramaters
    ----------
    ff : `~aplpy.FITSFigure`
        `~aplpy.FITSFigure` instance.
    rect : tuple
        Tuple that defines the rectangle like [x, y, width, height] in world
        coordinates.

    Returns
    -------
    rect : tuple
        Tuple that defines the rectangle like [x, y, width, height] in figure
        coordinates.
    """
    x, y, width, height = rect
    xf, yf = _world2fig(ff, [x, x + width], [y, y + height])
    return [xf[0], yf[0], abs(xf[1] - xf[0]), abs(yf[1] - yf[0])]


def _world2fig(ff, x, y):
    """
    Helper function to convert world to figure coordinates.

    Parameters
    ----------
    ff : `~aplpy.FITSFigure`
        `~aplpy.FITSFigure` instance.
    x : ndarray
        Array of x coordinates.
    y : ndarray
        Array of y coordinates.

    Returns
    -------
    coordsf : tuple
        Figure coordinates as tuple (xfig, yfig) of arrays.
    """
    # Convert world to pixel coordinates
    xp, yp = ff.world2pixel(x, y)

    # Pixel to Axes coordinates
    coordsa = ff._ax1.transData.transform(zip(xp, yp))

    # Axes to figure coordinates
    coordsf = ff._figure.transFigure.inverted().transform(coordsa)
    return coordsf[:, 0], coordsf[:, 1]


def grayify_colormap(cmap, mode='hsp'):
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

    if mode == 'skimage':
        from skimage.color import rgb2gray
        luminance = rgb2gray(np.array([colors]))
        colors[:, :3] = luminance[0][:, np.newaxis]
    elif mode == 'hsp':
        RGB_weight = [0.299, 0.587, 0.114]
        luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
        colors[:, :3] = luminance[:, np.newaxis]
    else:
        raise ValueError('Not a valid grayscale conversion mode.')

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
    show_cmap.imshow(im, cmap=cmap, origin='lower')
    show_cmap.set_xticklabels([])
    show_cmap.set_yticklabels([])
    show_cmap.set_yticks([])
    show_cmap.set_title('RGB & Gray Luminance of colormap {}'.format(cmap.name))

    # Show colormap gray
    show_cmap_gray = figure.add_axes([0.1, 0.72, 0.8, 0.09])
    show_cmap_gray.imshow(im, cmap=cmap_gray, origin='lower')
    show_cmap_gray.set_xticklabels([])
    show_cmap_gray.set_yticklabels([])
    show_cmap_gray.set_yticks([])

    # Plot RGB profiles
    plot_rgb = figure.add_axes([0.1, 0.1, 0.8, 0.6])
    plot_rgb.plot(v, [cmap(_)[0] for _ in v], color='#A60628')
    plot_rgb.plot(v, [cmap(_)[1] for _ in v], color='#467821')
    plot_rgb.plot(v, [cmap(_)[2] for _ in v], color='#348ABD')
    plot_rgb.plot(v, [cmap_gray(_)[0] for _ in v], color='k', linestyle='--')
    plot_rgb.set_ylabel('Luminance')
    plot_rgb.set_ylim(-0.005, 1.005)
