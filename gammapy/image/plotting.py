# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Helper functions and functions for plotting gamma-ray images.
"""
from __future__ import print_function, division
import numpy as np

__all__ = ['colormap_hess', 'colormap_milagro',
           'fits_to_png',
           'GalacticPlaneSurveyPanelPlot',
           ]

__doctest_requires__ = {('colormap_hess', 'colormap_milagro'): ['matplotlib']}


def colormap_hess(vmin, vmax, vtransition, width=0.1):
    """Colormap often used in H.E.S.S. collaboration publications.

    This colormap goes black -> blue -> red -> yellow -> white.

    A sharp blue -> red -> yellow transition is often used for significance images
    with a value of red at ``vtransition ~ 5`` or ``vtransition ~ 7``
    so that the following effect is achieved:

    - black, blue: non-significant features, not well visible
    - red: features at the detection threshold ``vtransition``
    - yellow, white: significant features, very well visible

    Parameters
    ----------
    vmin : float
        Minimum value (color: black)
    vmax : float
        Maximum value (color: white)
    vtransition : float
        Transition value (color: red).
    width : float
        Width of the blue-red color transition (fraction in ``vmax - vmin`` range).

    Returns
    -------
    colormap : `matplotlib.colors.LinearSegmentedColormap`
        Colormap

    Examples
    -------- 
    >>> from gammapy.image import colormap_hess
    >>> vmin, vmax, vtransition = -5, 15, 5
    >>> cmap = colormap_hess(vmin=vmin, vmax=vmax, vtransition=vtransition)

    .. plot::

        from gammapy.image import colormap_hess
        vmin, vmax, vtransition = -5, 15, 5
        cmap = colormap_hess(vmin=vmin, vmax=vmax, vtransition=vtransition)

        # This is how to plot a colorbar only with matplotlib
        # http://matplotlib.org/examples/api/colorbar_only.html
        import matplotlib.pyplot as plt
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.colors import Normalize
        norm = Normalize(vmin, vmax)
        fig = plt.figure(figsize=(8, 1))
        fig.add_axes([0.05, 0.3, 0.9, 0.6])
        ColorbarBase(plt.gca(), cmap, norm, orientation='horizontal')
        plt.show()
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Compute normalised values (range 0 to 1) that
    # correspond to red, blue, yellow.
    red = float(vtransition - vmin) / (vmax - vmin)

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


def colormap_milagro(vmin, vmax, vtransition, width=0.0001, huestart=0.6):
    """Colormap often used in Milagro collaboration publications.

    This colormap is gray below ``vtransition`` and similar to the jet colormap above.

    A sharp gray -> color transition is often used for significance images
    with a transition value of ``vtransition ~ 5`` or ``vtransition ~ 7``,
    so that the following effect is achieved:

    - gray: non-significant features are not well visible
    - color: significant features at the detection threshold ``vmid``

    Note that this colormap is often critizised for over-exaggerating small differences
    in significance below and above the gray - color transition threshold.

    Parameters
    ----------
    vmin : float
        Minimum value (color: black)
    vmax : float
        Maximum value
    vtransition : float
        Transition value (below: gray, above: color).
    width : float
        Width of the transition
    huestart : float
        Hue of the color at ``vtransition``

    Returns
    -------
    colormap : `matplotlib.colors.LinearSegmentedColormap`
        Colormap

    Examples
    --------
    >>> from gammapy.image import colormap_milagro
    >>> vmin, vmax, vtransition = -5, 15, 5
    >>> cmap = colormap_milagro(vmin=vmin, vmax=vmax, vtransition=vtransition)

    .. plot::

        from gammapy.image import colormap_milagro
        vmin, vmax, vtransition = -5, 15, 5
        cmap = colormap_milagro(vmin=vmin, vmax=vmax, vtransition=vtransition)

        # This is how to plot a colorbar only with matplotlib
        # http://matplotlib.org/examples/api/colorbar_only.html
        import matplotlib.pyplot as plt
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.colors import Normalize
        norm = Normalize(vmin, vmax)
        fig = plt.figure(figsize=(8, 1))
        fig.add_axes([0.05, 0.3, 0.9, 0.6])
        ColorbarBase(plt.gca(), cmap, norm, orientation='horizontal')
        plt.show()
    """
    from colorsys import hls_to_rgb
    from matplotlib.colors import LinearSegmentedColormap

    # Compute normalised red, blue, yellow values
    transition = float(vtransition - vmin) / (vmax - vmin)

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
    http://aplpy.readthedocs.org/en/latest/howto_subplot.html

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
                 xsize=10, xborder=0.5, yborder=0.5, yspacing=0.5):
        """Compute panel parameters and make a matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        self.panel_parameters = _panel_parameters(npanels=npanels,
                                                  center=center,
                                                  fov=fov,
                                                  xsize=xsize,
                                                  xborder=xborder,
                                                  yborder=yborder,
                                                  yspacing=yspacing)

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
        self.box = pp['boxes'][panel]

        # Execute user-defined plotting ...
        # This must set self.fits_figure
        self.main(self.figure, self.box)

        # self.fits_figure.set_auto_refresh(False)
        self.fits_figure.recenter(center[0], center[1],
                                  width=pp['width'], height=pp['height'])

        if panel == 0:
            self.bottom()
        if panel == (pp['npanels'] - 1):
            self.top()

        # fits_figure.refresh()
        # self.figure.canvas.draw()

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


def _panel_parameters(npanels, center, fov,
                      xsize, xborder, yborder, yspacing):
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
    xborder : float
        Free space to x border in inches
    yborder : float
        Free space to y border in inches
    yspacing : float
        Free space between slices in inches

    Returns
    -------
    panel_parameters : dict
        Dictionary of panel parameters
    """
    # Need floats for precise divisions
    center = [float(center[0]), float(center[1])]
    fov = [float(fov[0]), float(fov[1])]
    xsize = float(xsize)
    xborder = float(xborder)
    yborder = float(yborder)
    yspacing = float(yspacing)

    # Width and height in deg of a slice
    width = fov[0] / npanels
    height = fov[1]
    # Aspect ratio y:x of a slice
    aspectratio = fov[1] / (fov[0] / npanels)
    # Absolute figure dimensions
    ysize = (2 * yborder + (npanels - 1) * yspacing +
             npanels * aspectratio * (xsize - 2 * xborder))
    figsize = [xsize, ysize]

    # Relative slice subplot dimensions
    dx = 1 - 2 * xborder / xsize
    dy = aspectratio * dx * xsize / ysize
    dyspacing = yspacing / ysize

    # List of y slice offsets
    boxes = []
    box_centers = []
    for ii in range(npanels):
        box_center = [center[0] - fov[0] / 2 + (ii + 0.5) * width, center[1]]
        box = [xborder / xsize, yborder / ysize + ii * (dy + dyspacing), dx, dy]
        box_centers.append(box_center)
        boxes.append(box)

    pp = dict()
    pp['figsize'] = figsize
    pp['npanels'] = npanels
    pp['centers'] = box_centers
    pp['boxes'] = boxes
    pp['width'] = width
    pp['height'] = height

    return pp
