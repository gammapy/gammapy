# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Helper functions and functions for plotting gamma-ray images.
"""
from __future__ import print_function, division

__all__ = ['colormap_hess', 'colormap_milagro']

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
