# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Helper functions and functions for plotting Gamma-Ray data."""


def blue_red_colormap(value, width, data_min, data_max):
    """
    Create a blue-red colormap with well defined transition.

    Create a `matplotlib.colors.LinearSegmentedColormap` with a well defined
    transition between blue and red. Useful for illustrating data, with an
    'optical' threshold. E.g. residual significance maps where structures
    with sigma > 5 should appear red and structures with sigma < 5 should
    appear blue.

    Parameters
    ----------
    value : float
        Transition value between blue and red.
    width : float
        Width of the transition between blue and red.
    data_min : float
        Minimum of the data to be plotted.
    data_max : float
        Maximum of the data to be plotted.

    Examples
    --------
    Create a fake source measurement and show a nice significance map:

     .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D
        from astropy.convolution import Tophat2DKernel, Ring2DKernel, convolve
        from gammapy.image.plotting import blue_red_colormap
        from gammapy.stats.poisson import significance_on_off

        # Create a fake source
        source = Gaussian2D(6, 0, 0, 5, 5)
        y, x = np.mgrid[-50:51, -50:51]
        counts = np.random.poisson(source(x, y) + 3 * np.ones_like(x))

        # Setup ring and tophat convolution
        ring = Ring2DKernel(10, 5)
        ring.normalize(mode='peak')
        top = Tophat2DKernel(6)
        top.normalize(mode='peak')

        # Estimate on, off and alpha
        off = convolve(counts, ring, boundary='extend')
        on = convolve(counts, top, boundary='extend')
        alpha = top._array.sum() / ring._array.sum()

        # Compute significance
        signif = significance_on_off(on, off, alpha)

        # Plot with red blue colormap
        rb_cmap = blue_red_colormap(8, 0.1, -4., 16.)
        plt.imshow(signif, cmap=rb_cmap, vmin=-4, vmax=16)
        plt.title('Significance')
        plt.xlabel('x [pixels]')
        plt.ylabel('y [pixels]')
        plt.colorbar()
        plt.show()
    """
    from matplotlib.colors import LinearSegmentedColormap
    # Normalize transition value
    trans = (value - data_min) / (data_max - data_min)

    # Determine blue and yellow key values
    shiftb = trans - width
    shifty = 2. / 3. * (1 - trans) + trans

    if shiftb < 0:
        shiftb = 0.1 * trans

    # Create custom colormap
    blue = (0, 0, 0.8)
    yellow = (1., 1., 0)
    color_list = [(0, 'k'), (shiftb, blue), (trans, 'r'),
                  (shifty, yellow), (1., 'w')]
    return LinearSegmentedColormap.from_list("bluered", color_list)
