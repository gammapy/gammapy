# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Convenience functions for chi**2 and likelihood fitting.

TODO: Unusable at the moment. Refactor into classes and clean up.
"""
from __future__ import print_function, division
import numpy as np

__all__ = ['generate_MC_data',
           'plot_chi2',
           'plot_fit',
           'plot_model',
           'plot_points',
           ]


def set_off_diagonal_to_zero(matrix):
    """Sets the off-diagonal elements of a matrix
    """
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j:
                matrix[i, j] = 0
    return matrix


def generate_MC_data(model, xlim, yerr, npoints=5, verbose=0):
    """Generate points with equal-log spacing in x given by the model.

    model = [function, parameters, constants]
    xlim = [xmin, xmax]
    yerr = [ydn_err, yup_err] = fractional error on model y
    Returns:
    data = x, y, ydn, yup
    """
    # Generate equal-log spaced x points
    logx = np.linspace(np.log10(xlim[0]), np.log10(xlim[1]), npoints)
    x = 10 ** logx

    # Unpack model components
    f, p, c = model

    # Compute true y and asymmetric errors
    y = f(p, c)
    ydn = y * yerr[0]
    yup = y * yerr[1]

    #
    # Compute observed y by drawing a random value
    #

    # First decide if an up or down fluctuation occurs
    fluctuate_up = np.random.randint(0, 2, npoints)  # 1 = yes, 2 = no

    # Then draw a random y value
    yobs_dn = np.fabs(np.random.normal(0, ydn, size=npoints))
    yobs_up = np.fabs(np.random.normal(0, yup, size=npoints))
    yobs = y + np.where(fluctuate_up == 1, yobs_up, -yobs_dn)

    if verbose > 0:
        for i in range(npoints):
            fmt = '%2d' + ' %10g' * 7
            vals = i, x[i], y[i], ydn[i], yup[i], yobs_dn[i], yobs_up[i], yobs[i]
            print(fmt.format(vals))
    data = x, yobs, ydn, yup
    return data


def plot_points(data, xpower=0):
    """Make a nice plot
    """
    import matplotlib.pylab as plt
    x = data[0]
    y = data[1] * x ** xpower
    ydn = data[2] * x ** xpower
    yup = data[3] * x ** xpower
    plt.errorbar(x, y, [ydn, yup], fmt='o', color='k')


def plot_fit(f, c, fit_result, xlimits, ylimits,
             disregard_correlation=False,
             color='gray', alpha=0.3,
             fill_band=True,
             npoints=100):
    """Plot the error butterfly.

    Errors are propagated using the uncertainties module.
    If disregard_correlation == True, the off-diagonal elements of the
    covariance matrix are set to 0 before propagating the errors.
    """
    import matplotlib.pylab as plt
    import uncertainties
    # Choose equal-log x spacing
    logxlimits = np.log10(xlimits)
    logx = np.linspace(logxlimits[0], logxlimits[1], npoints)
    x = 10 ** logx
    popt = fit_result[0]
    pcov = fit_result[1]
    if disregard_correlation:
        pcov = set_off_diagonal_to_zero(pcov)
    y = f(popt, c, x)
    # Use uncertainties to compute an error band
    p_wu = uncertainties.correlated_values(popt, pcov)
    y_wu = f(p_wu, c, x)
    y_val = np.empty_like(y_wu)
    y_err = np.empty_like(y_wu)
    for i in range(y_wu.size):
        y_val[i] = y_wu[i].nominal_value
        y_err[i] = y_wu[i].std_dev
    # Need to clip values to frame so that no plotting artifacts occur
    y1 = np.maximum(y - y_err, ylimits[0])
    y2 = np.minimum(y + y_err, ylimits[1])

    # Plot error band
    if fill_band == True:
        plt.fill_between(x, y1, y2, color=color, alpha=alpha)
    else:
        plt.plot(x, y1, color=color, alpha=alpha)
        plt.plot(x, y2, color=color, alpha=alpha)

    # Plot best-fit spectrum
    plt.plot(x, y, color=color, alpha=alpha)


def plot_model(model, xlim, npoints=100):
    """Plot model curve.

    model = [function, parameters, constants]
    """
    import matplotlib.pylab as plt
    # Choose equal-log x spacing for plotting
    logxlim = np.log10(xlim)
    logx = np.linspace(logxlim[0], logxlim[1], npoints)
    x = 10 ** logx
    # Unpack model and compute y vector
    f, p, c = model
    y = f(p, c, x)
    plt.plot(x, y)


def plot_chi2(model, data, fit_result=None, limits=None,
              disregard_correlation=False,
              npoints=(100, 100), stddev_max=3, fmt='%d',
              linewidth=2):
    """Plot chi**2 contours and linear fit approxiation

    Note that chi**2 is related to likelihood in the following way:
    L = exp(-(1/2)*chi**2)
    This means that log(L) and chi**2 are identical up to a factor (-2).
    """
    import matplotlib.pylab as plt
    # Unpack model and fit result
    f, p, c = model
    # popt, pcov

    if disregard_correlation:
        fit_result[1] = set_off_diagonal_to_zero(fit_result[1])

    # If no limits are given, compute a good choice from fit_result

    # Set up a grid of points in parameter space
    p1_lim = limits[0]
    p2_lim = limits[1]
    p1 = np.linspace(p1_lim[0], p1_lim[1], npoints[0])
    p2 = np.linspace(p2_lim[0], p2_lim[1], npoints[1])
    P1, P2 = plt.meshgrid(p1, p2)

    # Compute chi**2 (called chi2) for each grid point
    # Looping can probably be avoided, but I don't know how.
    x2 = np.empty_like(P1)  # real chi**2
    x2lin = np.empty_like(P1)  # linear chi**2 approximation
    for i1 in range(p1.size):
        for i2 in range(p2.size):
            # Note the weird numpy indexing order.
            # i2,i1 seems wrong, but is correct:
            # TODO: implement
            pass
            # x2   [i2, i1] = chi2    ((p1[i1], p2[i2]), c, f, data)
            # x2lin[i2, i1] = chi2_lin((p1[i1], p2[i2]), fit_result)

    # Set the most likely point to chi**2 = 0
    x2 -= x2.min()

    # Use sqrt scale
    # x2 = np.sqrt(x2)
    # x2lin = np.sqrt(x2lin)

    # Plot likelihood as color landscape
    x2_image = plt.pcolor(P1, P2, x2, vmin=0,
                          vmax=stddev_max ** 2, cmap='gray')
    plt.colorbar()
    # Add marker at the minimum
    plt.plot(fit_result[0][0], fit_result[0][1],
            marker='*', markersize=12, color='r')

    # Add contour of real likelihood
    contour_levels = np.arange(1, stddev_max + 1) ** 2
    x2_cont = plt.contour(P1, P2, x2, contour_levels,
                         colors='r', linewidths=linewidth)
    plt.clabel(x2_cont, inline=1, fontsize=10,
               linewidths=linewidth, fmt=fmt)

    # Overplot linear approximation as contours
    x2lin_cont = plt.contour(P1, P2, x2lin, contour_levels,
                             colors='b', linewidths=linewidth)
    plt.clabel(x2lin_cont, inline=1, fontsize=10,
               linewidths=linewidth, fmt=fmt)

    # Add colorbar and labels
    # axcb = plt.colorbar()
    # axcb.set_label(r'$\chi^2$', size=15)
    plt.xlabel('Parameter P1')
    plt.ylabel('Parameter P2')
    plt.axis(p1_lim + p2_lim)
