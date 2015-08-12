# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Provide a Feldman Cousins algorithm."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.extern.six.moves import range

import logging
log = logging.getLogger(__name__)

__all__ = ['fc_find_confidence_interval_gauss',
           'fc_find_confidence_interval_poisson',
           'fc_construct_confidence_belt_pdfs',
           'fc_get_upper_and_lower_limit',
           'fc_fix_upper_and_lower_limit',
           'fc_find_limit',
           'fc_find_average_upper_limit',
           'fc_construct_confidence_belt',
          ]


def fc_find_confidence_interval_gauss(mu, sigma, x_bins, cl):
    """Analytically find confidence interval for Gaussian with boundary 
       at the origin

    Parameters
    ----------
    mu : double
        Mean of the Gaussian
    sigma : double
        Width of the Gaussian
    x_bins : array-like
        Bins in x
    cl : double
        Desired confidence level

    Returns
    -------
    (x_min, x_max) : tuple of floats
        Confidence interval
    """

    from scipy import stats

    dist = stats.norm(loc=mu, scale=sigma)

    x_bin_width = x_bins[1] - x_bins[0]

    p = []
    r = []

    for x in x_bins:
        p.append(dist.pdf(x)*x_bin_width)
        # This is the formula from the FC paper
        if mu == 0 and sigma == 1:
            if x < 0:
                r.append(np.exp(x*mu-mu*mu*0.5))
            else:
                r.append(np.exp(-0.5*np.power((x-mu),2)))
        # This is the more general formula
        else:
            # Implementing the boundary condition at zero
            muBest     = max(0, x)
            probMuBest = stats.norm.pdf(x, loc=muBest, scale=sigma)
            if probMuBest == 0.0:
                r.append(0.0);
            else:
                r.append(p[-1]/probMuBest)

    p = np.asarray(p)
    r = np.asarray(r)

    if sum(p) < cl:
        log.info("Bad choice of x-range for this mu!")
        log.info("Not enough probability in x bins to reach confidence level!")

    rank = stats.rankdata(-r, method='dense')

    index_array = np.arange(x_bins.size)

    rank_sorted, index_array_sorted = zip(*sorted(zip(rank, index_array)))

    index_min = index_array_sorted[0]
    index_max = index_array_sorted[0]

    p_sum = 0

    for i in range(len(rank_sorted)):
        if index_array_sorted[i] < index_min:
            index_min = index_array_sorted[i]
        if index_array_sorted[i] > index_max:
            index_max = index_array_sorted[i]
        p_sum += p[index_array_sorted[i]]
        if p_sum >= cl:
            break

    return x_bins[index_min], x_bins[index_max] + x_bin_width


def fc_find_confidence_interval_poisson(mu, background, x_bins, cl):
    """Analytically find confidence interval for Poisson process with background

    Parameters
    ----------
    mu : double
        Mean of the signal
    background : double
        Mean of the background
    x_bins : array-like
        Bins in x
    cl : double
        Desired confidence level

    Returns
    -------
    (x_min, x_max) : tuple of floats
        Confidence interval
    """

    from scipy import stats

    dist = stats.poisson(mu=mu+background)

    x_bin_width = x_bins[1] - x_bins[0]

    p = []
    r = []

    for x in x_bins:
        p.append(dist.pmf(x))
        # Implementing the boundary condition at zero
        muBest = max(0, x - background)
        probMuBest = stats.poisson.pmf(x, mu=muBest+background)
        if probMuBest == 0.0:
            r.append(0.0);
        else:
            r.append(p[-1]/probMuBest)

    p = np.asarray(p)
    r = np.asarray(r)

    if sum(p) < cl:
        log.info("Bad choice of x-range for this mu!")
        log.info("Not enough probability in x bins to reach confidence level!")

    rank = stats.rankdata(-r, method='dense')

    index_array = ny.arange(x_bins.size)

    rank_sorted, index_array_sorted = zip(*sorted(zip(rank, index_array)))

    index_min = index_array_sorted[0]
    index_max = index_array_sorted[0]

    p_sum = 0

    for i in range(len(rank_sorted)):
        if index_array_sorted[i] < index_min:
            index_min = index_array_sorted[i]
        if index_array_sorted[i] > index_max:
            index_max = index_array_sorted[i]
        p_sum += p[index_array_sorted[i]]
        if p_sum >= cl:
            break

    return x_bins[index_min], x_bins[index_max] + x_bin_width


def fc_construct_confidence_belt_pdfs(matrix, alpha):
    """Numerically choose bins a la Feldman Cousins ordering principle.

    Parameters
    ----------
    matrix : array-like
        A list of x PDFs for increasing values of mue.
    alpha : float
        Desired confidence level

    Returns
    -------
    distributions_scaled : ndarray
        Confidence intervals (1 means inside, 0 means outside)
    """

    number_mus    = len(matrix)
    number_x_bins = len(matrix[0])

    distributions_scaled    = np.asarray(matrix)
    distributions_re_scaled = np.asarray(matrix)
    summed_propability      = np.zeros(number_mus)

    # Step 1:
    # For each x, find the greatest likelihood in the mu direction.
    # greatest_likelihood is an array of length number_x_bins.
    greatest_likelihood = np.amax(distributions_scaled, axis=0)

    # Set to some value if none of the bins has an entry to avoid 
    # division by zero
    greatest_likelihood[greatest_likelihood == 0] = 1

    # Step 2:
    # Scale all entries by this value
    distributions_re_scaled /= greatest_likelihood

    # Step 3 (Feldman Cousins Ordering principel):
    # For each mu, get the largest entry
    largest_entry = np.argmax(distributions_re_scaled, axis = 1)
    # Set the rank to 1 and add probability
    for i in range(number_mus):
        distributions_re_scaled[i][largest_entry[i]] = 1
        summed_propability[i]  += np.sum(numpy.where(distributions_re_scaled[i] == 1, distributions_scaled[i], 0))
        distributions_scaled[i] = np.where(distributions_re_scaled[i] == 1, 1, distributions_scaled[i])

    # Identify next largest entry not yet ranked. While there are entries
    # smaller than 1, some bins don't have a rank yet.
    while numpy.amin(distributions_re_scaled) < 1:
        # For each mu, this is the largest rank attributed so far.
        largest_rank = np.amax(distributions_re_scaled, axis=1)
        # For each mu, this is the largest entry that is not yet a rank.
        largest_entry = np.where(distributions_re_scaled < 1, distributions_re_scaled, -1)
        # For each mu, this is the position of the largest entry that is not yet a rank.
        largest_entry_position = np.argmax(largest_entry, axis = 1)
        # Invalidate indices where there is no maximum (every entry is already a rank)
        largest_entry_position = [largest_entry_position[i] if largest_entry[i][largest_entry_position[i]] != -1 \
                                                            else -1 for i in range(len(largest_entry_position))]
        # Replace the largest entry with the highest rank so far plus one
        # Add the probability
        for i in range(number_mus):
            if largest_entry_position[i] == -1:
                continue
            distributions_re_scaled[i][largest_entry_position[i]] = largest_rank[i] + 1
            if summed_propability[i] < alpha:
                summed_propability[i] += distributions_scaled[i][largest_entry_position[i]]
                distributions_scaled[i][largest_entry_position[i]] = 1
            else:
                distributions_scaled[i][largest_entry_position[i]] = 0

    return distributions_scaled


def fc_get_upper_and_lower_limit(mu_bins, x_bins, confidence_interval, do_plot = False):
    """Find upper and lower limit from confidence interval.

    Parameters
    ----------
    mu_bins : array-like
        The bins used in mue direction.
    x_bins : array-like
        The bins of the x distribution
    confidence_interval : array-like
        The output of construct_confidence_belt_PDFs.
    do_plot : bool
        Draws the x values into the current canvas

    Returns
    -------
    upper_limit : array-like
        Feldman Cousins upper limit
    lower_limit : array-like
        Feldman Cousins lower limit
    """

    upper_limit = []
    lower_limit = []

    number_mu     = len(mu_bins)
    number_bins_x = len(x_bins)

    import matplotlib.pylab as plt

    for mu in range(number_mu):
        x_values = []
        upper_limit.append(-1)
        lower_limit.append(-1)
        for x in range(number_bins_x):
            #This point lies in the confidence interval
            if confidence_interval[mu][x] == 1:
                x_value = x_bins[x]
                x_values.append(x_value)
                # Upper limit is the first point where this condition is true
                if upper_limit[-1] == -1:
                    upper_limit[-1] = x_value
                # Lower limit is the first point after this condition is not true
                if x == number_bins_x - 1:
                    lower_limit[-1] = x_value
                else:
                    lower_limit[-1] = x_bins[x + 1]
        if do_plot:
            plt.plot(x_values, [mu_bins[mu] for i in range(len(x_values))], marker='.', ls='',color='black')

    return upper_limit, lower_limit


def fc_fix_upper_and_lower_limit(upper_limit, lower_limit):
    """Push limits outwards as described in the FC paper.

    Parameters
    ----------
    upper_limit : array-like
        Feldman Cousins upper limit
    lower_limit : array-like
        Feldman Cousins lower limit

    Returns
    -------
    upper_limit : array-like
        Feldman Cousins upper limit (fixed)
    lower_limit : array-like
        Feldman Cousins lower limit (fixed)
    """

    all_fixed = False

    while not all_fixed:
        all_fixed = True
        for j in range(1,len(upper_limit)):
            if upper_limit[j] < upper_limit[j-1]:
                upper_limit[j-1] = upper_limit[j]
                all_fixed = False
        for j in range(1,len(lower_limit)):
            if lower_limit[j] < lower_limit[j-1]:
                lower_limit[j] = lower_limit[j-1]
                all_fixed = False


def fc_find_limit(x_value, x_values_input, y_values_input, do_upper_edge = True):
    """Find the upper limit for a given x value.

    Parameters
    ----------
    x_value : double
        The measured x value for which the upper limit is wanted.
    x_values_input : array-like
        The x coordinates of the confidence belt
    y_values_input : array-like
        The y coordinates of the confidence belt
    do_upper_edge : bool
        If x_value lies on a bin border, use the upper edge of the belt.

    Returns
    -------
    limit : array-like
        The Feldman Cousins upper limit
    """

    limit = 0

    if do_upper_edge:
        previous_x = numpy.nan
        next_value = False
        identical = True
        x_values = x_values_input
        y_values = y_values_input
        for i in range(len(x_values)):
            current_x = x_values[i]
            # If the x_value did lie on the bin border, loop until the x value
            # is changing and take the last point (that is the highest point in
            # case points lie on top of each other.
            if next_value == True and current_x != previous_x:
                limit = y_values[i-1]
                break
            if x_value <= current_x:
                # If the x_value does not lie on the bin border, this should be
                # the upper limit
                if x_value != current_x:
                    limit = y_values[i]
                    break
                next_value = True
            previous_x = current_x
    else:
        x_values = numpy.flipud(x_values_input)
        y_values = numpy.flipud(y_values_input)
        for i in range(len(x_values)):
            current_x = x_values[i]
            if x_value >= current_x:
                limit = y_values[i]
                break

    return limit


def fc_find_average_upper_limit(x_bins, confidence_belt, upper_limit, mu_bins):
    """Function to calculate the average upper limit for a confidence belt.

    Parameters
    ----------
    x_bins : array-like
        Bins in x direction
    confidence_belt : array-like
        The output of construct_confidence_belt_PDFs.
    upper_limit : array-like
        Desired confidence level

    Returns
    -------
    average_limit : double
        Average upper limit
    """

    avergage_limit = 0
    number_points = len(distributions_scaled[0])

    for i in range(number_points):
        avergage_limit += confidence_belt[0][i]*find_limit(x_bins[i], upper_limit, mu_bins)

    return avergage_limit


def fc_construct_confidence_belt(distribution_dict, bins, alpha):
    """Convenience function that calculates the PDF for the user.

    Parameters
    ----------
    distribution_dict : `dict`
        Keys are mu values and value is an array-like list of x values
    bins : array-like
        The bins the x distribution will have
    alpha : float
        Desired confidence level

    Returns
    -------
    confidence_belt : ndarray
        Confidence interval (1 means inside, 0 means outside)
    """

    distributions_scaled = []

    # Histogram gets rid of the last bin, so add one extra
    bin_width = bins[1] - bins[0]
    new_bins = numpy.concatenate((bins, numpy.array([bins[-1]+bin_width])), axis=0)

    # Histogram and normalise each distribution so it is a real PDF
    for mu, distribution in iter(sorted(distribution_dict.iteritems())):
        entries = numpy.histogram(distribution, bins=new_bins)[0]
        integral = float(sum(entries))
        distributions_scaled.append(entries/integral)

    confidence_belt = construct_confidence_belt_pdfs(distributions_scaled, alpha)

    return confidence_belt
