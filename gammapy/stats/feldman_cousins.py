# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Feldman Cousins algorithm to compute parameter confidence limits."""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np

__all__ = [
    "fc_find_acceptance_interval_gauss",
    "fc_find_acceptance_interval_poisson",
    "fc_construct_acceptance_intervals_pdfs",
    "fc_get_limits",
    "fc_fix_limits",
    "fc_find_limit",
    "fc_find_average_upper_limit",
    "fc_construct_acceptance_intervals",
]

log = logging.getLogger(__name__)


def fc_find_acceptance_interval_gauss(mu, sigma, x_bins, alpha):
    r"""
    Analytical acceptance interval for Gaussian with boundary at the origin.

    .. math :: \int_{x_{min}}^{x_{max}} P(x|mu)\mathrm{d}x = alpha

    For more information see :ref:`documentation <feldman_cousins>`.

    Parameters
    ----------
    mu : float
        Mean of the Gaussian
    sigma : float
        Width of the Gaussian
    x_bins : array-like
        Bins in x
    alpha : float
        Desired confidence level

    Returns
    -------
    (x_min, x_max) : tuple of floats
        Acceptance interval
    """
    from scipy import stats

    dist = stats.norm(loc=mu, scale=sigma)

    x_bin_width = x_bins[1] - x_bins[0]

    p = []
    r = []

    for x in x_bins:
        p.append(dist.pdf(x) * x_bin_width)
        # This is the formula from the FC paper
        if mu == 0 and sigma == 1:
            if x < 0:
                r.append(np.exp(mu * (x - mu * 0.5)))
            else:
                r.append(np.exp(-0.5 * np.power((x - mu), 2)))
        # This is the more general formula
        else:
            # Implementing the boundary condition at zero
            mu_best = max(0, x)
            prob_mu_best = stats.norm.pdf(x, loc=mu_best, scale=sigma)
            # probMuBest should never be zero. Check it just in case.
            if prob_mu_best == 0.0:
                r.append(0.0)
            else:
                r.append(p[-1] / prob_mu_best)

    p = np.asarray(p)
    r = np.asarray(r)

    if sum(p) < alpha:
        raise ValueError(
            "X bins don't contain enough probability to reach "
            "desired confidence level for this mu!"
        )

    rank = stats.rankdata(-r, method="dense")

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
        if p_sum >= alpha:
            break

    return x_bins[index_min], x_bins[index_max] + x_bin_width


def fc_find_acceptance_interval_poisson(mu, background, x_bins, alpha):
    r"""Analytical acceptance interval for Poisson process with background.

    .. math :: \int_{x_{min}}^{x_{max}} P(x|mu)\mathrm{d}x = alpha

    For more information see :ref:`documentation <feldman_cousins>`.

    Parameters
    ----------
    mu : float
        Mean of the signal
    background : float
        Mean of the background
    x_bins : array-like
        Bins in x
    alpha : float
        Desired confidence level

    Returns
    -------
    (x_min, x_max) : tuple of floats
        Acceptance interval
    """
    from scipy import stats

    dist = stats.poisson(mu=mu + background)

    x_bin_width = x_bins[1] - x_bins[0]

    p = []
    r = []

    for x in x_bins:
        p.append(dist.pmf(x))
        # Implementing the boundary condition at zero
        muBest = max(0, x - background)
        probMuBest = stats.poisson.pmf(x, mu=muBest + background)
        # probMuBest should never be zero. Check it just in case.
        if probMuBest == 0.0:
            r.append(0.0)
        else:
            r.append(p[-1] / probMuBest)

    p = np.asarray(p)
    r = np.asarray(r)

    if sum(p) < alpha:
        raise ValueError(
            "X bins don't contain enough probability to reach "
            "desired confidence level for this mu!"
        )

    rank = stats.rankdata(-r, method="dense")

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
        if p_sum >= alpha:
            break

    return x_bins[index_min], x_bins[index_max] + x_bin_width


def fc_construct_acceptance_intervals_pdfs(matrix, alpha):
    r"""Numerically choose bins a la Feldman Cousins ordering principle.

    For more information see :ref:`documentation <feldman_cousins>`.

    Parameters
    ----------
    matrix : array-like
        A list of x PDFs for increasing values of mue.
    alpha : float
        Desired confidence level

    Returns
    -------
    distributions_scaled : ndarray
        Acceptance intervals (1 means inside, 0 means outside)
    """
    number_mus = len(matrix)

    distributions_scaled = np.asarray(matrix)
    distributions_re_scaled = np.asarray(matrix)
    summed_propability = np.zeros(number_mus)

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

    # Step 3 (Feldman Cousins Ordering principle):
    # For each mu, get the largest entry
    largest_entry = np.argmax(distributions_re_scaled, axis=1)
    # Set the rank to 1 and add probability
    for i in range(number_mus):
        distributions_re_scaled[i][largest_entry[i]] = 1
        summed_propability[i] += np.sum(
            np.where(distributions_re_scaled[i] == 1, distributions_scaled[i], 0)
        )
        distributions_scaled[i] = np.where(
            distributions_re_scaled[i] == 1, 1, distributions_scaled[i]
        )

    # Identify next largest entry not yet ranked. While there are entries
    # smaller than 1, some bins don't have a rank yet.
    while np.amin(distributions_re_scaled) < 1:
        # For each mu, this is the largest rank attributed so far.
        largest_rank = np.amax(distributions_re_scaled, axis=1)
        # For each mu, this is the largest entry that is not yet a rank.
        largest_entry = np.where(
            distributions_re_scaled < 1, distributions_re_scaled, -1
        )
        # For each mu, this is the position of the largest entry that is not yet a rank.
        largest_entry_position = np.argmax(largest_entry, axis=1)
        # Invalidate indices where there is no maximum (every entry is already a rank)
        largest_entry_position = [
            largest_entry_position[i]
            if largest_entry[i][largest_entry_position[i]] != -1
            else -1
            for i in range(len(largest_entry_position))
        ]
        # Replace the largest entry with the highest rank so far plus one
        # Add the probability
        for i in range(number_mus):
            if largest_entry_position[i] == -1:
                continue
            distributions_re_scaled[i][largest_entry_position[i]] = largest_rank[i] + 1
            if summed_propability[i] < alpha:
                summed_propability[i] += distributions_scaled[i][
                    largest_entry_position[i]
                ]
                distributions_scaled[i][largest_entry_position[i]] = 1
            else:
                distributions_scaled[i][largest_entry_position[i]] = 0

    return distributions_scaled


def fc_get_limits(mu_bins, x_bins, acceptance_intervals):
    r"""Find lower and upper limit from acceptance intervals.

    For more information see :ref:`documentation <feldman_cousins>`.

    Parameters
    ----------
    mu_bins : array-like
        The bins used in mue direction.
    x_bins : array-like
        The bins of the x distribution
    acceptance_intervals : array-like
        The output of fc_construct_acceptance_intervals_pdfs.

    Returns
    -------
    lower_limit : array-like
        Feldman Cousins lower limit x-coordinates
    upper_limit : array-like
        Feldman Cousins upper limit x-coordinates
    x_values : array-like
        All the points that are inside the acceptance intervals
    """
    upper_limit = []
    lower_limit = []
    x_values = []

    number_mu = len(mu_bins)
    number_bins_x = len(x_bins)

    for mu in range(number_mu):
        upper_limit.append(-1)
        lower_limit.append(-1)
        x_values.append([])
        acceptance_interval = acceptance_intervals[mu]
        for x in range(number_bins_x):
            # This point lies in the acceptance interval
            if acceptance_interval[x] == 1:
                x_value = x_bins[x]
                x_values[-1].append(x_value)
                # Upper limit is first point where this condition is true
                if upper_limit[-1] == -1:
                    upper_limit[-1] = x_value
                # Lower limit is first point after this condition is not true
                if x == number_bins_x - 1:
                    lower_limit[-1] = x_value
                else:
                    lower_limit[-1] = x_bins[x + 1]

    return lower_limit, upper_limit, x_values


def fc_fix_limits(lower_limit, upper_limit):
    r"""Push limits outwards as described in the FC paper.

    For more information see :ref:`documentation <feldman_cousins>`.

    Parameters
    ----------
    lower_limit : array-like
        Feldman Cousins lower limit x-coordinates
    upper_limit : array-like
        Feldman Cousins upper limit x-coordinates
    """
    all_fixed = False

    while not all_fixed:
        all_fixed = True
        for j in range(1, len(upper_limit)):
            if upper_limit[j] < upper_limit[j - 1]:
                upper_limit[j - 1] = upper_limit[j]
                all_fixed = False
        for j in range(1, len(lower_limit)):
            if lower_limit[j] < lower_limit[j - 1]:
                lower_limit[j] = lower_limit[j - 1]
                all_fixed = False


def fc_find_limit(x_value, x_values, y_values):
    r"""
    Find the limit for a given x measurement

    For more information see :ref:`documentation <feldman_cousins>`

    Parameters
    ----------
    x_value : float
        The measured x value for which the upper limit is wanted.
    x_values : array-like
        The x coordinates of the confidence belt.
    y_values : array-like
        The y coordinates of the confidence belt.

    Returns
    -------
    limit : float
        The Feldman Cousins limit
    """
    if x_value > max(x_values):
        raise ValueError("Measured x outside of confidence belt!")

    # Loop through the x-values in reverse order
    for i in reversed(range(len(x_values))):
        current_x = x_values[i]
        # The measured value sits on a bin edge. In this case we want the upper
        # most point to be conservative, so it's the first point where this
        # condition is true.
        if x_value == current_x:
            return y_values[i]
        # If the current value lies between two bins, take the higher y-value
        # in order to be conservative.
        if x_value > current_x:
            return y_values[i + 1]


def fc_find_average_upper_limit(x_bins, matrix, upper_limit, mu_bins, prob_limit=1e-5):
    r"""
    Function to calculate the average upper limit for a confidence belt

    For more information see :ref:`documentation <feldman_cousins>`

    Parameters
    ----------
    x_bins : array-like
        Bins in x direction
    matrix : array-like
        A list of x PDFs for increasing values of mue
        (same as for fc_construct_acceptance_intervals_pdfs).
    upper_limit : array-like
        Feldman Cousins upper limit x-coordinates
    mu_bins : array-like
        The bins used in mue direction.
    prob_limit : float
        Probability value at which x values are no longer considered for the
        average limit.

    Returns
    -------
    average_limit : float
        Average upper limit
    """
    average_limit = 0
    number_points = len(x_bins)

    for i in range(number_points):
        # Bins with very low probability will not contribute to average limit
        if matrix[0][i] < prob_limit:
            continue
        try:
            limit = fc_find_limit(x_bins[i], upper_limit, mu_bins)
        except:
            log.warning("Warning: Calculation of average limit incomplete!")
            log.warning("Add more bins in mu direction or decrease prob_limit.")
            return average_limit
        average_limit += matrix[0][i] * limit

    return average_limit


def fc_construct_acceptance_intervals(distribution_dict, bins, alpha):
    r"""Convenience function that calculates the PDF for the user.

    For more information see :ref:`documentation <feldman_cousins>`.

    Parameters
    ----------
    distribution_dict : dict
        Keys are mu values and value is an array-like list of x values
    bins : array-like
        The bins the x distribution will have
    alpha : float
        Desired confidence level

    Returns
    -------
    acceptance_intervals : ndarray
        Acceptance intervals (1 means inside, 0 means outside)
    """
    distributions_scaled = []

    # Histogram gets rid of the last bin, so add one extra
    bin_width = bins[1] - bins[0]
    new_bins = np.concatenate((bins, np.array([bins[-1] + bin_width])), axis=0)

    # Histogram and normalise each distribution so it is a real PDF
    for _, distribution in sorted(distribution_dict.items()):
        entries = np.histogram(distribution, bins=new_bins)[0]
        integral = float(sum(entries))
        distributions_scaled.append(entries / integral)

    acceptance_intervals = fc_construct_acceptance_intervals_pdfs(
        distributions_scaled, alpha
    )

    return acceptance_intervals
