"""
On-off statistics
"""
import numpy as np
from numpy import log, sqrt

__all__ = ['excess', 'excess_error',
           'significance', 'sensitivity',
           'DEFAULT_SIGNIFICANCE']

DEFAULT_SIGNIFICANCE = 5

def excess(on, off, alpha):
    """Compute excess for an on-off observation

    Parameters
    ----------
    on : list or ndarray
        Observerd number of counts in the on region
    off : list or ndarray
        Observed number of counts in the off region
    alpha : list or ndarray
        On / off region exposure ratio for background events

    Returns
    -------
    excess : ndarray
        Excess in the on region

    Examples
    --------
    >>> excess(10, 20, 0.1)
    8.0
    """
    on = np.asarray(on, dtype='f')
    off = np.asarray(off, dtype='f')
    alpha = np.asarray(alpha, dtype='f')

    return on - alpha * off


def excess_error(on, off, alpha):
    """Compute excess error for an on-off observation

    Parameters
    ----------
    on : list or ndarray
        Observerd number of counts in the on region
    off : list or ndarray
        Observed number of counts in the off region
    alpha : list or ndarray
        On / off region exposure ratio for background events

    Returns
    -------
    excess_error : ndarray
        Excess error

    Examples
    --------
    >>> excess_error(10, 20, 0.1)
    
    """
    on = np.asarray(on, dtype='f')
    off = np.asarray(off, dtype='f')
    alpha = np.asarray(alpha, dtype='f')

    raise NotImplementedError


def significance_lima(on, off, alpha):
    """Compute significance with the Li & Ma formula (17).

    Parameters
    ----------
    on : number, list or ndarray
        Observerd number of counts in the on region
    off : number, list or ndarray
        Observed number of counts in the off region
    alpha : number, list or ndarray
        On / off region exposure ratio for background events

    Returns
    -------
    significance : ndarray
        Significance according to the Li & Ma formula (17)

    References
    ----------
    .. [1] Li and Ma, "Analysis methods for results in gamma-ray
    astronomy", 1983ApJ...272..317L

    See Also
    --------
    significance, significance_simple

    Examples
    --------
    >>> significance_lima(10, 20, 0.1)
    3.6850322025333071
    """
    on = np.asarray(on, dtype='f')
    off = np.asarray(off, dtype='f')
    alpha = np.asarray(alpha, dtype='f')

    temp = (alpha + 1) / (on + off)
    l = on * log(on * temp / alpha)
    m = off * log(off * temp)
    sign = np.where(on - alpha * off > 0, 1, -1)

    return sign * sqrt(abs(2 * (l + m)))


def significance_simple(on, off, alpha):
    """Compute significance with a simple, somewhat biased formula.

    Notes
    -----
    This function implements formula (5) of Li & Ma.
    Li & Ma show that it is somewhat biased,
    but it does have the advantage of being analytically invertible,
    i.e. there is an analytical formula for sensitivity,
    which is often used in practice.

    Parameters
    ----------
    on : number, list or ndarray
        Observerd number of counts in the on region
    off : number, list or ndarray
        Observed number of counts in the off region
    alpha : number, list or ndarray
        On / off region exposure ratio for background events

    Returns
    -------
    significance : ndarray
        Significance according to the Li & Ma formula (5)

    References
    ----------
    .. [1] Li and Ma, "Analysis methods for results in gamma-ray
    astronomy", 1983ApJ...272..317L

    See Also
    --------
    significance, significance_lima

    Examples
    --------
    >>> significance_simple(10, 20, 0.1)
    2.5048971
    """
    on = np.asarray(on, dtype='f')
    off = np.asarray(off, dtype='f')
    alpha = np.asarray(alpha, dtype='f')

    excess = on - alpha * off
    variance = on + alpha ** 2 * off

    return excess / sqrt(variance)


def significance(on, off, alpha, method='lima'):
    """Compute significance of an on-off observation

    Parameters
    ----------
    on : number, list or ndarray
        Observerd number of counts in the on region
    off : number, list or ndarray
        Observed number of counts in the off region
    alpha : number, list or ndarray
        On / off region exposure ratio for background events
    method : str
        Method to use: 'lima' or 'simple'

    Returns
    -------
    significance : ndarray
        Significance according to the method chosen.

    See Also
    --------
    significance_simple, significance_lima

    Examples
    --------
    >>> significance(10, 20, 0.1, 'lima')
    3.6850322025333071
    >>> significance(10, 20, 0.1, 'simple')
    2.5048971
    """
    if method == 'lima':
        return significance_lima(on, off, alpha)
    elif method == 'simple':
        return significance_simple(on, off, alpha)
    else:
        raise Exception('Invalid method: %s' % method)


def sensitivity_lima(off, alpha, significance=DEFAULT_SIGNIFICANCE, guess=1e-3):
    """Compute sensitivity using the Li & Ma formula for significance.
    
	@note: in weird cases (e.g. on=0.1, off=0.1, alpha=0.001)
	fsolve does not find a solution.
	@todo: Is it possible to find a better starting point or
	implementation to make this robust and fast?
	values < guess are often not found using this cost function."""
    from scipy.optimize import fsolve
    def f(on):
        if on >= 0:
            return significance_lima(on, off, alpha) - significance
        else:
            return 1e100
    on = fsolve(f, guess)
    return excess(on, off, alpha)


def sensitivity_simple(off, alpha, significance=DEFAULT_SIGNIFICANCE):
    """Compute sensitivity using the simple formula for significance.
    
    Notes
    -----
    
    Solve[S == (Non - \[Alpha] Noff)/Sqrt[Non + \[Alpha]^2 Noff], Non]
    Non -> 1/2 (S^2 + 2 Noff \[Alpha] +
    S Sqrt[S^2 + 4 Noff \[Alpha] + 4 Noff \[Alpha]^2]"""
    """Compute sensitivity using significance_simple method.

    Parameters
    ----------
    off : number, list or ndarray
        Observed number of counts in the off region
    alpha : number, list or ndarray
        On / off region exposure ratio for background events
    significance : number, list or ndarray
        Desired significance level

    Returns
    -------
    sensitivity : ndarray
        Sensitivity, i.e. excess required to reach the desired
        significance level.

    See Also
    --------
    significance_simple, sensitivity

    Examples
    --------
    >>> sensitivity_simple(20, 0.1, 5)
    2.5048971
    """
    off = np.asarray(off, dtype='f')
    alpha = np.asarray(alpha, dtype='f')
    significance = np.asarray(significance, dtype='f')

    det = significance ** 2 + 4 * off * alpha * (1 + alpha)
    temp = significance ** 2 + 2 * off * alpha
    on = 0.5 * (temp + significance * sqrt(abs(det)))
    return excess(on, off, alpha)


def sensitivity(off, alpha, significance, method='lima'):
    """Compute sensitivity of an on-off observation

    Parameters
    ----------
    off : number, list or ndarray
        Observed number of counts in the off region
    alpha : number, list or ndarray
        On / off region exposure ratio for background events
    significance : number, list or ndarray
        Desired significance level
    method : str
        Significance method to use: 'lima' or 'simple'

    Returns
    -------
    sensitivity : ndarray
        Sensitivity according to the method chosen.

    See Also
    --------
    sensitivity_simple, sensitivity_lima

    Examples
    --------
    >>> sensitivity(20, 0.1, 5, 'lima')
    
    >>> sensitivity(20, 0.1, 5, 'simple')
    
    """
    if method == 'lima':
        return sensitivity_lima(off, alpha, significance)
    elif method == 'simple':
        return sensitivity_simple(off, alpha, significance)
    else:
        raise Exception('Invalid method: %s' % method)
