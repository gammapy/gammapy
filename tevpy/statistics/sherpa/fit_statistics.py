"""
Implementation of the Sherpa statistics with numpy

All functions return arrays:
>>> stat_per_bin = stat(...)
If you want the sum say so:
>>> total_stat = stat(...).sum()

WARNING: I just typed this, no documentation or testing yet ... 

Todo
----

- check corner cases (e.g. counts or exposure zero)
- exception handling for bad input?

References
----------

http://cxc.cfa.harvard.edu/sherpa/statistics
https://github.com/taldcroft/sherpa/blob/master/stats/__init__.py
sherpa/include/sherpa/stats.hh contains the C++ implementations of the Sherpa stats

Notes
-----

"""

import numpy as np
from numpy import log, sqrt

__all__ = ('cash', 'cstat', 'chi2constvar', 'chi2datavar', 
           'chi2gehrels', 'chi2modvar', 'chi2xspecvar')

def cash(D, M):
    D = np.asarray(D)
    M = np.asarray(M)
    return 2 * (M - D * log(M)) 

def cstat(D, M):
    D = np.asarray(D)
    M = np.asarray(M)
    return 2 * (M - D  + D * (log(D) - log(M)))

def chi2(N_S, B, S, sigma2):
    N_S = np.asarray(N_S)
    B = np.asarray(B)
    S = np.asarray(S)
    sigma2 = np.asarray(sigma2)
    return (N_S - B - S) ** 2 / sigma2

def chi2constvar(N_S, N_B, A_S, A_B):
    N_S = np.array(N_S)
    N_B = np.array(N_B)
    A_S = np.array(A_S)
    A_B = np.array(A_B)
    alpha2 = (A_S / A_B) ** 2
    # Need to mulitply with np.ones_like(N_S) here?
    sigma2 = (N_S + alpha2 * N_B).mean()
    return chi2(N_S, A_B, A_S, sigma2)

def chi2datavar(N_S, N_B, A_S, A_B):
    N_S = np.array(N_S)
    N_B = np.array(N_B)
    A_S = np.array(A_S)
    A_B = np.array(A_B)
    alpha2 = (A_S / A_B) ** 2
    sigma2 = N_S + alpha2 * N_B
    return chi2(N_S, A_B, A_S, sigma2)

def chi2gehrels(N_S, N_B, A_S, A_B):
    N_S = np.array(N_S)
    N_B = np.array(N_B)
    A_S = np.array(A_S)
    A_B = np.array(A_B)
    alpha2 = (A_S / A_B) ** 2
    sigma_S = 1 + sqrt(N_S + 0.75)
    sigma_B = 1 + sqrt(N_B + 0.75)
    sigma2 = sigma_S ** 2 + alpha2 * sigma_B ** 2
    return chi2(N_S, A_B, A_S, sigma2)

def chi2modvar(S, B, A_S, A_B):
    return chi2datavar(S, B, A_S, A_B)

def chi2xspecvar(N_S, N_B, A_S, A_B):
    # TODO: is this correct?
    mask = (N_S < 1) | (N_B < 1)
    #_stat = np.empty_like(mask, dtype='float')
    #_stat[mask] = 1
    return np.where(mask, 1, chi2datavar(N_S, N_B, A_S, A_B))

def print_info(label, data):
    print('*** %s ***' % label)
    print('Sum: %g' % data.sum())
    print(data)

def test_likelihood_stats():
    # Create some example input
    M = np.array([[0, 0.5, 1, 2],[10, 10.5, 100.5, 1000]])
    print_info('M', M)
    np.random.seed(0)
    D = np.random.poisson(M)
    print_info('D', D)
    
    print_info('cash', cash(D, M))
    print_info('cstat', cstat(D, M))

def test_chi2_stats():
    A_S = np.array([[0, 0.5, 1, 2],[10, 10.5, 100.5, 1000]])
    A_B = A_S
    N_S = A_S
    N_B = A_S
    print_info('chi2datavar', chi2datavar(N_S, N_B, A_S, A_B)) 

if __name__ == '__main__':
    test_likelihood_stats()
    test_chi2_stats()