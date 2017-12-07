"""XSPEC statistics tests.

Here you find a direct implementation of the formulae at
https://heasarc.nasa.gov/xanadu/xspec/manual/XSappendixStatistics.html

Then I'll try to translate to the notation I use in gammapy.stats
and get some test cases.
"""
from __future__ import print_function, division
import numpy as np
from numpy import log, sqrt


def xspec_cstat(t, m, S):
    return 2 * (t * m - S + S * (log(S) - log(t * m)))

def xspec_wstat_f(S, B, t_s, t_b, m, d):
    t = (t_s + t_b)
    temp = S + B - t * m
    temp_a = temp + d
    temp_b = temp - d
    term1 = np.where(temp_a > 0, temp_a, temp_b)
    term2 = 2 * t
    f = term1 / term2
    return f

def xspec_wstat_d(t_s, t_b, m, S, B):
    t = t_s + t_b
    term1 = t * m - S - B
    term2 = 4 * t * B * m
    d = sqrt(term1 ** 2 + term2)
    return d

def xspec_wstat(t_s, t_b, m, S, B):
    t = t_s + t_b
    if S == 0:
        W = t_s * m - B * log(t_b / t)
    elif B == 0:
        if m < S / t:
            W = -t_b * m - S * log(t_s / t)
        else:
            temp = log(S) - log(t_s * m) - 1
            W = t_s * m + S * temp
    else:
        d = xspec_wstat_d(t_s, t_b, m, S, B)
        f = xspec_wstat_f(S, B, t_s, t_b, m, d)
        term1 = t_s * m + t * f - S * log(t_s * m + t_s * f)
        term2 = -B * log(t_b * f) - S * (1 - log(S)) - B * (1 - log(B))
        W = term1 + term2
    return 2 * W

def xspec_wstat_limit(t_s, t_b, m, f, S, B):
    term1 = S - t_s * m - t_s * f
    term2 = t_s * (m + f)
    term3 = B - t_b * f
    term4 = t_b * f
    W = term1 ** 2 / term2 + term3 ** 2 / term4
    return W

def test_wstat():
    """Test case for wstat."""
    t_s = 4.2
    t_b = 2.4
    m = 5
    B = 4
    #f = 7
    S = 9

    W = xspec_wstat(t_s, t_b, m, S, B)
    print(W)


if __name__ == '__main__':
    test_wstat()
