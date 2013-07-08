# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Matched filter source detection methods"""
import numpy as np

__all__ = ['p_value']

def p_value(weights, counts, background):
    """Compute matched-filter p-value.

    Reference: Appendix B.1.2 from Stewart (2009)
        http://adsabs.harvard.edu/abs/2009A%26A...495..989S

    weights: array-like
        Matched filter weights image
    counts: array-like
        Counts image
    background : array-like
        Background image
    """
    from scipy.special import gammaincc as Q
    weights = np.asanyarray(weights)
    counts = np.asanyarray(counts)
    background = np.asanyarray(background)
    
    U = np.sum(weights * counts)
    B_prime = np.sum(weights * background)
    w_equiv = np.sum(weights * weights * background) / B_prime
    P = Q(B_prime / w_equiv, U / w_equiv)
    return P

def significance(weights, counts, background):
    from ..stats import p_to_s
    p = p_value(weights, counts, background)
    return p_to_s(p)