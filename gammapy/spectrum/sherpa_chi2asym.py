# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Implement asymmetric chi-square fit statistic in Sherpa.

To load the ``chi2asym`` fit statistic in your sherpa session::

    import sherpa_chi2asym
    sherpa_chi2asym.load_chi2asym_stat()
"""
from __future__ import print_function, division
import numpy as np

__all__ = ['check_chi2',
           'chi2asym_err_func',
           'chi2asym_stat_func',
           'load_chi2asym_stat',
           ]


def chi2asym_stat_func(data, model, staterror=None,
                       syserror=None, weight=None):
    """Define asymmetric chi-square errors.

    TODO: reference ROOT TGraphAsymErrors and add test against ROOT result.

    To make it fit into the Sherpa scheme we do this hack:
    * staterror = statistical down error
    * syserror = statistical up error
    """
    # The error is attached to the data point, so if model > data,
    # we have to use the up error, represented by syserror
    error = np.where(model > data, syserror, staterror)

    chi = ((data - model) / error)  # Chi per bin
    chi2 = chi ** 2  # Chi^2 per bin
    return chi2.sum(), chi2


def chi2asym_err_func(data):
    """Compute statistical error per bin from the data."""
    error = np.ones_like(data)
    return error


def check_chi2():
    """Execute this function after fitting to see if the
    best-fit chi2 reported matches the formula coded here"""
    import sherpa.astro.ui as sau
    chi2 = sau.get_fit_results().statval
    print('chi2 from fit: {0}'.format(chi2))
    data = sau.get_dep()
    model = sau.get_model_plot().y
    error = np.where(model > data, sau.get_syserror(), sau.get_staterror())

    chi = ((data - model) / error)  # Chi per bin
    chi2 = chi ** 2  # Chi^2 per bin
    print('chi2 re-computed: {0}'.format(chi2.sum()))


def load_chi2asym_stat():
    """"Load and set the chi2asym statistic"""
    import sherpa.astro.ui as sau
    sau.load_user_stat("chi2asym", chi2asym_stat_func, chi2asym_err_func)
    sau.set_stat(chi2asym)
