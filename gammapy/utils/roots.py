# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utils to find roots of a scalar function within a given range"""

import numpy as np
from scipy.optimize import RootResults, root_scalar
import astropy.units as u
from gammapy.utils.interpolation import interpolation_scale

BAD_RES = RootResults(root=np.nan, iterations=0, function_calls=0, flag=0)


def find_roots(
    f,
    lower_bound,
    upper_bound,
    nbin=100,
    points_scale="lin",
    args=(),
    method="brentq",
    fprime=None,
    fprime2=None,
    xtol=None,
    rtol=None,
    maxiter=None,
    options=None,
):
    """Find roots of a scalar function within a given range.

    Parameters
    ----------
    f : callable
        A function to find roots of. Its output should be unitless.
    lower_bound : `~astropy.units.Quantity`
        Lower bound of the search ranges to find roots.
        If an array is given search will be performed element-wise.
    upper_bound : `~astropy.units.Quantity`
        Uper bound of the search ranges to find roots.
        If an array is given search will be performed element-wise.
    nbin : int
        Number of bins to sample the search range, ignored if bounds are arrays
    points_scale : {"lin", "log", "sqrt"}
        Scale used to sample the search range. Default is linear ("lin")
    args : tuple, optional
        Extra arguments passed to the objective function and its derivative(s).
    method : str, optional
        Solver available in `~scipy.optimize.root_scalar`.  Should be one of :
            - 'brentq' (default),
            - 'brenth',
            - 'bisect',
            - 'ridder',
            - 'toms748',
            - 'newton',
            - 'secant',
            - 'halley',
    fprime : bool or callable, optional
        If `fprime` is a boolean and is True, `f` is assumed to return the
        value of the objective function and of the derivative.
        `fprime` can also be a callable returning the derivative of `f`. In
        this case, it must accept the same arguments as `f`.
    fprime2 : bool or callable, optional
        If `fprime2` is a boolean and is True, `f` is assumed to return the
        value of the objective function and of the
        first and second derivatives.
        `fprime2` can also be a callable returning the second derivative of `f`.
        In this case, it must accept the same arguments as `f`.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options : dict, optional
        A dictionary of solver options.
        See `~scipy.optimize.root_scalar` for details.

    Returns
    -------
    roots : `~astropy.units.Quantity`
        The function roots.

    results : `~numpy.array`
        An array of `~scipy.optimize.RootResults` which is an
        object containing information about the convergence.
        If the solver failed to converge in a bracketing range
        the corresponding `roots` array element is NaN.

    """

    kwargs = dict(
        args=args,
        method=method,
        fprime=fprime,
        fprime2=fprime2,
        xtol=xtol,
        rtol=rtol,
        maxiter=maxiter,
        options=options,
    )

    if isinstance(lower_bound, u.Quantity):
        unit = lower_bound.unit
        lower_bound = lower_bound.value
        upper_bound = u.Quantity(upper_bound).to_value(unit)
    else:
        unit = 1

    scale = interpolation_scale(points_scale)
    a = scale(lower_bound)
    b = scale(upper_bound)
    x = scale.inverse(np.linspace(a, b, nbin + 1))
    if len(x) > 2:
        signs = np.sign([f(xk, *args) for xk in x])
        ind = np.where(signs[:-1] != signs[1:])[0]
    else:
        ind = [0]
    nroots = max(1, len(ind))
    roots = np.ones(nroots) * np.nan
    results = np.array(nroots * [BAD_RES])

    for k, idx in enumerate(ind):
        bracket = [x[idx], x[idx + 1]]
        if method in ["bisection", "brentq", "brenth", "ridder", "toms748"]:
            kwargs["bracket"] = bracket
        elif method in ["secant", "newton", "halley"]:
            kwargs["x0"] = bracket[0]
            kwargs["x1"] = bracket[1]
        else:
            raise ValueError(f'Unknown solver "{method}"')
        try:
            res = root_scalar(f, **kwargs)
            results[k] = res
            if res.converged:
                roots[k] = res.root
        except (RuntimeError, ValueError):
            continue
    return roots * unit, results
