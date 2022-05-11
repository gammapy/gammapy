# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for testing"""
import os
import sys
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import matplotlib.pyplot as plt

__all__ = [
    "assert_quantity_allclose",
    "assert_skycoord_allclose",
    "assert_time_allclose",
    "Checker",
    "mpl_plot_check",
    "requires_data",
    "requires_dependency",
]

# Cache for `requires_dependency`
_requires_dependency_cache = {}


def requires_dependency(name):
    """Decorator to declare required dependencies for tests.

    Examples
    --------
    ::

        from gammapy.utils.testing import requires_dependency

        @requires_dependency('scipy')
        def test_using_scipy():
            import scipy
            ...
    """
    import pytest

    if name in _requires_dependency_cache:
        skip_it = _requires_dependency_cache[name]
    else:
        try:
            __import__(name)
            skip_it = False
        except ImportError:
            skip_it = True

        _requires_dependency_cache[name] = skip_it

    reason = f"Missing dependency: {name}"
    return pytest.mark.skipif(skip_it, reason=reason)


def has_data(name):
    """Is a certain set of data available?"""
    if name == "gammapy-extra":
        return "GAMMAPY_EXTRA" in os.environ
    elif name == "gammapy-data":
        return "GAMMAPY_DATA" in os.environ
    elif name == "gamma-cat":
        return "GAMMA_CAT" in os.environ
    elif name == "fermi-lat":
        return "GAMMAPY_FERMI_LAT_DATA" in os.environ
    else:
        raise ValueError(f"Invalid name: {name}")


def requires_data(name="gammapy-data"):
    """Decorator to declare required data for tests.

    Examples
    --------
    ::

        from gammapy.utils.testing import requires_data

        @requires_data()
        def test_using_data_files():
            filename = "$GAMMAPY_DATA/..."
            ...
    """
    import pytest

    if not isinstance(name, str):
        raise TypeError(
            "You must call @requires_data with a name (str). "
            "Usually this:  @requires_data()"
        )

    skip_it = not has_data(name)

    reason = f"Missing data: {name}"
    return pytest.mark.skipif(skip_it, reason=reason)


def run_cli(cli, args, exit_code=0):
    """Run Click command line tool.

    Thin wrapper around `click.testing.CliRunner`
    that prints info to stderr if the command fails.

    Parameters
    ----------
    cli : click.Command
        Click command
    args : list of str
        Argument list
    exit_code : int
        Expected exit code of the command

    Returns
    -------
    result : `click.testing.Result`
        Result
    """
    from click.testing import CliRunner

    result = CliRunner().invoke(cli, args, catch_exceptions=False)

    if result.exit_code != exit_code:
        sys.stderr.write("Exit code mismatch!\n")
        sys.stderr.write("Output:\n")
        sys.stderr.write(result.output)

    return result


def assert_skycoord_allclose(actual, desired):
    """Assert all-close for `astropy.coordinates.SkyCoord` objects.

    - Frames can be different, aren't checked at the moment.
    """
    assert isinstance(actual, SkyCoord)
    assert isinstance(desired, SkyCoord)
    assert_allclose(actual.data.lon.deg, desired.data.lon.deg)
    assert_allclose(actual.data.lat.deg, desired.data.lat.deg)


def assert_time_allclose(actual, desired, atol=1e-3):
    """Assert all-close for `astropy.time.Time` objects.

    atol is absolute tolerance in seconds.
    """
    assert isinstance(actual, Time)
    assert isinstance(desired, Time)
    assert actual.scale == desired.scale
    assert actual.format == desired.format
    dt = actual - desired
    assert_allclose(dt.sec, 0, rtol=0, atol=atol)


def assert_quantity_allclose(actual, desired, rtol=1.0e-7, atol=None, **kwargs):
    """Assert all-close for `astropy.units.Quantity` objects.

    Requires that ``unit`` is identical, not just that quantities
    are allclose taking different units into account.

    We prefer this kind of assert for testing, since units
    should only change on purpose, so this tests more behaviour.
    """
    # TODO: change this later to explicitly check units are the same!
    # assert actual.unit == desired.unit
    args = _unquantify_allclose_arguments(actual, desired, rtol, atol)
    assert_allclose(*args, **kwargs)


def _unquantify_allclose_arguments(actual, desired, rtol, atol):
    actual = u.Quantity(actual, subok=True, copy=False)

    desired = u.Quantity(desired, subok=True, copy=False)
    try:
        desired = desired.to(actual.unit)
    except u.UnitsError:
        raise u.UnitsError(
            "Units for 'desired' ({}) and 'actual' ({}) "
            "are not convertible".format(desired.unit, actual.unit)
        )

    if atol is None:
        # by default, we assume an absolute tolerance of 0
        atol = u.Quantity(0)
    else:
        atol = u.Quantity(atol, subok=True, copy=False)
        try:
            atol = atol.to(actual.unit)
        except u.UnitsError:
            raise u.UnitsError(
                "Units for 'atol' ({}) and 'actual' ({}) "
                "are not convertible".format(atol.unit, actual.unit)
            )

    rtol = u.Quantity(rtol, subok=True, copy=False)
    try:
        rtol = rtol.to(u.dimensionless_unscaled)
    except Exception:
        raise u.UnitsError("`rtol` should be dimensionless")

    return actual.value, desired.value, rtol.value, atol.value


def mpl_plot_check():
    """Matplotlib plotting test context manager.

    It create a new figure on __enter__ and calls savefig for the
    current figure in __exit__. This will trigger a render of the
    Figure, which can sometimes raise errors if there is a problem.

    This is writing to an in-memory byte buffer, i.e. is faster
    than writing to disk.
    """
    from io import BytesIO

    class MPLPlotCheck:
        def __enter__(self):
            plt.figure()

        def __exit__(self, type, value, traceback):
            plt.savefig(BytesIO(), format="png")
            plt.close()

    return MPLPlotCheck()


class Checker:
    """Base class for checker classes in Gammapy."""

    def run(self, checks="all"):
        if checks == "all":
            checks = self.CHECKS.keys()

        unknown_checks = sorted(set(checks).difference(self.CHECKS.keys()))
        if unknown_checks:
            raise ValueError(f"Unknown checks: {unknown_checks!r}")

        for check in checks:
            method = getattr(self, self.CHECKS[check])
            yield from method()
