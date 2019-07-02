# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""On-off bin stats computations."""

__all__ = ["Stats"]


class Stats:
    """Container for an on-off observation.

    Parameters
    ----------
    n_on : array_like
        Observed number of counts in the on region
    n_off : array_like
        Observed number of counts in the off region
    a_on : array_like
        Relative background exposure of the on region
    a_off : array_like
        Relative background exposure in the off region
    """

    # TODO: use numpy arrays and properties
    # TODO: add gamma exposure

    def __init__(self, n_on, n_off, a_on, a_off):
        self.n_on = n_on
        self.n_off = n_off
        self.a_on = a_on
        self.a_off = a_off

    @property
    def alpha(self):
        r"""Background exposure ratio (float).

        .. math:: \alpha = a_\mathrm{on} / a_\mathrm{off}
        """
        return self.a_on / self.a_off

    @property
    def background(self):
        r"""Background estimate (float).

        .. math:: \mu_\mathrm{bg} = \alpha\ n_\mathrm{off}
        """
        return self.alpha * self.n_off

    @property
    def excess(self):
        r"""Excess (float).

        .. math:: n_\mathrm{ex} = n_\mathrm{on} - \mu_\mathrm{bg}
        """
        return self.n_on - self.background

    def __str__(self):
        keys = ["n_on", "n_off", "a_on", "a_off", "alpha", "background", "excess"]
        values = [
            self.n_on,
            self.n_off,
            self.a_on,
            self.a_off,
            self.alpha,
            self.background,
            self.excess,
        ]
        return "\n".join(["{} = {}".format(k, v) for (k, v) in zip(keys, values)])
