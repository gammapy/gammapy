# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.table import Table
from ..utils.scripts import make_path
from .models import TableModel

__all__ = ["FluxPointProfiles"]


class FluxPointProfiles(object):
    """Flux point likelihood profiles.

    See :ref:`gadf:likelihood_sed`.

    TODO: bring the code from Fermipy into Gammapy.

    - https://github.com/fermiPy/fermipy/blob/master/fermipy/castro.py
    - https://github.com/fermiPy/fermipy/blob/master/fermipy/sed_plotting.py

    This is just a simple from-scratch implementation, because
    migrating the Fermipy code is a large project.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table holding the data
    """

    def __init__(self, table):
        self.table = table

    def __repr__(self):
        return "{}(n={!r})".format(self.__class__.__name__, self._n_bins)

    @property
    def _n_bins(self):
        return len(self.table)

    @property
    def _energy_edges(self):
        ebounds = list(self.table["e_min"])
        ebounds += [self.table["e_max"][-1]]
        return ebounds * self.table["e_min"].unit

    @property
    def _energy_ref(self):
        return self.table["e_ref"].quantity

    @property
    def _default_e2dnde_edges(self):
        e = self.table["ref_e2dnde"].quantity
        return (
            np.logspace(np.log10(0.1 * e.value.min()), np.log10(e.value.max()), 500)
            * e.unit
        )

    @classmethod
    def read(cls, filename, **kwargs):
        """Read from file."""
        filename = make_path(filename)
        table = Table.read(str(filename), **kwargs)
        return cls(table=table)

    def get_profile(self, idx):
        """Get 1D likelihood profile.

        Parameters
        ----------
        idx : int
            Profile energy bin index.

        Returns
        -------
        table : `~astropy.table.Table`
            Table with columns "norm" and "dloglike".
        """
        t = Table()
        t["norm"] = self.table["norm_scan"][idx]
        t["dloglike"] = self.table["dloglike_scan"][idx]
        return t

    def interp_profile(self, idx, norm):
        """Interpolate likelihood profile.

        Parameters
        ----------
        idx : int
            Profile energy bin index.
        norm : `~numpy.ndarray`
            Norm values at which to interpolate

        Returns
        -------
        table : `~astropy.table.Table`
            Table with columns "norm" (the input)
            and "dloglike" (the interpolated values).
        """
        from ..utils.interpolation import ScaledRegularGridInterpolator

        t = self.get_profile(idx)

        interp = ScaledRegularGridInterpolator(
            points=(t["norm"],), values=t["dloglike"], values_scale="sqrt"
        )

        dloglike = interp((norm,))

        t2 = Table()
        t2["norm"] = norm
        t2["dloglike"] = dloglike
        return t2

    def get_reference_spectrum(self, which="dnde"):
        """Get spectrum as a `~gammapy.spectrum.models.TableModel`.

        Parameters
        ----------
        which : {'dnde', 'flux', 'eflux'}
            Which reference spectrum representation?
            TODO: for now only dnde tested, not sure the others make sense
        """
        energy = self._energy_ref
        values = self.table["ref_" + which].quantity
        return TableModel(energy, values)

    def plot_sed(self, ax=None, e2dnde_edges=None, add_cbar=True, **kwargs):
        """Plot likelihood SED profiles.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axis object to plot on.
        e2dnde_edges : `~astropy.units.Quantity`
            Bin edges for the ``e2dnde`` y axis to use
            for the likelihood profile plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        if e2dnde_edges is None:
            e2dnde_edges = self._default_e2dnde_edges

        x = self._energy_edges
        y = e2dnde_edges

        y_unit = e2dnde_edges.unit

        # Compute likelihood "image" one energy bin at a time
        # by interpolating e2dnde at the log bin centers
        z = np.empty((self._n_bins, len(y) - 1))
        for idx in range(self._n_bins):
            e2dnde = np.sqrt(e2dnde_edges[:-1] * e2dnde_edges[1:])
            e2dnde_ref = self.table["ref_e2dnde"].quantity[idx]
            norm = (e2dnde / e2dnde_ref).to_value("")
            z[idx] = self.interp_profile(idx, norm)["dloglike"]

        kwargs.setdefault("vmax", 0)
        kwargs.setdefault("vmin", -4)
        kwargs.setdefault("zorder", 0)
        kwargs.setdefault("cmap", "Blues")
        kwargs.setdefault("linewidths", 0)

        # clipped values are set to NaN so that they appear white on the plot
        z[-z < kwargs["vmin"]] = np.nan
        caxes = ax.pcolormesh(x, y, -z.T, **kwargs)
        ax.set_xscale("log", nonposx="clip")
        ax.set_yscale("log", nonposy="clip")
        ax.set_xlabel("Energy ({})".format(self._energy_ref.unit))
        ax.set_ylabel("E^2 dN/dE ({})".format(y_unit))

        if add_cbar:
            label = "delta log-likelihood"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        return ax
