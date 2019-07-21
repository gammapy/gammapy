# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import copy
import numpy as np
from astropy.table import Table
from astropy.io import fits
import astropy.units as u
from ..maps import MapAxis
from ..maps.utils import edges_from_lo_hi
from ..utils.scripts import make_path
from ..utils.fits import energy_axis_to_ebounds, ebounds_to_energy_axis
from ..data import EventList

__all__ = ["CountsSpectrum"]

log = logging.getLogger("__name__")


class CountsSpectrum:
    """Generic counts spectrum.

    Parameters
    ----------
    energy_lo : `~astropy.units.Quantity`
        Lower bin edges of energy axis
    energy_hi : `~astropy.units.Quantity`
        Upper bin edges of energy axis
    data : `~numpy.ndarray`
        Spectrum data.
    unit : str or `~astropy.units.Unit`
        Data unit

    Examples
    --------
    .. plot::
        :include-source:

        from gammapy.spectrum import CountsSpectrum
        import numpy as np
        import astropy.units as u

        ebounds = np.logspace(0,1,11) * u.TeV
        data = np.arange(10)
        spec = CountsSpectrum(
            energy_lo=ebounds[:-1],
            energy_hi=ebounds[1:],
            data=data,
        )
        spec.plot(show_poisson_errors=True)
    """

    def __init__(self, energy_lo, energy_hi, data=None, unit=""):
        e_edges = edges_from_lo_hi(energy_lo, energy_hi)
        self.energy = MapAxis.from_edges(e_edges, interp="log", name="energy")

        if data is None:
            data = np.zeros(self.energy.nbin)

        self.data = np.array(data)
        if not self.energy.nbin == self.data.size:
            raise ValueError("Incompatible data and energy axis size.")

        self.unit = u.Unit(unit)

    @property
    def quantity(self):
        return self.data * self.unit

    @quantity.setter
    def quantity(self, quantity):
        self.data = quantity.value
        self.unit = quantity.unit

    @classmethod
    def from_hdulist(cls, hdulist, hdu1="COUNTS", hdu2="EBOUNDS"):
        """Read from HDU list in OGIP format."""
        counts_table = Table.read(hdulist[hdu1])
        counts = counts_table["COUNTS"].data
        ebounds = ebounds_to_energy_axis(hdulist[hdu2])
        return cls(data=counts, energy_lo=ebounds[:-1], energy_hi=ebounds[1:])

    @classmethod
    def read(cls, filename, hdu1="COUNTS", hdu2="EBOUNDS"):
        """Read from file."""
        filename = make_path(filename)
        with fits.open(str(filename), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist, hdu1=hdu1, hdu2=hdu2)

    def to_table(self):
        """Convert to `~astropy.table.Table`.

        Data format specification: :ref:`gadf:ogip-pha`
        """
        channel = np.arange(self.energy.nbin, dtype=np.int16)
        counts = np.array(self.data, dtype=np.int32)

        names = ["CHANNEL", "COUNTS"]
        meta = {"name": "COUNTS"}
        return Table([channel, counts], names=names, meta=meta)

    def to_hdulist(self, use_sherpa=False):
        """Convert to `~astropy.io.fits.HDUList`.

        This adds an ``EBOUNDS`` extension to the ``BinTableHDU`` produced by
        ``to_table``, in order to store the energy axis
        """
        table = self.to_table()
        name = table.meta["name"]
        hdu = fits.BinTableHDU(table, name=name)

        energy = self.energy.edges

        if use_sherpa:
            energy = energy.to("keV")

        ebounds = energy_axis_to_ebounds(energy)
        return fits.HDUList([fits.PrimaryHDU(), hdu, ebounds])

    def write(self, filename, use_sherpa=False, **kwargs):
        """Write to file."""
        filename = make_path(filename)
        self.to_hdulist(use_sherpa=use_sherpa).writeto(str(filename), **kwargs)

    def fill(self, events):
        """Fill with list of events.

        Parameters
        ----------
        events : `~astropy.units.Quantity`, `gammapy.data.EventList`,
            List of event energies
        """
        if isinstance(events, EventList):
            events = events.energy

        energy = events.to(self.energy.unit)
        binned_val = np.histogram(energy.value, self.energy.edges)[0]
        self.data = binned_val

    @property
    def total_counts(self):
        """Total number of counts."""
        return self.data.sum()

    def plot(
        self,
        ax=None,
        energy_unit="TeV",
        show_poisson_errors=False,
        show_energy=None,
        **kwargs
    ):
        """Plot as data points.

        kwargs are forwarded to `~matplotlib.pyplot.errorbar`

        Parameters
        ----------
        ax : `~matplotlib.axis` (optional)
            Axis instance to be used for the plot
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
        show_poisson_errors : bool, optional
            Show poisson errors on the plot
        show_energy : `~astropy.units.Quantity`, optional
            Show energy, e.g. threshold, as vertical line

        Returns
        -------
        ax: `~matplotlib.axis`
            Axis instance used for the plot
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax
        counts = self.data
        x = self.energy.center.to_value(energy_unit)
        bounds = self.energy.edges.to_value(energy_unit)
        xerr = [x - bounds[:-1], bounds[1:] - x]
        yerr = np.sqrt(counts) if show_poisson_errors else 0
        kwargs.setdefault("fmt", ".")
        ax.errorbar(x, counts, xerr=xerr, yerr=yerr, **kwargs)
        if show_energy is not None:
            ener_val = u.Quantity(show_energy).to_value(energy_unit)
            ax.vlines(ener_val, 0, 1.1 * max(self.data), linestyles="dashed")
        ax.set_xlabel("Energy [{}]".format(energy_unit))
        ax.set_ylabel("Counts")
        ax.set_xscale("log")
        return ax

    def plot_hist(self, ax=None, energy_unit="TeV", show_energy=None, **kwargs):
        """Plot as histogram.

        kwargs are forwarded to `~matplotlib.pyplot.hist`

        Parameters
        ----------
        ax : `~matplotlib.axis` (optional)
            Axis instance to be used for the plot
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
        show_energy : `~astropy.units.Quantity`, optional
            Show energy, e.g. threshold, as vertical line
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax
        kwargs.setdefault("lw", 2)
        kwargs.setdefault("histtype", "step")
        weights = self.data
        bins = self.energy.edges.to_value(energy_unit)
        x = self.energy.center.to_value(energy_unit)
        ax.hist(x, bins=bins, weights=weights, **kwargs)
        if show_energy is not None:
            ener_val = u.Quantity(show_energy).to_value(energy_unit)
            ax.vlines(ener_val, 0, 1.1 * max(self.data), linestyles="dashed")
        ax.set_xlabel("Energy [{}]".format(energy_unit))
        ax.set_ylabel("Counts")
        ax.set_xscale("log")
        return ax

    def peek(self, figsize=(5, 10)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self.plot_hist(ax=ax)
        return ax

    def copy(self):
        """A deep copy of self."""
        return copy.deepcopy(self)

    def downsample(self, factor):
        """Downsample spectrum.

        Parameters
        ----------
        factor : int
            Downsampling factor.

        Returns
        -------
        spectrum : `~gammapy.spectrum.CountsSpectrum`
            Downsampled spectrum.
        """
        from ..extern.skimage import block_reduce

        data = block_reduce(self.data, block_size=(factor,))
        energy = self.energy.downsample(factor).edges
        return self.__class__(energy_lo=energy[:-1], energy_hi=energy[1:], data=data)

    def energy_mask(self, emin=None, emax=None):
        """Create a mask for a given energy range.

        Parameters
        ----------
        emin, emax : `~astropy.units.Quantity`
            Energy range
        """
        edges = self.energy.edges

        # set default values
        emin = emin if emin is not None else edges[0]
        emax = emax if emax is not None else edges[-1]

        return (edges[:-1] >= emin) & (edges[1:] <= emax)

    def _arithmetics(self, operator, other, copy):
        """Perform arithmetics on maps after checking geometry consistency."""
        if isinstance(other, CountsSpectrum):
            q = other.quantity
        else:
            q = u.Quantity(other, copy=False)

        out = self.copy() if copy else self
        out.quantity = operator(out.quantity, q)
        return out

    def __add__(self, other):
        return self._arithmetics(np.add, other, copy=True)

    def __iadd__(self, other):
        return self._arithmetics(np.add, other, copy=False)

    def __sub__(self, other):
        return self._arithmetics(np.subtract, other, copy=True)

    def __isub__(self, other):
        return self._arithmetics(np.subtract, other, copy=False)

    def __mul__(self, other):
        return self._arithmetics(np.multiply, other, copy=True)

    def __imul__(self, other):
        return self._arithmetics(np.multiply, other, copy=False)

    def __truediv__(self, other):
        return self._arithmetics(np.true_divide, other, copy=True)

    def __itruediv__(self, other):
        return self._arithmetics(np.true_divide, other, copy=False)

    def __array__(self):
        return self.data
