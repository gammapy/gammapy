# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import logging
from astropy.table import Table
from astropy.io import fits
import astropy.units as u
from ..utils.nddata import NDDataArray, BinnedDataAxis
from ..utils.scripts import make_path
from ..utils.fits import energy_axis_to_ebounds, ebounds_to_energy_axis
from ..data import EventList

__all__ = ["CountsSpectrum", "PHACountsSpectrum", "PHACountsSpectrumList"]

log = logging.getLogger("__name__")


class CountsSpectrum(object):
    """Generic counts spectrum.

    Parameters
    ----------
    energy_lo : `~astropy.units.Quantity`
        Lower bin edges of energy axis
    energy_hi : `~astropy.units.Quantity`
        Upper bin edges of energy axis
    data : `~astropy.units.Quantity`, array-like
        Counts

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

    default_interp_kwargs = dict(bounds_error=False, method="nearest")
    """Default interpolation kwargs"""

    def __init__(self, energy_lo, energy_hi, data=None, interp_kwargs=None):
        axes = [
            BinnedDataAxis(
                energy_lo, energy_hi, interpolation_mode="log", name="energy"
            )
        ]

        if interp_kwargs is None:
            interp_kwargs = self.default_interp_kwargs
        self.data = NDDataArray(axes=axes, data=data, interp_kwargs=interp_kwargs)

    @property
    def energy(self):
        return self.data.axis("energy")

    @classmethod
    def from_hdulist(cls, hdulist, hdu1="COUNTS", hdu2="EBOUNDS"):
        """Read OGIP format hdulist"""
        counts_table = Table.read(hdulist[hdu1])
        counts = counts_table["COUNTS"].data
        ebounds = ebounds_to_energy_axis(hdulist[hdu2])
        return cls(
            data=counts, energy_lo=ebounds.lower_bounds, energy_hi=ebounds.upper_bounds
        )

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
        channel = np.arange(self.energy.nbins, dtype=np.int16)
        counts = np.array(self.data.data.value, dtype=np.int32)

        names = ["CHANNEL", "COUNTS"]
        meta = {"name": "COUNTS"}
        return Table([channel, counts], names=names, meta=meta)

    def to_hdulist(self):
        """Convert to `~astropy.io.fits.HDUList`.

        This adds an ``EBOUNDS`` extension to the ``BinTableHDU`` produced by
        ``to_table``, in order to store the energy axis
        """
        table = self.to_table()
        name = table.meta["name"]
        hdu = fits.BinTableHDU(table, name=name)
        ebounds = energy_axis_to_ebounds(self.energy.bins)
        return fits.HDUList([fits.PrimaryHDU(), hdu, ebounds])

    def write(self, filename, **kwargs):
        """Write to file."""
        filename = make_path(filename)
        self.to_hdulist().writeto(str(filename), **kwargs)

    def fill(self, events):
        """Fill with list of events.

        TODO: Move to `~gammapy.utils.nddata.NDDataArray`

        Parameters
        ----------
        events : `~astropy.units.Quantity`, `gammapy.data.EventList`,
            List of event energies
        """
        if isinstance(events, EventList):
            events = events.energy

        energy = events.to(self.energy.unit)
        binned_val = np.histogram(energy.value, self.energy.bins)[0]
        self.data.data = binned_val

    @property
    def total_counts(self):
        """Total number of counts
        """
        return self.data.data.sum()

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
        counts = self.data.data.value
        x = self.energy.nodes.to(energy_unit).value
        bounds = self.energy.bins.to(energy_unit).value
        xerr = [x - bounds[:-1], bounds[1:] - x]
        yerr = np.sqrt(counts) if show_poisson_errors else 0
        kwargs.setdefault("fmt", "")
        ax.errorbar(x, counts, xerr=xerr, yerr=yerr, **kwargs)
        if show_energy is not None:
            ener_val = u.Quantity(show_energy).to(energy_unit).value
            ax.vlines(ener_val, 0, 1.1 * max(self.data.data.value), linestyles="dashed")
        ax.set_xlabel("Energy [{}]".format(energy_unit))
        ax.set_ylabel("Counts")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.2 * max(self.data.data.value))
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
        weights = self.data.data.value
        bins = self.energy.bins.to(energy_unit).value
        x = self.energy.nodes.to(energy_unit).value
        ax.hist(x, bins=bins, weights=weights, **kwargs)
        if show_energy is not None:
            ener_val = u.Quantity(show_energy).to(energy_unit).value
            ax.vlines(ener_val, 0, 1.1 * max(self.data.data.value), linestyles="dashed")
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
        return deepcopy(self)

    def rebin(self, parameter):
        """Rebin.

        Parameters
        ----------
        parameter : int
            Number of bins to merge

        Returns
        -------
        rebinned_spectrum : `~gammapy.spectrum.CountsSpectrum`
            Rebinned spectrum
        """
        if len(self.data.data) % parameter != 0:
            raise ValueError(
                "Invalid rebin parameter: {}, nbins: {}".format(
                    parameter, len(self.data.data)
                )
            )

        # Copy to keep attributes
        retval = self.copy()
        energy = retval.energy
        energy.lo = energy.lo[0::parameter]
        energy.hi = energy.hi[parameter - 1 :: parameter]
        split_indices = np.arange(parameter, len(retval.data.data), parameter)
        counts_grp = np.split(retval.data.data, split_indices)
        counts_rebinned = np.sum(counts_grp, axis=1)
        retval.data.data = counts_rebinned

        return retval


class PHACountsSpectrum(CountsSpectrum):
    """Counts spectrum corresponding to OGIP PHA format.

    The ``bkg`` flag controls whether the PHA counts spectrum represents a
    background estimate or not (this slightly affects the FITS header
    information when writing to disk).

    Parameters
    ----------
    energy_lo : `~astropy.units.Quantity`
        Lower bin edges of energy axis
    energy_hi : `~astropy.units.Quantity`
        Upper bin edges of energy axis
    data : array-like, optional
        Counts
    quality : int, array-like, optional
        Mask bins in safe energy range (1 = bad, 0 = good)
    backscal : float, array-like, optional
        Background scaling factor
    areascal : float, array-like, optional
        Area scaling factor
    is_bkg : bool, optional
        Background or soure spectrum, default: False
    obs_id : int
        Observation identifier, optional
    livetime : `~astropy.units.Quantity`, optional
        Observation livetime
    offset : `~astropy.units.Quantity`, optional
        Field of view offset
    meta : dict, optional
        Meta information
    """

    def __init__(
        self,
        energy_lo,
        energy_hi,
        data=None,
        quality=None,
        backscal=None,
        areascal=None,
        is_bkg=False,
        obs_id=None,
        livetime=None,
        offset=None,
        meta=None,
    ):
        super(PHACountsSpectrum, self).__init__(energy_lo, energy_hi, data)
        if quality is None:
            quality = np.zeros(self.energy.nbins, dtype="i2")
        self._quality = quality
        if backscal is None:
            backscal = np.ones(self.energy.nbins)
        self.backscal = backscal
        if areascal is None:
            areascal = np.ones(self.energy.nbins)
        self.areascal = areascal
        self.is_bkg = is_bkg
        self.obs_id = obs_id
        self.livetime = livetime
        self.offset = offset
        self.meta = meta or OrderedDict()

    @property
    def quality(self):
        """Bins in safe energy range (1 = bad, 0 = good)"""
        return self._quality

    @quality.setter
    def quality(self, quality):
        self._quality = quality

    @property
    def phafile(self):
        """PHA file associated with the observation"""
        if isinstance(self.obs_id, list):
            filename = "pha_stacked.fits"
        else:
            filename = "pha_obs{}.fits".format(self.obs_id)
        return filename

    @property
    def arffile(self):
        """ARF associated with the observation"""
        return self.phafile.replace("pha", "arf")

    @property
    def rmffile(self):
        """RMF associated with the observation"""
        return self.phafile.replace("pha", "rmf")

    @property
    def bkgfile(self):
        """Background PHA files associated with the observation"""
        return self.phafile.replace("pha", "bkg")

    @property
    def bins_in_safe_range(self):
        """Indices of bins within the energy thresholds"""
        idx = np.where(np.array(self.quality) == 0)[0]
        return idx

    @property
    def counts_in_safe_range(self):
        """Counts with bins outside safe range set to 0"""
        data = self.data.data.copy()
        data[np.nonzero(self.quality)] = 0
        return data

    @property
    def lo_threshold(self):
        """Low energy threshold of the observation (lower bin edge)"""
        idx = self.bins_in_safe_range[0]
        return self.energy.lo[idx]

    @lo_threshold.setter
    def lo_threshold(self, thres):
        idx = np.where(self.energy.lo < thres)[0]
        self.quality[idx] = 1

    @property
    def hi_threshold(self):
        """High energy threshold of the observation (upper bin edge)"""
        idx = self.bins_in_safe_range[-1]
        return self.energy.hi[idx]

    @hi_threshold.setter
    def hi_threshold(self, thres):
        idx = np.where(self.energy.lo > thres)[0]
        if len(idx) != 0:
            idx = np.insert(idx, 0, idx[0] - 1)
        self.quality[idx] = 1

    def reset_thresholds(self):
        """Reset energy thresholds (i.e. declare all energy bins valid)"""
        self.quality = np.zeros_like(self.quality)

    def rebin(self, parameter):
        """Rebin.

        See `~gammapy.spectrum.CountsSpectrum`.
        This function treats the quality vector correctly
        """
        retval = super(PHACountsSpectrum, self).rebin(parameter)
        split_indices = np.arange(parameter, len(self.data.data), parameter)
        quality_grp = np.split(retval.quality, split_indices)
        quality_summed = np.sum(quality_grp, axis=1)
        # Exclude groups where not all bins are within the safe threshold
        condition = quality_summed == parameter
        quality_rebinned = np.where(
            condition, np.ones(len(retval.data.data)), np.zeros(len(retval.data.data))
        )
        retval.quality = np.array(quality_rebinned, dtype=int)

        # if backscal is not the same in all channels cannot merge
        if not np.isscalar(retval.backscal):
            if not np.isclose(np.diff(retval.backscal), 0).all():
                raise ValueError("Cannot merge energy dependent backscal")
            else:
                retval.meta.backscal = retval.backscal[0] * np.ones(retval.energy.nbins)

        # average areascal
        areascal_grp = np.split(retval.areascal, split_indices)
        retval.areascal = np.mean(areascal_grp, axis=1)

        return retval

    @property
    def _backscal_array(self):
        """Helper function to always return backscal as an array"""
        if np.isscalar(self.backscal):
            return np.ones(self.energy.nbins) * self.backscal
        else:
            return self.backscal

    def to_table(self):
        """Convert to `~astropy.table.Table`."""
        table = super(PHACountsSpectrum, self).to_table()

        table["QUALITY"] = self.quality
        table["BACKSCAL"] = self._backscal_array
        table["AREASCAL"] = self.areascal

        meta = OrderedDict()
        meta["name"] = "SPECTRUM"
        meta["hduclass"] = "OGIP"
        meta["hduclas1"] = "SPECTRUM"
        meta["corrscal"] = ""
        meta["chantype"] = "PHA"
        meta["detchans"] = self.energy.nbins
        meta["filter"] = "None"
        meta["corrfile"] = ""
        meta["poisserr"] = True
        meta["hduclas3"] = "COUNT"
        meta["hduclas4"] = "TYPE:1"
        meta["lo_thres"] = self.lo_threshold.to("TeV").value
        meta["hi_thres"] = self.hi_threshold.to("TeV").value
        meta["exposure"] = self.livetime.to("s").value
        meta["obs_id"] = self.obs_id

        if not self.is_bkg:
            if self.rmffile is not None:
                meta["respfile"] = self.rmffile

            meta["backfile"] = self.bkgfile
            meta["ancrfile"] = self.arffile
            meta["hduclas2"] = "TOTAL"
        else:
            meta["hduclas2"] = "BKG"

        meta.update(self.meta)

        table.meta = meta
        return table

    @classmethod
    def from_hdulist(cls, hdulist, hdu1="SPECTRUM", hdu2="EBOUNDS"):
        """Create from `~astropy.io.fits.HDUList`."""
        counts_table = Table.read(hdulist[hdu1])
        ebounds = Table.read(hdulist[2])
        emin = ebounds["E_MIN"].quantity
        emax = ebounds["E_MAX"].quantity

        # Check if column are present in the header
        quality = None
        areascal = None
        backscal = None
        if "QUALITY" in counts_table.colnames:
            quality = counts_table["QUALITY"].data
        if "AREASCAL" in counts_table.colnames:
            areascal = counts_table["AREASCAL"].data
        if "BACKSCAL" in counts_table.colnames:
            backscal = counts_table["BACKSCAL"].data

        kwargs = dict(
            data=counts_table["COUNTS"],
            backscal=backscal,
            energy_lo=emin,
            energy_hi=emax,
            quality=quality,
            areascal=areascal,
            livetime=counts_table.meta["EXPOSURE"] * u.s,
            obs_id=counts_table.meta["OBS_ID"],
        )
        if hdulist[1].header["HDUCLAS2"] == "BKG":
            kwargs["is_bkg"] = True
        return cls(**kwargs)

    @classmethod
    def read(cls, filename, hdu1="SPECTRUM", hdu2="EBOUNDS"):
        """Read from file."""
        filename = make_path(filename)
        with fits.open(str(filename), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist, hdu1=hdu1, hdu2=hdu2)

    def to_sherpa(self, name):
        """Convert to `sherpa.astro.data.DataPHA`.

        Parameters
        ----------
        name : str
            Instance name
        """
        from sherpa.utils import SherpaFloat
        from sherpa.astro.data import DataPHA

        table = self.to_table()

        # Workaround to avoid https://github.com/sherpa/sherpa/issues/248
        if np.isscalar(self.backscal):
            backscal = self.backscal
        else:
            backscal = self.backscal.copy()
            if np.allclose(backscal.mean(), backscal):
                backscal = backscal[0]

        return DataPHA(
            name=name,
            channel=(table["CHANNEL"].data + 1).astype(SherpaFloat),
            counts=table["COUNTS"].data.astype(SherpaFloat),
            quality=table["QUALITY"].data,
            exposure=self.livetime.to("s").value,
            backscal=backscal,
            areascal=self.areascal,
            syserror=None,
            staterror=None,
            grouping=None,
        )


class PHACountsSpectrumList(list):
    """List of `~gammapy.spectrum.PHACountsSpectrum` objects.

    All spectra must have the same energy binning.
    This represent the PHA type II data format.
    See https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007/node8.html
    """

    def write(self, outdir, **kwargs):
        """Write to file"""
        outdir = make_path(outdir)
        self.to_hdulist().writeto(str(outdir), **kwargs)

    def to_hdulist(self):
        """Convert to `~astropy.io.fits.HDUList`"""
        hdu = fits.BinTableHDU(self.to_table())
        ebounds = energy_axis_to_ebounds(self[0].energy.bins)
        return fits.HDUList([fits.PrimaryHDU(), hdu, ebounds])

    def to_table(self):
        """Convert to `~astropy.table.Table`."""
        is_bkg = self[0].is_bkg
        nbins = self[0].energy.nbins
        spec_num = np.empty([len(self), 1], dtype=np.int16)
        channel = np.empty([len(self), nbins], dtype=np.int16)
        counts = np.empty([len(self), nbins], dtype=np.int32)
        quality = np.empty([len(self), nbins], dtype=np.int32)
        backscal = np.empty([len(self), nbins], dtype=np.int32)
        backfile = list()
        for idx, pha in enumerate(self):
            t = pha.to_table()
            spec_num[idx] = pha.obs_id
            channel[idx] = t["CHANNEL"].data
            counts[idx] = t["COUNTS"].data
            quality[idx] = t["QUALITY"].data
            backscal[idx] = t["BACKSCAL"].data
            backfile.append("bkg.fits[{}]".format(idx))

        meta = self[0].to_table().meta
        meta["hduclas4"] = "TYPE:II"
        meta["ancrfile"] = "arf.fits"
        meta["respfile"] = "rmf.fits"

        data = [spec_num, channel, counts, quality, backscal]
        names = ["SPEC_NUM", "CHANNEL", "COUNTS", "QUALITY", "BACKSCAL"]
        table = Table(data, names=names, meta=meta)

        if not is_bkg:
            table.meta.pop("backfile")
            table["BACKFILE"] = backfile

        return table

    @classmethod
    def read(cls, filename):
        """Read from file."""
        filename = make_path(filename)
        with fits.open(str(filename), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist)

    @classmethod
    def from_hdulist(cls, hdulist):
        """Create from `~astropy.io.fits.HDUList`."""
        energy = ebounds_to_energy_axis(hdulist[2])
        kwargs = dict(energy_lo=energy[:-1], energy_hi=energy[1:])
        if hdulist[1].header["HDUCLAS2"] == "BKG":
            kwargs["is_bkg"] = True

        counts_table = Table.read(hdulist[1])
        speclist = cls()
        for row in counts_table:
            kwargs["data"] = row["COUNTS"]
            kwargs["backscal"] = row["BACKSCAL"]
            kwargs["quality"] = row["QUALITY"]
            kwargs["livetime"] = hdulist[1].header["EXPOSURE"] * u.s
            kwargs["obs_id"] = row["SPEC_NUM"]
            spec = PHACountsSpectrum(**kwargs)
            speclist.append(spec)

        return speclist
