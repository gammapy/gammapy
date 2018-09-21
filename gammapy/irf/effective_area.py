# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from ..utils.nddata import NDDataArray, BinnedDataAxis
from ..utils.energy import EnergyBounds
from ..utils.scripts import make_path

__all__ = ["EffectiveAreaTable", "EffectiveAreaTable2D"]


class EffectiveAreaTable(object):
    """Effective area table.

    TODO: Document

    Parameters
    -----------
    energy_lo : `~astropy.units.Quantity`
        Lower bin edges of energy axis
    energy_hi : `~astropy.units.Quantity`
        Upper bin edges of energy axis
    data : `~astropy.units.Quantity`
        Effective area

    Examples
    --------
    Plot parametrized effective area for HESS, HESS2 and CTA.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        from gammapy.irf import EffectiveAreaTable

        energy = np.logspace(-3, 3, 100) * u.TeV

        for instrument in ['HESS', 'HESS2', 'CTA']:
            aeff = EffectiveAreaTable.from_parametrization(energy, instrument)
            ax = aeff.plot(label=instrument)

        ax.set_yscale('log')
        ax.set_xlim([1e-3, 1e3])
        ax.set_ylim([1e3, 1e12])
        plt.legend(loc='best')
        plt.show()

    Find energy where the effective area is at 10% of its maximum value

    >>> import numpy as np
    >>> import astropy.units as u
    >>> from gammapy.irf import EffectiveAreaTable
    >>> energy = np.logspace(-1, 2) * u.TeV
    >>> aeff_max = aeff.max_area
    >>> print(aeff_max).to('m2')
    156909.413371 m2
    >>> energy_threshold = aeff.find_energy(0.1 * aeff_max)
    >>> print(energy_threshold)
    0.185368478744 TeV
    """

    def __init__(self, energy_lo, energy_hi, data, meta=None):
        axes = [
            BinnedDataAxis(
                energy_lo, energy_hi, interpolation_mode="log", name="energy"
            )
        ]
        self.data = NDDataArray(axes=axes, data=data)
        self.meta = OrderedDict(meta) if meta else OrderedDict()

    @property
    def energy(self):
        return self.data.axis("energy")

    def plot(self, ax=None, energy=None, show_energy=None, **kwargs):
        """Plot effective area.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        energy : `~astropy.units.Quantity`
            Energy nodes
        show_energy : `~astropy.units.Quantity`, optional
            Show energy, e.g. threshold, as vertical line

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        kwargs.setdefault("lw", 2)

        if energy is None:
            energy = self.energy.nodes
        eff_area = self.data.evaluate(energy=energy)
        xerr = (
            energy.value - self.energy.lo.value,
            self.energy.hi.value - energy.value,
        )
        ax.errorbar(energy.value, eff_area.value, xerr=xerr, **kwargs)
        if show_energy is not None:
            ener_val = u.Quantity(show_energy).to(self.energy.unit).value
            ax.vlines(ener_val, 0, 1.1 * self.max_area.value, linestyles="dashed")
        ax.set_xscale("log")
        ax.set_xlabel("Energy [{}]".format(self.energy.unit))
        ax.set_ylabel("Effective Area [{}]".format(self.data.data.unit))

        return ax

    @classmethod
    def from_parametrization(cls, energy, instrument="HESS"):
        """Get parametrized effective area.

        Parametrizations of the effective areas of different Cherenkov
        telescopes taken from Appendix B of Abramowski et al. (2010), see
        http://adsabs.harvard.edu/abs/2010MNRAS.402.1342A .

        .. math::
            A_{eff}(E) = g_1 \\left(\\frac{E}{\\mathrm{MeV}}\\right)^{-g_2}\\exp{\\left(-\\frac{g_3}{E}\\right)}

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy binning, analytic function is evaluated at log centers
        instrument : {'HESS', 'HESS2', 'CTA'}
            Instrument name
        """
        energy = EnergyBounds(energy)
        # Put the parameters g in a dictionary.
        # Units: g1 (cm^2), g2 (), g3 (MeV)
        # Note that whereas in the paper the parameter index is 1-based,
        # here it is 0-based
        pars = {
            "HESS": [6.85e9, 0.0891, 5e5],
            "HESS2": [2.05e9, 0.0891, 1e5],
            "CTA": [1.71e11, 0.0891, 1e5],
        }

        if instrument not in pars.keys():
            ss = "Unknown instrument: {}\n".format(instrument)
            ss += "Valid instruments: HESS, HESS2, CTA"
            raise ValueError(ss)

        xx = energy.log_centers.to("MeV").value

        g1 = pars[instrument][0]
        g2 = pars[instrument][1]
        g3 = -pars[instrument][2]

        value = g1 * xx ** (-g2) * np.exp(g3 / xx)

        data = value * u.cm ** 2

        return cls(
            energy_lo=energy.lower_bounds, energy_hi=energy.upper_bounds, data=data
        )

    @classmethod
    def from_table(cls, table):
        """Create from `~astropy.table.Table` in ARF format.

        Data format specification: :ref:`gadf:ogip-arf`
        """
        energy_lo = table["ENERG_LO"].quantity
        energy_hi = table["ENERG_HI"].quantity
        data = table["SPECRESP"].quantity
        return cls(energy_lo=energy_lo, energy_hi=energy_hi, data=data)

    @classmethod
    def from_hdulist(cls, hdulist, hdu="SPECRESP"):
        """Create from `~astropy.io.fits.HDUList`."""
        return cls.from_table(Table.read(hdulist[hdu]))

    @classmethod
    def read(cls, filename, hdu="SPECRESP"):
        """Read from file."""
        filename = make_path(filename)
        with fits.open(str(filename), memmap=False) as hdulist:
            try:
                aeff = cls.from_hdulist(hdulist, hdu=hdu)
            except KeyError:
                msg = 'File {} contains no HDU "{}"'.format(filename, hdu)
                msg += "\n Available {}".format([_.name for _ in hdulist])
                raise ValueError(msg)

        return aeff

    def to_table(self):
        """Convert to `~astropy.table.Table` in ARF format.

        Data format specification: :ref:`gadf:ogip-arf`
        """
        table = Table()
        table.meta = OrderedDict(
            [
                ("EXTNAME", "SPECRESP"),
                ("hduclass", "OGIP"),
                ("hduclas1", "RESPONSE"),
                ("hduclas2", "SPECRESP"),
            ]
        )
        table["ENERG_LO"] = self.energy.lo
        table["ENERG_HI"] = self.energy.hi
        table["SPECRESP"] = self.evaluate_fill_nan()
        return table

    def to_hdulist(self, name=None):
        """Convert to `~astropy.io.fits.HDUList`."""
        return fits.HDUList(
            [fits.PrimaryHDU(), fits.BinTableHDU(self.to_table(), name=name)]
        )

    def write(self, filename, **kwargs):
        """Write to file."""
        filename = make_path(filename)
        self.to_hdulist().writeto(str(filename), **kwargs)

    def evaluate_fill_nan(self, **kwargs):
        """Modified evaluate function.

        Calls :func:`gammapy.utils.nddata.NDDataArray.evaluate` and replaces
        possible nan values. Below the finite range the effective area is set
        to zero and above to value of the last valid note. This is needed since
        other codes, e.g. sherpa, don't like nan values in FITS files. Make
        sure that the replacement happens outside of the energy range, where
        the `~gammapy.irf.EffectiveAreaTable` is used.
        """
        retval = self.data.evaluate(**kwargs)
        idx = np.where(np.isfinite(retval))[0]
        retval[np.arange(idx[0])] = 0
        retval[np.arange(idx[-1], len(retval))] = retval[idx[-1]]
        return retval

    @property
    def max_area(self):
        """Maximum effective area."""
        cleaned_data = self.data.data[np.where(~np.isnan(self.data.data))]
        return cleaned_data.max()

    def find_energy(self, aeff, reverse=False):
        """Find energy for given effective area.

        A linear interpolation is performed between the two nodes closest to
        the desired effective area value. By default, the first match is
        returned (use `reverse` to search starting from the end of the array)

        TODO: Move to `~gammapy.utils.nddata.NDDataArray`

        Parameters
        ----------
        aeff : `~astropy.units.Quantity`
            Effective area value
        reverse : bool
            Reverse the direction, i.e. search starting from the end of the array

        Returns
        -------
        energy : `~astropy.units.Quantity`
            Energy corresponding to aeff
        """
        valid = np.where(self.data.data > aeff)[0]
        idx = valid[0]
        if reverse:
            idx = valid[-1]

        if not reverse:
            # Return lower edge if first bin is selected
            if idx == 0:
                energy = self.energy.lo[idx].value
            # Perform linear interpolation otherwise
            else:
                energy = np.interp(
                    aeff.value,
                    (self.data.data[[idx - 1, idx]].value),
                    (self.energy.nodes[[idx - 1, idx]].value),
                )
        else:
            # Return upper edge if last bin is selected
            if idx == self.data.data.size - 1:
                energy = self.energy.hi[idx].value
            # Perform linear interpolation otherwise
            else:
                energy = np.interp(
                    aeff.value,
                    (self.data.data[[idx, idx + 1]].value),
                    (self.energy.nodes[[idx, idx + 1]].value),
                )
        return energy * self.energy.unit

    def to_sherpa(self, name):
        """Convert to `~sherpa.astro.data.DataARF`

        Parameters
        ----------
        name : str
            Instance name
        """
        from sherpa.astro.data import DataARF

        table = self.to_table()
        return DataARF(
            name=name,
            energ_lo=table["ENERG_LO"].quantity.to("keV").value,
            energ_hi=table["ENERG_HI"].quantity.to("keV").value,
            specresp=table["SPECRESP"].quantity.to("cm2").value,
        )


class EffectiveAreaTable2D(object):
    """2D effective area table.

    Data format specification: :ref:`gadf:aeff_2d`

    Parameters
    -----------
    energy_lo, energy_hi : `~astropy.units.Quantity`
        Energy binning
    offset_lo, offset_hi : `~astropy.units.Quantity`
        Field of view offset angle.
    data : `~astropy.units.Quantity`
        Effective area

    Examples
    --------
    Here's an example you can use to learn about this class:

    >>> from gammapy.irf import EffectiveAreaTable2D
    >>> filename = '$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits'
    >>> aeff = EffectiveAreaTable2D.read(filename, hdu='EFFECTIVE AREA')
    >>> print(aeff)
    EffectiveAreaTable2D
    NDDataArray summary info
    energy         : size =    42, min =  0.014 TeV, max = 177.828 TeV
    offset         : size =     6, min =  0.500 deg, max =  5.500 deg
    Data           : size =   252, min =  0.000 m2, max = 5371581.000 m2

    Here's another one, created from scratch, without reading a file:

    >>> from gammapy.irf import EffectiveAreaTable2D
    >>> import astropy.units as u
    >>> import numpy as np
    >>> energy = np.logspace(0,1,11) * u.TeV
    >>> offset = np.linspace(0,1,4) * u.deg
    >>> data = np.ones(shape=(10,3)) * u.cm * u.cm
    >>> aeff = EffectiveAreaTable2D(energy_lo=energy[:-1], energy_hi=energy[1:], offset_lo=offset[:-1],
    >>>                             offset_hi=offset[1:], data= data)
    >>> print(aeff)
    Data array summary info
    energy         : size =    11, min =  1.000 TeV, max = 10.000 TeV
    offset         : size =     4, min =  0.000 deg, max =  1.000 deg
    Data           : size =    30, min =  1.000 cm2, max =  1.000 cm2
    """

    default_interp_kwargs = dict(bounds_error=False, fill_value=None)
    """Default Interpolation kwargs for `~NDDataArray`. Extrapolate."""

    def __init__(
        self,
        energy_lo,
        energy_hi,
        offset_lo,
        offset_hi,
        data,
        meta=None,
        interp_kwargs=None,
    ):

        if interp_kwargs is None:
            interp_kwargs = self.default_interp_kwargs
        axes = [
            BinnedDataAxis(
                energy_lo, energy_hi, interpolation_mode="log", name="energy"
            ),
            BinnedDataAxis(
                offset_lo, offset_hi, interpolation_mode="linear", name="offset"
            ),
        ]
        self.data = NDDataArray(axes=axes, data=data, interp_kwargs=interp_kwargs)
        self.meta = OrderedDict(meta) if meta else OrderedDict()

    def __str__(self):
        ss = self.__class__.__name__
        ss += "\n{}".format(self.data)
        return ss

    @property
    def low_threshold(self):
        """Low energy threshold"""
        return self.meta["LO_THRES"] * u.TeV

    @property
    def high_threshold(self):
        """High energy threshold"""
        return self.meta["HI_THRES"] * u.TeV

    @classmethod
    def from_table(cls, table):
        """Read from `~astropy.table.Table`."""
        return cls(
            energy_lo=table["ENERG_LO"].quantity[0],
            energy_hi=table["ENERG_HI"].quantity[0],
            offset_lo=table["THETA_LO"].quantity[0],
            offset_hi=table["THETA_HI"].quantity[0],
            data=table["EFFAREA"].quantity[0].transpose(),
            meta=table.meta,
        )

    @classmethod
    def from_hdulist(cls, hdulist, hdu="EFFECTIVE AREA"):
        """Create from `~astropy.io.fits.HDUList`."""
        return cls.from_table(Table.read(hdulist[hdu]))

    @classmethod
    def read(cls, filename, hdu="EFFECTIVE AREA"):
        """Read from file."""
        filename = make_path(filename)
        with fits.open(str(filename), memmap=False) as hdulist:
            aeff = cls.from_hdulist(hdulist, hdu=hdu)

        return aeff

    def to_effective_area_table(self, offset, energy=None):
        """Evaluate at a given offset and return `~gammapy.irf.EffectiveAreaTable`.

        Parameters
        ----------
        offset : `~astropy.coordinates.Angle`
            Offset
        energy : `~astropy.units.Quantity`
            Energy axis bin edges
        """
        if energy is None:
            energy = self.data.axis("energy").bins

        energy = EnergyBounds(energy)
        area = self.data.evaluate(offset=offset, energy=energy.log_centers)

        return EffectiveAreaTable(
            energy_lo=energy.lower_bounds, energy_hi=energy.upper_bounds, data=area
        )

    def plot_energy_dependence(self, ax=None, offset=None, energy=None, **kwargs):
        """Plot effective area versus energy for a given offset.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        offset : `~astropy.coordinates.Angle`
            Offset
        energy : `~astropy.units.Quantity`
            Energy axis
        kwargs : dict
            Forwarded tp plt.plot()

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        if offset is None:
            off_min, off_max = self.data.axis("offset").nodes[[0, -1]].value
            offset = np.linspace(off_min, off_max, 4) * self.data.axis("offset").unit

        if energy is None:
            energy = self.data.axis("energy").nodes

        for off in offset:
            area = self.data.evaluate(offset=off, energy=energy)
            label = "offset = {:.1f}".format(off)
            ax.plot(energy, area.value, label=label, **kwargs)

        ax.set_xscale("log")
        ax.set_xlabel("Energy [{}]".format(self.data.axis("energy").unit))
        ax.set_ylabel("Effective Area [{}]".format(self.data.data.unit))
        ax.set_xlim(min(energy.value), max(energy.value))
        ax.legend(loc="upper left")

        return ax

    def plot_offset_dependence(self, ax=None, offset=None, energy=None, **kwargs):
        """Plot effective area versus offset for a given energy.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        offset : `~astropy.coordinates.Angle`
            Offset axis
        energy : `~gammapy.utils.energy.Energy`
            Energy

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        if energy is None:
            e_min, e_max = np.log10(self.data.axis("energy").nodes[[0, -1]].value)
            energy = np.logspace(e_min, e_max, 4) * self.data.axis("energy").unit

        if offset is None:
            off_lo, off_hi = self.data.axis("offset").nodes[[0, -1]].to("deg").value
            offset = np.linspace(off_lo, off_hi, 100) * u.deg

        for ee in energy:
            area = self.data.evaluate(offset=offset, energy=ee)
            area /= np.nanmax(area)
            if np.isnan(area).all():
                continue
            label = "energy = {:.1f}".format(ee)
            ax.plot(offset, area, label=label, **kwargs)

        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Offset ({})".format(self.data.axis("offset").unit))
        ax.set_ylabel("Relative Effective Area")
        ax.legend(loc="best")

        return ax

    def plot(self, ax=None, add_cbar=True, **kwargs):
        """Plot effective area image."""
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        offset = self.data.axis("offset").bins
        energy = self.data.axis("energy").bins
        aeff = self.data.evaluate(offset=offset, energy=energy)

        vmin, vmax = np.nanmin(aeff.value), np.nanmax(aeff.value)

        kwargs.setdefault("cmap", "GnBu")
        kwargs.setdefault("edgecolors", "face")
        kwargs.setdefault("vmin", vmin)
        kwargs.setdefault("vmax", vmax)

        caxes = ax.pcolormesh(energy.value, offset.value, aeff.value.T, **kwargs)

        ax.set_xscale("log")
        ax.set_ylabel("Offset ({})".format(offset.unit))
        ax.set_xlabel("Energy ({})".format(energy.unit))

        xmin, xmax = energy.value.min(), energy.value.max()
        ax.set_xlim(xmin, xmax)

        if add_cbar:
            label = "Effective Area ({unit})".format(unit=aeff.unit)
            ax.figure.colorbar(caxes, ax=ax, label=label)

        return ax

    def peek(self, figsize=(15, 5)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        self.plot(ax=axes[2])
        self.plot_energy_dependence(ax=axes[0])
        self.plot_offset_dependence(ax=axes[1])
        plt.tight_layout()

    def to_table(self):
        """Convert to `~astropy.table.Table`."""
        meta = self.meta.copy()
        table = Table(meta=meta)
        table["ENERG_LO"] = self.data.axis("energy").lo[np.newaxis]
        table["ENERG_HI"] = self.data.axis("energy").hi[np.newaxis]
        table["THETA_LO"] = self.data.axis("offset").lo[np.newaxis]
        table["THETA_HI"] = self.data.axis("offset").hi[np.newaxis]
        table["EFFAREA"] = self.data.data.T[np.newaxis]
        return table

    def to_fits(self, name="EFFECTIVE AREA"):
        """Convert to `~astropy.io.fits.BinTable`."""
        return fits.BinTableHDU(self.to_table(), name=name)
