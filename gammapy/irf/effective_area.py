# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from gammapy.maps import MapAxis
from gammapy.maps.utils import edges_from_lo_hi
from gammapy.utils.energy import energy_logcenter
from gammapy.utils.nddata import NDDataArray
from gammapy.utils.scripts import make_path

__all__ = ["EffectiveAreaTable", "EffectiveAreaTable2D"]


class EffectiveAreaTable:
    """Effective area table.

    TODO: Document

    Parameters
    ----------
    energy_lo, energy_hi : `~astropy.units.Quantity`
        Energy axis bin edges
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

        e_edges = edges_from_lo_hi(energy_lo, energy_hi)
        energy_axis = MapAxis.from_edges(e_edges, interp="log", name="energy")

        interp_kwargs = {"extrapolate": False, "bounds_error": False}
        self.data = NDDataArray(
            axes=[energy_axis], data=data, interp_kwargs=interp_kwargs
        )
        self.meta = meta or {}

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
            energy = self.energy.center

        eff_area = self.data.evaluate(energy=energy)

        xerr = (
            (energy - self.energy.edges[:-1]).value,
            (self.energy.edges[1:] - energy).value,
        )

        ax.errorbar(energy.value, eff_area.value, xerr=xerr, **kwargs)
        if show_energy is not None:
            ener_val = u.Quantity(show_energy).to_value(self.energy.unit)
            ax.vlines(ener_val, 0, 1.1 * self.max_area.value, linestyles="dashed")
        ax.set_xscale("log")
        ax.set_xlabel(f"Energy [{self.energy.unit}]")
        ax.set_ylabel(f"Effective Area [{self.data.data.unit}]")

        return ax

    @classmethod
    def from_parametrization(cls, energy, instrument="HESS"):
        r"""Create parametrized effective area.

        Parametrizations of the effective areas of different Cherenkov
        telescopes taken from Appendix B of Abramowski et al. (2010), see
        https://ui.adsabs.harvard.edu/abs/2010MNRAS.402.1342A .

        .. math::
            A_{eff}(E) = g_1 \left(\frac{E}{\mathrm{MeV}}\right)^{-g_2}\exp{\left(-\frac{g_3}{E}\right)}

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy binning, analytic function is evaluated at log centers
        instrument : {'HESS', 'HESS2', 'CTA'}
            Instrument name
        """
        energy = u.Quantity(energy)
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
            ss = f"Unknown instrument: {instrument}\n"
            ss += "Valid instruments: HESS, HESS2, CTA"
            raise ValueError(ss)

        xx = energy_logcenter(energy).to_value("MeV")

        g1 = pars[instrument][0]
        g2 = pars[instrument][1]
        g3 = -pars[instrument][2]

        value = g1 * xx ** (-g2) * np.exp(g3 / xx)
        data = u.Quantity(value, "cm2", copy=False)

        return cls(energy_lo=energy[:-1], energy_hi=energy[1:], data=data)

    @classmethod
    def from_constant(cls, energy, value):
        """Create constant value effective area.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy binning, analytic function is evaluated at log centers
        value : `~astropy.units.Quantity`
            Effective area
        """
        data = np.ones((len(energy) - 1)) * u.Quantity(value)
        return cls(energy_lo=energy[:-1], energy_hi=energy[1:], data=data)

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
        with fits.open(filename, memmap=False) as hdulist:
            try:
                return cls.from_hdulist(hdulist, hdu=hdu)
            except KeyError:
                raise ValueError(
                    f"File {filename} contains no HDU {hdu!r}\n"
                    f"Available: {[_.name for _ in hdulist]}"
                )

    def to_table(self):
        """Convert to `~astropy.table.Table` in ARF format.

        Data format specification: :ref:`gadf:ogip-arf`
        """
        table = Table()
        table.meta = {
            "EXTNAME": "SPECRESP",
            "hduclass": "OGIP",
            "hduclas1": "RESPONSE",
            "hduclas2": "SPECRESP",
        }

        energy = self.energy.edges
        table["ENERG_LO"] = energy[:-1]
        table["ENERG_HI"] = energy[1:]
        table["SPECRESP"] = self.evaluate_fill_nan()
        return table

    def to_hdulist(self, name=None, use_sherpa=False):
        """Convert to `~astropy.io.fits.HDUList`."""
        table = self.to_table()

        if use_sherpa:
            table["ENERG_HI"] = table["ENERG_HI"].quantity.to("keV")
            table["ENERG_LO"] = table["ENERG_LO"].quantity.to("keV")
            table["SPECRESP"] = table["SPECRESP"].quantity.to("cm2")

        return fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(table, name=name)])

    def write(self, filename, use_sherpa=False, **kwargs):
        """Write to file."""
        filename = make_path(filename)
        self.to_hdulist(use_sherpa=use_sherpa).writeto(filename, **kwargs)

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

    def find_energy(self, aeff, emin=None, emax=None):
        """Find energy for a given effective area.

        In case the solution is not unique, provide the `emin` or `emax` arguments
        to limit the solution to the given range. By default the peak energy of the
        effective area is chosen as `emax`.

        Parameters
        ----------
        aeff : `~astropy.units.Quantity`
            Effective area value
        emin : `~astropy.units.Quantity`
            Lower bracket value in case solution is not unique.
        emax : `~astropy.units.Quantity`
            Upper bracket value in case solution is not unique.

        Returns
        -------
        energy : `~astropy.units.Quantity`
            Energy corresponding to the given aeff.
        """
        from gammapy.modeling.models import TemplateSpectralModel

        energy = self.energy.center

        if emin is None:
            emin = energy[0]
        if emax is None:
            # use the peak effective area as a default for the energy maximum
            emax = energy[np.argmax(self.data.data)]

        aeff_spectrum = TemplateSpectralModel(energy, self.data.data)
        return aeff_spectrum.inverse(aeff, emin=emin, emax=emax)


class EffectiveAreaTable2D:
    """2D effective area table.

    Data format specification: :ref:`gadf:aeff_2d`

    Parameters
    ----------
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

        e_edges = edges_from_lo_hi(energy_lo, energy_hi)
        energy_axis = MapAxis.from_edges(e_edges, interp="log", name="energy")

        # TODO: for some reason the H.E.S.S. DL3 files contain the same values for offset_hi and offset_lo
        if np.allclose(offset_lo.to_value("deg"), offset_hi.to_value("deg")):
            offset_axis = MapAxis.from_nodes(offset_lo, interp="lin", name="offset")
        else:
            offset_edges = edges_from_lo_hi(offset_lo, offset_hi)
            offset_axis = MapAxis.from_edges(offset_edges, interp="lin", name="offset")

        self.data = NDDataArray(
            axes=[energy_axis, offset_axis], data=data, interp_kwargs=interp_kwargs
        )
        self.meta = meta or {}

    def __str__(self):
        ss = self.__class__.__name__
        ss += f"\n{self.data}"
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
        with fits.open(make_path(filename), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist, hdu=hdu)

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
            energy = self.data.axis("energy").edges

        area = self.data.evaluate(offset=offset, energy=energy_logcenter(energy))

        return EffectiveAreaTable(
            energy_lo=energy[:-1], energy_hi=energy[1:], data=area
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
            off_min, off_max = self.data.axis("offset").center[[0, -1]]
            offset = np.linspace(off_min.value, off_max.value, 4) * off_min.unit

        if energy is None:
            energy = self.data.axis("energy").center

        for off in offset:
            area = self.data.evaluate(offset=off, energy=energy)
            label = f"offset = {off:.1f}"
            ax.plot(energy, area.value, label=label, **kwargs)

        ax.set_xscale("log")
        ax.set_xlabel(f"Energy [{self.data.axis('energy').unit}]")
        ax.set_ylabel(f"Effective Area [{self.data.data.unit}]")
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
        energy : `~astropy.units.Quantity`
            Energy

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        if energy is None:
            e_min, e_max = np.log10(self.data.axis("energy").center.value[[0, -1]])
            energy = np.logspace(e_min, e_max, 4) * self.data.axis("energy").unit

        if offset is None:
            offset = self.data.axis("offset").center

        for ee in energy:
            area = self.data.evaluate(offset=offset, energy=ee)
            area /= np.nanmax(area)
            if np.isnan(area).all():
                continue
            label = f"energy = {ee:.1f}"
            ax.plot(offset, area, label=label, **kwargs)

        ax.set_ylim(0, 1.1)
        ax.set_xlabel(f"Offset ({self.data.axis('offset').unit})")
        ax.set_ylabel("Relative Effective Area")
        ax.legend(loc="best")

        return ax

    def plot(self, ax=None, add_cbar=True, **kwargs):
        """Plot effective area image."""
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        energy = self.data.axis("energy").edges
        offset = self.data.axis("offset").edges
        aeff = self.data.evaluate(offset=offset, energy=energy[:, np.newaxis])

        vmin, vmax = np.nanmin(aeff.value), np.nanmax(aeff.value)

        kwargs.setdefault("cmap", "GnBu")
        kwargs.setdefault("edgecolors", "face")
        kwargs.setdefault("vmin", vmin)
        kwargs.setdefault("vmax", vmax)

        caxes = ax.pcolormesh(energy.value, offset.value, aeff.value.T, **kwargs)

        ax.set_xscale("log")
        ax.set_ylabel(f"Offset ({offset.unit})")
        ax.set_xlabel(f"Energy ({energy.unit})")

        xmin, xmax = energy.value.min(), energy.value.max()
        ax.set_xlim(xmin, xmax)

        if add_cbar:
            label = f"Effective Area ({aeff.unit})"
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

        energy = self.data.axis("energy").edges
        theta = self.data.axis("offset").edges

        table = Table(meta=meta)
        table["ENERG_LO"] = energy[:-1][np.newaxis]
        table["ENERG_HI"] = energy[1:][np.newaxis]
        table["THETA_LO"] = theta[:-1][np.newaxis]
        table["THETA_HI"] = theta[1:][np.newaxis]
        table["EFFAREA"] = self.data.data.T[np.newaxis]
        return table

    def to_fits(self, name="EFFECTIVE AREA"):
        """Convert to `~astropy.io.fits.BinTableHDU`."""
        return fits.BinTableHDU(self.to_table(), name=name)
