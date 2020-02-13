# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
import logging
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from regions import fits_region_objects_to_table, FITSRegionParser
from gammapy.maps import MapAxis
from gammapy.maps.utils import edges_from_lo_hi
from gammapy.utils.fits import ebounds_to_energy_axis, energy_axis_to_ebounds
from gammapy.utils.regions import compound_region_to_list, list_to_compound_region
from gammapy.utils.scripts import make_path

__all__ = ["CountsSpectrum"]

log = logging.getLogger(__name__)


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
    region : `~regions.SkyRegion`
        Region the spectrum is defined for.
    wcs : `~astropy.wcs.WCS`
        the wcs system used to perform region based event selection

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

    def __init__(self, energy_lo, energy_hi, data=None, unit="", region=None, wcs=None):
        e_edges = edges_from_lo_hi(energy_lo, energy_hi)
        self.energy = MapAxis.from_edges(e_edges, interp="log", name="energy")

        if data is None:
            data = np.zeros(self.energy.nbin)

        self.data = np.array(data)
        if not self.energy.nbin == self.data.size:
            raise ValueError("Incompatible data and energy axis size.")

        self.unit = u.Unit(unit)
        self.region = region
        self.wcs = wcs

    @property
    def quantity(self):
        return self.data * self.unit

    @quantity.setter
    def quantity(self, quantity):
        self.data = quantity.value
        self.unit = quantity.unit

    @staticmethod
    def read_region_table(hdu):
        """Read region table and convert it to region list."""
        region_table = Table.read(hdu)
        parser = FITSRegionParser(region_table)
        pix_region = parser.shapes.to_regions()
        wcs = WCS(region_table.meta)
        regions = []
        for reg in pix_region:
            regions.append(reg.to_sky(wcs))
        region = list_to_compound_region(regions)
        return region, wcs

    @classmethod
    def from_hdulist(cls, hdulist, hdu1="COUNTS", hdu2="EBOUNDS", hdu3="REGION"):
        """Read from HDU list in OGIP format."""
        table = Table.read(hdulist[hdu1])
        counts = table["COUNTS"].data
        ebounds = ebounds_to_energy_axis(hdulist[hdu2])

        # TODO: add region serilisation
        region = None
        wcs = None
        if hdu3 in hdulist:
            region, wcs = cls.read_region_table(hdulist[hdu3])

        return cls(
            data=counts,
            energy_lo=ebounds[:-1],
            energy_hi=ebounds[1:],
            region=region,
            wcs=wcs,
        )

    @classmethod
    def read(cls, filename, hdu1="COUNTS", hdu2="EBOUNDS", hdu3="REGION"):
        """Read from file."""
        with fits.open(make_path(filename), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist, hdu1=hdu1, hdu2=hdu2, hdu3=hdu3)

    def to_table(self):
        """Convert to `~astropy.table.Table`.

        Data format specification: :ref:`gadf:ogip-pha`
        """
        channel = np.arange(self.energy.nbin, dtype=np.int16)
        counts = np.array(self.data, dtype=np.int32)

        names = ["CHANNEL", "COUNTS"]
        meta = {"name": "COUNTS"}

        return Table([channel, counts], names=names, meta=meta)

    def _to_region_table(self):
        """Export region to a FITS region table."""
        region_list = compound_region_to_list(self.region)
        pixel_region_list = []
        for reg in region_list:
            pixel_region_list.append(reg.to_pixel(self.wcs))
        table = fits_region_objects_to_table(pixel_region_list)
        table.meta.update(self.wcs.to_header())
        return table

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

        region_table = self._to_region_table()
        region_hdu = fits.BinTableHDU(region_table, name="REGION")
        return fits.HDUList([fits.PrimaryHDU(), hdu, ebounds, region_hdu])

    def write(self, filename, use_sherpa=False, **kwargs):
        """Write to file."""
        filename = make_path(filename)
        self.to_hdulist(use_sherpa=use_sherpa).writeto(filename, **kwargs)

    def fill_events(self, events):
        """Fill events (`gammapy.data.EventList`)."""
        self.fill_energy(events.energy)

    def fill_energy(self, energy):
        """Fill energy values (`~astropy.units.Quantity`)"""
        energy = energy.to_value(self.energy.unit)
        edges = self.energy.edges.to_value(self.energy.unit)
        self.data = np.histogram(energy, edges)[0]

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
        **kwargs,
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
        ax.set_xlabel(f"Energy [{energy_unit}]")
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
        ax.set_xlabel(f"Energy [{energy_unit}]")
        ax.set_ylabel("Counts")
        ax.set_xscale("log")
        return ax

    def plot_region(self, ax=None, **kwargs):
        """Plot region

        Parameters
        ----------
        ax : `~astropy.vizualisation.WCSAxes`
            Axes to plot on.
        **kwargs : dict
            Keyword arguments forwarded to `~regions.PixelRegion.as_artist`
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection

        ax = plt.gca() or ax
        regions = compound_region_to_list(self.region)
        artists = [region.to_pixel(wcs=ax.wcs).as_artist() for region in regions]

        patches = PatchCollection(artists, **kwargs)
        ax.add_collection(patches)
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
        from gammapy.extern.skimage import block_reduce

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


class SpectrumEvaluator:
    """Calculate number of predicted counts (``npred``).

    The true and reconstructed energy binning are inferred from the provided IRFs.

    Parameters
    ----------
    model : `~gammapy.modeling.models.SkyModel`
        Spectral model
    aeff : `~gammapy.irf.EffectiveAreaTable`
        EffectiveArea
    edisp : `~gammapy.irf.EDispKernel`, optional
        Energy dispersion kernel.
    livetime : `~astropy.units.Quantity`
        Observation duration (may be contained in aeff)
    e_true : `~astropy.units.Quantity`, optional
        Desired energy axis of the prediced counts vector if no IRFs are given

    Examples
    --------
    Calculate predicted counts in a desired reconstruced energy binning

    .. plot::
        :include-source:

        import numpy as np
        import astropy.units as u
        import matplotlib.pyplot as plt
        from gammapy.irf import EffectiveAreaTable, EDispKernel
        from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
        from gammapy.spectrum import SpectrumEvaluator

        e_true = np.logspace(-2, 2.5, 109) * u.TeV
        e_reco = np.logspace(-2, 2, 73) * u.TeV

        aeff = EffectiveAreaTable.from_parametrization(energy=e_true)
        edisp = EDispKernel.from_gauss(e_true=e_true, e_reco=e_reco, sigma=0.3, bias=0)

        pwl = PowerLawSpectralModel(index=2.3, amplitude="2.5e-12 cm-2 s-1 TeV-1", reference="1 TeV")
        model = SkyModel(spectral_model=pwl)

        predictor = SpectrumEvaluator(model=model, aeff=aeff, edisp=edisp, livetime="1 hour")
        predictor.compute_npred().plot_hist()
        plt.show()
    """

    def __init__(self, model, aeff=None, edisp=None, livetime=None):
        self.model = model
        self.aeff = aeff
        self.edisp = edisp
        self.livetime = livetime

    def compute_npred(self):
        e_true = self.aeff.energy.edges
        integral_flux = self.model.spectral_model.integral(
            emin=e_true[:-1], emax=e_true[1:], intervals=True
        )

        true_counts = self.apply_aeff(integral_flux)
        return self.apply_edisp(true_counts)

    def apply_aeff(self, integral_flux):
        if self.aeff is not None:
            cts = integral_flux * self.aeff.data.data
        else:
            cts = integral_flux

        # Multiply with livetime if not already contained in aeff or model
        if cts.unit.is_equivalent("s-1"):
            cts *= self.livetime

        return cts.to("")

    def apply_edisp(self, true_counts):
        from . import CountsSpectrum

        if self.edisp is not None and self.model.apply_irf["edisp"] is True:
            cts = self.edisp.apply(true_counts)
            e_reco = self.edisp.e_reco.edges
        else:
            cts = true_counts
            e_reco = self.aeff.energy.edges

        return CountsSpectrum(data=cts, energy_lo=e_reco[:-1], energy_hi=e_reco[1:])
