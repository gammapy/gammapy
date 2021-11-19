# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from gammapy.utils.scripts import make_path
from .map import MapDataset, MapDatasetOnOff
from .utils import get_axes, get_figure

__all__ = ["SpectrumDatasetOnOff", "SpectrumDataset"]

log = logging.getLogger(__name__)


class PlotMixin:
    """Plot mixin for the spectral datasets"""

    def plot_fit(
        self,
        ax_spectrum=None,
        ax_residuals=None,
        kwargs_spectrum=None,
        kwargs_residuals=None,
    ):
        """Plot spectrum and residuals in two panels.

        Calls `~SpectrumDataset.plot_excess` and `~SpectrumDataset.plot_residuals`.

        Parameters
        ----------
        ax_spectrum : `~matplotlib.axes.Axes`
            Axes to plot spectrum on.
        ax_residuals : `~matplotlib.axes.Axes`
            Axes to plot residuals on.
        kwargs_spectrum : dict
            Keyword arguments passed to `~SpectrumDataset.plot_excess`.
        kwargs_residuals : dict
            Keyword arguments passed to `~SpectrumDataset.plot_residuals`.

        Returns
        -------
        ax_spectrum, ax_residuals : `~matplotlib.axes.Axes`
            Spectrum and residuals plots.
        """
        from matplotlib.gridspec import GridSpec

        gs = GridSpec(7, 1)
        ax_spectrum, ax_residuals = get_axes(
            ax_spectrum,
            ax_residuals,
            8,
            7,
            [gs[:5, :]],
            [gs[5:, :]],
            kwargs2={"sharex": ax_spectrum},
        )
        kwargs_spectrum = kwargs_spectrum or {}
        kwargs_residuals = kwargs_residuals or {}

        self.plot_excess(ax_spectrum, **kwargs_spectrum)

        self.plot_residuals_spectral(ax_residuals, **kwargs_residuals)

        method = kwargs_residuals.get("method", "diff")
        label = self._residuals_labels[method]
        ax_residuals.set_ylabel(f"Residuals\n{label}")

        return ax_spectrum, ax_residuals

    def plot_counts(
        self, ax=None, kwargs_counts=None, kwargs_background=None, **kwargs
    ):
        """Plot counts and background.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes to plot on.
        kwargs_counts: dict
            Keyword arguments passed to `~matplotlib.axes.Axes.hist` for the counts.
        kwargs_background: dict
            Keyword arguments passed to `~matplotlib.axes.Axes.hist` for the background.
        **kwargs: dict
            Keyword arguments passed to both `~matplotlib.axes.Axes.hist`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes object.
        """
        kwargs_counts = kwargs_counts or {}
        kwargs_background = kwargs_background or {}

        plot_kwargs = kwargs.copy()
        plot_kwargs.update(kwargs_counts)
        plot_kwargs.setdefault("label", "Counts")
        ax = self.counts.plot_hist(ax=ax, **plot_kwargs)

        plot_kwargs = kwargs.copy()
        plot_kwargs.update(kwargs_background)

        plot_kwargs.setdefault("label", "Background")
        self.background.plot_hist(ax=ax, **plot_kwargs)

        ax.legend(numpoints=1)
        return ax

    def plot_masks(self, ax=None, kwargs_fit=None, kwargs_safe=None):
        """Plot mask safe and mask fit

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes to plot on.
        kwargs_fit: dict
            Keyword arguments passed to `~RegionNDMap.plot_mask()` for mask fit.
        kwargs_safe: dict
            Keyword arguments passed to `~RegionNDMap.plot_mask()` for mask safe.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes object.
        """

        kwargs_fit = kwargs_fit or {}
        kwargs_safe = kwargs_safe or {}

        kwargs_fit.setdefault("label", "Mask fit")
        kwargs_fit.setdefault("color", "tab:green")
        kwargs_safe.setdefault("label", "Mask safe")
        kwargs_safe.setdefault("color", "black")

        if self.mask_fit:
            self.mask_fit.plot_mask(ax=ax, **kwargs_fit)

        if self.mask_safe:
            self.mask_safe.plot_mask(ax=ax, **kwargs_safe)

        return ax

    def plot_excess(
        self, ax=None, kwargs_excess=None, kwargs_npred_signal=None, **kwargs
    ):
        """Plot excess and predicted signal.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes to plot on.
        kwargs_excess: dict
            Keyword arguments passed to `~matplotlib.axes.Axes.errorbar` for
            the excess.
        kwargs_npred_signal : dict
            Keyword arguments passed to `~matplotlib.axes.Axes.hist` for the
            predicted signal.
        **kwargs: dict
            Keyword arguments passed to both plot methods.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes object.
        """
        kwargs_excess = kwargs_excess or {}
        kwargs_npred_signal = kwargs_npred_signal or {}

        plot_kwargs = kwargs.copy()
        plot_kwargs.update(kwargs_excess)
        plot_kwargs.setdefault("label", "Excess counts")
        ax = self.excess.plot(ax, yerr=np.sqrt(np.abs(self.excess.data)), **plot_kwargs)

        plot_kwargs = kwargs.copy()
        plot_kwargs.update(kwargs_npred_signal)
        plot_kwargs.setdefault("label", "Predicted signal counts")
        self.npred_signal().plot_hist(ax, **plot_kwargs)

        ax.legend(numpoints=1)
        return ax

    def peek(self, fig=None):
        """Quick-look summary plots.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            Figure to add AxesSubplot on.

        Returns
        -------
        ax1, ax2, ax3 : `~matplotlib.axes.AxesSubplot`
            Counts, effective area and energy dispersion subplots.
        """
        fig = get_figure(fig, 16, 4)
        ax1, ax2, ax3 = fig.subplots(1, 3)

        ax1.set_title("Counts")
        self.plot_counts(ax1)
        self.plot_masks(ax=ax1)
        ax1.legend()

        ax2.set_title("Exposure")
        self.exposure.plot(ax2, ls="-", markersize=0, xerr=None)

        ax3.set_title("Energy Dispersion")

        if self.edisp is not None:
            kernel = self.edisp.get_edisp_kernel()
            kernel.plot_matrix(ax=ax3, add_cbar=True)

        return ax1, ax2, ax3


class SpectrumDataset(PlotMixin, MapDataset):
    stat_type = "cash"
    tag = "SpectrumDataset"

    def write(self, *args, **kwargs):
        raise NotImplementedError("Method not supported on a spectrum dataset")

    def read(self, *args, **kwargs):
        raise NotImplementedError("Method not supported on a spectrum dataset")

    def to_hdulist(self, *args, **kwargs):
        raise NotImplementedError("Method not supported on a spectrum dataset")

    def from_hdulist(self, *args, **kwargs):
        raise NotImplementedError("Method not supported on a spectrum dataset")

    def from_dict(self, *args, **kwargs):
        raise NotImplementedError("Method not supported on a spectrum dataset")

    def cutout(self, *args, **kwargs):
        raise NotImplementedError("Method not supported on a spectrum dataset")

    def plot_residuals_spatial(self, *args, **kwargs):
        raise NotImplementedError("Method not supported on a spectrum dataset")

    def to_spectrum_dataset(self, *args, **kwargs):
        raise NotImplementedError("Already a Spectrum Dataset. Method not supported")


class SpectrumDatasetOnOff(PlotMixin, MapDatasetOnOff):
    stat_type = "wstat"
    tag = "SpectrumDatasetOnOff"

    def cutout(self, *args, **kwargs):
        raise NotImplementedError("Method not supported on a spectrum dataset")

    def plot_residuals_spatial(self, *args, **kwargs):
        raise NotImplementedError("Method not supported on a spectrum dataset")

    def to_hdulist(self, *args, **kwargs):
        raise NotImplementedError("Method not supported on a spectrum dataset")

    def from_hdulist(self, *args, **kwargs):
        raise NotImplementedError("Method not supported on a spectrum dataset")

    @classmethod
    def read(cls, filename):
        """Read from file

        For now, filename is assumed to the name of a PHA file where BKG file, ARF, and RMF names
        must be set in the PHA header and be present in the same folder.

        For formats specs see `OGIPDatasetReader.read`

        Parameters
        ----------
        filename : `~pathlib.Path` or str
            OGIP PHA file to read
        """
        from .io import OGIPDatasetReader

        reader = OGIPDatasetReader(filename=filename)
        return reader.read()

    def write(self, filename, overwrite=False, format="ogip"):
        """Write spectrum dataset on off to file.

        Currently only the OGIP format is supported

        For formats specs see `OGIPDatasetWriter`

        Parameters
        ----------
        filename : `~pathlib.Path` or str
            Filename to write to.
        overwrite : bool
            Overwrite existing file.
        format : {"ogip", "ogip-sherpa"}
            Format to use.
        """
        from .io import OGIPDatasetWriter

        writer = OGIPDatasetWriter(
            filename=filename, format=format, overwrite=overwrite
        )
        writer.write(self)

    @classmethod
    def from_dict(cls, data, **kwargs):
        """Create flux point dataset from dict.

        Parameters
        ----------
        data : dict
            Dict containing data to create dataset from.

        Returns
        -------
        dataset : `SpectrumDatasetOnOff`
            Spectrum dataset on off.

        """

        filename = make_path(data["filename"])
        dataset = cls.read(filename=filename)
        dataset.mask_fit = None
        return dataset

    def to_dict(self):
        """Convert to dict for YAML serialization."""
        filename = f"pha_obs{self.name}.fits"
        return {"name": self.name, "type": self.tag, "filename": filename}

    @classmethod
    def from_spectrum_dataset(cls, **kwargs):
        """Create spectrum dataseton off from another dataset.

        Parameters
        ----------
        dataset : `SpectrumDataset`
            Spectrum dataset defining counts, edisp, exposure etc.
        acceptance : `~numpy.array` or float
            Relative background efficiency in the on region.
        acceptance_off : `~numpy.array` or float
            Relative background efficiency in the off region.
        counts_off : `~gammapy.maps.RegionNDMap`
            Off counts spectrum . If the dataset provides a background model,
            and no off counts are defined. The off counts are deferred from
            counts_off / alpha.

        Returns
        -------
        dataset : `SpectrumDatasetOnOff`
            Spectrum dataset on off.

        """
        return cls.from_map_dataset(**kwargs)

    def to_spectrum_dataset(self, name=None):
        """Convert a SpectrumDatasetOnOff to a SpectrumDataset
        The background model template is taken as alpha*counts_off

        Parameters
        ----------
        name: str
            Name of the new dataset

        Returns
        -------
        dataset: `SpectrumDataset`
            SpectrumDatset with cash statistics
        """
        return self.to_map_dataset(name=name).to_spectrum_dataset(on_region=None)
