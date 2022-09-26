# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from gammapy.utils.scripts import make_path
from .map import MapDataset, MapDatasetOnOff
from .utils import get_axes

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

        Calls `~SpectrumDataset.plot_excess` and `~SpectrumDataset.plot_residuals_spectral`.

        Parameters
        ----------
        ax_spectrum : `~matplotlib.axes.Axes`
            Axes to plot spectrum on.
        ax_residuals : `~matplotlib.axes.Axes`
            Axes to plot residuals on.
        kwargs_spectrum : dict
            Keyword arguments passed to `~SpectrumDataset.plot_excess`.
        kwargs_residuals : dict
            Keyword arguments passed to `~SpectrumDataset.plot_residuals_spectral`.

        Returns
        -------
        ax_spectrum, ax_residuals : `~matplotlib.axes.Axes`
            Spectrum and residuals plots.

        Examples
        --------
        >>> #Creating a spectral dataset
        >>> from gammapy.datasets import SpectrumDatasetOnOff
        >>> from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
        >>> filename = "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits"
        >>> dataset = SpectrumDatasetOnOff.read(filename)
        >>> p = PowerLawSpectralModel()
        >>> dataset.models = SkyModel(spectral_model=p)
        >>> # optional configurations
        >>> kwargs_excess = {"color": "blue", "markersize":8, "marker":'s', }
        >>> kwargs_npred_signal = {"color": "black", "ls":"--"}
        >>> kwargs_spectrum = {"kwargs_excess":kwargs_excess, "kwargs_npred_signal":kwargs_npred_signal}  # noqa: E501
        >>> kwargs_residuals = {"color": "black", "markersize":4, "marker":'s', }  # optional configuration  # noqa: E501
        >>> dataset.plot_fit(kwargs_residuals=kwargs_residuals, kwargs_spectrum=kwargs_spectrum)  # doctest: +SKIP  noqa: E501
        """
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

        Examples
        --------
        >>> # Reading a spectral dataset
        >>> from gammapy.datasets import SpectrumDatasetOnOff
        >>> filename = "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits"
        >>> dataset = SpectrumDatasetOnOff.read(filename)
        >>> dataset.mask_fit = dataset.mask_safe.copy()
        >>> dataset.mask_fit.data[40:46] = False  # setting dummy mask_fit for illustration
        >>> # Plot the masks on top of the counts histogram
        >>> kwargs_safe = {"color":"green", "alpha":0.2} #optinonal arguments to configure
        >>> kwargs_fit = {"color":"pink", "alpha":0.2}
        >>> ax=dataset.plot_counts()  # doctest: +SKIP
        >>> dataset.plot_masks(ax=ax, kwargs_fit=kwargs_fit, kwargs_safe=kwargs_safe)  # doctest: +SKIP  # noqa: E501
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

        ax.legend()
        return ax

    def plot_excess(
        self, ax=None, kwargs_excess=None, kwargs_npred_signal=None, **kwargs
    ):
        """Plot excess and predicted signal.

        The error bars are computed with a symmetric assumption on the excess.

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

        Examples
        --------
        >>> #Creating a spectral dataset
        >>> from gammapy.datasets import SpectrumDatasetOnOff
        >>> from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
        >>> filename = "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits"
        >>> dataset = SpectrumDatasetOnOff.read(filename)
        >>> p = PowerLawSpectralModel()
        >>> dataset.models = SkyModel(spectral_model=p)
        >>> #Plot the excess in blue and the npred in black dotted lines
        >>> kwargs_excess = {"color": "blue", "markersize":8, "marker":'s', }
        >>> kwargs_npred_signal = {"color": "black", "ls":"--"}
        >>> dataset.plot_excess(kwargs_excess=kwargs_excess, kwargs_npred_signal=kwargs_npred_signal)  # doctest: +SKIP  # noqa: E501
        """
        kwargs_excess = kwargs_excess or {}
        kwargs_npred_signal = kwargs_npred_signal or {}

        # Determine the uncertainty on the excess
        yerr = self._counts_statistic.error

        plot_kwargs = kwargs.copy()
        plot_kwargs.update(kwargs_excess)
        plot_kwargs.setdefault("label", "Excess counts")
        ax = self.excess.plot(ax, yerr=yerr, **plot_kwargs)

        plot_kwargs = kwargs.copy()
        plot_kwargs.update(kwargs_npred_signal)
        plot_kwargs.setdefault("label", "Predicted signal counts")
        self.npred_signal().plot_hist(ax, **plot_kwargs)

        ax.legend(numpoints=1)
        return ax

    def peek(self, figsize=(16, 4)):
        """Quick-look summary plots.

        Parameters
        ----------
        figsize : tuple
            Size of the figure.

        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

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


class SpectrumDataset(PlotMixin, MapDataset):
    stat_type = "cash"
    tag = "SpectrumDataset"

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

    @classmethod
    def read(cls, filename, format="ogip", **kwargs):
        """Read from file

        For OGIP formats, filename is assumed to the name of a PHA file where
        BKG file, ARF, and RMF names
        must be set in the PHA header and be present in the same folder.
        For details, see `OGIPDatasetReader.read`

        For the GADF format, a MapDataset serialisation is used

        Parameters
        ----------
        filename : `~pathlib.Path` or str
            OGIP PHA file to read
        format : {"ogip", "ogip-sherpa", "gadf"}
            Format to use.
        kwargs : arguments passed to `MapDataset.read`
        """
        from .io import OGIPDatasetReader

        if format == "gadf":
            return super().read(filename, format="gadf", **kwargs)

        reader = OGIPDatasetReader(filename=filename)
        return reader.read()

    def write(self, filename, overwrite=False, format="ogip"):
        """Write spectrum dataset on off to file.

        Can be serialised either as a `MapDataset` with a `RegionGeom`
        following the GADF specifications, or as per the OGIP format.
        For OGIP formats specs see `OGIPDatasetWriter`

        Parameters
        ----------
        filename : `~pathlib.Path` or str
            Filename to write to.
        overwrite : bool
            Overwrite existing file.
        format : {"ogip", "ogip-sherpa", "gadf"}
            Format to use.
        """
        from .io import OGIPDatasetWriter

        if format == "gadf":
            super().write(filename=filename, overwrite=overwrite)
        elif format in ["ogip", "ogip-sherpa"]:
            writer = OGIPDatasetWriter(
                filename=filename, format=format, overwrite=overwrite
            )
            writer.write(self)
        else:
            raise ValueError(f"{format} is not a valid serialisation format")

    @classmethod
    def from_dict(cls, data, **kwargs):
        """Create spectrum dataset from dict.
        Reads file from the disk as specified in the dict

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
