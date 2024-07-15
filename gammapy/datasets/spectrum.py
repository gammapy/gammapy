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
    """Plot mixin for the spectral datasets."""

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
        ax_spectrum : `~matplotlib.axes.Axes`, optional
            Axes to plot spectrum on. Default is None.
        ax_residuals : `~matplotlib.axes.Axes`, optional
            Axes to plot residuals on. Default is None.
        kwargs_spectrum : dict, optional
            Keyword arguments passed to `~SpectrumDataset.plot_excess`. Default is None.
        kwargs_residuals : dict, optional
            Keyword arguments passed to `~SpectrumDataset.plot_residuals_spectral`. Default is None.

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
        >>> kwargs_spectrum = {"kwargs_excess":kwargs_excess, "kwargs_npred_signal":kwargs_npred_signal}
        >>> kwargs_residuals = {"color": "black", "markersize":4, "marker":'s', }
        >>> dataset.plot_fit(kwargs_residuals=kwargs_residuals, kwargs_spectrum=kwargs_spectrum)  # doctest: +SKIP
        """
        gs = GridSpec(7, 1)
        bool_visible_xticklabel = not (ax_spectrum is None and ax_residuals is None)
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
        plt.setp(ax_spectrum.get_xticklabels(), visible=bool_visible_xticklabel)

        return ax_spectrum, ax_residuals

    def plot_counts(
        self, ax=None, kwargs_counts=None, kwargs_background=None, **kwargs
    ):
        """Plot counts and background.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axes to plot on. Default is None.
        kwargs_counts : dict, optional
            Keyword arguments passed to `~matplotlib.axes.Axes.hist` for the counts. Default is None.
        kwargs_background : dict, optional
            Keyword arguments passed to `~matplotlib.axes.Axes.hist` for the background. Default is None.
        **kwargs : dict, optional
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
        """Plot safe mask and fit mask.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axes to plot on. Default is None.
        kwargs_fit : dict, optional
            Keyword arguments passed to `~RegionNDMap.plot_mask()` for mask fit. Default is None.
        kwargs_safe : dict, optional
            Keyword arguments passed to `~RegionNDMap.plot_mask()` for mask safe. Default is None.

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
        >>> dataset.plot_masks(ax=ax, kwargs_fit=kwargs_fit, kwargs_safe=kwargs_safe)  # doctest: +SKIP
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
        ax : `~matplotlib.axes.Axes`, optional
            Axes to plot on. Default is None.
        kwargs_excess : dict, optional
            Keyword arguments passed to `~matplotlib.axes.Axes.errorbar` for
            the excess. Default is None.
        kwargs_npred_signal : dict, optional
            Keyword arguments passed to `~matplotlib.axes.Axes.hist` for the
            predicted signal. Default is None.
        **kwargs : dict, optional
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
        >>> dataset.plot_excess(kwargs_excess=kwargs_excess, kwargs_npred_signal=kwargs_npred_signal)  # doctest: +SKIP
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
            Size of the figure. Default is (16, 4).

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
    """Main dataset for spectrum fitting (1D analysis).
    It bundles together binned counts, background, IRFs into `~gammapy.maps.RegionNDMap` (a Map with only one spatial bin).
    A safe mask and a fit mask can be added to exclude bins during the analysis.
    If models are assigned to it, it can compute the predicted number of counts and the statistic function,
    here the Cash statistic (see `~gammapy.stats.cash`).

    For more information see :ref:`datasets`.
    """

    stat_type = "cash"
    tag = "SpectrumDataset"

    def cutout(self, *args, **kwargs):
        """Not supported for `SpectrumDataset`"""
        raise NotImplementedError("Method not supported on a spectrum dataset")

    def plot_residuals_spatial(self, *args, **kwargs):
        """Not supported for `SpectrumDataset`"""
        raise NotImplementedError("Method not supported on a spectrum dataset")

    def to_spectrum_dataset(self, *args, **kwargs):
        """Not supported for `SpectrumDataset`"""
        raise NotImplementedError("Already a Spectrum Dataset. Method not supported")


class SpectrumDatasetOnOff(PlotMixin, MapDatasetOnOff):
    """Spectrum dataset for 1D on-off likelihood fitting.
    It bundles together the binned on and off counts, the binned IRFs as well as the on and off acceptances.

    A fit mask can be added to exclude bins during the analysis.

    It uses the Wstat statistic (see `~gammapy.stats.wstat`).

    For more information see :ref:`datasets`.
    """

    stat_type = "wstat"
    tag = "SpectrumDatasetOnOff"

    def cutout(self, *args, **kwargs):
        """Not supported for `SpectrumDatasetOnOff`."""
        raise NotImplementedError("Method not supported on a spectrum dataset")

    def plot_residuals_spatial(self, *args, **kwargs):
        """Not supported for `SpectrumDatasetOnOff`."""
        raise NotImplementedError("Method not supported on a spectrum dataset")

    @classmethod
    def read(cls, filename, format="ogip", checksum=False, **kwargs):
        """Read from file.

        For OGIP formats, filename is the name of a PHA file. The BKG, ARF, and RMF file names must be
        set in the PHA header and the files must be present in the same folder. For details, see `OGIPDatasetReader.read`.

        For the GADF format, a MapDataset serialisation is used.

        Parameters
        ----------
        filename : `~pathlib.Path` or str
            OGIP PHA file to read.
        format : {"ogip", "ogip-sherpa", "gadf"}
            Format to use. Default is "ogip".
        checksum : bool
            If True checks both DATASUM and CHECKSUM cards in the file headers. Default is False.
        kwargs : dict, optional
            Keyword arguments passed to `MapDataset.read`.
        """
        from .io import OGIPDatasetReader

        if format == "gadf":
            return super().read(filename, format="gadf", checksum=checksum, **kwargs)

        reader = OGIPDatasetReader(filename=filename, checksum=checksum)
        return reader.read()

    def write(self, filename, overwrite=False, format="ogip", checksum=False):
        """Write spectrum dataset on off to file.

        Can be serialised either as a `MapDataset` with a `RegionGeom`
        following the GADF specifications, or as per the OGIP format.
        For OGIP formats specs, see `OGIPDatasetWriter`.

        Parameters
        ----------
        filename : `~pathlib.Path` or str
            Filename to write to.
        overwrite : bool, optional
            Overwrite existing file. Default is False.
        format : {"ogip", "ogip-sherpa", "gadf"}
            Format to use. Default is "ogip".
        checksum : bool
            When True adds both DATASUM and CHECKSUM cards to the headers written to the file.
            Default is False.
        """
        from .io import OGIPDatasetWriter

        if format == "gadf":
            super().write(filename=filename, overwrite=overwrite, checksum=checksum)
        elif format in ["ogip", "ogip-sherpa"]:
            writer = OGIPDatasetWriter(
                filename=filename, format=format, overwrite=overwrite, checksum=checksum
            )
            writer.write(self)
        else:
            raise ValueError(f"{format} is not a valid serialisation format")

    @classmethod
    def from_dict(cls, data, **kwargs):
        """Create spectrum dataset from dict.
        Reads file from the disk as specified in the dict.

        Parameters
        ----------
        data : dict
            Dictionary containing data to create dataset from.

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
        """Create a SpectrumDatasetOnOff from a `SpectrumDataset` dataset.

        Parameters
        ----------
        dataset : `SpectrumDataset`
            Spectrum dataset defining counts, edisp, exposure etc.
        acceptance : `~numpy.array` or float
            Relative background efficiency in the on region.
        acceptance_off : `~numpy.array` or float
            Relative background efficiency in the off region.
        counts_off : `~gammapy.maps.RegionNDMap`
            Off counts spectrum. If the dataset provides a background model,
            and no off counts are defined. The off counts are deferred from
            counts_off / alpha.

        Returns
        -------
        dataset : `SpectrumDatasetOnOff`
            Spectrum dataset on off.
        """
        return cls.from_map_dataset(**kwargs)

    def to_spectrum_dataset(self, name=None):
        """Convert a SpectrumDatasetOnOff to a SpectrumDataset.
        The background model template is taken as alpha*counts_off.

        Parameters
        ----------
        name : str, optional
            Name of the new dataset. Default is None.

        Returns
        -------
        dataset : `SpectrumDataset`
            SpectrumDataset with Cash statistic.
        """
        return self.to_map_dataset(name=name).to_spectrum_dataset(on_region=None)
