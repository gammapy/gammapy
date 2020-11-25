# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from pathlib import Path
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from gammapy.data import GTI
from gammapy.irf import EDispKernel, EDispKernelMap
from gammapy.maps import RegionNDMap
from gammapy.utils.scripts import make_name, make_path
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
        ax_spectrum.label_outer()

        self.plot_residuals_spectral(ax_residuals, **kwargs_residuals)
        method = kwargs_residuals.get("method", "diff")
        label = self._residuals_labels[method]
        ax_residuals.set_ylabel(f"Residuals\n{label}")

        return ax_spectrum, ax_residuals

    @property
    def _energy_unit(self):
        return self._geom.axes[0].unit

    def _plot_energy_range(self, ax):
        energy_min, energy_max = self.energy_range
        kwargs = {"color": "black", "linestyle": "dashed"}
        ax.axvline(energy_min.to_value(self._energy_unit), label="fit range", **kwargs)
        ax.axvline(energy_max.to_value(self._energy_unit), **kwargs)

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
        ax = self.counts.plot_hist(ax, **plot_kwargs)

        plot_kwargs = kwargs.copy()
        plot_kwargs.update(kwargs_background)

        plot_kwargs.setdefault("label", "Background")
        self.background.plot_hist(ax, **plot_kwargs)

        self._plot_energy_range(ax)
        energy_min, energy_max = self.energy_range
        ax.set_xlim(0.7 * energy_min.value, 1.3 * energy_max.value)

        ax.legend(numpoints=1)
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
        ax = self.excess.plot(
            ax, yerr=np.sqrt(np.abs(self.excess.data.flatten())), **plot_kwargs
        )

        plot_kwargs = kwargs.copy()
        plot_kwargs.update(kwargs_npred_signal)
        plot_kwargs.setdefault("label", "Predicted signal counts")
        self.npred_signal().plot_hist(ax, **plot_kwargs)

        self._plot_energy_range(ax)
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

        ax2.set_title("Exposure")
        self.exposure.plot(ax2)
        self._plot_energy_range(ax2)
        energy_min, energy_max = self.energy_range
        ax2.set_xlim(0.7 * energy_min.value, 1.3 * energy_max.value)

        ax3.set_title("Energy Dispersion")
        if self.edisp is not None:
            kernel = self.edisp.get_edisp_kernel()
            kernel.plot_matrix(ax3, vmin=0, vmax=1)

        # TODO: optimize layout
        fig.subplots_adjust(wspace=0.3)
        return ax1, ax2, ax3


class SpectrumDataset(PlotMixin, MapDataset):
    stat_type = "cash"
    tag = "SpectrumDataset"

    def write(self, *args, **kwargs):
        raise NotImplementedError

    def read(self, *args, **kwargs):
        raise NotImplementedError

    def to_hdulist(self, *args, **kwargs):
        raise NotImplementedError

    def from_hdulist(self, *args, **kwargs):
        raise NotImplementedError

    def from_dict(self, *args, **kwargs):
        raise NotImplementedError

    def cutout(self, *args, **kwargs):
        raise NotImplementedError

    def plot_residuals_spatial(self, *args, **kwargs):
        raise NotImplementedError


class SpectrumDatasetOnOff(PlotMixin, MapDatasetOnOff):
    stat_type = "wstat"
    tag = "SpectrumDatasetOnOff"

    def cutout(self, *args, **kwargs):
        raise NotImplementedError

    def plot_residuals_spatial(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def read(cls, filename):
        """Read from file

        For now, filename is assumed to the name of a PHA file where BKG file, ARF, and RMF names
        must be set in the PHA header and be present in the same folder

        Parameters
        ----------
        filename : str
            OGIP PHA file to read
        """
        raise NotImplementedError(
            "To read from an OGIP fits file use SpectrumDatasetOnOff.from_ogip_files."
        )

    def to_ogip_files(self, outdir=None, use_sherpa=False, overwrite=False):
        """Write OGIP files.

        If you want to use the written files with Sherpa you have to set the
        ``use_sherpa`` flag. Then all files will be written in units 'keV' and
        'cm2'.

        The naming scheme is fixed, with {name} the dataset name:

        * PHA file is named pha_obs{name}.fits
        * BKG file is named bkg_obs{name}.fits
        * ARF file is named arf_obs{name}.fits
        * RMF file is named rmf_obs{name}.fits

        Parameters
        ----------
        outdir : `pathlib.Path`
            output directory, default: pwd
        use_sherpa : bool, optional
            Write Sherpa compliant files, default: False
        overwrite : bool
            Overwrite existing files?
        """
        # TODO: refactor and reduce amount of code duplication
        outdir = Path.cwd() if outdir is None else make_path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        phafile = f"pha_obs{self.name}.fits"

        bkgfile = phafile.replace("pha", "bkg")
        arffile = phafile.replace("pha", "arf")
        rmffile = phafile.replace("pha", "rmf")

        counts_table = self.counts.to_table()
        counts_table["QUALITY"] = np.logical_not(self.mask_safe.data[:, 0, 0])
        counts_table["BACKSCAL"] = self.acceptance.data[:, 0, 0]
        counts_table["AREASCAL"] = np.ones(self.acceptance.data.size)
        meta = self._ogip_meta()

        meta["respfile"] = rmffile
        meta["backfile"] = bkgfile
        meta["ancrfile"] = arffile
        meta["hduclas2"] = "TOTAL"
        counts_table.meta = meta

        name = counts_table.meta["name"]
        hdu = fits.BinTableHDU(counts_table, name=name)

        energy_axis = self.counts.geom.axes[0]

        hdu_format = "ogip-sherpa" if use_sherpa else "ogip"

        hdulist = fits.HDUList(
            [fits.PrimaryHDU(), hdu, energy_axis.to_table_hdu(format=hdu_format)]
        )

        if self.gti is not None:
            hdu = fits.BinTableHDU(self.gti.table, name="GTI")
            hdulist.append(hdu)

        if self.counts.geom._region is not None and self.counts.geom.wcs is not None:
            region_table = self.counts.geom._to_region_table()
            region_hdu = fits.BinTableHDU(region_table, name="REGION")
            hdulist.append(region_hdu)

        hdulist.writeto(str(outdir / phafile), overwrite=overwrite)

        aeff = self.exposure / self.exposure.meta["livetime"]

        aeff.write(
            outdir / arffile,
            overwrite=overwrite,
            format=hdu_format,
            ogip_column="SPECRESP",
        )

        if self.counts_off is not None:
            counts_off_table = self.counts_off.to_table()
            counts_off_table["QUALITY"] = np.logical_not(self.mask_safe.data[:, 0, 0])
            counts_off_table["BACKSCAL"] = self.acceptance_off.data[:, 0, 0]
            counts_off_table["AREASCAL"] = np.ones(self.acceptance.data.size)
            meta = self._ogip_meta()
            meta["hduclas2"] = "BKG"

            counts_off_table.meta = meta
            name = counts_off_table.meta["name"]
            hdu = fits.BinTableHDU(counts_off_table, name=name)
            hdulist = fits.HDUList(
                [fits.PrimaryHDU(), hdu, energy_axis.to_table_hdu(format=hdu_format)]
            )
            if (
                self.counts_off.geom._region is not None
                and self.counts_off.geom.wcs is not None
            ):
                region_table = self.counts_off.geom._to_region_table()
                region_hdu = fits.BinTableHDU(region_table, name="REGION")
                hdulist.append(region_hdu)

            hdulist.writeto(str(outdir / bkgfile), overwrite=overwrite)

        if self.edisp is not None:
            kernel = self.edisp.get_edisp_kernel()
            kernel.write(outdir / rmffile, overwrite=overwrite, use_sherpa=use_sherpa)

    def _ogip_meta(self):
        """Meta info for the OGIP data format"""
        try:
            livetime = self.exposure.meta["livetime"]
        except KeyError:
            raise ValueError(
                "Storing in ogip format require the livetime "
                "to be defined in the exposure meta data"
            )
        return {
            "name": "SPECTRUM",
            "hduclass": "OGIP",
            "hduclas1": "SPECTRUM",
            "corrscal": "",
            "chantype": "PHA",
            "detchans": self.counts.geom.axes[0].nbin,
            "filter": "None",
            "corrfile": "",
            "poisserr": True,
            "hduclas3": "COUNT",
            "hduclas4": "TYPE:1",
            "lo_thres": self.energy_range[0].to_value("TeV"),
            "hi_thres": self.energy_range[1].to_value("TeV"),
            "exposure": livetime.to_value("s"),
            "obs_id": self.name,
        }

    @classmethod
    def from_ogip_files(cls, filename):
        """Read `~gammapy.datasets.SpectrumDatasetOnOff` from OGIP files.

        BKG file, ARF, and RMF must be set in the PHA header and be present in
        the same folder.

        The naming scheme is fixed to the following scheme:

        * PHA file is named ``pha_obs{name}.fits``
        * BKG file is named ``bkg_obs{name}.fits``
        * ARF file is named ``arf_obs{name}.fits``
        * RMF file is named ``rmf_obs{name}.fits``
          with ``{name}`` the dataset name.

        Parameters
        ----------
        filename : str
            OGIP PHA file to read
        """
        filename = make_path(filename)
        dirname = filename.parent

        with fits.open(str(filename), memmap=False) as hdulist:
            counts = RegionNDMap.from_hdulist(hdulist, format="ogip")
            acceptance = RegionNDMap.from_hdulist(
                hdulist, format="ogip", ogip_column="BACKSCAL"
            )
            livetime = counts.meta["EXPOSURE"] * u.s

            if "GTI" in hdulist:
                gti = GTI(Table.read(hdulist["GTI"]))
            else:
                gti = None

            mask_safe = RegionNDMap.from_hdulist(
                hdulist, format="ogip", ogip_column="QUALITY"
            )
            mask_safe.data = np.logical_not(mask_safe.data)

        phafile = filename.name

        try:
            rmffile = phafile.replace("pha", "rmf")
            kernel = EDispKernel.read(dirname / rmffile)
            edisp = EDispKernelMap.from_edisp_kernel(kernel, geom=counts.geom)

        except OSError:
            # TODO : Add logger and echo warning
            edisp = None

        try:
            bkgfile = phafile.replace("pha", "bkg")
            with fits.open(str(dirname / bkgfile), memmap=False) as hdulist:
                counts_off = RegionNDMap.from_hdulist(hdulist, format="ogip")
                acceptance_off = RegionNDMap.from_hdulist(
                    hdulist, ogip_column="BACKSCAL"
                )
        except OSError:
            # TODO : Add logger and echo warning
            counts_off, acceptance_off = None, None

        arffile = phafile.replace("pha", "arf")
        aeff = RegionNDMap.read(dirname / arffile, format="ogip-arf")
        exposure = aeff * livetime
        exposure.meta["livetime"] = livetime

        if edisp is not None:
            edisp.exposure_map.data = exposure.data[:, :, np.newaxis, :]

        return cls(
            counts=counts,
            exposure=exposure,
            counts_off=counts_off,
            edisp=edisp,
            mask_safe=mask_safe,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
            name=str(counts.meta["OBS_ID"]),
            gti=gti,
        )

    def to_dict(self, filename, *args, **kwargs):
        """Convert to dict for YAML serialization."""
        outdir = Path(filename).parent
        filename = str(outdir / f"pha_obs{self.name}.fits")

        return {"name": self.name, "type": self.tag, "filename": filename}

    def write(self, filename, overwrite):
        """Write spectrum dataset on off to file.

        Currently only the OGIP format is supported

        Parameters
        ----------
        filename : str
            Filename to write to.
        overwrite : bool
            Overwrite existing file.
        """
        outdir = Path(filename).parent
        self.to_ogip_files(outdir=outdir, overwrite=overwrite)

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
        dataset = cls.from_ogip_files(filename=filename)
        dataset.mask_fit = None
        return dataset

    @classmethod
    def from_spectrum_dataset(
        cls, **kwargs
    ):
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
        """ Convert a SpectrumDatasetOnOff to a SpectrumDataset
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
        name = make_name(name)
        return SpectrumDataset(
            counts=self.counts,
            exposure=self.exposure,
            edisp=self.edisp,
            name=name,
            gti=self.gti,
            mask_fit=self.mask_fit,
            mask_safe=self.mask_safe,
            meta_table=self.meta_table,
            background=self.background,
        )
