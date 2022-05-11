# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from gammapy.data import GTI
from gammapy.irf import EDispKernel, EDispKernelMap
from gammapy.maps import RegionNDMap
from gammapy.utils.scripts import make_path
from .spectrum import SpectrumDatasetOnOff

__all__ = [
    "DatasetReader",
    "DatasetWriter",
    "OGIPDatasetReader",
    "OGIPDatasetWriter",
]


class DatasetReader(abc.ABC):
    """Dataset reader base class"""

    @property
    @abc.abstractmethod
    def tag(self):
        pass

    @abc.abstractmethod
    def read(self):
        pass


class DatasetWriter(abc.ABC):
    """Dataset writer base class"""

    @property
    @abc.abstractmethod
    def tag(self):
        pass

    @abc.abstractmethod
    def write(self, dataset):
        pass


class OGIPDatasetWriter(DatasetWriter):
    """Write OGIP files.

    If you want to use the written files with Sherpa you have to use the
    ``ogip-sherpa`` format. Then all files will be written in units of 'keV' and
    'cm2'.

    The naming scheme is fixed as following:

    * PHA file is named filename.fits
    * BKG file is named filename_bkg.fits
    * ARF file is named filename_arf.fits
    * RMF file is named filename_rmf.fits

    Parameters
    ----------
    filename : `pathlib.Path` or str
        Filename.
    format : {"ogip", "ogip-sherpa"}
        Which format to use.
    overwrite : bool
        Overwrite existing files?
    """

    tag = ["ogip", "ogip-sherpa"]

    def __init__(self, filename, format="ogip", overwrite=False):
        filename = make_path(filename)
        filename.parent.mkdir(exist_ok=True, parents=True)

        self.filename = filename
        self.format = format
        self.overwrite = overwrite

    @staticmethod
    def get_filenames(filename):
        """Get filenames

        Parameters
        ----------
        filename : `~pathlib.Path`
            Filename

        Returns
        -------
        filenames : dict
            Dict of filenames.
        """
        suffix = "".join(filename.suffixes)
        name = filename.name.replace(suffix, "")
        name = f"{name}{{}}{suffix}"
        return {
            "respfile": name.format("_rmf"),
            "backfile": name.format("_bkg"),
            "ancrfile": name.format("_arf"),
        }

    def get_ogip_meta(self, dataset, is_bkg=False):
        """Meta info for the OGIP data format"""
        try:
            livetime = dataset.exposure.meta["livetime"]
        except KeyError:
            raise ValueError(
                "Storing in ogip format require the livetime "
                "to be defined in the exposure meta data"
            )

        hdu_class = "BKG" if is_bkg else "TOTAL"

        meta = {
            "HDUCLAS2": hdu_class,
            "HDUCLAS3": "COUNT",
            "HDUCLAS4": "TYPE:1",
            "EXPOSURE": livetime.to_value("s"),
            "OBS_ID": dataset.name,
        }

        filenames = OGIPDatasetWriter.get_filenames(self.filename)
        meta["ANCRFILE"] = filenames["ancrfile"]

        if dataset.edisp:
            meta["BACKFILE"] = filenames["backfile"]

        if dataset.counts_off:
            meta["RESPFILE"] = filenames["respfile"]

        return meta

    def write(self, dataset):
        """Write dataset to files

        Parameters
        ----------
        dataset : `SpectrumDatasetOnOff`
            Dataset to write
        """
        filenames = self.get_filenames(self.filename)

        self.write_pha(dataset, filename=self.filename)

        path = self.filename.parent
        self.write_arf(dataset, filename=path / filenames["ancrfile"])

        if dataset.counts_off:
            self.write_bkg(dataset, filename=path / filenames["backfile"])

        if dataset.edisp:
            self.write_rmf(dataset, filename=path / filenames["respfile"])

    def write_rmf(self, dataset, filename):
        """Write energy dispersion.

        Parameters
        ----------
        dataset : `SpectrumDatasetOnOff`
            Dataset to write
        filename : str or `Path`
            Filename to use.
        """
        kernel = dataset.edisp.get_edisp_kernel()
        kernel.write(filename=filename, overwrite=self.overwrite, format=self.format)

    def write_arf(self, dataset, filename):
        """Write effective area

        Parameters
        ----------
        dataset : `SpectrumDatasetOnOff`
            Dataset to write
        filename : str or `Path`
            Filename to use.

        """
        aeff = dataset.exposure / dataset.exposure.meta["livetime"]
        aeff.write(
            filename=filename,
            overwrite=self.overwrite,
            format=self.format.replace("ogip", "ogip-arf"),
        )

    def to_counts_hdulist(self, dataset, is_bkg=False):
        """Convert counts region map to hdulist

        Parameters
        ----------
        dataset : `SpectrumDatasetOnOff`
            Dataset to write
        is_bkg : bool
            Whether to use counts off.
        """
        counts = dataset.counts_off if is_bkg else dataset.counts
        acceptance = dataset.acceptance_off if is_bkg else dataset.acceptance

        hdulist = counts.to_hdulist(format=self.format)

        table = Table.read(hdulist["SPECTRUM"])
        meta = self.get_ogip_meta(dataset, is_bkg=is_bkg)

        if dataset.mask_safe is not None:
            mask_array = dataset.mask_safe.data[:, 0, 0]
        else:
            mask_array = np.ones(acceptance.data.size)

        table["QUALITY"] = np.logical_not(mask_array)
        del table.meta["QUALITY"]

        table["BACKSCAL"] = acceptance.data[:, 0, 0]
        del table.meta["BACKSCAL"]

        # adapt meta data
        table.meta.update(meta)
        hdulist["SPECTRUM"] = fits.BinTableHDU(table)
        return hdulist

    def write_pha(self, dataset, filename):
        """Write counts file

        Parameters
        ----------
        dataset : `SpectrumDatasetOnOff`
            Dataset to write
        filename : str or `Path`
            Filename to use.

        """
        hdulist = self.to_counts_hdulist(dataset)

        if dataset.gti:
            hdu = fits.BinTableHDU(dataset.gti.table, name="GTI")
            hdulist.append(hdu)

        hdulist.writeto(filename, overwrite=self.overwrite)

    def write_bkg(self, dataset, filename):
        """Write off counts file

        Parameters
        ----------
        dataset : `SpectrumDatasetOnOff`
            Dataset to write
        filename : str or `Path`
            Filename to use.
        """
        hdulist = self.to_counts_hdulist(dataset, is_bkg=True)
        hdulist.writeto(filename, overwrite=self.overwrite)


class OGIPDatasetReader(DatasetReader):
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
    filename : str or `~pathlib.Path`
        OGIP PHA file to read
    """

    tag = "ogip"

    def __init__(self, filename):
        self.filename = make_path(filename)

    def get_valid_path(self, filename):
        """Get absolute or relative path

        The relative path is with respect to the name of the reference file.

        Parameters
        ----------
        filename : str or `Path`
            Filename

        Returns
        -------
        filename : `Path`
            Valid path
        """
        filename = make_path(filename)

        if not filename.exists():
            return self.filename.parent / filename
        else:
            return filename

    def get_filenames(self, pha_meta):
        """Get filenames

        Parameters
        ----------
        pha_meta : dict
            Meta data from the PHA file

        Returns
        -------
        filenames : dict
            Dict with filenames of "arffile", "rmffile" (optional)
            and "bkgfile" (optional)
        """
        filenames = {"arffile": self.get_valid_path(pha_meta["ANCRFILE"])}

        if "BACKFILE" in pha_meta:
            filenames["bkgfile"] = self.get_valid_path(pha_meta["BACKFILE"])

        if "RESPFILE" in pha_meta:
            filenames["rmffile"] = self.get_valid_path(pha_meta["RESPFILE"])

        return filenames

    @staticmethod
    def read_pha(filename):
        """Read PHA file

        Parameters
        ----------
        filename : str or `Path`
            PHA file name

        Returns
        -------
        data : dict
            Dict with counts, acceptance and mask_safe
        """
        data = {}

        with fits.open(filename, memmap=False) as hdulist:
            data["counts"] = RegionNDMap.from_hdulist(hdulist, format="ogip")
            data["acceptance"] = RegionNDMap.from_hdulist(
                hdulist, format="ogip", ogip_column="BACKSCAL"
            )

            if "GTI" in hdulist:
                data["gti"] = GTI(Table.read(hdulist["GTI"]))

            data["mask_safe"] = RegionNDMap.from_hdulist(
                hdulist, format="ogip", ogip_column="QUALITY"
            )

        return data

    @staticmethod
    def read_bkg(filename):
        """Read PHA background file

        Parameters
        ----------
        filename : str or `Path`
            PHA file name

        Returns
        -------
        data : dict
            Dict with counts_off and acceptance_off
        """
        with fits.open(filename, memmap=False) as hdulist:
            counts_off = RegionNDMap.from_hdulist(hdulist, format="ogip")
            acceptance_off = RegionNDMap.from_hdulist(
                hdulist, ogip_column="BACKSCAL", format="ogip"
            )
        return {"counts_off": counts_off, "acceptance_off": acceptance_off}

    @staticmethod
    def read_rmf(filename, exposure):
        """Read RMF file

        Parameters
        ----------
        filename : str or `Path`
            PHA file name
        exposure : `RegionNDMap`
            Exposure map

        Returns
        -------
        data : `EDispKernelMap`
            Dict with edisp
        """
        kernel = EDispKernel.read(filename)
        edisp = EDispKernelMap.from_edisp_kernel(kernel, geom=exposure.geom)

        # TODO: resolve this separate handling of exposure for edisp
        edisp.exposure_map.data = exposure.data[:, :, np.newaxis, :]
        return edisp

    @staticmethod
    def read_arf(filename, livetime):
        """Read ARF file

        Parameters
        ----------
        filename : str or `Path`
            PHA file name
        livetime : `Quantity`
            Livetime

        Returns
        -------
        data : `RegionNDMap`
            Exposure map
        """
        aeff = RegionNDMap.read(filename, format="ogip-arf")
        exposure = aeff * livetime
        exposure.meta["livetime"] = livetime
        return exposure

    def read(self):
        """Read dataset

        Returns
        -------
        dataset : SpectrumDatasetOnOff
            Spectrum dataset
        """
        kwargs = self.read_pha(self.filename)
        pha_meta = kwargs["counts"].meta

        name = str(pha_meta["OBS_ID"])
        livetime = pha_meta["EXPOSURE"] * u.s

        filenames = self.get_filenames(pha_meta=pha_meta)
        exposure = self.read_arf(filenames["arffile"], livetime=livetime)

        if "bkgfile" in filenames:
            bkg = self.read_bkg(filenames["bkgfile"])
            kwargs.update(bkg)

        if "rmffile" in filenames:
            kwargs["edisp"] = self.read_rmf(filenames["rmffile"], exposure=exposure)

        return SpectrumDatasetOnOff(name=name, exposure=exposure, **kwargs)
