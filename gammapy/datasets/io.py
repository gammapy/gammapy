# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
from pathlib import Path
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from gammapy.data import GTI
from gammapy.utils.scripts import make_path
from gammapy.maps import RegionNDMap
from gammapy.irf import EDispKernelMap, EDispKernel
from .spectrum import SpectrumDatasetOnOff


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

    The naming scheme is fixed, with {name} the dataset name:

    * PHA file is named pha_obs{name}.fits
    * BKG file is named bkg_obs{name}.fits
    * ARF file is named arf_obs{name}.fits
    * RMF file is named rmf_obs{name}.fits

    Parameters
    ----------
    outdir : `pathlib.Path` or str
        output directory, default: pwd
    format : {"ogip", "ogip-sherpa"}
        Which format to use.
    overwrite : bool
        Overwrite existing files?
    """
    tag = ["ogip", "ogip-sherpa"]

    def __init__(self, outdir=None, format="ogip", overwrite=False):
        outdir = Path.cwd() if outdir is None else make_path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        self.outdir = outdir
        self.format = format
        self.overwrite = overwrite

    @staticmethod
    def get_filenames(outdir, name):
        """Get filenames

        Parameters
        ----------
        outdir : `Path`
            Base directory
        name : str
            Dataset name

        """
        # TODO: allow arbitrary filenames?
        phafile = f"pha_obs{name}.fits"
        return {
            "phafile": str(outdir / phafile),
            "respfile": str(outdir / phafile.replace("pha", "rmf")),
            "backfile": str(outdir / phafile.replace("pha", "bkg")),
            "ancrfile": str(outdir / phafile.replace("pha", "arf"))
        }

    @staticmethod
    def get_ogip_meta(dataset, is_bkg=False):
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
            "name": "SPECTRUM",
            "hduclass": "OGIP",
            "hduclas1": "SPECTRUM",
            "hduclas2": hdu_class,
            "corrscal": "",
            "chantype": "PHA",
            "detchans": dataset.counts.geom.axes[0].nbin,
            "filter": "None",
            "corrfile": "",
            "poisserr": True,
            "hduclas3": "COUNT",
            "hduclas4": "TYPE:1",
            "lo_thres": dataset.energy_range[0].to_value("TeV"),
            "hi_thres": dataset.energy_range[1].to_value("TeV"),
            "exposure": livetime.to_value("s"),
            "obs_id": dataset.name,
        }

        if not is_bkg:
            filenames = OGIPDatasetWriter.get_filenames(outdir=Path(""), name=dataset.name)
            meta.update(filenames)

        return meta

    def write(self, dataset):
        """Write dataset to files

        Parameters
        ----------
        dataset : `SpectrumDatasetOnOff`
            Dataset to write
        """
        filenames = self.get_filenames(outdir=self.outdir, name=dataset.name)

        self.write_pha(dataset, filename=filenames["phafile"])
        self.write_arf(dataset, filename=filenames["ancrfile"])

        if dataset.counts_off:
            self.write_bkg(dataset, filename=filenames["backfile"])

        if dataset.edisp:
            self.write_rmf(dataset, filename=filenames["respfile"])

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
        kernel.write(
            filename=filename,
            overwrite=self.overwrite,
            format=self.format
        )

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
            format=self.format,
            ogip_column="SPECRESP",
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
        hdulist = fits.HDUList()

        counts = dataset.counts_off if is_bkg else dataset.counts
        acceptance = dataset.acceptance_off if is_bkg else dataset.acceptance

        table = counts.to_table()
        table["QUALITY"] = np.logical_not(dataset.mask_safe.data[:, 0, 0])
        table["BACKSCAL"] = acceptance.data[:, 0, 0]
        table["AREASCAL"] = np.ones(acceptance.data.size)

        # prepare meta data
        table.meta = self.get_ogip_meta(dataset, is_bkg=is_bkg)
        hdulist.append(fits.BinTableHDU(table, name=table.meta["name"]))

        hdulist_geom = counts.geom.to_hdulist(format=self.format)[1:]

        hdulist.extend(hdulist_geom)
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
    filename : str
        OGIP PHA file to read
    """
    tag = "ogip"

    def __init__(self, filename):
        self.filename = make_path(filename)
        self.path = self.filename.parent

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
        if not Path(filename).exists():
            return {"counts_off": None, "acceptance_off": None}

        with fits.open(filename, memmap=False) as hdulist:
            counts_off = RegionNDMap.from_hdulist(hdulist, format="ogip")
            acceptance_off = RegionNDMap.from_hdulist(
                hdulist, ogip_column="BACKSCAL"
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
        if not Path(filename).exists():
            return

        kernel = EDispKernel.read(filename)
        edisp = EDispKernelMap.from_edisp_kernel(kernel, geom=exposure.geom)
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
        pha = self.read_pha(self.filename)
        meta = pha["counts"].meta

        name = str(meta["OBS_ID"])
        livetime = meta["EXPOSURE"] * u.s

        filenames = OGIPDatasetWriter.get_filenames(self.path, name)

        exposure = self.read_arf(filenames["ancrfile"], livetime=livetime)
        bkg = self.read_bkg(filenames["backfile"])
        edisp = self.read_rmf(filenames["respfile"], exposure=exposure)

        return SpectrumDatasetOnOff(
            name=name, exposure=exposure, edisp=edisp, **pha, **bkg,
        )
