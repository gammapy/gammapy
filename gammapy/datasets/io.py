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


__all__ = ["DatasetWriter", "DatasetReader", "OGIPDatasetReader", "OGIPDatasetWriter"]


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
    tag = "ogip"

    def __init__(self, outdir=None, use_sherpa=False, overwrite=False):
        outdir = Path.cwd() if outdir is None else make_path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        self.outdir = outdir
        self.use_sherpa = use_sherpa
        self.overwrite = overwrite

    @staticmethod
    def get_filenames(dataset):
        """Get filenames

        Parameters
        ----------
        dataset : `SpectrumDataset`
            Dataset to write

        """
        # TODO: allow
        filenames = {}
        phafile = f"pha_obs{dataset.name}.fits"

        bkgfile = phafile.replace("pha", "bkg")
        arffile = phafile.replace("pha", "arf")
        rmffile = phafile.replace("pha", "rmf")

        filenames["phafile"] = phafile
        filenames["respfile"] = rmffile
        filenames["backfile"] = bkgfile
        filenames["ancrfile"] = arffile
        return filenames

    @staticmethod
    def get_ogip_meta(dataset):
        """Meta info for the OGIP data format"""
        try:
            livetime = dataset.exposure.meta["livetime"]
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

    def write(self, dataset):
        """Write dataset to files

        Parameters
        ----------
        dataset : `SpectrumDatasetOnOff`
            Dataset to write
        """
        filenames = self.get_filenames(dataset)

        self.write_counts(dataset, filename=filenames["phafile"])
        self.write_aeff(dataset, filename=filenames["ancrfile"])

        if dataset.counts_off:
            self.write_counts_off(dataset, filename=filenames["backfile"])

        if dataset.edisp:
            self.write_edisp(dataset, filename=filenames["respfile"])

    def write_edisp(self, dataset, filename):
        """Write energy dispersion.

        Parameters
        ----------
        dataset : `SpectrumDatasetOnOff`
            Dataset to write
        filename : str
            Filename to use.
        """
        kernel = dataset.edisp.get_edisp_kernel()
        kernel.write(self.outdir / filename, overwrite=self.overwrite, use_sherpa=self.use_sherpa)

    def write_aeff(self, dataset, filename):
        """Write effective area

        Parameters
        ----------
        dataset : `SpectrumDatasetOnOff`
            Dataset to write
        filename : str
            Filename to use.

        """
        aeff = dataset.exposure / dataset.exposure.meta["livetime"]

        hdu_format = "ogip-sherpa" if self.use_sherpa else "ogip"

        aeff.write(
            self.outdir / filename,
            overwrite=self.overwrite,
            format=hdu_format,
            ogip_column="SPECRESP",
        )

    def to_counts_hdulist(self, dataset, is_counts_off=False):
        """Convert counts region map to hdulist

        Parameters
        ----------
        dataset : `SpectrumDatasetOnOff`
            Dataset to write
        is_counts_off : bool
            Whether to use counts off.
        """
        counts = dataset.counts_off if is_counts_off else dataset.counts
        acceptance = dataset.acceptance_off if is_counts_off else dataset.acceptance

        table = counts.to_table()
        table["QUALITY"] = np.logical_not(dataset.mask_safe.data[:, 0, 0])
        table["BACKSCAL"] = acceptance.data[:, 0, 0]
        table["AREASCAL"] = np.ones(acceptance.data.size)

        # prepare meta data
        meta = self.get_ogip_meta(dataset)

        if is_counts_off:
            meta["hduclas2"] = "BKG"
        else:
            meta.update(self.get_filenames(dataset))
            meta["hduclas2"] = "TOTAL"

        table.meta = meta

        name = table.meta["name"]
        hdu = fits.BinTableHDU(table, name=name)

        energy_axis = dataset.counts.geom.axes[0]

        hdu_format = "ogip-sherpa" if self.use_sherpa else "ogip"

        hdulist = fits.HDUList(
            [fits.PrimaryHDU(), hdu, energy_axis.to_table_hdu(format=hdu_format)]
        )

        if counts.geom.region:
            region_table = counts.geom._to_region_table()
            region_hdu = fits.BinTableHDU(region_table, name="REGION")
            hdulist.append(region_hdu)

        return hdulist

    def write_counts(self, dataset, filename):
        """Write counts file

        Parameters
        ----------
        dataset : `SpectrumDatasetOnOff`
            Dataset to write
        filename : str
            Filename to use.

        """
        hdulist = self.to_counts_hdulist(dataset)

        if dataset.gti:
            hdu = fits.BinTableHDU(dataset.gti.table, name="GTI")
            hdulist.append(hdu)

        hdulist.writeto(str(self.outdir / filename), overwrite=self.overwrite)

    def write_counts_off(self, dataset, filename):
        """Write off counts file

        Parameters
        ----------
        dataset : `SpectrumDatasetOnOff`
            Dataset to write
        filename : str
            Filename to use.
        """
        hdulist = self.to_counts_hdulist(dataset, is_counts_off=True)
        hdulist.writeto(str(self.outdir / filename), overwrite=self.overwrite)


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
        self.filename = filename

    def read(self):
        """Read dataset"""
        filename = make_path(self.filename)
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

        return SpectrumDatasetOnOff(
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
