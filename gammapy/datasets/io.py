# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path
from gammapy.utils.scripts import make_path
import numpy as np
from astropy.io import fits


class DatasetIO:
    pass


class DatasetIOOGIP(DatasetIO):
    tag = "ogip"

    def __init__(self, outdir=None, use_sherpa=False, overwrite=False):
        outdir = Path.cwd() if outdir is None else make_path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        self.outdir = outdir
        self.use_sherpa = use_sherpa
        self.overwrite = overwrite

    def get_filenames(self, dataset):
        """"""
        filenames = {}
        phafile = f"pha_obs{dataset.name}.fits"

        bkgfile = phafile.replace("pha", "bkg")
        arffile = phafile.replace("pha", "arf")
        rmffile = phafile.replace("pha", "rmf")

        filenames["respfile"] = rmffile
        filenames["backfile"] = bkgfile
        filenames["ancrfile"] = arffile
        return filenames

    @staticmethod
    def write_edisp(dataset, filename, overwrite):
        """"""
        kernel = self.edisp.get_edisp_kernel()
        kernel.write(outdir / rmffile, overwrite=overwrite, use_sherpa=use_sherpa)

    def write_aeff(self, dataset):
        """Write effective area"""
        aeff = dataset.exposure / dataset.exposure.meta["livetime"]

        hdu_format = "ogip-sherpa" if use_sherpa else "ogip"

        aeff.write(
            filename,
            overwrite=overwrite,
            format=hdu_format,
            ogip_column="SPECRESP",
        )

    def to_table_hdu(self, dataset):
        """"""
        counts_table = self.counts.to_table()
        counts_table["QUALITY"] = np.logical_not(self.mask_safe.data[:, 0, 0])
        counts_table["BACKSCAL"] = self.acceptance.data[:, 0, 0]
        counts_table["AREASCAL"] = np.ones(self.acceptance.data.size)
        meta = self._ogip_meta()

    def write(self, dataset):
        """"""
        self.write_counts()
        self.write_counts_off()

        self.write_aeff()
        self.write_edisp()
        pass

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


class DatasetIOGADF(DatasetIO):
    tag = "gadf"


    def to_hdulist(self):
        pass


    def from_hdulist(self):
        pass


    def read(self):
        pass


    def write(self):
        pass
