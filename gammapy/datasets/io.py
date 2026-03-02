# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import numpy as np
from pathlib import Path
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from gammapy.data import GTI
from gammapy.irf import EDispKernel, EDispKernelMap, PSFMap
from gammapy.maps import RegionNDMap, Map
from gammapy.modeling.models import create_fermi_isotropic_diffuse_model, Models
from gammapy.utils.scripts import read_yaml, make_path
from gammapy.utils.metadata import CreatorMetaData
from .spectrum import SpectrumDatasetOnOff
from .utils import create_map_dataset_from_dl4

__all__ = [
    "DatasetReader",
    "DatasetWriter",
    "OGIPDatasetReader",
    "OGIPDatasetWriter",
    "FermipyDatasetsReader",
]


class DatasetReader(abc.ABC):
    """Dataset reader base class."""

    @property
    @abc.abstractmethod
    def tag(self):
        pass

    @abc.abstractmethod
    def read(self):
        pass


class DatasetWriter(abc.ABC):
    """Dataset writer base class."""

    @property
    @abc.abstractmethod
    def tag(self):
        pass

    @abc.abstractmethod
    def write(self, dataset):
        pass


class OGIPDatasetWriter(DatasetWriter):
    """Write OGIP files.

    If you want to use the written files with Sherpa, you have to use the
    ``ogip-sherpa`` format. Then all files will be written in units of 'keV' and
    'cm2'.

    The naming scheme is fixed as following:

    * PHA file is named filename.fits
    * BKG file is named filename_bkg.fits
    * ARF file is named filename_arf.fits
    * RMF file is named filename_rmf.fits

    Parameters
    ----------
    filename : `~pathlib.Path` or str
        Filename.
    format : {"ogip", "ogip-sherpa"}
        Which format to use. Default is 'ogip'.
    overwrite : bool, optional
        Overwrite existing files. Default is False.
    checksum : bool
        When True adds both DATASUM and CHECKSUM cards to the headers written to the files.
        Default is False.
    creation : `~gammapy.utils.metadata.CreatorMetaData`, optional.
        The creation metadata to write to disk. If None, use default creator metadata object.
    """

    tag = ["ogip", "ogip-sherpa"]

    def __init__(
        self, filename, format="ogip", overwrite=False, checksum=False, creation=None
    ):
        filename = make_path(filename)
        filename.parent.mkdir(exist_ok=True, parents=True)

        self.filename = filename
        self.format = format
        self.overwrite = overwrite
        self.checksum = checksum
        self.creation = creation or CreatorMetaData()

    @staticmethod
    def get_filenames(filename):
        """Get filenames.

        Parameters
        ----------
        filename : `~pathlib.Path`
            Filename.

        Returns
        -------
        filenames : dict
            Dictionary of filenames.
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
        """Meta info for the OGIP data format."""
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

        if dataset.counts_off:
            meta["BACKFILE"] = filenames["backfile"]

        if dataset.edisp:
            meta["RESPFILE"] = filenames["respfile"]

        return meta

    def write(self, dataset):
        """Write dataset to file.

        Parameters
        ----------
        dataset : `SpectrumDatasetOnOff`
            Dataset to write.
        """
        filenames = self.get_filenames(self.filename)

        self.creation.update_time()

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
            Dataset to write.
        filename : str or `~pathlib.Path`
            Filename to use.
        """
        kernel = dataset.edisp.get_edisp_kernel()
        kernel.write(
            filename=filename,
            format=self.format,
            checksum=self.checksum,
            overwrite=self.overwrite,
        )

    def write_arf(self, dataset, filename):
        """Write effective area.

        Parameters
        ----------
        dataset : `SpectrumDatasetOnOff`
            Dataset to write.
        filename : str or `~pathlib.Path`
            Filename to use.

        """
        aeff = dataset.exposure / dataset.exposure.meta["livetime"]

        filename = make_path(filename)
        hdulist = aeff.to_hdulist(format=self.format.replace("ogip", "ogip-arf"))

        for hdu in hdulist:
            hdu.header.update(self.creation.to_header())

        hdulist.writeto(filename, overwrite=self.overwrite, checksum=self.checksum)

    def to_counts_hdulist(self, dataset, is_bkg=False):
        """Convert counts region map to hdulist.

        Parameters
        ----------
        dataset : `SpectrumDatasetOnOff`
            Dataset to write.
        is_bkg : bool
            Whether to use counts off. Default is False.
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

        for hdu in hdulist:
            hdu.header.update(self.creation.to_header())
        return hdulist

    def write_pha(self, dataset, filename):
        """Write counts file.

        Parameters
        ----------
        dataset : `SpectrumDatasetOnOff`
            Dataset to write.
        filename : str or `~pathlib.Path`
            Filename to use.

        """
        hdulist = self.to_counts_hdulist(dataset)

        if dataset.gti:
            hdu = dataset.gti.to_table_hdu()
            hdulist.append(hdu)

        hdulist.writeto(filename, overwrite=self.overwrite, checksum=self.checksum)

    def write_bkg(self, dataset, filename):
        """Write off counts file.

        Parameters
        ----------
        dataset : `SpectrumDatasetOnOff`
            Dataset to write.
        filename : str or `~pathlib.Path`
            Filename to use.
        """
        hdulist = self.to_counts_hdulist(dataset, is_bkg=True)
        hdulist.writeto(filename, overwrite=self.overwrite, checksum=self.checksum)


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
        OGIP PHA file to read.
    checksum : bool
        If True checks both DATASUM and CHECKSUM cards in the file headers. Default is False.
    """

    tag = "ogip"

    def __init__(self, filename, checksum=False, name=None):
        self.filename = make_path(filename)
        self.checksum = checksum
        self.name = name

    def get_valid_path(self, filename):
        """Get absolute or relative path.

        The relative path is with respect to the name of the reference file.

        Parameters
        ----------
        filename : str or `~pathlib.Path`
            Filename.

        Returns
        -------
        filename : `~pathlib.Path`
            Valid path.
        """
        filename = make_path(filename)

        if not filename.exists():
            return self.filename.parent / filename
        else:
            return filename

    def get_filenames(self, pha_meta):
        """Get filenames.

        Parameters
        ----------
        pha_meta : dict
            Metadata from the PHA file.

        Returns
        -------
        filenames : dict
            Dictionary with filenames of "arffile", "rmffile" (optional)
            and "bkgfile" (optional).
        """
        filenames = {"arffile": self.get_valid_path(pha_meta["ANCRFILE"])}

        if "BACKFILE" in pha_meta:
            filenames["bkgfile"] = self.get_valid_path(pha_meta["BACKFILE"])

        if "RESPFILE" in pha_meta:
            filenames["rmffile"] = self.get_valid_path(pha_meta["RESPFILE"])

        return filenames

    @staticmethod
    def read_pha(filename, checksum=False):
        """Read PHA file.

        Parameters
        ----------
        filename : str or `~pathlib.Path`
            PHA file name.
        checksum : bool
            If True checks both DATASUM and CHECKSUM cards in the file headers. Default is False.

        Returns
        -------
        data : dict
            Dictionary with counts, acceptance and mask_safe.
        """
        data = {}

        with fits.open(filename, memmap=False, checksum=checksum) as hdulist:
            data["counts"] = RegionNDMap.from_hdulist(hdulist, format="ogip")
            data["acceptance"] = RegionNDMap.from_hdulist(
                hdulist, format="ogip", ogip_column="BACKSCAL"
            )

            if "GTI" in hdulist:
                data["gti"] = GTI.from_table_hdu(hdulist["GTI"])

            data["mask_safe"] = RegionNDMap.from_hdulist(
                hdulist, format="ogip", ogip_column="QUALITY"
            )

        return data

    @staticmethod
    def read_bkg(filename, checksum=False):
        """Read PHA background file.

        Parameters
        ----------
        filename : str or `~pathlib.Path`
            PHA file name.
        checksum : bool
            If True checks both DATASUM and CHECKSUM cards in the file headers. Default is False.

        Returns
        -------
        data : dict
            Dictionary with counts_off and acceptance_off.
        """
        with fits.open(filename, memmap=False, checksum=checksum) as hdulist:
            counts_off = RegionNDMap.from_hdulist(hdulist, format="ogip")
            acceptance_off = RegionNDMap.from_hdulist(
                hdulist, ogip_column="BACKSCAL", format="ogip"
            )
        return {"counts_off": counts_off, "acceptance_off": acceptance_off}

    @staticmethod
    def read_rmf(filename, exposure, checksum=False):
        """Read RMF file.

        Parameters
        ----------
        filename : str or `~pathlib.Path`
            PHA file name.
        exposure : `RegionNDMap`
            Exposure map.
        checksum : bool
            If True checks both DATASUM and CHECKSUM cards in the file headers. Default is False.

        Returns
        -------
        data : `EDispKernelMap`
            Dictionary with edisp.
        """
        kernel = EDispKernel.read(filename, checksum=checksum)
        edisp = EDispKernelMap.from_edisp_kernel(kernel, geom=exposure.geom)

        # TODO: resolve this separate handling of exposure for edisp
        edisp.exposure_map.data = exposure.data[:, :, np.newaxis, :]
        return edisp

    @staticmethod
    def read_arf(filename, livetime, checksum=False):
        """Read ARF file.

        Parameters
        ----------
        filename : str or `~pathlib.Path`
            PHA file name.
        livetime : `Quantity`
            Livetime.
        checksum : bool
            If True checks both DATASUM and CHECKSUM cards in the file headers. Default is False.

        Returns
        -------
        data : `RegionNDMap`
            Exposure map.
        """
        aeff = RegionNDMap.read(filename, format="ogip-arf", checksum=checksum)
        exposure = aeff * livetime
        exposure.meta["livetime"] = livetime
        return exposure

    def read(self):
        """Read dataset.

        Returns
        -------
        dataset : SpectrumDatasetOnOff
            Spectrum dataset.
        """
        kwargs = self.read_pha(self.filename, checksum=self.checksum)
        pha_meta = kwargs["counts"].meta

        if self.name is not None:
            name = self.name
        else:
            name = str(pha_meta["OBS_ID"])
        livetime = pha_meta["EXPOSURE"] * u.s

        filenames = self.get_filenames(pha_meta=pha_meta)
        exposure = self.read_arf(
            filenames["arffile"], livetime=livetime, checksum=self.checksum
        )

        if "bkgfile" in filenames:
            bkg = self.read_bkg(filenames["bkgfile"], checksum=self.checksum)
            kwargs.update(bkg)

        if "rmffile" in filenames:
            kwargs["edisp"] = self.read_rmf(
                filenames["rmffile"], exposure=exposure, checksum=self.checksum
            )

        return SpectrumDatasetOnOff(name=name, exposure=exposure, **kwargs)


class FermipyDatasetsReader(DatasetReader):
    """Create datasets from Fermi-LAT files.

    Parameters
    ----------
    filename : str
        Path to Fermipy configuration file (tested only for v1.3.1).
    edisp_bins : int
        Number of margin bins to slice in energy. Default is 0.
        For now only maps created with edisp_bins=0 in fermipy configuration are supported,
        in that case the emin/emax in the fermipy configuration will correspond to the true energy range for gammapy,
        and  a value edisp_bins>0 should be set here in order to apply the energy dispersion correctly.
        With a binning of 8 to 10 bins per decade, it is recommended to use edisp_bins ≥ 2
        (See https://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Pass8_edisp_usage.html)

    """

    tag = "fermipy"

    def __init__(self, filename, edisp_bins=0):
        self.filename = make_path(filename)
        self.edisp_bins = edisp_bins

    @staticmethod
    def create_dataset(
        counts_file,
        exposure_file,
        psf_file,
        edisp_file,
        isotropic_file=None,
        edisp_bins=0,
        name=None,
        gti_file=None,
    ):
        """Create a map dataset from Fermi-LAT files.

         Parameters
         ----------
         counts_file : str
             Counts file path.
         exposure_file : str
             Exposure file path.
         psf_file : str
             Point spread function file path.
         edisp_file : str
             Energy dispersion file path.
         isotropic_file : str, optional
             Isotropic file path. Default is None
         edisp_bins : int
             Number of margin bins to slice in energy. Default is 0.
             For now only maps created with edisp_bins=0 in fermipy configuration are supported,
             in that case the emin/emax in the fermipy configuration will correspond to the true energy range for gammapy,
             and  a value edisp_bins>0 should be set here in order to apply the energy dispersion correctly.
             With a binning of 8 to 10 bins per decade, it is recommended to use edisp_bins ≥ 2
             (See https://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Pass8_edisp_usage.html)
        name : str, optional
            Dataset name. The default is None, and the name is randomly generated.
        gti_file : str, optional
             GTI file path. Default is None


         Returns
         -------
         dataset : `~gammapy.datasets.MapDataset`
             Map dataset.

        """
        from gammapy.datasets import MapDataset

        counts = Map.read(counts_file)
        exposure = Map.read(exposure_file)
        psf = PSFMap.read(psf_file, format="gtpsf")
        edisp = EDispKernelMap.read(edisp_file, format="gtdrm")

        # check that fermipy edisp_bins are matching between edisp and exposure
        # as we will interp to edisp axis the exposure axis must be larger or equal
        edisp_axes = edisp.edisp_map.geom.axes
        if len(edisp_axes["energy_true"].center) > len(
            exposure.geom.axes["energy_true"].center
        ):
            raise ValueError(
                "Energy true axes of exposure and DRM do not match. Check fermipy configuration."
            )

        psf_r68s = psf.containment_radius(
            0.68,
            edisp_axes["energy_true"].center,
            position=counts.geom.center_skydir,
        )
        # check that pdf is well defined (fails if edisp_bins>0 in fermipy)
        if np.any(np.isclose(psf_r68s.value, 0.0)):
            raise ValueError(
                "PSF is not defined for all true energies. Check fermipy configuration."
            )

        # change counts energy axis unit keV->MeV
        energy_axis = counts.geom.axes["energy"]._init_copy(
            nodes=edisp_axes["energy"].edges
        )
        geom = counts.geom.to_image().to_cube([energy_axis])
        counts = Map.from_geom(geom, data=counts.data)

        # get gtis from gtmktime evtfile if given
        if gti_file:
            gtis = GTI.read(gti_file)
        else:
            gtis = None

        # standardize dataset interpolating to same geom and axes
        dataset = MapDataset(
            counts=counts,
            exposure=exposure,
            psf=psf,
            edisp=edisp,
            name=name,
            gti=gtis,
        )
        dataset = create_map_dataset_from_dl4(
            dataset, geom=counts.geom, name=dataset.name
        )

        if edisp_bins > 0:  # slice edisp_bins
            dataset = dataset.slice_by_idx(
                dict(energy=slice(edisp_bins, -edisp_bins)), name=dataset.name
            )

        if isotropic_file:
            model = create_fermi_isotropic_diffuse_model(
                isotropic_file, datasets_names=[dataset.name]
            )
            dataset.models = Models([model])
        return dataset

    def read(self):
        """Create Fermi-LAT map datasets from Fermipy configuration file.

        Returns
        -------
        dataset : `~gammapy.datasets.Datasets`
            Map datasets.

        """
        from gammapy.datasets import Datasets

        filename = self.filename.resolve()
        data = read_yaml(filename)

        components = self._get_components(data)

        datasets = Datasets()
        for file_id, component in enumerate(components):
            if "fileio" in component and "outdir" in component["fileio"]:
                path = Path(component["fileio"]["outdir"])
            elif "fileio" in data and "outdir" in data["fileio"]:
                path = Path(data["fileio"]["outdir"])
            else:
                path = Path(filename.parent)
            if not path.is_absolute():
                path = Path(filename.parent) / path

            if "model" in component and "isodiff" in component["model"]:
                isotropic_file = self._get_isodiff(component["model"]["isodiff"])
                name = isotropic_file.stem[4:]
            elif "model" in data and "isodiff" in data["model"]:
                isotropic_file = self._get_isodiff(data["model"]["isodiff"])
                name = isotropic_file.stem[4:]
            else:
                isotropic_file = None
                name = None

            datasets.append(
                self.create_dataset(
                    counts_file=path / f"ccube_0{str(file_id)}.fits",
                    exposure_file=path / f"bexpmap_0{str(file_id)}.fits",
                    psf_file=path / f"psf_0{str(file_id)}.fits",
                    edisp_file=path / f"drm_0{str(file_id)}.fits",
                    isotropic_file=isotropic_file,
                    edisp_bins=self.edisp_bins,
                    name=name,
                    gti_file=path / f"ft1_0{str(file_id)}.fits",
                )
            )
        return datasets

    @staticmethod
    def _get_components(data):
        components = data.get("components")
        if isinstance(components, list) and len(components) > 0:
            return components
        return [data]

    @staticmethod
    def _get_isodiff(data):
        if isinstance(data, str):
            return Path(data)

        if not isinstance(data, list):
            raise ValueError("Invalid isodiff filename.")

        if len(data) != 1:
            raise ValueError("Only one isodiff filename per component should be given.")
        if not isinstance(data[0], str):
            raise ValueError("Invalid isodiff filename.")
        return Path(data[0])
