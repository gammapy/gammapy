#!/usr/bin/env python
"""Make the gammapy.org static webpage.

This is very much work in progress.
Probably we should add a static website build step.
"""
import logging
import json
import os
from pathlib import Path
import click
import hashlib

log = logging.getLogger(__name__)


def hashmd5(path):
    md5_hash = hashlib.md5()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()


class DownloadDataset:
    """DownloadDataset base class.

    The DownloadDataset class has a local_repo property where to scan the content
    and a base_url to build access links for each file.

    A DownloadDataset has a name as identifier.
    It also has a description and list of files, each file has a given URL
    and a path that tells you where the file will be placed when downloaded.

    If you want to add a DownloadDataset, make a new class and add it to the list below.

    The followLinks flag declares how to build the destination paths in the local desktop
    according to the datasets paths used in the tutorials.
        GAMMAPY-EXTRA: datasets stored in Gammapy-extra reporsitory
        JOINT-CRAB: datasets stored in Joint-crab reporsitory
        OTHERS: datasets stored in others repositories
    """

    followLinks = "GAMMAPY-EXTRA"
    base_url = "https://github.com/gammapy/gammapy-extra/raw/master/datasets"
    local_repo = Path(os.environ["GAMMAPY_EXTRA"]) / "datasets"

    @property
    def record(self):
        return {
            "name": self.name,
            "description": self.description,
            "files": list(self.files),
        }

    @property
    def pathlist(self):
        for itempath in (self.local_repo / self.name).glob("**/*.*"):
            if not itempath.name.startswith("."):
                yield itempath.as_posix().replace(self.local_repo.as_posix() + "/", "")

    @property
    def files(self):
        for item in self.pathlist:

            if self.followLinks == "GAMMAPY-EXTRA":
                jsonpath = str(Path(item))
            elif self.followLinks == "JOINT-CRAB":
                jsonpath = str(Path("joint-crab") / Path("spectra") / Path(item))
            else:
                jsonpath = str(Path(self.name) / Path(item).name)

            itempath = self.local_repo / item
            urlpath = itempath.as_posix().replace(self.local_repo.as_posix(), "")
            filesize = os.path.getsize(itempath)
            md5 = hashmd5(itempath)
            yield {
                "path": jsonpath,
                "url": self.base_url + urlpath,
                "filesize": filesize,
                "hashmd5": md5,
            }


class DatasetCTA1DC(DownloadDataset):
    name = "cta-1dc"
    description = "tbd"


class DatasetDarkMatter(DownloadDataset):
    name = "dark_matter_spectra"
    description = "tbd"


class DatasetCatalogs(DownloadDataset):
    name = "catalogs"
    description = "tbd"


class DatasetFermi2FHL(DownloadDataset):
    name = "fermi_2fhl"
    description = "tbd"


class DatasetFermi3FHL(DownloadDataset):
    name = "fermi_3fhl"
    description = "tbd"


class DatasetFermiSurvey(DownloadDataset):
    name = "fermi_survey"
    description = "tbd"


class DatasetHESSDL3DR1(DownloadDataset):
    name = "hess-dl3-dr1"
    description = "tbd"


class DatasetImages(DownloadDataset):
    name = "images"
    description = "tbd"


class DatasetEBL(DownloadDataset):
    name = "ebl"
    description = "tbd"


class DatasetTests(DownloadDataset):
    name = "tests"
    description = "tbd"


class DatasetFigures(DownloadDataset):
    name = "figures"
    description = "tbd"
    base_url = "https://github.com/gammapy/gammapy-extra/raw/master"
    local_repo = Path(os.environ["GAMMAPY_EXTRA"])


class DatasetJointCrab(DownloadDataset):
    name = "joint-crab"
    description = "tbd"
    base_url = (
        "https://github.com/open-gamma-ray-astro/joint-crab/raw/master/results/spectra"
    )
    local_repo = Path(os.environ["JOINT_CRAB"]) / "results" / "spectra"

    followLinks = "JOINT-CRAB"
    pathlist = []
    for itempath in (local_repo).glob("**/*.*"):
        if not itempath.name.startswith("."):
            pathlist.append(
                itempath.as_posix().replace(local_repo.as_posix() + "/", "")
            )


class DatasetGammaCat(DownloadDataset):
    name = "gamma-cat"
    description = "tbd"
    base_url = "https://github.com/gammapy/gamma-cat/raw/master"
    local_repo = Path(os.environ["GAMMA_CAT"])

    followLinks = "Others"
    pathlist = [str(Path("output") / "gammacat.fits.gz")]


class DatasetFermiLat(DownloadDataset):

    name = "fermi-3fhl"
    description = "tbd"
    base_url = "https://github.com/gammapy/gammapy-fermi-lat-data/raw/master"
    local_repo = Path(os.environ["GAMMAPY_FERMI_LAT_DATA"])

    followLinks = "Others"
    pathlist = [
        str(Path("3fhl") / "allsky" / "fermi_3fhl_events_selected.fits.gz"),
        str(Path("3fhl") / "allsky" / "fermi_3fhl_exposure_cube_hpx.fits.gz"),
        str(Path("3fhl") / "allsky" / "fermi_3fhl_psf_gc.fits.gz"),
        str(Path("isodiff") / "iso_P8R2_SOURCE_V6_v06.txt"),
    ]


class DatasetFermi3FHLGC(DownloadDataset):

    name = "fermi-3fhl-gc"
    description = "Prepared Fermi-LAT 3FHL dataset of the Galactic center region"
    base_url = "https://github.com/gammapy/gammapy-fermi-lat-data/raw/master"
    local_repo = Path(os.environ["GAMMAPY_FERMI_LAT_DATA"])
    basepath = Path("3fhl") / "galactic-center"

    followLinks = "Others"
    pathlist = [
        str(basepath / "fermi-3fhl-gc-background.fits.gz"),
        str(basepath / "fermi-3fhl-gc-background-cube.fits.gz"),
        str(basepath / "fermi-3fhl-gc-counts.fits.gz"),
        str(basepath / "fermi-3fhl-gc-counts-cube.fits.gz"),
        str(basepath / "fermi-3fhl-gc-events.fits.gz"),
        str(basepath / "fermi-3fhl-gc-exposure-cube.fits.gz"),
        str(basepath / "fermi-3fhl-gc-exposure.fits.gz"),
        str(basepath / "fermi-3fhl-gc-psf.fits.gz"),
        str(basepath / "fermi-3fhl-gc-psf-cube.fits.gz"),
        str(basepath / "gll_iem_v06_gc.fits.gz"),
    ]

class DatasetFermi3FHLcrab(DownloadDataset):

    name = "fermi-3fhl-crab"
    description = "Prepared Fermi-LAT 3FHL dataset of the Crab Nebula region"
    base_url = "https://github.com/gammapy/gammapy-fermi-lat-data/raw/master"
    local_repo = Path(os.environ["GAMMAPY_FERMI_LAT_DATA"])
    basepath = Path("3fhl") / "crab"

    followLinks = "Others"
    pathlist = [
        str(basepath / "Fermi-LAT-3FHL_data_Fermi-LAT.fits"),
        str(basepath / "Fermi-LAT-3FHL_datasets.yaml"),
        str(basepath / "Fermi-LAT-3FHL_models.yaml"),
    ]


class DatasetHAWCcrab(DownloadDataset):
    name = "hawc_crab"
    description = "tbd"


class DownloadDatasetIndex:
    path = Path(__file__).parent / "gammapy-data-index.json"
    download_datasets = [
        DatasetCTA1DC,
        DatasetDarkMatter,
        DatasetCatalogs,
        DatasetFermi3FHL,
        DatasetHESSDL3DR1,
        DatasetImages,
        DatasetJointCrab,
        DatasetEBL,
        DatasetGammaCat,
        DatasetFermi3FHLGC,
        DatasetFermi3FHLcrab,
        DatasetHAWCcrab,
        DatasetTests,
        DatasetFigures,
    ]

    def make(self):
        records = list(self.make_records())
        txt = json.dumps(records, indent=True)
        log.info("Writing {}".format(self.path))
        Path(self.path).write_text(txt)

    def make_records(self):
        for cls in self.download_datasets:
            yield cls().record


@click.group()
def cli():
    """Make a dataset index JSON file to download datasets with gammapy download datasets"""
    logging.basicConfig(level="INFO")


@cli.command("all")
@click.pass_context
def cli_all(ctx):
    """Run all steps"""
    ctx.invoke(cli_download_dataset_index)


@cli.command("dataset-index")
def cli_download_dataset_index():
    """Generate dataset index JSON file"""
    DownloadDatasetIndex().make()


if __name__ == "__main__":
    cli()
