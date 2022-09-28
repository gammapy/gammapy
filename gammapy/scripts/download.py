# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to download datasets and notebooks"""
import logging
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from astropy.utils import lazyproperty
import click
from gammapy import __version__

log = logging.getLogger(__name__)

BUNDLESIZE = 152  # in MB
GAMMAPY_BASE_URL = "https://gammapy.org/download/"

RELEASE = __version__

if "dev" in __version__:
    RELEASE = "dev"


class DownloadIndex:
    """Download index"""

    _notebooks_key = "notebooks"
    _datasets_key = "datasets"
    _environment_key = "conda-environment"
    _index_json = "index.json"

    def __init__(self, release=RELEASE):
        self.release = release

    @lazyproperty
    def index(self):
        """Index for a given release"""
        import requests

        response = requests.get(GAMMAPY_BASE_URL + self._index_json)
        data = response.json()

        if self.release not in data:
            raise ValueError(
                f"Download not available for release {self.release}, "
                f"choose from: {list(data.keys())}"
            )

        return data[self.release]

    @property
    def notebooks_url(self):
        """Notebooks URL"""
        return self.index[self._notebooks_key]

    @property
    def environment_url(self):
        """Environment URL"""
        return self.index[self._environment_key]

    @property
    def datasets_url(self):
        """Datasets URL"""
        return self.index[self._datasets_key]


def progress_download(source, destination):
    import requests
    from tqdm import tqdm

    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(source, stream=True) as r:
        total_size = (
            int(r.headers.get("content-length"))
            if r.headers.get("content-length")
            else BUNDLESIZE * 1024 * 1024
        )
        progress_bar = tqdm(
            total=total_size, unit="B", unit_scale=True, unit_divisor=1024
        )
        with open(destination, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
    progress_bar.close()


def members(tf):
    list_members = tf.getmembers()
    root_folder = list_members[0].name
    for member in list_members:
        if member.path.startswith(root_folder):
            member.path = member.path[len(root_folder) + 1 :]  # noqa: E203
            yield member


def extract_bundle(bundle, destination):
    with tarfile.open(bundle) as tar:
        tar.extractall(path=destination, members=members(tar))


def show_info_notebooks(outfolder, release):
    print("")
    print(
        "*** Enter the following commands below to get started with this version of Gammapy"
    )
    print(f"cd {outfolder}")
    if __version__ != release:
        print(f"conda env create -f gammapy-{release}-environment.yml")
        print(f"conda activate gammapy-{release}")
    print("jupyter lab")
    print("")


def show_info_datasets(outfolder):
    print("")
    print("*** You might want to declare GAMMAPY_DATA env variable")
    print(f"export GAMMAPY_DATA={outfolder}")
    print("")


@click.command(name="notebooks")
@click.option(
    "--release",
    default=RELEASE,
    help="Number of stable release - ex: 0.18.2)",
    show_default=True,
)
@click.option(
    "--out",
    default="gammapy-notebooks",
    help="Path where the versioned notebook files will be copied.",
    show_default=True,
)
def cli_download_notebooks(release, out):
    """Download notebooks"""
    index = DownloadIndex(release=release)

    path = Path(out) / index.release

    filename = path / f"gammapy-{index.release}-environment.yml"
    progress_download(index.environment_url, filename)

    url_path = urlparse(index.notebooks_url).path
    filename_destination = path / Path(url_path).name
    progress_download(index.notebooks_url, filename_destination)

    if "zip" in index.notebooks_url:
        archive = zipfile.ZipFile(filename_destination, "r")
    else:
        archive = tarfile.open(filename_destination)

    with archive as ref:
        ref.extractall(path)

    # delete file
    filename_destination.unlink()

    show_info_notebooks(path, release)


@click.command(name="datasets")
@click.option(
    "--release",
    default=RELEASE,
    help="Number of stable release - ex: 0.18.2)",
    show_default=True,
)
@click.option(
    "--out",
    default="gammapy-datasets",
    help="Destination folder.",
    show_default=True,
)
def cli_download_datasets(release, out):
    """Download datasets"""
    index = DownloadIndex(release=release)

    localfolder = Path(out) / index.release
    log.info(f"Downloading datasets from {index.datasets_url}")
    tar_destination_file = localfolder / "datasets.tar.gz"
    progress_download(index.datasets_url, tar_destination_file)

    log.info(f"Extracting {tar_destination_file}")
    extract_bundle(tar_destination_file, localfolder)
    Path(tar_destination_file).unlink()
    show_info_datasets(localfolder)
