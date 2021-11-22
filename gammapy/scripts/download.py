# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to download datasets and notebooks"""
import logging
import tarfile
from pathlib import Path
import click
from gammapy import __version__

log = logging.getLogger(__name__)

BUNDLESIZE = 152  # in MB
ENVS_BASE_URL = "https://gammapy.org/download/install"
NBTAR_BASE_URL = "https://docs.gammapy.org"
TAR_DATASETS = "https://github.com/gammapy/gammapy-data/tarball/master"


def progress_download(source, destination):
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        log.error("To use gammapy download install the tqdm and requests packages")
        return

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
            member.path = member.path[len(root_folder) + 1 :]
            yield member


def extract_bundle(bundle, destination):
    with tarfile.open(bundle) as tar:
        tar.extractall(path=destination, members=members(tar))
    Path(bundle).unlink()


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
@click.option("--release", required=True, help="Number of stable release - ex: 0.18.2)")
@click.option(
    "--out",
    default="gammapy-notebooks",
    help="Path where the versioned notebook files will be copied.",
    show_default=True,
)
def cli_download_notebooks(release, out):
    """Download notebooks"""
    localfolder = Path(out) / release
    url_file_env = f"{ENVS_BASE_URL}/gammapy-{release}-environment.yml"
    yaml_destination_file = localfolder / f"gammapy-{release}-environment.yml"
    progress_download(url_file_env, yaml_destination_file)
    url_tar_notebooks = f"{NBTAR_BASE_URL}/{release}/_downloads/notebooks-{release}.tar"
    tar_destination_file = localfolder / f"notebooks_{release}.tar"
    progress_download(url_tar_notebooks, tar_destination_file)
    extract_bundle(tar_destination_file, localfolder)
    show_info_notebooks(localfolder, release)


@click.command(name="datasets")
@click.option(
    "--out",
    default="gammapy-datasets",
    help="Destination folder.",
    show_default=True,
)
def cli_download_datasets(out):
    """Download datasets"""
    localfolder = Path(out)
    log.info(f"Downloading datasets from {TAR_DATASETS}")
    tar_destination_file = localfolder / "datasets.tar.gz"
    progress_download(TAR_DATASETS, tar_destination_file)
    log.info(f"Extracting {tar_destination_file}")
    extract_bundle(tar_destination_file, localfolder)
    show_info_datasets(localfolder)
