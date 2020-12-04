# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to download datasets and notebooks"""
import logging
import tarfile
from pathlib import Path
import click
from gammapy import __version__

log = logging.getLogger(__name__)

BUNDLESIZE = 152  # in MB
RELEASES_BASE_URL = "https://gammapy.org/download"
TAR_DATASETS = "https://github.com/gammapy/gammapy-data/tarball/master"


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
            member.path = member.path[len(root_folder) + 1 :]
            yield member


def extract_bundle(bundle, destination):
    with tarfile.open(bundle) as tar:
        tar.extractall(path=destination, members=members(tar))
    Path(bundle).unlink()


def get_release_number():
    if 'dev' in __version__:
        print("You are working with a not stable version of Gammapy")
        print("Please specify a published notebooks release")
        exit()
    else:
        release = __version__.split('.dev', 1)[0]
        return release


def show_info_notebooks(outfolder, release):
    print("")
    print(
        "*** Enter the following commands below to get started with this version of Gammapy"
    )
    print(f"cd {outfolder}")
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
    "--out",
    default="gammapy-notebooks",
    help="Path where the versioned notebook files will be copied.",
    show_default=True,
)
@click.option("--release", default="", help="Number of release - ex: 0.18.2)")
def cli_download_notebooks(out, release):
    """Download notebooks"""
    release = get_release_number() if not release else release
    localfolder = Path(out) / release
    filename_env = f"gammapy-{release}-environment.yml"
    url_file_env = f"{RELEASES_BASE_URL}/install/{filename_env}"
    log.info(f"Downloading {url_file_env}")
    progress_download(url_file_env, localfolder/filename_env)
    filename_tar = f"notebooks-{release}.tar"
    tar_notebooks = f"{RELEASES_BASE_URL}/notebooks/{filename_tar}"
    tar_destination_file = localfolder / "notebooks.tar"
    log.info(f"Downloading {tar_notebooks}")
    progress_download(tar_notebooks, tar_destination_file)
    extract_bundle(tar_destination_file, localfolder)
    show_info_notebooks(localfolder, release)


@click.command(name="datasets")
@click.option(
    "--out", default="gammapy-datasets", help="Destination folder.", show_default=True,
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
