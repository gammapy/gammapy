# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to download datasets and notebooks"""
import logging
import tarfile
from pathlib import Path
import click
from .downloadclasses import ComputePlan, ParallelDownload

BUNDLESIZE = 152  # in MB
log = logging.getLogger(__name__)

TAR_DATASETS = "https://github.com/gammapy/gammapy-data/tarball/master"  # curated datasets bundle


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


@click.command(name="notebooks")
@click.option(
    "--out",
    default="gammapy-notebooks",
    help="Path where the versioned notebook files will be copied.",
    show_default=True,
)
@click.option("--release", default="", help="Number of release - ex: 0.12)")
@click.option("--modetutorials", default=False, hidden=True)
def cli_download_notebooks(out, release, modetutorials):
    """Download notebooks"""
    plan = ComputePlan(out, release, "notebooks")
    if release:
        plan.getenvironment()
    down = ParallelDownload(
        plan.getfilelist(),
        plan.getlocalfolder(),
        release,
        "notebooks",
        modetutorials
    )
    down.run()
    print("")


@click.command(name="datasets")
@click.option(
    "--out", default="gammapy-datasets", help="Destination folder.", show_default=True,
)
@click.option("--release", default="", hidden=True)
@click.option("--modetutorials", default=False, hidden=True)
def cli_download_datasets(out, release, modetutorials):
    """Download datasets"""
    localfolder = Path(out) / "datasets" if modetutorials else Path(out)
    log.info(f"Downloading datasets from {TAR_DATASETS}")
    tar_destination_file = localfolder / "datasets.tar.gz"
    progress_download(TAR_DATASETS, tar_destination_file)
    log.info(f"Extracting {tar_destination_file}")
    extract_bundle(tar_destination_file, localfolder)
    show_info_datasets(localfolder, modetutorials, release)


@click.command(name="tutorials")
@click.pass_context
@click.option(
    "--out",
    default="gammapy-tutorials",
    help="Path where notebooks and datasets folders will be copied.",
    show_default=True,
)
@click.option("--release", default="", help="Number of release - ex: 0.12")
@click.option("--modetutorials", default=True, hidden=True)
def cli_download_tutorials(ctx, out, release, modetutorials):
    """Download notebooks and datasets"""
    ctx.forward(cli_download_notebooks)
    ctx.forward(cli_download_datasets)


def show_info_datasets(outfolder, modetutorials, release):
    print("")
    if modetutorials and release:
        print(
            "*** Enter the following commands below to get started with this version of Gammapy"
        )
        print(f"cd {outfolder.parent}")
        print(f"conda env create -f gammapy-{release}-environment.yml")
        print(f"conda activate gammapy-{release}")
        print("jupyter lab")
        print("")
    print("*** You might want to declare GAMMAPY_DATA env variable")
    print(f"export GAMMAPY_DATA={outfolder}")
    print("")
