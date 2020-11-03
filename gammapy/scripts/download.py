# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to download datasets and notebooks"""
import logging
import tarfile
from pathlib import Path
import click
from .downloadclasses import ComputePlan, ParallelDownload

BUNDLESIZE = 152  # in MB
log = logging.getLogger(__name__)


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
@click.option("--src", default="", help="Specific notebook to download.")
@click.option(
    "--out",
    default="gammapy-notebooks",
    help="Path where the versioned notebook files will be copied.",
    show_default=True,
)
@click.option("--release", default="", help="Number of release - ex: 0.12)")
@click.option(
    "--all",
    default=True,
    is_flag=True,
    help="Consider also other notebooks than tutorials",
)
@click.option("--modetutorials", default=False, hidden=True)
@click.option("--silent", default=True, is_flag=True, hidden=True)
def cli_download_notebooks(src, out, release, all, modetutorials, silent):
    """Download notebooks"""
    plan = ComputePlan(src, out, release, "notebooks", all_notebooks=all)
    if release:
        plan.getenvironment()
    down = ParallelDownload(
        plan.getfilelist(),
        plan.getlocalfolder(),
        release,
        "notebooks",
        modetutorials,
        silent,
    )
    down.run()
    print("")


@click.command(name="scripts")
@click.option("--src", default="", help="Specific script to download.")
@click.option(
    "--out",
    default="gammapy-scripts",
    help="Path where the versioned python scripts will be copied.",
    show_default=True,
)
@click.option("--release", default="", help="Number of release - ex: 0.12")
@click.option("--modetutorials", default=False, hidden=True)
@click.option("--silent", default=True, is_flag=True, hidden=False)
def cli_download_scripts(src, out, release, modetutorials, silent):
    """Download scripts"""
    plan = ComputePlan(src, out, release, "scripts")
    if release:
        plan.getenvironment()
    down = ParallelDownload(
        plan.getfilelist(),
        plan.getlocalfolder(),
        release,
        "scripts",
        modetutorials,
        silent,
    )
    down.run()
    print("")


@click.command(name="datasets")
@click.option("--src", default="", help="Specific dataset to download.")
@click.option(
    "--out", default="gammapy-datasets", help="Destination folder.", show_default=True,
)
@click.option("--release", default="", help="Number of release - ex: 0.12")
@click.option("--modetutorials", default=False, hidden=True)
@click.option("--silent", default=True, is_flag=True, hidden=True)
@click.option(
    "--tests",
    default=True,
    is_flag=True,
    help="Include datasets needed for tests. [default: True]",
    hidden=True,
)
def cli_download_datasets(src, out, release, modetutorials, silent, tests):
    """Download datasets"""
    plan = ComputePlan(
        src, out, release, "datasets", modetutorials=modetutorials, download_tests=tests
    )
    filelist = plan.getfilelist()
    localfolder = plan.getlocalfolder()
    down = ParallelDownload(
        filelist, localfolder, release, "datasets", modetutorials, silent,
    )
    # tar bundle
    if "bundle" in filelist:
        log.info(f"Downloading datasets from {filelist['bundle']['url']}")
        tar_destination_file = Path(localfolder) / "datasets.tar.gz"
        progress_download(filelist["bundle"]["url"], tar_destination_file)
        log.info(f"Extracting {tar_destination_file}")
        extract_bundle(tar_destination_file, localfolder)

    # specific collection
    else:
        down.run()
    down.show_info_datasets()


@click.command(name="tutorials")
@click.pass_context
@click.option("--src", default="", help="Specific tutorial to download.")
@click.option(
    "--out",
    default="gammapy-tutorials",
    help="Path where notebooks and datasets folders will be copied.",
    show_default=True,
)
@click.option("--release", default="", help="Number of release - ex: 0.12")
@click.option("--modetutorials", default=True, hidden=True)
@click.option("--silent", default=True, is_flag=True, hidden=True)
def cli_download_tutorials(ctx, src, out, release, modetutorials, silent):
    """Download notebooks, scripts and datasets"""
    ctx.forward(cli_download_notebooks)
    ctx.forward(cli_download_scripts)
    ctx.forward(cli_download_datasets)
