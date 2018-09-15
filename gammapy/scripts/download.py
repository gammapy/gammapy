# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to download datasets and notebooks from gammapy-extra GitHub repo.
GitHub REST API is used to scan the tree-folder strucutre and get commmit hash.
https://developer.github.com/v3/
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import click
import logging
from ..extern.pathlib import Path
from .downloadclass import DownloadProcess

log = logging.getLogger(__name__)


@click.command(name="notebooks")
@click.pass_context
def cli_download_notebooks(ctx):
    """Download notebooks"""

    downloadproc = DownloadProcess(
        "gammapy",
        "tutorials",
        ctx.obj["specfile"],
        ctx.obj["specfold"],
        ctx.obj["release"],
        ["../environment.yml"],
        Path(ctx.obj["localfold"]),
        ctx.obj["recursive"],
    )

    downloadproc.check_hash()
    downloadproc.label_version()
    downloadproc.build_folders()
    downloadproc.build_files()
    downloadproc.run()


@click.command(name="datasets")
@click.pass_context
def cli_download_datasets(ctx):
    """Download datasets"""

    downloadproc = DownloadProcess(
        "gammapy-extra",
        "datasets",
        ctx.obj["specfile"],
        ctx.obj["specfold"],
        ctx.obj["release"],
        [],
        Path(ctx.obj["localfold"]) / "datasets",
        ctx.obj["recursive"],
    )

    downloadproc.check_hash()
    downloadproc.build_folders()
    downloadproc.build_files()
    downloadproc.run()


@click.command(name="tutorials")
@click.pass_context
def cli_download_tutorials(ctx):
    """Download tutorial notebooks and datasets"""

    if ctx.obj["specfile"] or ctx.obj["specfile"]:
        log.info("--file and --foder are not allowed options for tutorials.")
    if not ctx.obj["recursive"]:
        log.info("--recursive is True for tutorials.")

    downnotebooks = DownloadProcess(
        "gammapy",
        "tutorials",
        "",
        "",
        ctx.obj["release"],
        ["../environment.yml"],
        Path(ctx.obj["localfold"]),
        True,
    )

    downnotebooks.check_hash()
    downnotebooks.label_version()
    downnotebooks.build_folders()
    downnotebooks.build_files(tutorials=True)
    downnotebooks.run()

    downdatasets = DownloadProcess(
        "gammapy-extra",
        "datasets",
        "",
        "",
        ctx.obj["release"],
        [],
        Path(ctx.obj["localfold"]) / "datasets",
        True,
    )

    downdatasets.check_hash()
    downdatasets.build_folders()
    downdatasets.build_files(tutorials=True, datasets=True)
    downdatasets.run()
