# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to download datasets and notebooks from gammapy-extra GitHub repo.
GitHub REST API is used to scan the tree-folder strucutre and get commmit hash.
https://developer.github.com/v3/
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import click
import logging
from .downloadclass import DownloadProcess

log = logging.getLogger(__name__)


@click.command(name="notebooks")
@click.option("--src", default="", help="Specific notebook to download.")
@click.option(
    "--out",
    default="gammapy-notebooks",
    help="Path where the versioned notebook files will be copied.",
    show_default=True,
)
@click.option("--release", default="", help="Gammapy release environment.")
def cli_download_notebooks(src, out, release):
    """Download notebooks"""
    downloadproc = DownloadProcess(src, out, release, "notebooks", False)

    downloadproc.setup()
    downloadproc.files()
    downloadproc.run()


@click.command(name="datasets")
@click.option("--src", default="", help="Specific dataset to download.")
@click.option(
    "--out",
    default="gammapy-datasets",
    help="Path where datasets will be copied.",
    show_default=True,
)
def cli_download_datasets(src, out):
    """Download datasets"""
    downloadproc = DownloadProcess(src, out, "", "datasets", False)

    downloadproc.setup()
    downloadproc.files()
    downloadproc.run()

    downloadproc.show_info()


@click.command(name="tutorials")
@click.option("--src", default="", help="Specific tutorial to download.")
@click.option(
    "--out",
    default="gammapy-tutorials",
    help="Path where notebooks and datasets folders will be copied.",
    show_default=True,
)
@click.option("--release", default="", help="Gammapy release environment.")
def cli_download_tutorials(src, out, release):
    """Download tutorial notebooks and datasets"""
    downnotebooks = DownloadProcess(src, out, release, "notebooks", True)
    downnotebooks.setup()
    downnotebooks.files()
    downnotebooks.run()

    downdatasets = DownloadProcess(src, out, release, "datasets", True)
    downdatasets.setup()
    downdatasets.files()
    downdatasets.run()

    downnotebooks.show_info()
