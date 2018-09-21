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
        ctx.obj["src"], ctx.obj["out"], ctx.obj["release"], "notebooks"
    )

    downloadproc.setup()
    downloadproc.files()
    downloadproc.run()


@click.command(name="datasets")
@click.pass_context
def cli_download_datasets(ctx):
    """Download datasets"""

    downloadproc = DownloadProcess(
        ctx.obj["src"], ctx.obj["out"], ctx.obj["release"], "datasets"
    )

    downloadproc.setup()
    downloadproc.files()
    downloadproc.run()

    downloadproc.show_info()


@click.command(name="tutorials")
@click.pass_context
def cli_download_tutorials(ctx):
    """Download tutorial notebooks and datasets"""

    downnotebooks = DownloadProcess(
        ctx.obj["src"], ctx.obj["out"], ctx.obj["release"], "notebooks"
    )
    downnotebooks.setup()
    downnotebooks.files()
    downnotebooks.run()

    downdatasets = DownloadProcess(
        ctx.obj["src"], ctx.obj["out"], ctx.obj["release"], "tutorials"
    )
    downdatasets.setup()
    downdatasets.files()
    downdatasets.run()

    downnotebooks.show_info()
