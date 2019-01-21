# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to download datasets and notebooks"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import click
from .downloadclasses import ComputePlan, ParallelDownload

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

    plan = ComputePlan(src, out, release, "notebooks")
    outfolder = plan.getlocalfolder()
    if release:
        plan.getenvironment()
    fl = plan.getfilelist()
    if fl:
        down = ParallelDownload(fl, outfolder, release, "notebooks")
        down.run()
        down.show_info()

@click.command(name="scripts")
@click.option("--src", default="", help="Specific script to download.")
@click.option(
    "--out",
    default="gammapy-scripts",
    help="Path where the versioned python scripts will be copied.",
    show_default=True,
)
@click.option("--release", default="", help="Gammapy release environment.")
def cli_download_scripts(src, out, release):
    """Download scripts"""

    plan = ComputePlan(src, out, release, "scripts")
    outfolder = plan.getlocalfolder()
    if release:
        plan.getenvironment()
    fl = plan.getfilelist()
    if fl:
        down = ParallelDownload(fl, outfolder, release, "scripts")
        down.run()
        down.show_info()

@click.command(name="datasets")
@click.option("--src", default="", help="Specific dataset to download.")
@click.option("--release", default="", help="Gammapy release environment.")
@click.option(
    "--out",
    default="gammapy-datasets",
    help="Path where datasets will be copied.",
    show_default=True,
)
def cli_download_datasets(src, out, release):
    """Download datasets"""

    plan = ComputePlan(src, out, release, "datasets")
    outfolder = plan.getlocalfolder()
    fl = plan.getfilelist()
    if fl:
        down = ParallelDownload(fl, outfolder, release, "datasets")
        down.run()
        down.show_info()


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

    plan = ComputePlan(src, out, release, "notebooks")
    outfolder = plan.getlocalfolder()
    if release:
        plan.getenvironment()
    fl = plan.getfilelist()
    if fl:
        down = ParallelDownload(fl, outfolder, release, "notebooks")
        down.run()

    plan = ComputePlan(src, out, release, "scripts")
    outfolder = plan.getlocalfolder()
    fl = plan.getfilelist()
    if fl:
        down = ParallelDownload(fl, outfolder, release, "scripts")
        down.run()

    plan = ComputePlan(src, out, release, "datasets", modetutorials=True)
    outfolder = plan.getlocalfolder()
    fl = plan.getfilelist()
    if fl:
        down = ParallelDownload(fl, outfolder, release, "all")
        down.run()
        down.show_info()
