# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to download datasets and notebooks"""
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
@click.option("--modetutorials", default=False, hidden=True)
def cli_download_notebooks(src, out, release, modetutorials):
    """Download notebooks"""
    plan = ComputePlan(src, out, release, "notebooks")
    if release:
        plan.getenvironment()
    down = ParallelDownload(
        plan.getfilelist(), plan.getlocalfolder(), release, "notebooks", modetutorials
    )
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
@click.option("--modetutorials", default=False, hidden=True)
def cli_download_scripts(src, out, release, modetutorials):
    """Download scripts"""
    plan = ComputePlan(src, out, release, "scripts")
    if release:
        plan.getenvironment()
    down = ParallelDownload(
        plan.getfilelist(), plan.getlocalfolder(), release, "scripts", modetutorials
    )
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
@click.option("--modetutorials", default=False, hidden=True)
def cli_download_datasets(src, out, release, modetutorials):
    """Download datasets"""
    plan = ComputePlan(src, out, release, "datasets", modetutorials=modetutorials)
    down = ParallelDownload(
        plan.getfilelist(), plan.getlocalfolder(), release, "datasets", modetutorials
    )
    down.run()
    down.show_info()


@click.command(name="tutorials")
@click.pass_context
@click.option("--src", default="", help="Specific tutorial to download.")
@click.option(
    "--out",
    default="gammapy-tutorials",
    help="Path where notebooks and datasets folders will be copied.",
    show_default=True,
)
@click.option("--release", default="", help="Gammapy release environment.")
@click.option("--modetutorials", default=True, hidden=True)
def cli_download_tutorials(ctx, src, out, release, modetutorials):
    """Download notebooks, scripts and datasets"""
    ctx.forward(cli_download_notebooks)
    ctx.forward(cli_download_scripts)
    ctx.forward(cli_download_datasets)
