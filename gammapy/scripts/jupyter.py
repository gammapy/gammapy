# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to perform actions on jupyter notebooks."""

from __future__ import absolute_import, division, print_function, unicode_literals
import click
import logging
import os
import subprocess
import sys
import time

log = logging.getLogger(__name__)
OFF = [
    "CTADATA",
    "GAMMA_CAT",
    "GAMMAPY_DATA",
    "GAMMAPY_EXTRA",
    "GAMMAPY_FERMI_LAT_DATA",
]


@click.command(name="run")
@click.pass_context
@click.option(
    "--tutor",
    is_flag=True,
    default=False,
    help="Tutorials environment?",
    show_default=True,
)
def cli_jupyter_run(ctx, tutor):
    """Execute Jupyter notebooks."""

    with environment(OFF, tutor, ctx):
        for path in ctx.obj["paths"]:
            execute_notebook(path)


def execute_notebook(path, loglevel=20):
    """Execute a Jupyter notebook."""

    kernel_name = "python3"
    if sys.version_info[0] < 3:
        kernel_name = "python2"

    try:
        t = time.time()
        subprocess.call(
            "jupyter nbconvert "
            "--allow-errors "
            "--log-level={} "
            "--ExecutePreprocessor.timeout=None "
            "--ExecutePreprocessor.kernel_name={} "
            "--to notebook "
            "--inplace "
            "--execute '{}'".format(loglevel, kernel_name, path),
            shell=True,
        )
        t = (time.time() - t) / 60
        log.info("   ... Executing duration: {:.2f} mn".format(t))
    except Exception as ex:
        log.error("Error executing file {}".format(str(path)))
        log.error(ex)


@click.command(name="strip")
@click.pass_context
def cli_jupyter_strip(ctx):
    """Strip output cells."""
    import nbformat

    for path in ctx.obj["paths"]:
        rawnb = nbformat.read(str(path), as_version=nbformat.NO_CONVERT)

        for cell in rawnb.cells:
            if cell["cell_type"] == "code":
                cell["execution_count"] = None
                cell["outputs"] = []

        nbformat.write(rawnb, str(path))
        log.info("Jupyter notebook {} stripped out.".format(str(path)))


@click.command(name="black")
@click.pass_context
def cli_jupyter_black(ctx):
    """Format code cells with black."""
    import nbformat

    for path in ctx.obj["paths"]:
        rawnb = nbformat.read(str(path), as_version=nbformat.NO_CONVERT)
        blacknb = BlackNotebook(rawnb)
        blacknb.blackformat()
        rawnb = blacknb.rawnb
        nbformat.write(rawnb, str(path))
        log.info("Jupyter notebook {} blacked.".format(str(path)))


class BlackNotebook:
    """Manage the process of black formatting."""

    MAGIC_TAG = "###-MAGIC TAG-"

    def __init__(self, rawnb):

        self.rawnb = rawnb

    def blackformat(self):
        """Format code cells."""
        from black import format_str

        for cell in self.rawnb.cells:
            fmt = cell["source"]
            if cell["cell_type"] == "code":
                try:
                    fmt = "\n".join(self.tag_magics(fmt))
                    has_semicolon = fmt.endswith(";")
                    fmt = format_str(src_contents=fmt, line_length=79).rstrip()
                    if has_semicolon:
                        fmt += ";"
                except Exception as ex:
                    logging.info(ex)
                fmt = fmt.replace(self.MAGIC_TAG, "")
            cell["source"] = fmt

    def tag_magics(self, cellcode):
        """Comment magic commands."""

        lines = cellcode.splitlines(False)
        for line in lines:
            if line.startswith("%") or line.startswith("!"):
                magic_line = self.MAGIC_TAG + line
                yield magic_line
            else:
                yield line


@click.command(name="test")
@click.pass_context
@click.option(
    "--tutor",
    is_flag=True,
    default=False,
    help="Tutorials environment?",
    show_default=True,
)
def cli_jupyter_test(ctx, tutor):
    """Check if Jupyter notebooks are broken."""

    with environment(OFF, tutor, ctx):
        for path in ctx.obj["paths"]:
            notebook_test(path)


def notebook_test(path):
    """Execute and parse a Jupyter notebook exposing broken cells."""
    import nbformat

    passed = True
    log.info("   ... TESTING {}".format(str(path)))
    execute_notebook(path, 30)
    rawnb = nbformat.read(str(path), as_version=nbformat.NO_CONVERT)

    for cell in rawnb.cells:
        if "outputs" in cell.keys():
            for output in cell["outputs"]:
                if output["output_type"] == "error":
                    passed = False
                    traceitems = ["--TRACEBACK: "]
                    for o in output["traceback"]:
                        traceitems.append("{}".format(o))
                    traceback = "\n".join(traceitems)
                    infos = "\n\n{} in cell [{}]\n\n" "--SOURCE CODE: \n{}\n\n".format(
                        output["ename"], cell["execution_count"], cell["source"]
                    )
                    report = infos + traceback
                    break
        if not passed:
            break

    if passed:
        log.info("   ... PASSED")
        return True
    else:
        log.info("   ... FAILED")
        log.info(report)
        return False


class environment:
    """
    Helper for setting environmental variables
    """

    def __init__(self, envs, tutor, ctx):
        self.envs = envs
        self.tutor = tutor
        self.ctx = ctx

    def __enter__(self):

        self.old = os.environ
        if self.tutor:
            for item in self.envs:
                if item in os.environ:
                    del os.environ[item]
                    logging.info("Unsetting {} environment variable.".format(item))
            abspath = self.ctx.obj["pathsrc"].absolute()
            datapath = abspath.parent / "datasets"
            if abspath.is_file():
                datapath = abspath.parent.parent / "datasets"
            os.environ["GAMMAPY_DATA"] = str(datapath)
            logging.info("Setting GAMMAPY_DATA={}".format(os.environ["GAMMAPY_DATA"]))

    def __exit__(self, type, value, traceback):
        if self.tutor:
            os.environ = self.old
            logging.info("Environment variables recovered.")
