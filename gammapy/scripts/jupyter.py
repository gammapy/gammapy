# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to perform devel management actions on jupyter notebooks."""

from __future__ import absolute_import, division, print_function, unicode_literals
from black import format_str
import click
import logging
import nbformat
import subprocess
import time
from ..extern.pathlib import Path

log = logging.getLogger(__name__)


@click.command(name='black')
@click.pass_context
def cli_jupyter_black(ctx):
    """Format code cells with black."""

    for path in build_nblist(ctx):
        rawnb = read_notebook(path)

        for cell in rawnb.cells:
            fmt = cell['source']
            if cell['cell_type'] == 'code':
                try:
                    fmt = '\n'.join(tag_magics(fmt))
                    has_semicolon = fmt.endswith(';')
                    fmt = format_str(src_contents=fmt,
                                     line_length=79).rstrip()
                    if has_semicolon:
                        fmt += ';'
                except Exception as ex:
                    logging.info(ex)
                fmt = fmt.replace("###-MAGIC COMMAND-", "")
            cell['source'] = fmt

        if rawnb:
            nbformat.write(rawnb, str(path))
            log.info('Jupyter notebook {} painted in black.'.format(str(path)))


@click.command(name='stripout')
@click.pass_context
def cli_jupyter_stripout(ctx):
    """Strip output cells."""

    for path in build_nblist(ctx):
        rawnb = read_notebook(path)

        for cell in rawnb.cells:
            if cell['cell_type'] == 'code':
                cell['outputs'] = []

        if rawnb:
            nbformat.write(rawnb, str(path))
            log.info('Jupyter notebook {} output stripped.'.format(str(path)))


@click.command(name='execute')
@click.pass_context
def cli_jupyter_execute(ctx):
    """Execute jupyter notebook."""

    for path in build_nblist(ctx):
        run_notebook(path)


@click.command(name='test')
@click.pass_context
def cli_jupyter_test(ctx):
    """Check if Jupyter notebook is broken."""

    for path in build_nblist(ctx):
        run_notebook(path)
        rawnb = read_notebook(path)

        if rawnb:
            report = ""
            passed = True
            log.info("   ... Testing {}".format(str(path)))
            for cell in rawnb.cells:
                if 'outputs' in cell.keys():
                    for output in cell['outputs']:
                        if output['output_type'] == 'error':
                            passed = False
                            traceitems = ["--TRACEBACK: "]
                            for o in output['traceback']:
                                traceitems.append("{}".format(o))
                            traceback = "\n".join(traceitems)
                            infos = "\n\n{} in cell [{}]\n\n" \
                                    "--SOURCE CODE: \n{}\n\n".format(
                                        output['ename'],
                                        cell['execution_count'],
                                        cell['source']
                                    )
                            report = report + infos + traceback
            if passed:
                log.info("   ... {} Passed".format(str(path)))
            else:
                log.info("   ... {} Failed".format(str(path)))
                log.info(report)


def tag_magics(cellcode):
    """Comment magic commands when formatting cells."""

    lines = cellcode.splitlines(False)
    for line in lines:
        if line.startswith("%") or line.startswith("!"):
            magic_line = "###-MAGIC COMMAND-" + line
            yield magic_line
        else:
            yield line


def build_nblist(ctx):
    """Fill list of notebooks in a the given folder."""

    if ctx.obj['file']:
        yield Path(ctx.obj['file'])
    else:
        for f in Path(ctx.obj['fold']).iterdir():
            if f.name.endswith('.ipynb'):
                yield f


def run_notebook(path):
    """Execute a Jupyter notebook."""

    try:
        t = time.time()
        subprocess.call(
            "jupyter nbconvert "
            "--allow-errors "
            "--ExecutePreprocessor.timeout=None "
            "--ExecutePreprocessor.kernel_name=python3 "
            "--to notebook "
            "--inplace "
            "--execute '{}'".format(path),
            shell=True)
        t = (time.time() - t) / 60
        log.info('Executing duration: {:.2f} mn'.format(t))
    except Exception as ex:
        log.error('Error executing file {}'.format(str(path)))
        log.error(ex)


def read_notebook(path):
    """Read a Jupyter notebook raw structure."""
    try:
        return nbformat.read(str(path), as_version=nbformat.NO_CONVERT)
    except Exception as ex:
        log.error('Error parsing file {}'.format(str(path)))
        log.error(ex)
        return
