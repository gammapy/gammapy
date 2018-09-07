# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to perform devel management actions on jupyter notebooks."""

from __future__ import absolute_import, division, print_function, unicode_literals
from black import format_str
import click
import logging
import nbformat
import subprocess
import sys
import testipynb
import time
from unittest import TestCase
from ..extern.pathlib import Path


log = logging.getLogger(__name__)


@click.command(name='black')
@click.pass_context
def cli_jupyter_black(ctx):
    """Format code cells with black."""

    for path in build_nblist(ctx):
        try:
            nb = nbformat.read(str(path), as_version=nbformat.NO_CONVERT)
        except Exception as ex:
            log.error('Error parsing file {}'.format(str(path)))
            log.error(ex)
            sys.exit()

        for cell in nb.cells:
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

        nbformat.write(nb, str(path))
        log.info('Jupyter notebook {} painted in black.'.format(str(path)))


@click.command(name='stripout')
@click.pass_context
def cli_jupyter_stripout(ctx):
    """Strip output cells."""

    for path in build_nblist(ctx):
        try:
            nb = nbformat.read(str(path), as_version=nbformat.NO_CONVERT)
        except Exception as ex:
            log.error('Error parsing file {}'.format(str(path)))
            log.error(ex)
            sys.exit()

        for cell in nb.cells:
            if cell['cell_type'] == 'code':
                cell['outputs'] = []

        nbformat.write(nb, str(path))
        log.info('Jupyter notebook {} output stripped.'.format(str(path)))


@click.command(name='execute')
@click.pass_context
def cli_jupyter_execute(ctx):
    """Execute jupyter notebook."""

    for path in build_nblist(ctx):
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


@click.command(name='test')
@click.pass_context
def cli_jupyter_test(ctx):
    """Check if jupyter notebook is broken."""

    path = Path(ctx.obj['file'])
    folder = Path(ctx.obj['fold'])
    ignorelist = []

    if ctx.obj['file']:
        # ignore all files except --file
        for f in path.parent.iterdir():
            if path.name != f.name and f.name.endswith('.ipynb'):
                nbname = f.name.replace('.ipynb', '')
                ignorelist.append(nbname)
        folder = path.parent

    testnb = testipynb.TestNotebooks(
        directory=str(folder), ignore=ignorelist)
    TestCase.assertTrue(testnb, testnb.run_tests())


def tag_magics(input):
    """Comment magic commands when formatting cells."""

    lines = input.splitlines(False)
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
