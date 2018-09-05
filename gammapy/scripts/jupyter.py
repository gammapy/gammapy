# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to perform devel management actions on jupyter notebooks."""

from __future__ import absolute_import, division, print_function, unicode_literals
from black import format_str
import click
import logging
import nbformat
import sys
from ..extern.pathlib import Path

log = logging.getLogger(__name__)


@click.command(name='black')
@click.pass_context
def cli_jupyter_black(ctx):
    """Format notebook cells with black."""

    jupyterfile = Path(ctx.obj['file'])

    try:
        nb = nbformat.read(str(jupyterfile), as_version=nbformat.NO_CONVERT)
    except Exception as ex:
        log.error('Error parsing file {}'.format(str(jupyterfile)))
        log.error(ex)
        sys.exit()

    # paint cells in black
    for cellnumber, cell in enumerate(nb.cells):
        fmt = nb.cells[cellnumber]['source']
        if nb.cells[cellnumber]['cell_type'] == 'code':
            try:
                fmt = comment_magics(fmt)
                fmt = format_str(src_contents=fmt,
                                 line_length=79).rstrip()
            except Exception as ex:
                logging.info(ex)
            fmt = fmt.replace("###-MAGIC COMMAND-", "")
        nb.cells[cellnumber]['source'] = fmt

    # write formatted notebook
    nbformat.write(nb, str(jupyterfile))

    # inform
    print('Jupyter notebook {} painted in black.'.format(str(jupyterfile)))


def comment_magics(input):
    """Coment magic commands when formatting cells."""

    lines = input.splitlines(True)
    output = ""
    for line in lines:
        new_line = ""
        if line.startswith("%") or line.startswith("!"):
            new_line = new_line + "###-MAGIC COMMAND-" + line
        if new_line:
            output = output + new_line
        else:
            output = output + line
    return output
