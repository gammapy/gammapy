#!/usr/bin/env python
"""Format code in notebooks cells using black."""

from black import format_str
from gammapy.extern.pathlib import Path
import logging
import nbformat
import os
import sys
import yaml

logging.basicConfig(level=logging.INFO)


def comment_magics(input):
    """Coment magic commands."""
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


# check gammapy-extra
if 'GAMMAPY_EXTRA' not in os.environ:
    logging.info('GAMMAPY_EXTRA environment variable not set.')
    logging.info('Running notebook tests requires gammapy-extra.')
    logging.info('Exiting now.')
    sys.exit()

# get list of notebooks
dirnbs = Path(os.environ['GAMMAPY_EXTRA']) / 'notebooks'
yamlfile = Path(os.environ['GAMMAPY_EXTRA']) / \
    'notebooks' / 'notebooks.yaml'
with open(str(yamlfile)) as fh:
    notebooks = yaml.safe_load(fh)

# scan notebooks
for notebook in notebooks:

    if not notebook['test']:
        logging.info(
            'Skipping notebook {} because test=false.'.format(notebook['name']))
        continue

    notebookfile = notebook['name'] + '.ipynb'
    filepath = dirnbs / notebookfile

    # read not formatted notebook
    nb = nbformat.read(str(filepath), as_version=nbformat.NO_CONVERT)

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
    nbformat.write(nb, str(filepath))
