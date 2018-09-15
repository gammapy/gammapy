#!/usr/bin/env python
"""
Test if IPython notebooks work.
"""
import os
import sys
import logging
from pkg_resources import working_set
from gammapy.extern.pathlib import Path
from gammapy.scripts.jupyter import test_notebook
import yaml

logging.basicConfig(level=logging.INFO)


def get_notebooks():
    """Read `notebooks.yaml` info."""
    filename = str(Path('tutorials') / 'notebooks.yaml')
    with open(filename) as fh:
        notebooks = yaml.safe_load(fh)
    return notebooks


def requirement_missing(notebook):
    """Check if one of the requirements is missing."""
    if notebook['requires'] is None:
        return False

    for package in notebook['requires'].split():
        try:
            working_set.require(package)
        except Exception as ex:
            return True
    return False


if 'GAMMAPY_EXTRA' not in os.environ:
    logging.info('GAMMAPY_EXTRA environment variable not set.')
    logging.info('Running notebook tests requires gammapy-extra.')
    logging.info('Exiting now.')
    sys.exit()

try:
    path_datasets = Path(os.environ['GAMMAPY_EXTRA']) / 'datasets'
    os.symlink(str(path_datasets), 'datasets')
except Exception as ex:
    logging.error('It was not possible to create a /datasets symlink')
    logging.error(ex)
    sys.exit()


passed = True
yamlfile = get_notebooks()
dirnbs = Path('tutorials')

for notebook in yamlfile:
    if requirement_missing(notebook):
        logging.info('Skipping notebook {} because requirement is missing.'.format(
            notebook['name']))
        continue

    filename = notebook['name'] + '.ipynb'
    path = dirnbs / filename

    if not test_notebook(path):
        passed = False

os.unlink('datasets')
assert passed
