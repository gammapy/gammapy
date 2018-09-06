#!/usr/bin/env python
"""
Test if IPython notebooks work.
"""
import os
import sys
import unittest
import logging
from pkg_resources import working_set
from pprint import pprint
from gammapy.extern.pathlib import Path
from gammapy.extern import testipynb
import yaml

logging.basicConfig(level=logging.INFO)

if 'GAMMAPY_EXTRA' not in os.environ:
    logging.info('GAMMAPY_EXTRA environment variable not set.')
    logging.info('Running notebook tests requires gammapy-extra.')
    logging.info('Exiting now.')
    sys.exit()


def get_notebooks():
    """Read `notebooks.yaml` info."""
    filename = str(
        Path(os.environ['GAMMAPY_EXTRA']) / 'notebooks' / 'notebooks.yaml')
    logging.info('')
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
            logging.warning('Skipping notebook {} because dependency {} is missing.'.format(
                notebook['name'], package))
            return True

    return False


class TestNotebooks(unittest.TestCase):

    def test_notebooks(self):

        logging.info('Python executable: {}'.format(sys.executable))
        logging.info('Python version: {}'.format(sys.version))
        logging.info('Testing IPython notebooks...')

        dirnbs = Path(os.environ['GAMMAPY_EXTRA']) / 'notebooks'
        nbfiles = [f.name for f in dirnbs.iterdir() if
                   f.name.endswith('.ipynb')]
        notebooks = get_notebooks()
        pprint(notebooks)
        ignorelist = []
        yamllist = []

        for notebook in notebooks:

            notebookfile = notebook['name']
            yamllist.append(notebookfile)

            if not notebook['test']:
                logging.info(
                    'Skipping notebook {} because test=false.'.format(notebook['name']))
                ignorelist.append(notebookfile)
                continue

            if requirement_missing(notebook):
                logging.info('Skipping notebook {} because requirement is missing.'.format(
                    notebook['name']))
                ignorelist.append(notebookfile)
                continue

        for nbfile in nbfiles:
            nbname = nbfile.replace('.ipynb', '')
            if nbname not in yamllist:
                ignorelist.append(nbname)

        testnb = testipynb.TestNotebooks(
            directory=str(dirnbs), ignore=ignorelist)
        self.assertTrue(testnb.run_tests())


if __name__ == "__main__":
    unittest.main()
