#!/usr/bin/env python
"""
Test if IPython notebooks work.
"""
import os
import sys
import subprocess
import logging
from pkg_resources import working_set
from pprint import pprint
import yaml

logging.basicConfig(level=logging.INFO)

status_codes = dict(
    success=0,
    error_nb_failed=1,
    error_no_gammapy_extra=2,
)

if 'GAMMAPY_EXTRA' not in os.environ:
    logging.info('GAMMAPY_EXTRA environment variable not set.')
    logging.info('Running notebook tests requires gammapy-extra.')
    logging.info('Exiting now.')
    sys.exit(status_codes['error_no_gammapy_extra'])

status_code = status_codes['success']


def get_notebooks():
    """Read `notebooks.yaml` info."""
    filename = os.environ['GAMMAPY_EXTRA'] + '/notebooks/notebooks.yaml'
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
            logging.warning('Skipping notebook {} because dependency {} is missing.'.format(notebook['name'], package))
            return True

    return False


notebooks = get_notebooks()
pprint(notebooks)

logging.info('Python executable: {}'.format(sys.executable))
logging.info('Python version: {}'.format(sys.version))
logging.info('Testing IPython notebooks ...')

for notebook in notebooks:

    if not notebook['test']:
        logging.info('Skipping notebook {} because test=false.'.format(notebook['name']))
        continue

    if requirement_missing(notebook):
        continue

    # For testing how `subprocess.Popen` works:
    # cmd = 'pwd && echo "hi" && asdf'

    cmd = 'pwd; '
    cmd += 'echo $GAMMAPY_EXTRA; '
    # cmd += 'export GAMMAPY_EXTRA={}; '.format(os.environ['GAMMAPY_EXTRA'])
    # cmd += 'cd $GAMMAPY_EXTRA/notebooks; '
    cmd += sys.executable + ' -m runipy.main {}.ipynb'.format(notebook['name'])
    logging.info('Executing: {}'.format(cmd))
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
        cwd=os.environ['GAMMAPY_EXTRA'] + '/notebooks',
    )
    stdout, stderr = proc.communicate()
    logging.info('Exit status code: {}'.format(proc.returncode))
    logging.info('stdout:\n{}'.format(stdout.decode('utf8')))
    logging.info('stderr:\n{}'.format(stderr.decode('utf8')))

    if proc.returncode != status_codes['success']:
        status_code = status_codes['error_nb_failed']

logging.info('... finished testing IPython notebooks.')
logging.info('Total exit status code of test_notebook.py is: {}'.format(status_code))
sys.exit(status_code)
