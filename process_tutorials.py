#!/usr/bin/env python
"""
Process tutorials notebooks for publication in documentation.
"""
import logging
import os
import subprocess
import sys
from shutil import copytree, rmtree
from gammapy.extern.pathlib import Path
from gammapy.scripts.jupyter import test_notebook

logging.basicConfig(level=logging.INFO)

if 'GAMMAPY_EXTRA' not in os.environ:
    logging.info('GAMMAPY_EXTRA environment variable not set.')
    logging.info('Running notebook tests requires gammapy-extra.')
    logging.info('Exiting now.')
    sys.exit()

# make datasets symlink
try:
    path_datasets = Path(os.environ['GAMMAPY_EXTRA']) / 'datasets'
    os.symlink(str(path_datasets), 'datasets')
except Exception as ex:
    logging.error('It was not possible to create a /datasets symlink')
    logging.error(ex)
    sys.exit()

# prepare folder structure
path_temp = Path('temp')
path_empty_nbs = Path('tutorials')
path_filled_nbs = Path('docs') / 'notebooks'
path_static_nbs = Path('docs') / '_static' / 'notebooks'

rmtree(str(path_temp), ignore_errors=True)
rmtree(str(path_filled_nbs), ignore_errors=True)
rmtree(str(path_static_nbs), ignore_errors=True)

# work in temporal folder


def ignorefiles(d, files): return [
    f
    for f in files
    if os.path.isfile(os.path.join(d, f))
    and f[-6:] != '.ipynb'
    and f[-4:] != '.png'
]


copytree(str(path_empty_nbs), str(path_temp), ignore=ignorefiles)

# test /run
passed = True
for path in path_temp.glob("*.ipynb"):
    if not test_notebook(path):
        passed = False

if passed:
    # convert into scripts
    def ignoreall(d, files): return [
        f
        for f in files
        if os.path.isfile(os.path.join(d, f))
        and f[-6:] != '.ipynb'
    ]
    copytree(str(path_empty_nbs), str(path_static_nbs), ignore=ignoreall)
    for path in path_static_nbs.glob("*.ipynb"):
        subprocess.call(
            "jupyter nbconvert --to script '{}'".format(str(path)),
            shell=True)

    # copy filled notebooks
    copytree(str(path_temp), str(path_filled_nbs), ignore=ignorefiles)

else:
    logging.info('Tests have not passed.')
    logging.info('Tutorials not ready for documentation building process.')
    rmtree(str(path_static_nbs), ignore_errors=True)

# tear down
rmtree(str(path_temp), ignore_errors=True)
os.unlink('datasets')
