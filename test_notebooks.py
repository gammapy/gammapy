#!/usr/bin/env python
"""
Auto-test IPython notebooks.
(this runs on travis-ci)
"""
import os
from astropy.extern.six import PY2

notebooks_for_py2 = [
    'notebooks/hess_spectrum_analysis.ipynb',
]

notebooks_for_py3 = [

]

notebooks_for_py23 = [
    'index.ipynb',
]

if PY2:
    notebooks = notebooks_for_py23 + notebooks_for_py2
else:
    notebooks = notebooks_for_py23 + notebooks_for_py3

print('*** Testing IPython notebooks ...')

for notebook in notebooks:
    cmd = 'cd $GAMMAPY_EXTRA; '
    cmd += 'runipy --quiet {}'.format(notebook)
    print('*** Executing: {}'.format(cmd))
    os.system(cmd)

print('*** ... finished testing IPython notebooks.')
