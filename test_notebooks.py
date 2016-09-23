#!/usr/bin/env python
"""
Auto-test IPython notebooks.
(this runs on travis-ci)
"""
import os
import sys
import subprocess

notebooks_for_py2 = [
    'hess_spectrum_analysis.ipynb',
]

notebooks_for_py3 = [

]

notebooks_for_py23 = [
    'hess_image_analysis.ipynb',
    'fermi_ts_image.ipynb',
    'fermi_2fhl.ipynb',
]

print('*** Python executable: {}'.format(sys.executable))
print('*** Python version: {}'.format(sys.version))

if sys.version_info.major == 2:
    print('*** This is Python 2')
    notebooks = notebooks_for_py23 + notebooks_for_py2
else:
    notebooks = notebooks_for_py23 + notebooks_for_py3

print('*** Testing IPython notebooks ...')

status_code = 0  # mean no errors so far.

for notebook in notebooks:
    # For testing how `subprocess.Popen` works:
    # cmd = 'pwd && echo "hi" && asdf'

    cmd = 'which python; '
    cmd += 'echo $GAMMAPY_EXTRA; '
    cmd += 'pwd; '
    # cmd += 'export GAMMAPY_EXTRA={}; '.format(os.environ['GAMMAPY_EXTRA'])
    # cmd += 'cd $GAMMAPY_EXTRA/notebooks; '
    cmd += 'runipy {}'.format(notebook)
    print('*** Executing: {}'.format(cmd))
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
    print('*** Exit status code: {}'.format(proc.returncode))
    print('*** stdout:\n{}'.format(stdout.decode('utf8')))
    print('*** stderr:\n{}'.format(stderr.decode('utf8')))

    if proc.returncode != 0:
        status_code = 1  # If one test fails, return fail as total exit status.

print('*** ... finished testing IPython notebooks.')
print('*** Total exit status code of test_notebook.py is: {}'.format(status_code))
sys.exit(status_code)
