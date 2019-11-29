# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test if Jupyter notebooks work."""
import logging
import os
import shutil
import sys
from pathlib import Path
import yaml
from gammapy.scripts.jupyter import notebook_test

log = logging.getLogger(__name__)


def get_notebooks():
    """Read `notebooks.yaml` info."""
    path = Path("tutorials") / "notebooks.yaml"
    with path.open() as fh:
        return yaml.safe_load(fh)


def main():
    logging.basicConfig(level=logging.INFO)

    if "GAMMAPY_DATA" not in os.environ:
        log.info("GAMMAPY_DATA environment variable not set.")
        log.info("Running notebook tests requires this environment variable.")
        log.info("Exiting now.")
        sys.exit()

    passed = True

    # setup
    path_temp = Path("temp")
    path_empty_nbs = Path("tutorials")
    shutil.rmtree(path_temp, ignore_errors=True)
    shutil.copytree(path_empty_nbs, path_temp)

    for notebook in get_notebooks():

        filename = notebook["name"] + ".ipynb"
        path = path_temp / filename

        if not notebook_test(path):
            passed = False

    # tear down
    shutil.rmtree(path_temp, ignore_errors=True)

    if not passed:
        sys.exit("Some tests failed. Existing now.")


if __name__ == "__main__":
    main()
