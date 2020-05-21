# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test if Jupyter notebooks work."""
import logging
import os
import shutil
import sys
from pathlib import Path
import pkg_resources
import yaml
from gammapy.scripts.jupyter import notebook_test

log = logging.getLogger(__name__)


def get_notebooks():
    """Read `notebooks.yaml` info."""
    path = Path("tutorials") / "notebooks.yaml"
    with path.open() as fh:
        return yaml.safe_load(fh)


def requirement_missing(notebook):
    """Check if one of the requirements is missing."""
    if "requires" in notebook:
        if notebook["requires"] is None:
            return False
        for package in notebook["requires"].split():
            try:
                pkg_resources.working_set.require(package)
            except Exception:
                return True
    return False


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
    shutil.rmtree(path_temp, ignore_errors=True)

    for notebook in get_notebooks():
        if requirement_missing(notebook):
            log.info(f"Skipping notebook (requirement missing): {notebook['name']}")
            continue

        filename = notebook["name"] + ".ipynb"
        path = path_temp / filename

        # TODO: get notebook path from dict and copy file into temp
        # path_empty_nbs = Path("tutorials")
        # shutil.copy(src_path, path)

        if not notebook_test(path):
            passed = False

    # tear down
    shutil.rmtree(path_temp, ignore_errors=True)

    if not passed:
        sys.exit("Some tests failed. Existing now.")


if __name__ == "__main__":
    main()
