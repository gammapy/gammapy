# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test if Jupyter notebooks work."""
import os
import sys
import logging
from pathlib import Path
from pkg_resources import working_set
from shutil import copytree, rmtree
import yaml
from ..scripts.jupyter import notebook_test

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
                working_set.require(package)
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
    path_empty_nbs = Path("tutorials")
    rmtree(str(path_temp), ignore_errors=True)
    copytree(str(path_empty_nbs), str(path_temp))

    for notebook in get_notebooks():
        if requirement_missing(notebook):
            log.info(
                "Skipping notebook {} because requirement is missing.".format(
                    notebook["name"]
                )
            )
            continue

        filename = notebook["name"] + ".ipynb"
        path = path_temp / filename

        if not notebook_test(path):
            passed = False

    # tear down
    rmtree(str(path_temp), ignore_errors=True)

    if not passed:
        sys.exit("Some tests failed. Existing now.")


if __name__ == "__main__":
    main()
