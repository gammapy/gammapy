#!/usr/bin/env python
"""
Test if IPython notebooks work.
"""
import os
import sys
import logging
from pkg_resources import working_set
from gammapy.extern.pathlib import Path
from gammapy.scripts.jupyter import notebook_test
import yaml

logging.basicConfig(level=logging.INFO)


def get_notebooks():
    """Read `notebooks.yaml` info."""
    filename = str(Path("tutorials") / "notebooks.yaml")
    with open(filename) as fh:
        notebooks = yaml.safe_load(fh)
    return notebooks


def requirement_missing(notebook):
    """Check if one of the requirements is missing."""
    if "requires" in notebook:
        if notebook["requires"] is None:
            return False
        for package in notebook["requires"].split():
            try:
                working_set.require(package)
            except Exception as ex:
                return True
    return False


def main():

    if "GAMMAPY_DATA" not in os.environ:
        logging.info("GAMMAPY_DATA environment variable not set.")
        logging.info("Running notebook tests requires this environment variable.")
        logging.info("Exiting now.")
        sys.exit()

    passed = True
    yamlfile = get_notebooks()
    dirnbs = Path("tutorials")

    for notebook in yamlfile:
        if requirement_missing(notebook):
            logging.info(
                "Skipping notebook {} because requirement is missing.".format(
                    notebook["name"]
                )
            )
            continue

        filename = notebook["name"] + ".ipynb"
        path = dirnbs / filename

        if not notebook_test(path):
            passed = False

    assert passed


if __name__ == "__main__":
    main()
