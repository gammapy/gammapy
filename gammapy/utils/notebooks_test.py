# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test if Jupyter notebooks work."""
import logging
import os
import shutil
import sys
from configparser import ConfigParser
from pathlib import Path
from gammapy.scripts.jupyter import notebook_run
from gammapy.utils.scripts import get_notebooks_paths

log = logging.getLogger(__name__)
PATH_CFG = Path(__file__).resolve().parent / ".." / ".."

# fetch params from setup.cfg
conf = ConfigParser()
conf.read(PATH_CFG / "setup.cfg")
setup_cfg = dict(conf.items("metadata"))
URL_GAMMAPY_MASTER = setup_cfg["url_raw_github"]


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
    path_temp.mkdir()

    try:
        for nb_path in get_notebooks_paths():
            path_dest = path_temp / nb_path.name
            shutil.copyfile(nb_path, path_dest)
            if not notebook_run(path_dest):
                passed = False
    finally:
        # tear down
        shutil.rmtree(path_temp, ignore_errors=True)

    if not passed:
        sys.exit("Some tests failed. Existing now.")


if __name__ == "__main__":
    main()
