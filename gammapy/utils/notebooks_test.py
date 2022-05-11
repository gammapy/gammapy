# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test if Jupyter notebooks work."""
import logging
import os
import shutil
import sys
import tempfile
from argparse import ArgumentParser
from configparser import ConfigParser
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path
from gammapy.scripts.jupyter import notebook_run
from gammapy.utils.scripts import get_notebooks_paths

parser = ArgumentParser()
parser.add_argument("-j", "--n-jobs", type=int)

log = logging.getLogger(__name__)
PATH_CFG = Path(__file__).resolve().parent / ".." / ".."

# fetch params from setup.cfg
conf = ConfigParser()
conf.read(PATH_CFG / "setup.cfg")
setup_cfg = dict(conf.items("metadata"))
URL_GAMMAPY_MASTER = setup_cfg["url_raw_github"]


def run_notebook(notebook_path, tmp_dir):
    path_dest = tmp_dir / notebook_path.name
    shutil.copyfile(notebook_path, path_dest)
    return notebook_run(path_dest)


def main():
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if "GAMMAPY_DATA" not in os.environ:
        log.info("GAMMAPY_DATA environment variable not set.")
        log.info("Running notebook tests requires this environment variable.")
        log.info("Exiting now.")
        sys.exit(1)

    notebooks = list(get_notebooks_paths())
    log.info("Found %d notebooks", len(notebooks))

    with tempfile.TemporaryDirectory(suffix="_gammapy_nb_test") as tmp_dir:
        tmp_dir = Path(tmp_dir)

        with Pool(args.n_jobs) as pool:
            run_nb = partial(run_notebook, tmp_dir=tmp_dir)
            passed = pool.map(run_nb, notebooks)

    if not all(passed):
        sys.exit("Some tests failed. Existing now.")


if __name__ == "__main__":
    main()
