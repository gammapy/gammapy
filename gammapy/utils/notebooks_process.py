# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Process tutorials notebooks for publication in documentation."""
import argparse
import logging
import os
import shutil
import subprocess
import sys
from configparser import ConfigParser
from distutils.util import strtobool
from pathlib import Path
import nbformat
from nbformat.v4 import new_markdown_cell
from gammapy import __version__
from gammapy.scripts.jupyter import notebook_test
from gammapy.utils.notebooks_test import get_notebooks

log = logging.getLogger(__name__)
PATH_CFG = Path(__file__).resolve().parent / ".." / ".."
SETUP_FILE = "setup.cfg"

# fetch params from setup.cfg
conf = ConfigParser()
conf.read(PATH_CFG / SETUP_FILE)
setup_cfg = dict(conf.items("metadata"))
URL_GAMMAPY_MASTER = setup_cfg["url_raw_github"]
build_docs_cfg = dict(conf.items("build_docs"))
DOWN_NBS = build_docs_cfg["downloadable-notebooks"]
PATH_NBS = Path(build_docs_cfg["source-dir"]) / DOWN_NBS
PATH_SOURCE_IMAGES = Path(build_docs_cfg["source-dir"]) / "tutorials" / "images"
PATH_DEST_IMAGES = Path(build_docs_cfg["source-dir"]) / DOWN_NBS / "images"
GITHUB_TUTOS_URL = "https://github.com/gammapy/gammapy/tree/master/docs/tutorials"
BINDER_BADGE_URL = "https://static.mybinder.org/badge.svg"
BINDER_URL = "https://mybinder.org/v2/gh/gammapy/gammapy-webpage"


def setup_sphinx_params(args):
    """Set Sphinx params in config file setup.cfg"""

    flagnotebooks = "True"
    if not args.nbs:
        flagnotebooks = "False"
    build_notebooks_line = f"build_notebooks = {flagnotebooks}\n"

    file_str = ""
    with open(SETUP_FILE) as f:
        for line in f:
            if line.startswith("build_notebooks ="):
                line = build_notebooks_line
            file_str += line

    with open(SETUP_FILE, "w") as f:
        f.write(file_str)


def fill_notebook(nb_path, args):
    """Code formatting, strip output, file copy, execution and script conversion."""

    if not Path(nb_path).exists():
        log.info(f"File {nb_path} does not exist.")
        return

    if args.fmt:
        subprocess.run(
            [sys.executable, "-m", "gammapy", "jupyter", "--src", nb_path, "black"]
        )
    subprocess.run(
        [sys.executable, "-m", "gammapy", "jupyter", "--src", nb_path, "strip"]
    )

    log.info(f"Copying notebook {nb_path} to {PATH_NBS}")
    shutil.copy(nb_path, PATH_NBS)

    # execute notebook
    notebook_test(nb_path)

    static_nb_path = PATH_NBS / Path(nb_path).absolute().name
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "script",
            static_nb_path,
        ]
    )


def add_box(nb_path):
    """Adds box with downloadable links and binder."""

    nb_path = Path(nb_path)
    log.info(f"Adding box in {nb_path}")
    release_number_binder = f"v{__version__}"
    if "dev" in __version__:
        release_number_binder = "master"

    DOWNLOAD_CELL = """
<div class="alert alert-info">

**This is a fixed-text formatted version of a Jupyter notebook**

- Try online [![Binder]({BINDER_BADGE_URL})]({BINDER_URL}/{release_number_binder}?urlpath=lab/tree/{nb_filename})
- You can contribute with your own notebooks in this
[GitHub repository]({GITHUB_TUTOS_URL}).
- **Source files:**
[{nb_filename}](../{DOWN_NBS}/{nb_filename}) |
[{py_filename}](../{DOWN_NBS}/{py_filename})
</div>
"""

    # add binder cell
    nb_filename = nb_path.absolute().name
    py_filename = nb_filename.replace("ipynb", "py")
    ctx = dict(
        nb_filename=nb_filename,
        py_filename=py_filename,
        release_number_binder=release_number_binder,
        DOWN_NBS=DOWN_NBS,
        BINDER_BADGE_URL=BINDER_BADGE_URL,
        BINDER_URL=BINDER_URL,
        GITHUB_TUTOS_URL=GITHUB_TUTOS_URL,
    )
    strcell = DOWNLOAD_CELL.format(**ctx)
    rawnb = nbformat.read(nb_path, as_version=nbformat.NO_CONVERT)

    if "nbsphinx" not in rawnb.metadata:
        rawnb.metadata["nbsphinx"] = {"orphan": bool("true")}
        rawnb.cells.insert(0, new_markdown_cell(strcell))

        # add latex format
        for cell in rawnb.cells:
            if "outputs" in cell.keys():
                for output in cell["outputs"]:
                    if (
                        output["output_type"] == "execute_result"
                        and "text/latex" in output["data"].keys()
                    ):
                        output["data"]["text/latex"] = output["data"][
                            "text/latex"
                        ].replace("$", "$$")
        nbformat.write(rawnb, nb_path)


def build_notebooks(args):
    if "GAMMAPY_DATA" not in os.environ:
        log.info("GAMMAPY_DATA environment variable not set.")
        log.info("Running notebook tests requires this environment variable.")
        log.info("Exiting now.")
        sys.exit()

    PATH_NBS.mkdir(parents=True, exist_ok=True)

    if args.src:
        pathsrc = Path(args.src)
        fill_notebook(pathsrc, args)
        add_box(pathsrc)
    else:
        for notebook in get_notebooks():
            nb_path = notebook["url"].replace(URL_GAMMAPY_MASTER, "")
            fill_notebook(nb_path, args)
            add_box(nb_path)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="Tutorial notebook to process")
    parser.add_argument("--nbs", help="Notebooks are considered in Sphinx")
    parser.add_argument("--fmt", help="Black format notebooks")
    args = parser.parse_args()

    if not args.nbs:
        args.nbs = "True"
    if not args.fmt:
        args.fmt = "True"

    try:
        args.nbs = strtobool(args.nbs)
        args.fmt = strtobool(args.fmt)
    except Exception as ex:
        log.error(ex)
        sys.exit()

    setup_sphinx_params(args)

    if args.nbs:
        build_notebooks(args)
        shutil.rmtree(PATH_DEST_IMAGES, ignore_errors=True)
        shutil.copytree(PATH_SOURCE_IMAGES, PATH_DEST_IMAGES)


if __name__ == "__main__":
    main()
