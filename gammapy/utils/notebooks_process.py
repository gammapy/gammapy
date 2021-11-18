# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Process tutorials notebooks for publication in documentation."""
import argparse
import logging
import os
import shutil
import subprocess
import sys
from configparser import ConfigParser
from pathlib import Path
import nbformat
from nbformat.v4 import new_markdown_cell
from gammapy import __version__
from gammapy.scripts.jupyter import notebook_run
from gammapy.utils.scripts import get_notebooks_paths

log = logging.getLogger(__name__)
PATH_CFG = Path(__file__).resolve().parent / ".." / ".."
SETUP_FILE = "setup.cfg"

# fetch params from setup.cfg
conf = ConfigParser()
conf.read(PATH_CFG / SETUP_FILE)
setup_cfg = dict(conf.items("metadata"))
build_docs_cfg = dict(conf.items("build_docs"))
DOWN_NBS = build_docs_cfg["downloadable-notebooks"]
PATH_NBS = Path(build_docs_cfg["source-dir"]) / DOWN_NBS
BINDER_URL = "https://mybinder.org/v2/gh/gammapy/gammapy-webpage"
BINDER_BADGE = "https://static.mybinder.org/badge.svg"


def copy_clean_notebook(nb_path):
    """Strip output, copy file and convert to script."""

    subprocess.run(
        [sys.executable, "-m", "gammapy", "jupyter", "--src", nb_path, "strip"]
    )
    log.info(f"Copying notebook {nb_path} to {PATH_NBS}")
    shutil.copy(nb_path, PATH_NBS)
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

    # add binder cell
    path_tail = str(nb_path).split(f"/{build_docs_cfg['source-dir']}")[1]
    level_depth = path_tail.count("/") - 1
    start_link = level_depth * "../"
    nb_filename = nb_path.absolute().name
    py_filename = nb_filename.replace("ipynb", "py")
    release_number = __version__
    BINDER_LINK = f"- Try online[![Binder]({BINDER_BADGE})]({BINDER_URL}/v{__version__}?urlpath=lab/tree{path_tail})"
    if "dev" in __version__:
        BINDER_LINK = ""
        release_number = "dev"

    BOX_CELL = f"""
<div class="alert alert-info">

**This is a fixed-text formatted version of a Jupyter notebook**

{BINDER_LINK}
- You may download all the notebooks in the documentation as a
[tar file]({start_link}_downloads/notebooks-{release_number}.tar).
- **Source files:**
[{nb_filename}]({start_link}{DOWN_NBS}/{nb_filename}) |
[{py_filename}]({start_link}{DOWN_NBS}/{py_filename})
</div>
"""

    rawnb = nbformat.read(nb_path, as_version=nbformat.NO_CONVERT)

    if "nbsphinx" not in rawnb.metadata:
        rawnb.metadata["nbsphinx"] = {"orphan": bool("true")}
        rawnb.cells.insert(0, new_markdown_cell(BOX_CELL))

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

    for nb_path in get_notebooks_paths():
        if args.src and Path(args.src).resolve() != nb_path:
            continue
        skip = False
        copy_clean_notebook(nb_path)
        rawnb = nbformat.read(nb_path, as_version=nbformat.NO_CONVERT)
        if "gammapy" in rawnb.metadata and "skip_run" in rawnb.metadata["gammapy"]:
            skip = rawnb.metadata["gammapy"]["skip_run"]
        if not skip:
            notebook_run(nb_path)
            add_box(nb_path)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="Tutorial notebook to process")
    args = parser.parse_args()
    build_notebooks(args)


if __name__ == "__main__":
    main()
