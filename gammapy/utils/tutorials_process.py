"""Process tutorials notebooks for publication in documentation."""
import logging
import os
import subprocess
import argparse
import sys
from pathlib import Path
from shutil import copyfile, copytree, rmtree
from distutils.util import strtobool
from ..scripts.jupyter import notebook_test

log = logging.getLogger(__name__)


def ignorefiles(d, files):
    return [
        f
        for f in files
        if os.path.isfile(os.path.join(d, f))
        and f[-6:] != ".ipynb"
        and f[-4:] != ".png"
    ]


def ignoreall(d, files):
    return [
        f for f in files if os.path.isfile(os.path.join(d, f)) and f[-6:] != ".ipynb"
    ]


def setup_sphinx_params(args):
    flagnotebooks = "True"
    setupfilename = "setup.cfg"
    if not args.nbs:
        flagnotebooks = "False"
    build_notebooks_line = "build_notebooks = {}\n".format(flagnotebooks)
    git_commit_line = "git_commit = {}\n".format(args.release)

    file_str = ""
    with open(setupfilename) as f:
        for line in f:
            if line.startswith("build_notebooks ="):
                line = build_notebooks_line
            if line.startswith("git_commit ="):
                line = git_commit_line
            file_str += line

    with open(setupfilename, "w") as f:
        f.write(file_str)


def build_notebooks(args):
    if "GAMMAPY_DATA" not in os.environ:
        log.info("GAMMAPY_DATA environment variable not set.")
        log.info("Running notebook tests requires this environment variable.")
        log.info("Exiting now.")
        sys.exit()

    # prepare folder structure
    pathsrc = Path(args.src)
    path_temp = Path("temp")
    path_empty_nbs = Path("tutorials")
    path_filled_nbs = Path("docs") / "notebooks"
    path_static_nbs = Path("docs") / "_static" / "notebooks"

    rmtree(str(path_temp), ignore_errors=True)
    path_temp.mkdir(parents=True, exist_ok=True)
    path_filled_nbs.mkdir(parents=True, exist_ok=True)
    path_static_nbs.mkdir(parents=True, exist_ok=True)

    if pathsrc == path_empty_nbs:
        rmtree(str(path_temp), ignore_errors=True)
        rmtree(str(path_static_nbs), ignore_errors=True)
        rmtree(str(path_filled_nbs), ignore_errors=True)
        copytree(str(path_empty_nbs), str(path_temp), ignore=ignorefiles)
    elif pathsrc.exists():
        notebookname = pathsrc.name
        pathdest = path_temp / notebookname
        copyfile(str(pathsrc), str(pathdest))
    else:
        log.info("Notebook file does not exist.")
        sys.exit()

    # strip and blackformat
    subprocess.call(
        [sys.executable, "-m", "gammapy", "jupyter", "--src", "temp", "black"]
    )
    subprocess.call(
        [sys.executable, "-m", "gammapy", "jupyter", "--src", "temp", "strip"]
    )

    # test /run
    passed = True
    for path in path_temp.glob("*.ipynb"):
        if not notebook_test(path):
            passed = False

    # convert into scripts
    # copy generated filled notebooks to doc
    # if passed:
    try:
        import jupyter
        jupyter_module = "jupyter"
    except ImportError:
        import jupyter_core
        jupyter_module = "jupyter_core"

    if pathsrc == path_empty_nbs:
        # copytree is needed to copy subfolder images
        copytree(str(path_empty_nbs), str(path_static_nbs), ignore=ignoreall)
        for path in path_static_nbs.glob("*.ipynb"):
            subprocess.call(
                [
                    sys.executable,
                    "-m",
                    jupyter_module,
                    "nbconvert",
                    "--to",
                    "script",
                    "{}".format(str(path)),
                ]
            )
        copytree(str(path_temp), str(path_filled_nbs), ignore=ignorefiles)
    else:
        pathsrc = path_temp / notebookname
        pathdest = path_static_nbs / notebookname
        copyfile(str(pathsrc), str(pathdest))
        subprocess.call(
            [
                sys.executable,
                "-m",
                jupyter_module,
                "nbconvert",
                "--to",
                "script",
                "{}".format(str(pathdest)),
            ]
        )
        pathdest = path_filled_nbs / notebookname
        copyfile(str(pathsrc), str(pathdest))

    # else:
    #    log.info("Tests have not passed.")
    #    log.info("Tutorials not ready for documentation building process.")
    #    rmtree(str(path_static_nbs), ignore_errors=True)

    # tear down
    rmtree(str(path_temp), ignore_errors=True)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="Tutorial notebook or folder to process")
    parser.add_argument("--release", help="Release tag for Binder links")
    parser.add_argument("--nbs", help="Notebooks are considered in Sphinx")
    args = parser.parse_args()

    if not args.src:
        args.src = "tutorials"
    if not args.release:
        args.release = "master"
    if not args.nbs:
        args.nbs = "True"

    try:
        args.nbs = strtobool(args.nbs)
    except Exception as ex:
        log.error(ex)
        sys.exit()
    # if not args.release.startswith("v") and args.release != "master":
    #    args.release = "v" + args.release

    setup_sphinx_params(args)

    if args.nbs:
        build_notebooks(args)


if __name__ == "__main__":
    main()
