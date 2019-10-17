# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to perform actions on jupyter notebooks."""
import logging
import os
import subprocess
import sys
import time
import click

log = logging.getLogger(__name__)

OFF = ["GAMMA_CAT", "GAMMAPY_DATA", "GAMMAPY_EXTRA", "GAMMAPY_FERMI_LAT_DATA"]


@click.command(name="run")
@click.pass_context
@click.option(
    "--tutor",
    is_flag=True,
    default=False,
    help="Tutorials environment?",
    show_default=True,
)
@click.option("--kernel", default="python3", help="Kernel name", show_default=True)
def cli_jupyter_run(ctx, tutor, kernel):
    """Execute Jupyter notebooks."""
    with environment(OFF, tutor, ctx):
        for path in ctx.obj["paths"]:
            notebook_test(path, kernel)


def execute_notebook(path, kernel="python3", loglevel=30):
    """Execute a Jupyter notebook."""
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--allow-errors",
        f"--log-level={loglevel}",
        "--ExecutePreprocessor.timeout=None",
        f"--ExecutePreprocessor.kernel_name={kernel}",
        "--to",
        "notebook",
        "--inplace",
        "--execute",
        f"{path}",
    ]
    t = time.time()
    completed_process = subprocess.run(cmd)
    t = time.time() - t
    if completed_process.returncode:
        log.error(f"Error executing file: {path}")
        return False
    else:
        log.info(f"   ... Executing duration: {t:.1f} seconds")
        return True


@click.command(name="strip")
@click.pass_context
def cli_jupyter_strip(ctx):
    """Strip output cells."""
    import nbformat

    for path in ctx.obj["paths"]:
        rawnb = nbformat.read(str(path), as_version=nbformat.NO_CONVERT)

        rawnb["metadata"].pop("pycharm", None)

        for cell in rawnb.cells:
            if cell["cell_type"] == "code":
                cell["metadata"].pop("pycharm", None)
                cell["execution_count"] = None
                cell["outputs"] = []

        nbformat.write(rawnb, str(path))
        log.info(f"Strip output cells in notebook: {path}")


@click.command(name="black")
@click.pass_context
def cli_jupyter_black(ctx):
    """Format code cells with black."""
    import nbformat

    for path in ctx.obj["paths"]:
        rawnb = nbformat.read(str(path), as_version=nbformat.NO_CONVERT)
        blacknb = BlackNotebook(rawnb)
        blacknb.blackformat()
        rawnb = blacknb.rawnb
        nbformat.write(rawnb, str(path))
        log.info(f"Applied black to notebook: {path}")


class BlackNotebook:
    """Manage the process of black formatting.

    Probably this will become available directly in the future.

    See https://github.com/ambv/black/issues/298#issuecomment-476960082
    """

    MAGIC_TAG = "###-MAGIC TAG-"

    def __init__(self, rawnb):
        self.rawnb = rawnb

    def blackformat(self):
        """Format code cells."""
        import black

        for cell in self.rawnb.cells:
            fmt = cell["source"]
            if cell["cell_type"] == "code":
                try:
                    fmt = "\n".join(self.tag_magics(fmt))
                    has_semicolon = fmt.endswith(";")
                    fmt = black.format_str(
                        src_contents=fmt, mode=black.FileMode(line_length=79)
                    ).rstrip()
                    if has_semicolon:
                        fmt += ";"
                except Exception as ex:
                    log.info(ex)
                fmt = fmt.replace(self.MAGIC_TAG, "")
            cell["source"] = fmt

    def tag_magics(self, cellcode):
        """Comment magic commands."""
        lines = cellcode.splitlines(False)
        for line in lines:
            if line.startswith("%") or line.startswith("!"):
                magic_line = self.MAGIC_TAG + line
                yield magic_line
            else:
                yield line


@click.command(name="test")
@click.pass_context
@click.option(
    "--tutor",
    is_flag=True,
    default=False,
    help="Tutorials environment?",
    show_default=True,
)
@click.option("--kernel", default="python3", help="Kernel name", show_default=True)
def cli_jupyter_test(ctx, tutor, kernel):
    """Check if Jupyter notebooks are broken."""
    with environment(OFF, tutor, ctx):
        for path in ctx.obj["paths"]:
            notebook_test(path, kernel)


def notebook_test(path, kernel="python3"):
    """Execute and parse a Jupyter notebook exposing broken cells."""
    import nbformat

    log.info(f"   ... EXECUTING: {path}")
    passed = execute_notebook(path, kernel)
    rawnb = nbformat.read(str(path), as_version=nbformat.NO_CONVERT)
    report = ""

    for cell in rawnb.cells:
        if "outputs" in cell.keys():
            for output in cell["outputs"]:
                if output["output_type"] == "error":
                    passed = False
                    traceitems = ["--TRACEBACK: "]
                    for o in output["traceback"]:
                        traceitems.append(f"{o}")
                    traceback = "\n".join(traceitems)
                    infos = "\n\n{} in cell [{}]\n\n" "--SOURCE CODE: \n{}\n\n".format(
                        output["ename"], cell["execution_count"], cell["source"]
                    )
                    report = infos + traceback
                    break
        if not passed:
            break

    if passed:
        log.info("   ... PASSED")
        return True
    else:
        log.info("   ... FAILED")
        log.info(report)
        return False


class environment:
    """Helper for setting environmental variables."""

    def __init__(self, envs, tutor, ctx):
        self.envs = envs
        self.tutor = tutor
        self.ctx = ctx

    def __enter__(self):
        self.old = os.environ
        if self.tutor:
            for item in self.envs:
                if item in os.environ:
                    del os.environ[item]
                    log.info(f"Unsetting {item} environment variable.")
            abspath = self.ctx.obj["pathsrc"].absolute()
            datapath = abspath.parent / "datasets"
            if abspath.is_file():
                datapath = abspath.parent.parent / "datasets"
            os.environ["GAMMAPY_DATA"] = str(datapath)
            log.info(f"Setting GAMMAPY_DATA={datapath}")

    def __exit__(self, type, value, traceback):
        if self.tutor:
            os.environ = self.old
            log.info("Environment variables recovered.")
