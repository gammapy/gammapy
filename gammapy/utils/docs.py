# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Gammapy documentation generator utilities.

This is for the Sphinx docs build of Gammapy itself,
it is used from ``docs/conf.py``.

It should not be imported or used from other scripts or packages,
because we will change it for our use case whenever we like.

Docutils / Sphinx is notoriously hard to understand / learn about.
Here's some good resources with working examples:

- https://doughellmann.com/blog/2010/05/09/defining-custom-roles-in-sphinx/
- http://docutils.sourceforge.net/docs/howto/rst-directives.html
- https://github.com/docutils-mirror/docutils/blob/master/docutils/parsers/rst/directives/images.py
- https://github.com/sphinx-doc/sphinx/blob/master/sphinx/directives/other.py
- https://github.com/bokeh/bokeh/tree/master/bokeh/sphinxext
"""
import os
from distutils.util import strtobool
from pathlib import Path
import nbformat
from docutils.parsers.rst.directives import register_directive
from docutils.parsers.rst.directives.body import CodeBlock
from docutils.parsers.rst.directives.images import Image
from docutils.parsers.rst.directives.misc import Include
from nbformat.v4 import new_markdown_cell
from sphinx.util import logging
from gammapy import __version__
from gammapy.analysis import AnalysisConfig

try:
    gammapy_data_path = Path(os.environ["GAMMAPY_DATA"])
    HAS_GP_DATA = True
except KeyError:
    HAS_GP_DATA = False

log = logging.getLogger(__name__)


class HowtoHLI(Include):
    """Directive to insert how-to for high-level interface"""

    def run(self):
        raw = ""
        section = self.arguments[0]
        doc = AnalysisConfig._get_doc_sections()
        for keyword in doc.keys():
            if section == "" or section == keyword:
                raw += doc[keyword]
        include_lines = raw.splitlines()
        codeblock = CodeBlock(
            self.name,
            [],
            self.options,
            include_lines,  # content
            self.lineno,
            self.content_offset,
            self.block_text,
            self.state,
            self.state_machine,
        )
        return codeblock.run()


class DocsImage(Image):
    """Directive to add optional images from gammapy-data"""

    def run(self):
        filename = self.arguments[0]

        if HAS_GP_DATA:
            path = gammapy_data_path / "figures" / filename
            if not path.is_file():
                msg = "Error in {} directive: File not found: {}".format(
                    self.name, path
                )
                raise self.error(msg)
            # Usually Sphinx doesn't support absolute paths
            # But passing a POSIX string of the absolute path
            # with an extra "/" at the start seems to do the trick
            self.arguments[0] = "/" + path.absolute().as_posix()
        else:
            self.warning(
                "GAMMAPY_DATA not available. "
                "Missing image: name: {!r} filename: {!r}".format(self.name, filename)
            )
            self.options["alt"] = self.arguments[1] if len(self.arguments) > 1 else ""

        return super().run()


def gammapy_sphinx_ext_activate():
    if HAS_GP_DATA:
        log.info(f"*** Found GAMMAPY_DATA = {gammapy_data_path}")
        log.info("*** Nice!")
    else:
        log.info("*** gammapy-data *not* found.")
        log.info("*** Set the GAMMAPY_DATA environment variable!")
        log.info("*** Docs build will be incomplete.")

    # Register our directives and roles with Sphinx
    register_directive("gp-image", DocsImage)
    register_directive("gp-howto-hli", HowtoHLI)


def parse_notebooks(folder, url_docs):
    """
    Modifies raw and html-fixed notebooks so they will not have broken links
    to other files in the documentation. Adds a box to the sphinx formatted
    notebooks with info and links to the *.ipynb and *.py files.
    """
    release_number_binder = __version__
    if "dev" in __version__:
        release_number_binder = "master"

    DOWNLOAD_CELL = """
<div class="alert alert-info">

**This is a fixed-text formatted version of a Jupyter notebook**

- Try online [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/gammapy/gammapy-webpage/{release_number_binder}?urlpath=lab/tree/{nb_filename})
- You can contribute with your own notebooks in this
[GitHub repository](https://github.com/gammapy/gammapy/tree/master/tutorials).
- **Source files:**
[{nb_filename}](../_static/notebooks/{nb_filename}) |
[{py_filename}](../_static/notebooks/{py_filename})
</div>
"""

    for nbpath in list(folder.glob("*.ipynb")):
        if str(folder) == "notebooks":

            # add binder cell
            nb_filename = str(nbpath).replace("notebooks/", "")
            py_filename = nb_filename.replace("ipynb", "py")
            ctx = dict(
                nb_filename=nb_filename,
                py_filename=py_filename,
                release_number_binder=release_number_binder,
            )
            strcell = DOWNLOAD_CELL.format(**ctx)
            rawnb = nbformat.read(str(nbpath), as_version=nbformat.NO_CONVERT)

            if "nbsphinx" not in rawnb.metadata:
                rawnb.metadata["nbsphinx"] = {"orphan": bool("true")}
                rawnb.cells.insert(0, new_markdown_cell(strcell))

                # add latex format
                for cell in rawnb.cells:
                    if "outputs" in cell.keys():
                        for output in cell["outputs"]:
                            if output["output_type"] == "execute_result":
                                if "text/latex" in output["data"].keys():
                                    output["data"]["text/latex"] = output["data"][
                                        "text/latex"
                                    ].replace("$", "$$")
                nbformat.write(rawnb, str(nbpath))


def gammapy_sphinx_notebooks(setup_cfg):
    """Manage creation of Sphinx-formatted notebooks."""
    if not strtobool(setup_cfg["build_notebooks"]):
        log.info("Config build_notebooks is False; skipping notebook processing")
        return

    url_docs = setup_cfg["url_docs"]

    # fix links
    filled_notebooks_folder = Path("notebooks")
    download_notebooks_folder = Path("_static") / "notebooks"

    if filled_notebooks_folder.is_dir():
        parse_notebooks(filled_notebooks_folder, url_docs)
        parse_notebooks(download_notebooks_folder, url_docs)
