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
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import re
from shutil import copytree, rmtree
from distutils.util import strtobool
from docutils.parsers.rst.directives.images import Image
from docutils.parsers.rst.directives import register_directive
from docutils.parsers.rst import roles
from docutils import nodes
from sphinx.util import logging
import nbformat
from nbformat.v4 import new_markdown_cell
from nbconvert.exporters import PythonExporter
from ..extern.pathlib import Path

try:
    gammapy_extra_path = Path(os.environ["GAMMAPY_EXTRA"])
    HAS_GP_EXTRA = True
except KeyError:
    HAS_GP_EXTRA = False

log = logging.getLogger("__name__")


class ExtraImage(Image):
    """Directive to add optional images from gammapy-extra"""

    def run(self):
        filename = self.arguments[0]

        if HAS_GP_EXTRA:
            path = gammapy_extra_path / "figures" / filename
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
                "GAMMAPY_EXTRA not available. "
                "Missing image: name: {!r} filename: {!r}".format(self.name, filename)
            )
            self.options["alt"] = self.arguments[1]

        return super(ExtraImage, self).run()


def notebook_role(name, rawtext, notebook, lineno, inliner, options={}, content=[]):
    """Link to a notebook on gammapy-extra"""

    # check if file exists in local notebooks folder
    nbfolder = Path("notebooks")
    nbfilename = notebook + ".ipynb"
    nbfile = nbfolder / nbfilename

    if not nbfile.is_file():
        msg = inliner.reporter.error(
            "Unknown notebook {}".format(notebook), line=lineno
        )
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]
    else:
        refuri = inliner.document.settings._source
        app = inliner.document.settings.env.app
        node = make_link_node(rawtext, app, refuri, notebook, options)
        return [node], []


def make_link_node(rawtext, app, refuri, notebook, options):
    # base = 'https://github.com/gammapy/gammapy-extra/tree/master/notebooks/'
    # base = 'https://nbviewer.jupyter.org/github/gammapy/gammapy-extra/blob/master/notebooks/'

    relpath = refuri.split(str(Path("/gammapy/docs")))[1]
    foldersplit = relpath.split(os.sep)
    base = ((".." + os.sep) * (len(foldersplit) - 2)) + "notebooks" + os.sep
    full_name = notebook + ".html"
    ref = base + full_name
    roles.set_classes(options)
    node = nodes.reference(rawtext, full_name, refuri=ref, **options)
    return node


def gammapy_sphinx_ext_activate():
    if HAS_GP_EXTRA:
        log.info("*** Found GAMMAPY_EXTRA = {}".format(gammapy_extra_path))
        log.info("*** Nice!")
    else:
        log.info("*** gammapy-extra *not* found.")
        log.info("*** Set the GAMMAPY_EXTRA environment variable!")
        log.info("*** Docs build will be incomplete.")
        log.info("*** Notebook links will not be verified.")

    # Register our directives and roles with Sphinx
    register_directive("gp-extra-image", ExtraImage)
    roles.register_local_role("gp-extra-notebook", notebook_role)


def modif_nb_links(folder, url_docs, git_commit):
    """
    Modifies links in raw and sphinx formatted notebooks and so they
    point to and from the same version of the documentation. Adds a box to the
    sphinx formatted notebooks with info and link to the ipynb file.
    """

    DOWNLOAD_CELL = """
<script type="text/javascript" src="../_static/linksdl.js"></script>
<div class='alert alert-info'>
**This is a fixed-text formatted version of a Jupyter notebook.**

- Try online [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/gammapy/gammapy-extra/{git_commit}?urlpath=lab/tree/{nb_filename})
- You can contribute with your own notebooks in this
[GitHub repository](https://github.com/gammapy/gammapy-extra/tree/master/notebooks).
- **Source files:**
[{nb_filename}](../_static/notebooks/{nb_filename}) |
[{py_filename}](../_static/notebooks/{py_filename})
</div>
"""

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath) and filepath[-6:] == ".ipynb":
            if folder == "notebooks":
                py_filename = filename.replace("ipynb", "py")
                ctx = dict(
                    nb_filename=filename, py_filename=py_filename, git_commit=git_commit
                )
                strcell = DOWNLOAD_CELL.format(**ctx)
                nb = nbformat.read(filepath, as_version=nbformat.NO_CONVERT)
                nb.metadata["nbsphinx"] = {"orphan": bool("true")}
                nb.cells.insert(0, new_markdown_cell(strcell))
                nbformat.write(nb, filepath)

            txt = Path(filepath).read_text(encoding="utf-8")

            if folder == "notebooks":
                repl = r"..\/\1rst\2"
            elif folder == "_static/notebooks":
                repl = r"..\/..\/\1html\2"

            txt = re.sub(
                pattern=url_docs + "(.*?)html(\)|#)",
                repl=repl,
                string=txt,
                flags=re.M | re.I,
            )

            Path(filepath).write_text(txt, encoding="utf-8")


def convert_nb_to_script(path):
    """Convert notebook to Python script using the nbconvert API.

    Before we were shelling out to call ``nbconvert``, but that
    didn't always work because the cli tool is sometimes called
    differently, e.g. ``nbconvert-3.6``. This should always work
    and makes sure the right Python / nbconvert is used.
    """
    # https://nbconvert.readthedocs.io/en/latest/execute_api.html#executing-notebooks-using-the-python-api-interface
    # https://stackoverflow.com/a/38248141/498873
    txt = path.read_text(encoding="utf-8")

    nb = nbformat.reads(txt, nbformat.NO_CONVERT)

    exporter = PythonExporter()
    source, meta = exporter.from_notebook_node(nb)

    path = path.with_suffix(".py")
    log.info("Writing {}".format(path))
    path.write_text(source, encoding="utf-8")


def gammapy_sphinx_notebooks(setup_cfg):
    """
    Manages the processes for the building of sphinx formatted notebooks
    """

    if not strtobool(setup_cfg["build_notebooks"]):
        log.info("Config build_notebooks is False; skipping notebook processing")
        return

    if not HAS_GP_EXTRA:
        log.info("No GAMMAPY_EXTRA found; skipping notebook processing")
        return

    url_docs = setup_cfg["url_docs"]
    git_commit = setup_cfg["git_commit"]

    # copy and build notebooks
    gammapy_extra_notebooks_folder = Path(os.environ["GAMMAPY_EXTRA"]) / "notebooks"

    if gammapy_extra_notebooks_folder.is_dir():

        ignorefiles = lambda d, files: [
            f
            for f in files
            if os.path.isfile(os.path.join(d, f))
            and f[-6:] != ".ipynb"
            and f[-4:] != ".png"
        ]
        log.info("*** Converting notebooks to scripts")

        path_nbs = Path("notebooks")
        path_static_nbs = Path("_static") / "notebooks"

        # remove existing notebooks
        rmtree(str(path_static_nbs), ignore_errors=True)
        rmtree("notebooks", ignore_errors=True)

        # copy notebooks
        copytree(str(gammapy_extra_notebooks_folder), str(path_nbs), ignore=ignorefiles)
        copytree(
            str(gammapy_extra_notebooks_folder),
            str(path_static_nbs),
            ignore=ignorefiles,
        )

        for path in path_static_nbs.glob("*.ipynb"):
            convert_nb_to_script(path)

        modif_nb_links("notebooks", url_docs, git_commit)
        modif_nb_links("_static/notebooks", url_docs, git_commit)
