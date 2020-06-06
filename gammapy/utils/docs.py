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
from configparser import ConfigParser
from pathlib import Path
from docutils.parsers.rst.directives import register_directive
from docutils.parsers.rst.directives.body import CodeBlock
from docutils.parsers.rst.directives.images import Image
from docutils.parsers.rst.directives.misc import Include
from sphinx.util import logging
from gammapy.analysis import AnalysisConfig

try:
    gammapy_data_path = Path(os.environ["GAMMAPY_DATA"])
    HAS_GP_DATA = True
except KeyError:
    HAS_GP_DATA = False

log = logging.getLogger(__name__)
PATH_CFG = Path(__file__).resolve().parent / ".." / ".."

# fetch params from setup.cfg
conf = ConfigParser()
conf.read(PATH_CFG / "setup.cfg")
build_docs_cfg = dict(conf.items("build_docs"))
PATH_NBS = build_docs_cfg["downloadable-notebooks"]


class HowtoHLI(Include):
    """Directive to insert how-to for high-level interface"""

    def run(self):
        raw = ""
        section = self.arguments[0]
        doc = AnalysisConfig._get_doc_sections()
        for keyword in doc.keys():
            if section in ["", keyword]:
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
