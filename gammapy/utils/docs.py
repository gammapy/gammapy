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
- https://github.com/docutils/docutils/blob/master/docutils/docutils/parsers/rst/directives/images.py
- https://github.com/sphinx-doc/sphinx/blob/master/sphinx/directives/other.py
- https://github.com/bokeh/bokeh/tree/master/src/bokeh/sphinxext
"""

import os
import re
from pathlib import Path
from docutils.parsers.rst import Directive
from docutils.parsers.rst.directives import register_directive
from docutils.parsers.rst.directives.body import CodeBlock
from docutils.parsers.rst.directives.images import Image
from docutils.parsers.rst.directives.misc import Include, Raw
from docutils import nodes
from sphinx.util import logging
from sphinx.transforms.post_transforms import SphinxPostTransform
from gammapy.analysis import AnalysisConfig

try:
    gammapy_data_path = Path(os.environ["GAMMAPY_DATA"])
    HAS_GP_DATA = True
except KeyError:
    HAS_GP_DATA = False

log = logging.getLogger(__name__)


class AccordionHeader(Directive):
    """Insert HTML code to open an accordion box in the How To."""

    option_spec = {"id": str, "title": str, "link": str}

    def run(self):
        raw = f"""
            <div id="accordion" class="shadow tutorial-accordion">
        <div class="card tutorial-card">
            <div class="card-header d-flex justify-content-between align-items-center">
                <div class="d-flex align-items-center" data-bs-toggle="collapse"
                data-bs-target="#{self.options["id"]}" style="cursor: pointer;">
                    <button class="btn btn-dark btn-sm me-2"
                    style="background-color: #150458; border-radius: 50%; width: 24px; height: 24px;">
                        <span style="font-size: 14px;">+</span>
                    </button>
                    {self.options["title"]}
                </div>
        """
        if self.options.get("link", None):
            raw += f"""
             <span class="badge gs-badge-link">
             <a class="reference external" href="{self.options["link"]}">Straight to tutorial…</a>
             </span>
             """
        raw += f"""

            </div>
            <div id="{self.options["id"]}" class="collapse" data-parent="#accordion">
                <div class="card-body">
        """
        include_lines = raw.splitlines()
        c = Raw(
            self.name,
            ["html"],
            self.options,
            include_lines,  # content
            self.lineno,
            self.content_offset,
            self.block_text,
            self.state,
            self.state_machine,
        )
        return c.run()


class AccordionFooter(Directive):
    """Insert HTML code to close an accordion box in the How To."""

    def run(self):
        raw = """
                    </div>
                </div>
            </div>
        </div>
        """
        include_lines = raw.splitlines()
        c = Raw(
            self.name,
            ["html"],
            self.options,
            include_lines,  # content
            self.lineno,
            self.content_offset,
            self.block_text,
            self.state,
            self.state_machine,
        )
        return c.run()


class HowtoHLI(Include):
    """Directive to insert how-to for high-level interface."""

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
    """Directive to add optional images from gammapy-data."""

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


class SubstitutionCodeBlock(CodeBlock):
    """Similar to CodeBlock but replaces placeholders with variables."""

    def run(self):
        """Replace placeholders with given variables."""
        app = self.state.document.settings.env.app
        new_content = []
        self.content = self.content
        existing_content = self.content
        for item in existing_content:
            for pair in app.config.substitutions:
                original, replacement = pair
                item = item.replace(original, replacement)
            new_content.append(item)

        self.content = new_content
        return list(CodeBlock.run(self))


class DynamicPRLinkTransform(SphinxPostTransform):
    """
    A Sphinx post-transform that converts [#XXXX] into clickable links to GitHub PR pages.
    """

    default_priority = 800
    PR_PATTERN = re.compile(r"\[#(\d+)\]")  # Matches [#XXXX]
    GITHUB_PR_BASE_URL = "https://github.com/gammapy/gammapy/pull/"  # Fixed URL

    def apply(self):
        """Apply the transform to convert PR references into links."""
        for text_node in self.document.traverse(nodes.Text):
            if "[#" not in text_node:  # Skip nodes unlikely to match
                continue

            content, last_idx = [], 0
            # Find the associated pattern
            for match in self.PR_PATTERN.finditer(text_node.astext()):
                # Add preceding text
                content.append(nodes.Text(text_node[last_idx : match.start()]))

                # Create hyperlink node for the PR
                pr_number = match.group(1)
                pr_url = f"{self.GITHUB_PR_BASE_URL}{pr_number}"
                link_node = nodes.reference(text=f"[#{pr_number}]", refuri=pr_url)
                content.append(link_node)

                last_idx = match.end()

            # Add any remaining text after the last match
            content.append(nodes.Text(text_node[last_idx:]))

            # Replace the original text node with the new processed nodes
            text_node.parent.replace(text_node, content)


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
    register_directive("accordion-header", AccordionHeader)
    register_directive("accordion-footer", AccordionFooter)
