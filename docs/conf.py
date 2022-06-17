# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Gammapy documentation build configuration file.

import datetime
import os

# Get configuration information from setup.cfg
from configparser import ConfigParser

from pkg_resources import get_distribution

# Load all the global Astropy configuration
from sphinx_astropy.conf import *

# Sphinx-gallery config
from sphinx_gallery.sorting import FileNameSortKey

# Load utils docs functions
from gammapy.utils.docs import gammapy_sphinx_ext_activate

conf = ConfigParser()
conf.read([os.path.join(os.path.dirname(__file__), "..", "setup.cfg")])
setup_cfg = dict(conf.items("metadata"))

# -- General configuration ----------------------------------------------------

# Matplotlib directive sets whether to show a link to the source in HTML
plot_html_show_source_link = False

# If true, figures, tables and code-blocks are automatically numbered if they have a caption
numfig = False

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.1'

# We currently want to link to the latest development version of the astropy docs,
# so we override the `intersphinx_mapping` entry pointing to the stable docs version
# that is listed in `astropy/sphinx/conf.py`.
intersphinx_mapping.pop("h5py", None)
intersphinx_mapping["matplotlib"] = ("https://matplotlib.org/", None)
intersphinx_mapping["astropy"] = ("http://docs.astropy.org/en/latest/", None)
intersphinx_mapping["regions"] = (
    "https://astropy-regions.readthedocs.io/en/latest/",
    None,
)
intersphinx_mapping["reproject"] = ("https://reproject.readthedocs.io/en/latest/", None)
intersphinx_mapping["naima"] = ("https://naima.readthedocs.io/en/latest/", None)
intersphinx_mapping["gadf"] = (
    "https://gamma-astro-data-formats.readthedocs.io/en/latest/",
    None,
)
intersphinx_mapping["iminuit"] = ("https://iminuit.readthedocs.io/en/latest/", None)
intersphinx_mapping["pandas"] = ("https://pandas.pydata.org/pandas-docs/stable/", None)

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns.append("_templates")
exclude_patterns.append("_static")
exclude_patterns.append("**.ipynb_checkpoints")
exclude_patterns.append("user-guide/model-gallery/*/*.ipynb")
exclude_patterns.append("user-guide/model-gallery/*/*.md5")
exclude_patterns.append("user-guide/model-gallery/*/*.py")

extensions.extend(
    [
        "nbsphinx",
        "sphinx_click.ext",
        "IPython.sphinxext.ipython_console_highlighting",
        "sphinx.ext.mathjax",
        "sphinx_gallery.gen_gallery",
        "sphinx.ext.doctest",
        "sphinx_panels",
        "sphinx_copybutton",
    ]
)
nbsphinx_execute = "never"
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
# --

# This is added to the end of RST files - a good place to put substitutions to
# be used globally.
rst_epilog += """
"""

# This is added to keep the links to PRs in release notes
changelog_links_docpattern = ['.*changelog.*', 'whatsnew/.*', 'release-notes/.*']

# -- Project information ------------------------------------------------------

# This does not *have* to match the package name, but typically does
project = setup_cfg["name"]
author = setup_cfg["author"]
copyright = "{}, {}".format(datetime.datetime.now().year, setup_cfg["author"])

version = get_distribution(project).version
release = version
switch_version = version
if len(version) > 5:
    switch_version = "dev"

# -- Options for HTML output ---------------------------------------------------

# A NOTE ON HTML THEMES
# The global astropy configuration uses a custom theme, 'bootstrap-astropy',
# which is installed along with astropy. A different theme can be used or
# the options for this theme can be modified by overriding some
# variables set in the global configuration. The variables set in the
# global configuration are listed below, commented out.

# Add any paths that contain custom themes here, relative to this directory.
# To use a different custom theme, add the directory containing the theme.
# html_theme_path = []

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes. To override the custom theme, set this to the
# name of a builtin theme or the name of a custom theme in html_theme_path.
html_theme = "pydata_sphinx_theme"

# Static files to copy after template files
html_static_path = ["_static"]

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_logo = os.path.join(html_static_path[0], "gammapy_logo_nav.png")
html_favicon = os.path.join(html_static_path[0], "gammapy_logo.ico")

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "search": "search-field.html",
    "navigation": "sidebar-nav-bs.html",
}

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = ''

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "{} v{}".format(project, release)

# Output file base name for HTML help builder.
htmlhelp_basename = f"{project}doc"

html_theme_options = {
    # toc options
    "collapse_navigation": False,
    "navigation_depth": 2,
    "show_prev_next": False,
    # links in menu
    "icon_links": [
        {
            "name": "Github",
            "url": "https://github.com/gammapy/gammapy",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/gammapyST",
            "icon": "fab fa-twitter-square",
        },
        {
            "name": "Slack",
            "url": "https://gammapy.slack.com/",
            "icon": "fab fa-slack",
        },
    ],
    "switcher": {
        "json_url": "https://docs.gammapy.org/stable/switcher.json",
        "version_match": switch_version,
    },
    "navbar_end": ["version-switcher", "navbar-icon-links"],
}

# Theme style
# html_style = ''
html_css_files = ["gammapy.css"]

gammapy_sphinx_ext_activate()

# -- Options for LaTeX output --------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ("index", f"{project}.tex", f"{project} Documentation", author, "manual")
]

# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [("index", project.lower(), f"{project} Documentation", [author], 1)]


# -- Other options --

github_issues_url = "https://github.com/gammapy/gammapy/issues/"

# http://sphinx-automodapi.readthedocs.io/en/latest/automodapi.html
# show inherited members for classes
automodsumm_inherited_members = True

# In `about.rst` and `references.rst` we are giving lists of citations
# (e.g. papers using Gammapy) that partly aren't referenced from anywhere
# in the Gammapy docs. This is normal, but Sphinx emits a warning.
# The following config option suppresses the warning.
# http://www.sphinx-doc.org/en/stable/rest.html#citations
# http://www.sphinx-doc.org/en/stable/config.html#confval-suppress_warnings
suppress_warnings = ["ref.citation"]

# nitpicky = True

sphinx_gallery_conf = {
    "examples_dirs": ["../examples/models"],  # path to your example scripts
    "gallery_dirs": [
        "user-guide/model-gallery"
    ],  # path to where to save gallery generated output
    "within_subsection_order": FileNameSortKey,
    "download_all_examples": False,
    "capture_repr": (),
    "min_reported_time": 10000,
    "show_memory": False,
    "line_numbers": False,
    "reference_url": {
        # The module you locally document uses None
        "gammapy": None,
    },
}
