# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Gammapy documentation build configuration file.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this file.
#
# All configuration values have a default.
#
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('..'))


import datetime
import sys
import os
import gammapy

# Get configuration information from setup.cfg
from configparser import ConfigParser

# Sphinx-gallery config
from sphinx_gallery.sorting import ExplicitOrder

# Load utils docs functions
from gammapy.utils.docs import SubstitutionCodeBlock, DynamicPRLinkTransform, gammapy_sphinx_ext_activate

# flake8: noqa

# Add our custom directives to Sphinx
def setup(app):
    """
    Add the custom directives to Sphinx.
    """
    app.add_config_value("substitutions", [], "html")
    app.add_directive("substitution-code-block", SubstitutionCodeBlock)
    app.add_post_transform(DynamicPRLinkTransform)


conf = ConfigParser()
conf.read([os.path.join(os.path.dirname(__file__), "..", "setup.cfg")])
setup_cfg = dict(conf.items("metadata"))

sys.path.insert(0, os.path.dirname(__file__))

linkcheck_anchors_ignore = []
linkcheck_ignore = [
    "http://gamma-sky.net/#",
    "https://forge.in2p3.fr/projects/data-challenge-1-dc-1/wiki",
    "https://indico.cta-observatory.org/event/2070/",
    "https://groups.google.com/forum/#!forum/astropy-dev",
    "https://data.hawc-observatory.org/datasets/3hwc-survey/index.php",
    "https://gammapy.slack.com/join/signup#/domain-signup",
    # invalid certificates
    "https://www.hawc-observatory.org/",
    "https://data.hawc-observatory.org/datasets/crab_events_pass4/index.php",
    "https://ipython.org",
    "https://jupyter.org",
    # private pages
    "https://cchesswiki.in2p3.fr/*",
    "https://github.com/VERITAS-Observatory/VEGAS",
]

# the buttons link to html pages which are auto-generated...
linkcheck_exclude_documents = [r"getting-started/.*"]

# -- General configuration ----------------------------------------------------

# By default, highlight as Python 3.
highlight_language = "python3"

# Matplotlib directive sets whether to show a link to the source in HTML
plot_html_show_source_link = False

# If true, figures, tables and code-blocks are automatically numbered if they have a caption
numfig = False

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# Allow to add the canonical html flag on all the pages
# This allows you to indicate to the web search engine that the pages for the stable version are privileged
html_baseurl = 'https://docs.gammapy.org/stable/'

# The reST default role (used for this markup: `text`) to use for all
# documents. Set to the "smart" one.
default_role = 'obj'

# Add any Sphinx extension module names here, as strings.
extensions = [
    # Order for sphinx_automodapi is important
    "sphinx.ext.autosummary",
    "sphinx_automodapi.automodapi", # This should come after autosummary
    "sphinx_automodapi.smart_resolver",
    "sphinx_click.ext",
    'sphinx_copybutton',
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    'sphinx.ext.viewcode',
    # Allows for mapping to other documentation projects
    "sphinx.ext.intersphinx",
    # Allows for Numpy docstring format
    "numpydoc",
    # Needed for the plot:: functionality in rst
    "matplotlib.sphinxext.plot_directive",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    "**.ipynb_checkpoints",
    "user-guide/model-gallery/*/*.ipynb",
    "user-guide/model-gallery/*/*.md5",
    "user-guide/model-gallery/*/*.py",
    "_build",
]

# Define intersphinx_mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
	"matplotlib": ("https://matplotlib.org/", None),
	"astropy": ("https://docs.astropy.org/en/stable/", None),
	"regions": ("https://astropy-regions.readthedocs.io/en/latest/", None),
	"reproject": ("https://reproject.readthedocs.io/en/latest/", None),
	"naima": ("https://naima.readthedocs.io/en/latest/", None),
	"gadf": ("https://gamma-astro-data-formats.readthedocs.io/en/latest/", None),
	"iminuit": ("https://iminuit.readthedocs.io/en/latest/", None),
	"pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
	}

# -- Options for autosummary/autodoc output ------------------------------------
# Enable generation of stub files
autosummary_generate = True

# Document inherited members
automodsumm_inherited_members = True

# Include class and __init__ docstrings
autoclass_content = "both"

# Directory for API docs
automodapi_toctreedirnm = 'api'

# Suppress member summaries
numpydoc_show_class_members = False

# autodoc options to make the AnalysisConfig look nice
autodoc_class_signature = "separated"
autodoc_typehints = "description"

# Ensures that when users click the "Copy" button, only the actual code is copied,
# excluding interactive prompts and indentation markers
# https://sphinx-copybutton.readthedocs.io/en/latest/use.html#using-regexp-prompt-identifiers
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# This is added to the end of RST files - a good place to put substitutions to
# be used globally.
rst_epilog = """
.. |Table| replace:: :class:`~astropy.table.Table`
.. |QTable| replace:: :class:`~astropy.table.QTable`
.. |BaseFrame| replace:: :class:`~astropy.coordinates.BaseCoordinateFrame`
.. |SkyCoord| replace:: :class:`~astropy.coordinates.SkyCoord`
"""

# This is added to keep the links to PRs in release notes
changelog_links_docpattern = [".*changelog.*", "whatsnew/.*", "release-notes/.*"]

# -- Project information -------------------------------------------------------

# This does not *have* to match the package name, but typically does
project = setup_cfg["name"]
author = setup_cfg["author"]
copyright = "{}, {}".format(datetime.datetime.now().year, setup_cfg["author"])


version = str(gammapy.__version__)
release = "X.Y.Z" if "dev" in version else version
switch_version = "dev" if "dev" in version else release

substitutions = [
    ("|release|", release),
]
# -- Options for HTML output ---------------------------------------------------

html_theme = "pydata_sphinx_theme"

# Static files to copy after template files
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["matomo.js"]
templates_path = ["_templates"]


# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_logo = os.path.join(html_static_path[0], "gammapy_logo.png")
html_favicon = os.path.join(html_static_path[0], "gammapy_logo.ico")

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "search": ["search-field.html"],
    "navigation": ["sidebar-nav-bs.html"],
}

# If not "", a "Last updated on:" timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = ""

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "{} v{}".format(project, release)

# Output file base name for HTML help builder.
htmlhelp_basename = f"{project}doc"

# Default for the configuration can be found here
# https://github.com/pydata/pydata-sphinx-theme/blob/main/src/pydata_sphinx_theme/theme/pydata_sphinx_theme/theme.conf
html_theme_options = {
    "header_links_before_dropdown": 6,
    "show_toc_level": 2,
    # False = don't show "Previous" and "Next" buttons at the bottom of each page
    "show_prev_next": False,
    # links in menu
    "icon_links": [
        {
            "name": "Github",
            "url": "https://github.com/gammapy/gammapy",
            "icon": "fab fa-github-square",
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
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "navigation_with_keys": True,
    # footers
    "footer_start": ["copyright","custom-footer.html"],
    "footer_center": ["last-updated"],
    "footer_end": ["sphinx-version", "theme-version"]
}

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


# -- Other options -------------------------------------------------------------

github_issues_url = "https://github.com/gammapy/gammapy/issues/"

# In `about.rst` and `references.rst` we are giving lists of citations
# (e.g. papers using Gammapy) that partly aren't referenced from anywhere
# in the Gammapy docs. This is normal, but Sphinx emits a warning.
# The following config option suppresses the warning.
# http://www.sphinx-doc.org/en/stable/rest.html#citations
# http://www.sphinx-doc.org/en/stable/config.html#confval-suppress_warnings
suppress_warnings = ["ref.citation"]

branch = "main" if switch_version == "dev" else f"v{switch_version}"

binder_config = {
    # Required keys
    "org": "gammapy",
    "repo": "gammapy-webpage",
    "branch": branch,  # Can be any branch, tag, or commit hash. Use a branch that hosts your docs.
    "binderhub_url": "https://mybinder.org",  # Any URL of a binderhub deployment. Must be full URL (e.g. https://mybinder.org).
    "dependencies": "./binder/requirements.txt",
    "notebooks_dir": f"notebooks/{switch_version}",
    "use_jupyter_lab": True,
}

sphinx_gallery_conf = {
    # Remove the sphinx comments i.e. sphinx_gallery_thumbnail_number in tutorials
    "remove_config_comments": True,
    "examples_dirs": [
        "../examples/models",
        "../examples/tutorials",
    ],  # path to your example scripts
    "gallery_dirs": [
        "user-guide/model-gallery",
        "tutorials",
    ],  # path to where to save gallery generated output
    "subsection_order": ExplicitOrder(
        [
            "../examples/models/spatial",
            "../examples/models/spectral",
            "../examples/models/temporal",
            "../examples/tutorials/starting",
            "../examples/tutorials/data",
            "../examples/tutorials/model-gallery",
            "../examples/tutorials/details",
            "../examples/tutorials/analysis-1d",
            "../examples/tutorials/analysis-2d",
            "../examples/tutorials/analysis-3d",
            "../examples/tutorials/analysis-time",
            "../examples/tutorials/astrophysics",
            "../examples/tutorials/scripts",
        ]
    ),
    "binder": binder_config,
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("gammapy",),
    "exclude_implicit_doc": {},
    "filename_pattern": r"\.py",
    "reset_modules": ("matplotlib",),
    "within_subsection_order": "sphinxext.TutorialExplicitOrder",
    "download_all_examples": True,
    "capture_repr": ("_repr_html_", "__repr__"),
    # Show sidebar dropdowns for menu
    "nested_sections": True,
    "min_reported_time": 10,
    "show_memory": False,
    "line_numbers": False,
    "reference_url": {
        # The module you locally document uses None
        "gammapy": None,
    },
}

html_context = {
    "default_mode": "light",
}

