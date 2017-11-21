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
import re
import nbformat
from nbformat.v4 import new_markdown_cell
from shutil import copytree, rmtree
from docutils.parsers.rst.directives.images import Image
from docutils.parsers.rst.directives import register_directive
from docutils.parsers.rst import roles
from docutils import nodes
from .scripts import read_yaml
from ..extern.pathlib import Path

try:
    gammapy_extra_path = Path(os.environ['GAMMAPY_EXTRA'])
    HAS_GP_EXTRA = True
except KeyError:
    HAS_GP_EXTRA = False


class ExtraImage(Image):
    """Directive to add optional images from gammapy-extra"""

    def run(self):
        filename = self.arguments[0]

        if HAS_GP_EXTRA:
            path = gammapy_extra_path / 'figures' / filename
            if not path.is_file():
                msg = 'Error in {} directive: File not found: {}'.format(self.name, path)
                raise self.error(msg)
            # Usually Sphinx doesn't support absolute paths
            # But passing a POSIX string of the absolute path
            # with an extra "/" at the start seems to do the trick
            self.arguments[0] = '/' + path.absolute().as_posix()
        else:
            self.warning('GAMMAPY_EXTRA not available. Missing image: {}'.format(self.name, filename))
            self.options['alt'] = self.arguments[1]

        return super(ExtraImage, self).run()


def notebook_role(name, rawtext, notebook, lineno, inliner, options={}, content=[]):
    """Link to a notebook on gammapy-extra"""
    if HAS_GP_EXTRA:
        available_notebooks = read_yaml('$GAMMAPY_EXTRA/notebooks/notebooks.yaml')
        exists = notebook in [_['name'] for _ in available_notebooks]
    else:
        exists = True

    if not exists:
        msg = inliner.reporter.error(
            'Unknown notebook {}'.format(notebook),
            line=lineno,
        )
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]
    else:
        app = inliner.document.settings.env.app
        node = make_link_node(rawtext, app, notebook, options)
        return [node], []


def make_link_node(rawtext, app, notebook, options):
    # base = 'https://github.com/gammapy/gammapy-extra/tree/master/notebooks/'
    base = 'https://nbviewer.jupyter.org/github/gammapy/gammapy-extra/blob/master/notebooks/'
    full_name = notebook + '.ipynb'
    ref = base + full_name
    roles.set_classes(options)
    node = nodes.reference(rawtext, full_name, refuri=ref, **options)
    return node


def gammapy_sphinx_ext_activate():
    if HAS_GP_EXTRA:
        print('*** Found GAMMAPY_EXTRA = {}'.format(gammapy_extra_path))
        print('*** Nice!')
    else:
        print('*** gammapy-extra *not* found.')
        print('*** Set the GAMMAPY_EXTRA environment variable!')
        print('*** Docs build will be incomplete.')
        print('*** Notebook links will not be verified.')

    # Register our directives and roles with Sphinx
    register_directive('gp-extra-image', ExtraImage)
    roles.register_local_role('gp-extra-notebook', notebook_role)

def modif_nb_links(folder, url_docs):
    """
    Modifies links in raw and sphinx formatted notebooks and so they
    point to and from the same version of the documentation. Adds a box to the
    sphinx formatted notebooks with info and link to the ipynb file.
    """

    DOWNLOAD_CELL = """
<div class='admonition note'>
This is a *fixed-text* formatted version of a Jupyter notebook.

You can download for each version of *gammapy* a
[HTMLZip pack](http://readthedocs.org/projects/gammapy/downloads/) containing
the whole documentation and full collection of notebooks, so you can execute
them in your local _static/notebooks/ folder. You can also contribute with
your own notebooks in this
[GitHub repository](https://github.com/gammapy/gammapy-extra/tree/master/notebooks).

**Download source files:**
[{nb_filename}](../_static/notebooks/{nb_filename}) |
[{py_filename}](../_static/notebooks/{py_filename})
</div>"""

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath) and filepath[-6:] == '.ipynb':
            if folder=='notebooks':
                py_filename = filename.replace('ipynb', 'py')
                ctx = dict(nb_filename=filename, py_filename=py_filename)
                strcell = DOWNLOAD_CELL.format(**ctx)
                nb = nbformat.read(filepath, as_version=nbformat.NO_CONVERT)
                nb.cells.insert(0, new_markdown_cell(strcell))
                nbformat.write(nb, filepath)
            with open(filepath) as f:
                txt = f.read()
            if folder=='notebooks':
                txt = re.sub(url_docs+'(.*?)html(\)|#)',r'..\1rst\2', txt, flags=re.M|re.I)
            if folder=='_static/notebooks':
                txt = re.sub(url_docs+'(.*?)html(\)|#)',r'..\/..\1html\2', txt, flags=re.M|re.I)
            with open(filepath, "w") as f:
                f.write(txt)

def gammapy_sphinx_notebooks(setup_cfg):
    """
    Manages the processes for the building of sphinx formatted notebooks
    """

    url_docs = setup_cfg.get('url_docs')

    # remove existing notebooks if rebuilding
    if bool(setup_cfg.get('clean_notebooks')):
        print('*** Cleaning notebooks')
        rmtree('notebooks', ignore_errors=True)
        rmtree('_static/notebooks', ignore_errors=True)

    # copy and build notebooks if empty
    if os.environ.get('GAMMAPY_EXTRA') and not os.path.isdir("notebooks"):
        gammapy_extra_notebooks_folder = os.environ['GAMMAPY_EXTRA'] + '/notebooks'
        if os.path.isdir(gammapy_extra_notebooks_folder):
            ignorefiles = lambda d, files: [f for f in files
                if os.path.isfile(os.path.join(d, f)) and f[-6:] != '.ipynb' and f[-4:] != '.png']
            print('*** Converting notebooks to scripts')
            copytree(gammapy_extra_notebooks_folder, 'notebooks', ignore=ignorefiles)
            copytree(gammapy_extra_notebooks_folder, '_static/notebooks')
            os.system('jupyter nbconvert --to script _static/notebooks/*.ipynb')
            modif_nb_links('notebooks', url_docs)
            modif_nb_links('_static/notebooks', url_docs)
