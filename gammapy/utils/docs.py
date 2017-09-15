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
