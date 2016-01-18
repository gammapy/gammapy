# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from flask import Flask, Blueprint, render_template, redirect, url_for, session
from flask_bootstrap import Bootstrap
from .forms import CatalogBrowserForm

__all__ = ['catalog_browser']

log = logging.getLogger(__name__)

catalog_browser = Blueprint('catalog_browser', __name__, static_folder='static')


@catalog_browser.route('/', methods=['GET', 'POST'])
def view_index():
    form = CatalogBrowserForm()

    if not 'catalog_name' in session:
        # This should only be executed once on start-up
        log.info('Using default values for session!')
        # TODO: this is an odd pattern to copy over the default.
        # There must be a better way!
        session['catalog_name'] = CatalogBrowserForm.catalog_name.kwargs['default']
        session['source_name'] = CatalogBrowserForm.source_name.kwargs['default']
        session['info_display'] = CatalogBrowserForm.info_display.kwargs['default']

    if form.validate_on_submit():
        session['catalog_name'] = form.catalog_name.data
        session['source_name'] = form.source_name.data
        session['info_display'] = form.info_display.data

        redirect(url_for('catalog_browser.view_index'))

    cat = source_catalogs[form.catalog_name.data]
    source = cat[form.source_name.data]
    session['source_id'] = source.index

    if session['catalog_name'] in ['3fgl', '2fhl']:
        template_name = 'fermi.html'
    elif session['catalog_name'] == 'hgps':
        template_name = 'hgps.html'
    else:
        raise ValueError('Invalid catalog_name: {}'.format(session['catalog_name']))

    return render_template(template_name,
                           form=form,
                           session=session,
                           )


@catalog_browser.route('/test')
def view_test():
    """A test page to try stuff out.
    """
    name = 'Johnny'
    return render_template('test.html', name=name)


def create_catalog_browser(config):
    try:
        import matplotlib
        matplotlib.use('agg')
    except ImportError:
        log.warning('Matplotlib not available.')

    app = Flask(__name__)
    app.register_blueprint(catalog_browser)
    app.secret_key = 'development key'
    Bootstrap(app)
    app.config['BOOTSTRAP_SERVE_LOCAL'] = True

    return app


from .api import *
