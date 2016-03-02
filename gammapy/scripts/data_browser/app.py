# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Useful tutorial to make a Flask apps: https://github.com/hplgit/web4sciapps
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from astropy.extern.six import BytesIO
try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
from flask import Flask, render_template, send_file, redirect, url_for, session
from flask_bootstrap import Bootstrap
from flask_nav import Nav
from flask_nav.elements import Navbar, View
from flask_wtf import Form
from wtforms import SelectField
from wtforms.validators import InputRequired
from ...data import DataStore, EventListDataset

# __all__ = ['run_data_browser']

log = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'development key'
Bootstrap(app)
app.config['BOOTSTRAP_SERVE_LOCAL'] = True


# TODO: communicating via app.config doesn't work!?
dir = '/Users/deil/work/_Data/hess/HESSFITS/fits_prod02/pa/Model_Deconvoluted_Prod26/Mpp_Std/'
# dir = app.config['GAMMAPY_DATA_DIR']
datastore = DataStore(dir=dir)


obs_id_choices = [23037, 23038]

class ObsForm(Form):
    obs_id = SelectField(
        label='obs_id', default=23037,
        choices=list(zip(obs_id_choices, [str(_) for _ in obs_id_choices])),
        validators=[InputRequired()]
    )

@app.route('/test/', methods=['GET', 'POST'])
def view_test():
    """Use this view to test stuff out.

    TODO: Removed once everything works.
    """
    form = ObsForm()
    # obs_id_list = datastore.index_table['OBS_ID'][:100]
    # form.obs_id.choices = list(zip(obs_id_list, obs_id_list))

    if form.validate_on_submit():
        idx = form.obs_id.data
        session['obs_id'] = idx # obs_id_list[idx][0]
        redirect(url_for('view_test'))

    return render_template('test.html',
                           form=form,
                           obs_id=session.get('obs_id'),
                           obs_id_choices=obs_id_choices,
                           )


@app.route('/')
def view_about():
    return render_template('about.html')


@app.route('/data')
def view_data():
    table = datastore.index_table
    # TODO: this should be computed on table construction or be a cached attribute
    table['DEAD_FRAC'] = 100. * (1 - table['LIVETIME'] / table['ONTIME'])
    table['DEAD_FRAC'].unit = '%'
    cols = 'OBS_ID TSTART ONTIME DEAD_FRAC RA DEC GLON GLAT ALT AZ N_TELS TEL_LIST QUALITY'
    for col in ['RA', 'DEC', 'GLON', 'GLAT', 'ALT', 'AZ']:
        table[col].format = '.3f'
    kwargs = dict(html=True, max_lines=-1, max_width=-1)
    obs_table = '\n'.join(table[cols.split()].pformat(**kwargs)[1:-1])
    return render_template('data.html', datastore=datastore, obs_table=obs_table)


@app.route('/events', methods=('GET', 'POST'))
def view_events():
    obs_id = 23037
    obs = datastore.index_table.select_obs_subset(obs_id)[0]
    filename = datastore.filename(obs_id, filetype='events')
    events = EventListDataset.read(filename).event_list

    info = events.info('stats', out=None)

    cols = 'BUNCH_ID EVENT_ID TIME ENERGY RA DEC DETX DETY'
    kwargs = dict(html=True, max_lines=-1, max_width=-1)
    event_data_table = '\n'.join(events[cols.split()].pformat(**kwargs)[1:-1])
    return render_template('events.html', obs=obs, events=events,
                           event_data_table=event_data_table)


@app.route('/irfs', methods=('GET', 'POST'))
def view_irfs():
    form = ObsForm(csrf_enabled=False)
    if form.validate_on_submit():
        result = 'hi'
    else:
        result = None

    return render_template('irfs.html', form=form, result=result)


@app.route('/image/<image_type>/<int:obs_id>')
def view_image(image_type, obs_id):
    """Generate image.
    """
    import matplotlib.pyplot as plt
    from ... import irf

    fig, ax = plt.subplots()

    if image_type == 'aeff-energy':
        filename = datastore.filename(obs_id, filetype='effective area')
        log.info('Reading {}'.format(filename))
        aeff2d = irf.EffectiveAreaTable2D.read(filename)
        aeff2d.plot_energy_dependence(ax=ax)
    else:
        log.error('Invalid image_type: {}'.format(image_type))
        # Return a dummy image ... not sure if that's a useful thing to do ...
        ax.plot([1, 3, 2, 4, obs_id])


    img = BytesIO()
    fig.savefig(img)
    img.seek(0)

    # TODO: check somehow if this leaves IRF objects or open file handles
    # hanging around ...
    # The Mac activity app shows that this file is opened many times (once per request):
    # /opt/local/Library/Frameworks/Python.framework/Versions/3.4/lib/python3.4/site-packages/matplotlib/mpl-data/fonts/ttf/Vera.ttf
    # Deleting fix and ax seems to have no effect ...
    # del fig, ax

    return send_file(img, mimetype='image/png')


topbar = Navbar('',
    View('About', 'view_about'),
    View('Data', 'view_data'),
    View('Events', 'view_events'),
    View('IRFs', 'view_irfs'),
)


nav = Nav()
nav.register_element('top', topbar)
nav.init_app(app)

# def run_data_browser(data_dir):
#     """Run a data browser web app.
#     """
#
#     log.info('data_dir = {}'.format(data_dir))
#     # http://flask.pocoo.org/docs/0.10/api/#flask.Flask.add_url_rule
#     app.add_url_rule('/', view_func=view_about, methods=['GET', 'POST'])
#     app.add_url_rule('/obs', view_func=view_obs, methods=['GET', 'POST'])
#     app.add_url_rule('/events', view_func=view_events, methods=['GET', 'POST'])
#     app.add_url_rule('/irfs', view_func=view_irfs, methods=['GET', 'POST'])
#     app.add_url_rule('/image/<int:obs_id>', view_func=view_image)
