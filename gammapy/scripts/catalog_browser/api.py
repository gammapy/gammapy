# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from flask import send_file, jsonify
from io import BytesIO
from ...catalog import source_catalogs
from .app import catalog_browser


@catalog_browser.route('/api/data/<catalog_name>/<int:source_id>')
def view_api_data(catalog_name, source_id):
    """Source table data in JSON format that DataTables understands.
    https://datatables.net/examples/data_sources/ajax.html
    """
    catalog = source_catalogs[catalog_name]
    source = catalog[source_id]

    data = []
    for key, value in source.data.items():
        # Only include data that's easy to display in a table (not vector columns)
        if catalog.name in ['3fgl', '2fhl'] and key in ['Flux_History', 'Unc_Flux_History']:
            continue

        data.append([key, str(value)])
    return jsonify(dict(data=data))


@catalog_browser.route('/api/image/<image_type>/<catalog_name>/<int:source_id>')
def view_api_image(image_type, catalog_name, source_id):
    """Source spectrum image."""
    import matplotlib.pyplot as plt

    catalog = source_catalogs[catalog_name]
    source = catalog[source_id]

    plt.style.use('fivethirtyeight')

    if image_type == 'spectrum':
        fig, ax = plt.subplots()
        source.plot_spectrum(ax=ax)
    if image_type == 'lightcurve':
        fig, ax = plt.subplots()
        source.plot_lightcurve(ax=ax)
    elif image_type == 'test':
        fig, ax = plt.subplots()
        ax.plot([2, 4, 3])
    else:
        raise ValueError('Invalid image_type: {}'.format(image_type))

    fig.tight_layout()
    # fig.canvas.draw()

    img = BytesIO()
    fig.savefig(img)
    img.seek(0)

    del fig, ax

    return send_file(img, mimetype='image/png')
