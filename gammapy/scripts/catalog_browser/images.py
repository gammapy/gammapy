"""Catalog browser view functions that return images.
"""
from ...datasets import fermi
from .app import catalog_browser


@catalog_browser.route('/image/<catalog_name>/<int:source_index>')
def view_image_spectrum(catalog_name, source_index):
    """Generate image.
    """
    import matplotlib.pyplot as plt

    if catalog_name == '3FGL':
        catalog_object = fermi.Fermi3FGLObject(source)
    else:
        raise ValueError('Invalid catalog: {}'.format(catalog_name))

    plt.style.use('fivethirtyeight')

    ax = catalog_object.plot_spectrum()
    ax.plot()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()

    img = BytesIO()
    fig.savefig(img)
    img.seek(0)

    return send_file(img, mimetype='image/png')
