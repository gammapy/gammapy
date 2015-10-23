# Licensed under a 3-clause BSD style license - see LICENSE.rst


def get_package_data():
    files = ['datasets.yaml',
             'data/README.rst',
             'data/fermi/*',
             'data/fermi_vela/*',
             'data/hess/*',
             'data/poisson_stats_image/*',
             'data/tev_spectra/*',
             'data/tev_catalogs/*',
             'data/irfs/*',
             'data/atnf/*'
             ]
    return {'gammapy.datasets': files}
