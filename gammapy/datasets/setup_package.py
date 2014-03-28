# Licensed under a 3-clause BSD style license - see LICENSE.rst

def get_package_data():
    files = ['README.rst',
             'poisson_stats_image/*',
             'fermi/*',
             'tev_spectra/*'
             ]
    return {'gammapy.datasets': files}
