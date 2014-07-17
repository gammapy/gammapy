# Licensed under a 3-clause BSD style license - see LICENSE.rst


def get_package_data():
    files = ['README.rst',
             'data/README.rst',
             'data/fermi/*',
             'data/poisson_stats_image/*',
             'data/tev_spectra/*.txt',
             'data/tev_spectra/*.fits.gz',
             ]
    return {'gammapy.datasets': files}
