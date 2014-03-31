# Licensed under a 3-clause BSD style license - see LICENSE.rst

def get_package_data():
    files = ['data/*.xml', 'data/*.json', 'data/*.conf']
    return {'gammapy.morphology.tests': files}
