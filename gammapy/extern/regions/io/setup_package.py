# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os

def get_package_data():
    parser_test = ['data/*.reg']
    return {'regions.io.tests': parser_test}
