# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to download datasets and notebooks from gammapy-extra GitHub repo.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import click

import os
import sys
import requests
import json

log = logging.getLogger(__name__)

apigitUrl = 'https://api.github.com/repos/gammapy/gammapy-extra/git/trees/master:'
rawgitUrl = 'https://raw.githubusercontent.com/gammapy/gammapy-extra/master/'
gammapy_extra = os.environ.get('GAMMAPY_EXTRA')
list_files = ['environment.yml']


@click.option('--extra_folder', prompt='$GAMMAPY_EXTRA', required=True,
              help='Value of $GAMMAPY_EXTRA variable', default=gammapy_extra)
@click.command(name='extra')
def cli_download_extra(extra_folder):
    """Downloads datasets and notebooks"""

    # fix env var value
    if not extra_folder:
        extra_folder = os.path.join(os.getcwd(), extra_folder)

    # github rest api
    json_datasets = get_json_tree('datasets')
    json_notebooks = get_json_tree('notebooks')

    # parse json responses: fill list_files and build folders if needed
    parse_json_tree(json_datasets, extra_folder, 'datasets')
    parse_json_tree(json_notebooks, extra_folder, 'notebooks')

    # make env folder
    create_folder(extra_folder)

    # download files with progressbar
    with click.progressbar(list_files, label='Downloading files') as bar:
        for f in bar:
            get_file(extra_folder, f)

    # process finished
    show_info(extra_folder)


def create_folder(folder):
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
            log.warning(folder + ' folder created.')
    except Exception as ex:
        log.error('Failed: error creating directory. ' + folder)
        sys.exit()


def get_file(extra_folder, filename):

    url = rawgitUrl + filename
    filepath = os.path.join(extra_folder, filename)

    try:
        with open(filepath, 'wb') as f:
            response = requests.get(url, stream=True)
            cl = response.headers.get('content-length')

            if cl is None:  # no content length header
                f.write(response.content)
            else:
                dl = 0
                cl = int(cl)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
    except Exception as ex:
        log.warning(filepath+' could not be copied')


def get_json_tree(folder):

    url = apigitUrl + folder + '?recursive=1'

    try:
        r = requests.get(url)
        json_items = json.loads(r.text)
        return json_items
    except Exception as ex:
        log.error('Failed: bad response from GitHub API')
        sys.exit()


def parse_json_tree(json_tree, extra_folder, folder):

    create_folder(os.path.join(extra_folder, folder))

    for item in json_tree['tree']:

        ipath = os.path.join(folder, item['path'])
        ifolder = os.path.join(extra_folder, folder, item['path'])

        if item['type'] == 'tree':
            create_folder(ifolder)
        else:
            list_files.append(ipath)


def show_info(extra_folder):

    # explain how to access datasets
    print('The datasets and notebooks may be found in folder: '+extra_folder)
    print('Process finished.')
    if gammapy_extra != extra_folder:
        print('-')
        print('In order to access datasets from scripts you need to have GAMMAPY_EXTRA shell environment variable set.')
        print('You should have the following line in you .bashrc or .profile files:')
        print('export GAMMAPY_EXTRA='+extra_folder)
