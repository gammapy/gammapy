# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to download datasets and notebooks from gammapy-extra GitHub repo."""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import click

import sys
import requests
import json
from ..extern.pathlib import Path
from ..extern.six.moves.urllib.request import urlretrieve

log = logging.getLogger(__name__)

apigitUrl = 'https://api.github.com/repos/gammapy/gammapy-extra/git/trees/master:'
rawgitUrl = 'https://raw.githubusercontent.com/gammapy/gammapy-extra/master/'
localfolder = Path('./gammapy-extra')


@click.command(name='notebooks')
def cli_download_notebooks():
    """Download notebooks"""

    downloadproc = DownloadProcess('notebooks', ['environment.yml'])
    downloadproc.go()


@click.command(name='datasets')
def cli_download_datasets():
    """Download datasets"""

    downloadproc = DownloadProcess('datasets', [])
    downloadproc.go()


def create_folder(folder):

    try:
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            log.info(str(folder) + ' folder created.')
    except Exception as ex:
        log.error('Failed: error creating directory. ' + str(folder))
        sys.exit()


def get_file(filename):

    url = rawgitUrl + filename
    filepath = localfolder / filename

    try:
        urlretrieve(url, str(filepath))
    except Exception as ex:
        log.error(str(filepath) + ' could not be copied')


def show_info():

    print('The files have been downloaded in folder {}.'.format(localfolder))
    print('Process finished.')


class DownloadProcess:
    """Manages the process of downloading the folder of the Github repository"""

    def __init__(self, repofold, listfiles):

        self.repofold = repofold
        self.listfiles = listfiles

    def go(self):

        json_files = self.get_json_tree()
        self.parse_json_tree(json_files)

        # download files with progressbar
        with click.progressbar(self.listfiles, label='Downloading files') as bar:
            for f in bar:
                get_file(f)

        # process finished
        show_info()

    def get_json_tree(self):

        url = apigitUrl + self.repofold + '?recursive=1'

        try:
            r = requests.get(url)
            json_items = json.loads(r.text)
            return json_items
        except Exception as ex:
            log.error('Failed: bad response from GitHub API')
            sys.exit()

    def parse_json_tree(self, json_files):

        for item in json_files['tree']:

            ipath = self.repofold + '/' + item['path']
            ifolder = localfolder / Path(self.repofold) / Path(item['path'])

            if item['type'] == 'tree':
                create_folder(ifolder)
            else:
                self.listfiles.append(ipath)