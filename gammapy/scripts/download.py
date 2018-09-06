# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to download datasets and notebooks from gammapy-extra GitHub repo.
GitHub REST API is used to access tree-folder and file lists for a specific commit/tag.
https://developer.github.com/v3/
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import click

import sys
import json
from ..extern.pathlib import Path
from ..extern.six.moves.urllib.request import urlretrieve, urlopen

log = logging.getLogger(__name__)

apigitUrl = 'https://api.github.com/repos/gammapy/gammapy-extra/git/trees/master:'
rawgitUrl = 'https://raw.githubusercontent.com/gammapy/gammapy-extra/master/'


@click.command(name='notebooks')
@click.pass_context
def cli_download_notebooks(ctx):
    """Download notebooks"""

    localfolder = Path(ctx.obj['localfolder'])
    downloadproc = DownloadProcess('notebooks', ['environment.yml'], localfolder)
    downloadproc.go()


@click.command(name='datasets')
@click.pass_context
def cli_download_datasets(ctx):
    """Download datasets"""

    localfolder = Path(ctx.obj['localfolder'])
    downloadproc = DownloadProcess('datasets', [], localfolder)
    downloadproc.go()


class DownloadProcess:
    """Manages the process of downloading the folder of the Github repository"""

    def __init__(self, repofold, listfiles, localfolder):

        self.repofold = repofold
        self.listfiles = listfiles
        self.localfolder = localfolder

    def go(self):

        json_files = self.get_json_tree()
        self.parse_json_tree(json_files)

        # download files with progressbar
        with click.progressbar(self.listfiles, label='Downloading files') as bar:
            for f in bar:
                self.get_file(f)

        # process finished
        self.show_info()

    def get_json_tree(self):

        url = apigitUrl + self.repofold + '?recursive=1'

        try:
            r = urlopen(url)
            json_items = json.loads(r.read())
            return json_items
        except Exception as ex:
            log.error('Failed: bad response from GitHub API')
            sys.exit()

    def parse_json_tree(self, json_files):

        for item in json_files['tree']:

            ipath = self.repofold + '/' + item['path']
            ifolder = self.localfolder / self.repofold / item['path']

            if item['type'] == 'tree':
                ifolder.mkdir(parents=True, exist_ok=True)
            else:
                self.listfiles.append(ipath)

    def get_file(self, filename):

        url = rawgitUrl + filename
        filepath = self.localfolder / filename

        try:
            urlretrieve(url, str(filepath))
        except Exception as ex:
            log.error(str(filepath) + ' could not be copied')

    def show_info(self,):

        print('The files have been downloaded in folder {}.'.format(self.localfolder))
        print('Process finished.')
