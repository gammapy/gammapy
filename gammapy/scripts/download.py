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
# from gammapy import version
from ..extern.pathlib import Path
from ..extern.six.moves.urllib.request import urlretrieve, urlopen

log = logging.getLogger(__name__)

apigitUrl = 'https://api.github.com/repos/gammapy/'
rawgitUrl = 'https://raw.githubusercontent.com/gammapy/'


@click.command(name="notebooks")
@click.pass_context
def cli_download_notebooks(ctx):
    """Download notebooks"""

    downloadproc = DownloadProcess(
        'gammapy-extra',
        'notebooks',
        ctx.obj['hash'],
        ['../environment.yml'],
        ctx.obj['localfolder']
    )
    downloadproc.go()

    # TODO
    # rename environment.yml with downloadproc.hash


@click.command(name="datasets")
@click.pass_context
def cli_download_datasets(ctx):
    """Download datasets"""

    downloadproc = DownloadProcess(
        'gammapy-extra',
        'datasets',
        ctx.obj['hash'],
        [],
        ctx.obj['localfolder']
    )
    downloadproc.go()


class DownloadProcess:
    """Manages the process of downloading the folder of the Github repository"""

    def __init__(self, repo, repofold, hash, listfiles, localfolder):

        self.repo = repo
        self.repofold = repofold
        self.hash = hash
        self.listfiles = listfiles
        self.localfolder = Path(localfolder) / repofold

    def go(self):

        # scan Github repo
        self.check_hash()
        json_files = self.get_json_tree()
        self.parse_json_tree(json_files)

        # download files with progressbar
        with click.progressbar(self.listfiles, label="Downloading files") as bar:
            for f in bar:
                self.get_file(f)

        # process finished
        self.show_info()

    def check_hash(self):

        # master
        # notebooks still in gammapy-extra repo
        if self.hash == '':
            # installed gammapy version
            # uncomment when notebooks moved to gammapy repo
            # self.hash = version.githash
            self.hash = 'master'
            refhash = 'refs/heads/' + self.hash
        # version
        elif self.hash.startswith('v.'):
            refhash = 'refs/tags/' + self.hash
        # commit hash
        elif len(self.hash) == 40:
            refhash = 'commits/' + self.hash
        # branch
        else:
            refhash = 'refs/heads/' + self.hash

        # check hash
        url = apigitUrl + self.repo + '/git/' + refhash
        try:
            urlopen(url)
        except Exception as ex:
            log.error('Failed: bad response from GitHub API')
            log.error(
                'Bad value for release, branch or hash: ' + self.hash)
            log.error(ex)
            sys.exit()

        # name versioned notebooks folder
        if self.repofold == 'notebooks':
            v_nbfolder = str(self.localfolder) + '-' + self.hash
            self.localfolder = Path(v_nbfolder)

    def get_json_tree(self):

        url = apigitUrl + self.repo + '/git/trees/' + self.hash + ':' + \
            self.repofold + '?recursive=1'

        try:
            r = urlopen(url)
            json_items = json.loads(r.read())
            return json_items
        except Exception as ex:
            log.error('Failed: bad response from GitHub API')
            log.error(ex)
            sys.exit()

    def parse_json_tree(self, json_files):

        for item in json_files['tree']:
            ifolder = self.localfolder / item['path']
            if item['type'] == 'tree':
                ifolder.mkdir(parents=True, exist_ok=True)
            else:
                self.listfiles.append(item['path'])

    def get_file(self, filename):

        url = rawgitUrl + self.repo + \
            '/' + self.hash + '/' + self.repofold + '/' + filename
        filepath = self.localfolder / filename

        try:
            urlretrieve(url, str(filepath))
        except Exception as ex:
            log.error(str(filepath) + ' could not be copied')
            log.error(ex)

    def show_info(self):

        print('The files have been downloaded in folder {}.'.format(
            self.localfolder))
        print('Process finished.')
