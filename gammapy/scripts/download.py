# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to download datasets and notebooks from gammapy-extra GitHub repo.
GitHub REST API is used to scan the tree-folder strucutre and get commmit hash.
https://developer.github.com/v3/
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import click
import sys
import json
# from .. import version
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
        ctx.obj['specfile'],
        ctx.obj['specfold'],
        ctx.obj['release'],
        ['../environment.yml'],
        Path(ctx.obj['localfold']) / 'notebooks',
        ctx.obj['recursive']
    )

    # valid hash
    downloadproc.check_hash()

    # get version label
    version = ctx.obj['release']
    if ctx.obj['release'] == '':
        # uncomment when notebooks moved to gammapy repo
        # version = version.version
        version = 'master'

    # rename notebooks folder with version label
    verfolder = str(
        Path(ctx.obj['localfold']) / 'notebooks-') + version
    downloadproc.localfold = Path(verfolder)

    # download
    downloadproc.run()

    # rename environment.yml with version label
    envfile = Path(ctx.obj['localfold']) / 'environment.yml'
    verfile = str(
        Path(ctx.obj['localfold']) / 'environment-') + version + '.yml'
    envfile.rename(verfile)


@click.command(name="datasets")
@click.pass_context
def cli_download_datasets(ctx):
    """Download datasets"""

    downloadproc = DownloadProcess(
        'gammapy-extra',
        'datasets',
        ctx.obj['specfile'],
        ctx.obj['specfold'],
        ctx.obj['release'],
        [],
        Path(ctx.obj['localfold']) / 'datasets',
        ctx.obj['recursive']
    )

    # valid hash
    downloadproc.check_hash()

    # download
    downloadproc.run()


class DownloadProcess:
    """Manages the process of downloading content from the Github repository"""

    def __init__(self, repo, repofold, specfile, specfold,
                 release, listfiles, localfold, recursive):

        self.repo = repo
        self.repofold = repofold
        self.specfile = specfile
        self.specfold = specfold
        self.release = release
        self.listfiles = listfiles
        self.localfold = localfold
        self.recursive = recursive

        if specfile and specfold:
            log.error(
                '--file and --foder are mutually exclusive options, only one is allowed.')
            sys.exit()

    def run(self):

        # fill list of files
        self.build_files()
        print('Content will be downloaded in {}'.format(self.localfold))

        # download files
        if self.listfiles:
            with click.progressbar(self.listfiles, label='Downloading files') as bar:
                for f in bar:
                    self.get_file(f)
        else:
            sys.exit()

    def check_hash(self):

        # installed local gammapy hash
        if self.release == '' or self.release == 'master':
            # uncomment when notebooks moved to gammapy repo
            # self.release = version.githash
            # refhash = 'refs/commits/' + self.release
            self.release = 'master'
            refhash = 'refs/heads/' + self.release
        # release
        elif self.release.startswith('v') and len(self.release) == 4:
            refhash = 'refs/tags/' + self.release
        # commit hash
        elif len(self.release) == 40:
            refhash = 'commits/' + self.release
        # not allowed
        else:
            refhash = 'refs/tags/notallowed'

        # check hash
        url = apigitUrl + self.repo + '/git/' + refhash

        try:
            urlopen(url)
        except Exception as ex:
            log.error('Bad response from GitHub API.')
            log.error(
                'Bad value for release or commit hash: ' + self.release)
            sys.exit()

    def build_files(self):

        if self.specfile:
            ifolder = self.localfold / Path(self.specfile).parent
            ifolder.mkdir(parents=True, exist_ok=True)
            self.listfiles.append(self.specfile)
        else:
            json_files = self.get_json_tree()
            self.parse_json_tree(json_files)

    def get_json_tree(self):

        url = apigitUrl + self.repo + '/git/trees/' + self.release + ':' + self.repofold
        if self.specfold:
            url = url + '/' + self.specfold
        if self.recursive:
            url = url + '?recursive=1'

        try:
            r = urlopen(url)
            json_items = json.loads(r.read())
            return json_items
        except Exception as ex:
            log.error('Bad response from GitHub API.')
            log.error(ex)
            sys.exit()

    def parse_json_tree(self, json_files):

        if self.specfold:
            ifolder = self.localfold / Path(self.specfold)
        else:
            ifolder = self.localfold
        ifolder.mkdir(parents=True, exist_ok=True)

        for item in json_files['tree']:
            if self.specfold:
                item['path'] = self.specfold + '/' + item['path']
            ifolder = self.localfold / Path(item['path'])
            if item['type'] == 'tree':
                if self.recursive:
                    ifolder.mkdir(parents=True, exist_ok=True)
            else:
                self.listfiles.append(item['path'])

    def get_file(self, filename):

        url = rawgitUrl + self.repo + \
            '/' + self.release + '/' + self.repofold + '/' + filename
        filepath = self.localfold / filename

        try:
            urlretrieve(url, str(filepath))
        except Exception as ex:
            log.error(str(filepath) + ' could not be copied.')
            log.error(ex)
