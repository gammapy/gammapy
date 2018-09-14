# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Download class for gammapy download CLI."""
from __future__ import absolute_import, division, print_function, unicode_literals
import click
import json
import logging
import os
import sys
from ..extern.six.moves.urllib.request import urlretrieve, urlopen
from ..extern.pathlib import Path

# from .. import version

log = logging.getLogger(__name__)

APIGIT_URL = "https://api.github.com/repos/gammapy/"
RAWGIT_URL = "https://raw.githubusercontent.com/gammapy/"
YAMLFLNAME = "notebooks.yaml"


class DownloadProcess(object):
    """Manages the process of downloading content from the Github repository"""

    def __init__(
        self,
        repo,
        repofold,
        specfile,
        specfold,
        release,
        listfiles,
        localfold,
        recursive,
    ):

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
                "--file and --foder are mutually exclusive options, only one is allowed."
            )
            sys.exit()

    def run(self):

        log.info("Content will be downloaded in {}".format(self.localfold))

        # download files
        if self.listfiles:
            with click.progressbar(self.listfiles, label="Downloading files") as bar:
                for f in bar:
                    self.get_file(f)
        else:
            sys.exit()

    def check_hash(self):

        # installed local gammapy hash
        if self.release == "" or self.release == "master":
            # uncomment when notebooks moved to gammapy repo
            # self.release = version.githash
            # refhash = 'refs/commits/' + self.release
            self.release = "master"
            refhash = "refs/heads/" + self.release
        # release
        elif self.release.startswith("v") and len(self.release) == 4:
            refhash = "refs/tags/" + self.release
        # commit hash
        elif len(self.release) == 40:
            refhash = "commits/" + self.release
        # not allowed
        else:
            refhash = "refs/tags/notallowed"

        # check hash
        url = APIGIT_URL + self.repo + "/git/" + refhash
        try:
            urlopen(url)
        except Exception as ex:
            log.error("Bad response from GitHub API.")
            log.error("Bad value for release or commit hash: " + self.release)
            log.error(url)
            sys.exit()

    def label_version(self):

        # rename notebooks folder with version label
        verfolder = str(Path(self.localfold) / "notebooks-") + self.release
        self.localfold = Path(verfolder)

    def build_folders(self):

        # make base folders
        if self.specfile:
            ifolder = self.localfold / Path(self.specfile).parent
            ifolder.mkdir(parents=True, exist_ok=True)
            self.listfiles.append(self.specfile)
        elif self.specfold:
            ifolder = self.localfold / Path(self.specfold)
            ifolder.mkdir(parents=True, exist_ok=True)
        else:
            ifolder = self.localfold
            ifolder.mkdir(parents=True, exist_ok=True)

    def build_files(self, tutorials=False, datasets=False):

        # fill files list
        if not self.specfile:
            if tutorials:
                for folder in self.parse_yaml(datasets):
                    self.specfold = folder
                    ifolder = self.localfold / Path(self.specfold)
                    ifolder.mkdir(parents=True, exist_ok=True)
                    json_files = self.get_json_tree()
                    self.parse_json_tree(json_files)
            else:
                json_files = self.get_json_tree()
                self.parse_json_tree(json_files)

    def get_json_tree(self):

        url = (
            APIGIT_URL + self.repo + "/git/trees/" + self.release + ":" + self.repofold
        )
        if self.specfold:
            url = url + "/" + self.specfold
        if self.recursive:
            url = url + "?recursive=1"

        try:
            r = urlopen(url)
            json_items = json.loads(r.read())
            return json_items
        except Exception as ex:
            log.error("Bad response from GitHub API.")
            log.error(url)
            log.error(ex)
            sys.exit()

    def parse_json_tree(self, json_files):

        if json_files:
            for item in json_files["tree"]:
                if self.specfold:
                    item["path"] = self.specfold + "/" + item["path"]
                ifolder = self.localfold / Path(item["path"])
                if item["type"] == "tree":
                    if self.recursive:
                        ifolder.mkdir(parents=True, exist_ok=True)
                else:
                    self.listfiles.append(item["path"])

    def parse_yaml(self, datasets=False):
        import yaml

        if datasets:
            notebooksfolder = "notebooks-" + self.release
            yamlpath = self.localfold.parent / notebooksfolder / YAMLFLNAME
        else:
            self.localfold.mkdir(parents=True, exist_ok=True)
            self.get_file("notebooks.yaml")
            yamlpath = self.localfold / YAMLFLNAME

        try:
            with open(str(yamlpath)) as fh:
                confignbs = yaml.safe_load(fh)
        except IOError as ex:
            log.error("{} could not be read.".format(str(yamlpath)))
            sys.exit()

        set_folders = set()
        for nb in confignbs:
            if nb["published"]:
                if not datasets:
                    self.listfiles.append(nb["name"] + ".ipynb")
                else:
                    if nb["datasets"]:
                        for ds in nb["datasets"]:
                            name, ext = os.path.splitext(ds)
                            if not ext:
                                set_folders.add(ds)
                            else:
                                if "/" in ds:
                                    ifolder = self.localfold / Path(ds).parent
                                    ifolder.mkdir(parents=True, exist_ok=True)
                                self.listfiles.append(ds)
        return set_folders

    def get_file(self, filename):

        url = (
            RAWGIT_URL
            + self.repo
            + "/"
            + self.release
            + "/"
            + self.repofold
            + "/"
            + filename
        )
        filepath = str(self.localfold / filename)

        if filepath.endswith("../environment.yml"):
            verfilename = "environment-" + self.release + ".yml"
            filepath = filepath.replace("environment.yml", verfilename)

        try:
            urlretrieve(url, filepath)
        except Exception as ex:
            log.error(filepath + " could not be copied.")
            log.error(ex)
