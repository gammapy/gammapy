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
from .. import version

log = logging.getLogger(__name__)

RELEASES = ["0.8"]
BASE_URL = "http://gammapy.org/download"


class DownloadProcess(object):
    """Manages the process of downloading content"""

    def __init__(self, src, out, release, option):

        self.src = src
        self.localfold = Path(out)
        self.release = release
        self.option = option
        self.listfiles = {}

    def setup(self):

        if self.release == "":
            self.release = str(version.major) + "." + str(version.minor)

        filename_env = "gammapy-" + self.release + "-environment.yml"
        filepath_env = str(self.localfold / filename_env)
        url_env = BASE_URL + "/install/" + filename_env

        if self.option == "datasets" or self.option == "tutorials":
            self.localfold = self.localfold / "datasets"
        else:
            try:
                urlopen(url_env)
            except Exception as ex:
                log.info("Release {} not found".format(self.release))
                exit()
            nbfolder = "notebooks-" + self.release
            self.localfold = self.localfold / nbfolder
            self.get_file(url_env, filepath_env)

    def files(self):

        self.parse_yaml()
        filename_dat = "gammapy-data-index.json"
        url_dat = BASE_URL + "/data/" + filename_dat
        jsondata = json.loads(urlopen(url_dat).read())

        if self.option == "notebooks" or self.option == "tutorials":
            if self.src != "":
                keyrec = "nb: " + self.src
                if keyrec in self.listfiles:
                    record = self.listfiles[keyrec]
                    self.listfiles = {}
                    self.listfiles[keyrec] = record
                else:
                    log.info("Notebook {} not found".format(self.src))
                    sys.exit()

            imagefiles = self.parse_imagefiles()
            self.listfiles.update(imagefiles)

        if self.option == "datasets" or self.option == "tutorials":
            datafound = {}

            if self.option == "datasets":
                datafound.update(self.parse_datafiles(self.src, jsondata))

            if not self.src:
                for item in self.listfiles:
                    record = self.listfiles[item]
                    if "datasets" in record:
                        if record["datasets"] != "":
                            for ds in record["datasets"]:
                                datafound.update(self.parse_datafiles(ds, jsondata))

            if not datafound:
                log.info("Dataset {} not found".format(self.src))
                sys.exit()

            self.listfiles = datafound

    def run(self):

        log.info("Content will be downloaded in {}".format(self.localfold))

        with click.progressbar(self.listfiles, label="Downloading files") as bar:
            for f in bar:
                url = self.listfiles[f]["url"]
                path = self.localfold / self.listfiles[f]["path"]
                self.get_file(url, str(path))

    def show_info(self):

        localfolder = self.localfold.parent
        condaname = "gammapy-" + self.release
        envfilename = condaname + "-environment.yml"
        GAMMAPY_DATA = Path.cwd() / localfolder / "datasets"

        print("")
        print("")
        print("***** Enter the following commands below to play with tutorials")
        print("cd {}".format(localfolder))
        print("conda env create -f {}".format(envfilename))
        print("conda activate {}".format(condaname))
        print("export GAMMAPY_DATA={}".format(GAMMAPY_DATA))
        print("jupyter lab")
        print("")

    def parse_yaml(self):
        import yaml

        filename_nbs = "gammapy-" + self.release + "-tutorials.yml"
        url_nbs = BASE_URL + "/tutorials/" + filename_nbs
        r = urlopen(url_nbs)

        for nb in yaml.safe_load(r.read()):
            path = nb["name"] + ".ipynb"
            label = "nb: " + nb["name"]
            self.listfiles[label] = {}
            self.listfiles[label]["url"] = nb["url"]
            self.listfiles[label]["path"] = path
            self.listfiles[label]["datasets"] = []
            self.listfiles[label]["images"] = []
            if "datasets" in nb:
                if nb["datasets"]:
                    for ds in nb["datasets"]:
                        self.listfiles[label]["datasets"].append(ds)
            if "images" in nb:
                if nb["images"]:
                    for im in nb["images"]:
                        self.listfiles[label]["images"].append(im)

    def parse_imagefiles(self):

        imagefiles = {}
        for item in self.listfiles:
            record = self.listfiles[item]
            if "images" in record:
                if record["images"] != "":
                    for im in record["images"]:
                        label = "im: " + im
                        path = "images/" + im + ".png"
                        url = str(Path(record["url"]).parent)
                        url = url.replace(":/", "://")
                        url = url + "/" + path
                        imagefiles[label] = {}
                        imagefiles[label]["url"] = url
                        imagefiles[label]["path"] = path
        return imagefiles

    def parse_datafiles(self, df, jsondata):

        datafiles = {}
        for dataset in jsondata:
            if df == dataset["name"]:
                if dataset["files"]:
                    for ds in dataset["files"]:
                        label = ds["path"]
                        datafiles[label] = {}
                        datafiles[label]["url"] = ds["url"]
                        datafiles[label]["path"] = ds["path"]

        return datafiles

    @staticmethod
    def get_file(url, filepath):

        try:
            ifolder = Path(filepath).parent
            ifolder.mkdir(parents=True, exist_ok=True)
            urlretrieve(url, filepath)
        except Exception as ex:
            log.error(filepath + " could not be copied.")
            log.error(url)
            log.error(ex)
