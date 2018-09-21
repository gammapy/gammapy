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

BASE_URL = "http://gammapy.org/download"
YAML_URL = "https://raw.githubusercontent.com/gammapy/gammapy/master/tutorials/notebooks.yaml"


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

        if self.option == "tutorials":
            self.localfold = self.localfold / "datasets"
        if self.option == "notebooks":
            try:
                urlopen(url_env)
            except Exception as ex:
                log.error(ex)
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

            dssearch = ''
            if self.option == "datasets":
                dssearch = self.src
                datafound.update(self.parse_datafiles(dssearch, jsondata))

            if not dssearch:
                if self.option == "tutorials":
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
        if self.option == "datasets":
            print("***** You might want to declare GAMMAPY_DATA env variable")
            print("export GAMMAPY_DATA={}".format(GAMMAPY_DATA))
        else:
            print("***** Enter the following commands below to play with tutorials")
            print("cd {}".format(localfolder))
            print("conda env create -f {}".format(envfilename))
            print("conda activate {}".format(condaname))
            print("export GAMMAPY_DATA={}".format(GAMMAPY_DATA))
            print("jupyter lab")
        print("")

    def parse_yaml(self):
        import yaml

        if version.release:
            filename_nbs = "gammapy-" + self.release + "-tutorials.yml"
            url_nbs = BASE_URL + "/tutorials/" + filename_nbs
        else:
            url_nbs = YAML_URL

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
            if df == dataset["name"] or df == "":
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
            log.error(ex)
