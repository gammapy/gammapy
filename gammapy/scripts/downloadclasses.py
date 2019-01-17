# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Download class for gammapy download CLI."""
from __future__ import absolute_import, division, print_function, unicode_literals
import json
import logging
import sys
import multiprocessing
from ..extern.six.moves.urllib.request import urlretrieve, urlopen
from ..extern.pathlib import Path
from .. import version

log = logging.getLogger(__name__)

BASE_URL = "https://gammapy.org/download"
DEV_NBS_YAML_URL = (
    "https://raw.githubusercontent.com/gammapy/gammapy/master/tutorials/notebooks.yaml"
)


def get_file(ftuple):
    url, filepath = ftuple
    try:
        urlretrieve(url, filepath)
    except Exception as ex:
        log.error(filepath + " could not be copied.")
        log.error(ex)


def parse_datafiles(datasearch, datasetslist):
    datafiles = {}
    for dataset in datasetslist:
        if datasearch == dataset["name"] or datasearch == "":
            if dataset["files"]:
                for ds in dataset["files"]:
                    label = ds["path"]
                    datafiles[label] = {}
                    datafiles[label]["url"] = ds["url"]
                    datafiles[label]["path"] = ds["path"]
    return datafiles


def parse_imagefiles(notebookslist):
    imagefiles = {}
    for item in notebookslist:
        record = notebookslist[item]
        if "images" in record:
            if record["images"]:
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


class ComputePlan(object):
    """Generates the whole list of files to download"""

    def __init__(self, src, outfolder, version, option, modetutorials=False):
        self.src = src
        self.outfolder = Path(outfolder)
        self.version = version
        self.release = version
        self.option = option
        self.modetutorials = modetutorials
        self.listfiles = {}

    def getlocalfolder(self):
        if self.version == "":
            self.version = version.version

        if self.option == "datasets" and self.modetutorials:
            self.outfolder = self.outfolder / "datasets"

        if self.option == "notebooks":
            nbfolder = "notebooks-" + self.version
            self.outfolder = self.outfolder / nbfolder
            if self.release:
                filename_env = "gammapy-" + self.version + "-environment.yml"
                url_file_env = BASE_URL + "/install/" + filename_env
                filepath_env = str(self.outfolder.parent / filename_env)
                try:
                    log.info("Downloading {}".format(url_file_env))
                    urlopen(url_file_env)
                    ifolder = Path(filepath_env).parent
                    ifolder.mkdir(parents=True, exist_ok=True)
                    get_file((url_file_env, filepath_env))
                except Exception as ex:
                    log.error(ex)
                    exit()
        return self.outfolder

    def getfilelist(self):
        if self.option == "notebooks" or self.modetutorials:
            self.parse_notebooks_yaml()
            if self.src != "":
                keyrec = "nb: " + self.src
                if keyrec in self.listfiles:
                    record = self.listfiles[keyrec]
                    self.listfiles = {}
                    self.listfiles[keyrec] = record
                else:
                    log.info("Notebook {} not found".format(self.src))
                    sys.exit()

            imagefiles = parse_imagefiles(self.listfiles)
            self.listfiles.update(imagefiles)

        if self.option == "datasets":
            datafound = {}

            url = BASE_URL + "/data/gammapy-data-index.json"
            log.info("Reading {}".format(url))
            txt = urlopen(url).read().decode("utf-8")
            datasets = json.loads(txt)

            if not self.modetutorials:
                datafound.update(parse_datafiles(self.src, datasets))
            else:
                for item in self.listfiles:
                    record = self.listfiles[item]
                    if "datasets" in record:
                        if record["datasets"] != "":
                            for ds in record["datasets"]:
                                datafound.update(parse_datafiles(ds, datasets))

            if not datafound:
                log.info("Dataset {} not found".format(self.src))
                sys.exit()
            self.listfiles = datafound

        return self.listfiles

    def parse_notebooks_yaml(self):
        import yaml

        if version.release:
            filename_nbs = "gammapy-" + self.version + "-tutorials.yml"
            url = BASE_URL + "/tutorials/" + filename_nbs
        else:
            url = DEV_NBS_YAML_URL

        log.info("Reading {}".format(url))
        txt = urlopen(url).read().decode("utf-8")

        for nb in yaml.safe_load(txt):
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


class ParallelDownload(object):
    """Manages the process of downloading files"""

    def __init__(self, listfiles, outfolder, release, opt):
        self.listfiles = listfiles
        self.outfolder = outfolder
        self.release = release
        self.opt = opt
        self.bar = 0

    def run(self):
        log.info("Content will be downloaded in {}".format(self.outfolder))

        pool = multiprocessing.Pool(5)
        for rec in self.listfiles:
            url = self.listfiles[rec]["url"]
            path = self.outfolder / self.listfiles[rec]["path"]
            ifolder = Path(path).parent
            ifolder.mkdir(parents=True, exist_ok=True)
            ftuple = (url, str(path))
            pool.apply_async(get_file, args=(ftuple,), callback=self.progressbar)
        pool.close()
        pool.join()
        pool.close()

    def show_info(self):
        print("")
        if self.opt == "datasets":
            GAMMAPY_DATA = Path.cwd() / self.outfolder / "datasets"
        if self.opt == "all":
            GAMMAPY_DATA = Path.cwd() / self.outfolder.parent / "datasets"
        if self.opt == "datasets" or self.opt == "all":
            print("*** You might want to declare GAMMAPY_DATA env variable")
            print("export GAMMAPY_DATA={}".format(GAMMAPY_DATA))
            print("")
        if self.release:
            print(
                "*** Enter the following commands below to get started with this version of Gammapy"
            )
            print("cd {}".format(self.outfolder.parent))
            condaname = "gammapy-" + self.release
            envfilename = condaname + "-environment.yml"
            print("conda env create -f {}".format(envfilename))
            print("conda activate {}".format(condaname))
            print("jupyter lab")
            print("")

    def progressbar(self, args):
        self.bar += 1
        barLength, status = 50, ""
        progress = self.bar / len(self.listfiles)
        if progress >= 1.0:
            progress, status = 1, "\r\n"
        block = int(round(barLength * progress))
        text = "\rDownloading files [{}] {:.0f}% {}".format(
            "=" * block + "." * (barLength - block), round(progress * 100, 0), status
        )
        sys.stdout.write(text)
        sys.stdout.flush()
