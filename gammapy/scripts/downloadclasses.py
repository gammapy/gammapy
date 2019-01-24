# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Download class for gammapy download CLI."""
from __future__ import absolute_import, division, print_function, unicode_literals
import hashlib
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
DEV_SCRIPTS_YAML_URL = (
    "https://raw.githubusercontent.com/gammapy/gammapy/master/examples/scripts.yaml"
)
DEV_DATA_JSON_URL = (
    "https://raw.githubusercontent.com/gammapy/gammapy-webpage/gh-pages/download/data/gammapy-data-index.json"
)

def hashmd5(path):
    md5_hash = hashlib.md5()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    md5 = md5_hash.hexdigest()
    return md5


def get_file(ftuple):
    url, filepath, md5server = ftuple

    retrieve = True
    if md5server and Path(filepath).exists():
        md5local = hashmd5(filepath)
        if md5local == md5server:
            retrieve = False

    if retrieve:
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
                    if "hashmd5" in ds:
                        datafiles[label]["hashmd5"] = ds["hashmd5"]
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
                    filename_img = record["url"][record["url"].rfind("/") :]
                    url = record["url"].replace(filename_img, "")
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
        log.info("Looking for {}...".format(self.option))

    def getlocalfolder(self):
        namefolder = ""

        if self.release == "":
            self.version = version.version

        if self.option == "notebooks":
            namefolder = "notebooks-" + self.version

        if self.option == "scripts":
            namefolder = "scripts-" + self.version

        if self.option == "datasets":
            if self.modetutorials:
                if self.release:
                    namefolder = "datasets-" + self.version
                else:
                    namefolder = "datasets"
            else:
                if self.release:
                    self.outfolder = Path("gammapy-datasets-" + self.version)

        if namefolder:
            self.outfolder = self.outfolder / namefolder

        return self.outfolder

    def getenvironment(self):
        filename_env = "gammapy-" + self.version + "-environment.yml"
        url_file_env = BASE_URL + "/install/" + filename_env
        filepath_env = str(self.outfolder.parent / filename_env)
        try:
            log.info("Downloading {}".format(url_file_env))
            urlopen(url_file_env)
            ifolder = Path(filepath_env).parent
            ifolder.mkdir(parents=True, exist_ok=True)
            get_file((url_file_env, filepath_env, ""))
        except Exception as ex:
            log.error(ex)
            exit()

    def getfilelist(self):
        found = False
        if self.option == "notebooks" or self.modetutorials:
            self.parse_notebooks_yaml()
            if self.src != "":
                keyrec = "nb: " + self.src
                if keyrec in self.listfiles:
                    record = self.listfiles[keyrec]
                    self.listfiles = {}
                    self.listfiles[keyrec] = record
                    found = True

            imagefiles = parse_imagefiles(self.listfiles)
            self.listfiles.update(imagefiles)

        if self.option == "scripts" or self.modetutorials:
            self.parse_scripts_yaml()
            if self.src != "":
                keyrec = "sc: " + self.src
                if keyrec in self.listfiles:
                    record = self.listfiles[keyrec]
                    self.listfiles = {}
                    self.listfiles[keyrec] = record
                    found = True

        if self.src != "" and not found:
            if self.option == "notebooks":
                log.warning("Notebook {} not found".format(self.src))
            if self.option == "scripts":
                log.warning("Script {} not found".format(self.src))
            return []

        if self.option == "datasets":
            datafound = {}

            if self.release:
                filename_datasets = "gammapy-" + self.version + "-data-index.json"
                url = BASE_URL + "/data/" + filename_datasets
            else:
                url = DEV_DATA_JSON_URL

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

        if self.release:
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

    def parse_scripts_yaml(self):
        import yaml

        if self.release:
            filename_scripts = "gammapy-" + self.version + "-scripts.yml"
            url = BASE_URL + "/tutorials/" + filename_scripts
        else:
            url = DEV_SCRIPTS_YAML_URL

        log.info("Reading {}".format(url))
        txt = urlopen(url).read().decode("utf-8")

        for sc in yaml.safe_load(txt):
            path = sc["name"] + ".py"
            label = "sc: " + sc["name"]
            self.listfiles[label] = {}
            self.listfiles[label]["url"] = sc["url"]
            self.listfiles[label]["path"] = path
            self.listfiles[label]["datasets"] = []
            if "datasets" in sc:
                if sc["datasets"]:
                    for ds in sc["datasets"]:
                        self.listfiles[label]["datasets"].append(ds)


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
            md5 = ""
            if "hashmd5" in self.listfiles[rec]:
                md5 = self.listfiles[rec]["hashmd5"]
            ifolder = Path(path).parent
            ifolder.mkdir(parents=True, exist_ok=True)
            ftuple = (url, str(path), md5)
            pool.apply_async(get_file, args=(ftuple,), callback=self.progressbar)
        pool.close()
        pool.join()
        pool.close()

    def show_info(self):
        print("")
        if self.opt == "datasets":
            GAMMAPY_DATA = Path.cwd() / self.outfolder
        if self.opt == "all":
            GAMMAPY_DATA = Path.cwd() / self.outfolder.parent / "datasets"
        if self.opt == "datasets" or self.opt == "all":
            print("*** You might want to declare GAMMAPY_DATA env variable")
            print("export GAMMAPY_DATA={}".format(GAMMAPY_DATA))
            print("")
        if self.release and self.opt != "datasets":
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
