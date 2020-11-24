# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Download class for gammapy download CLI."""
import hashlib
import json
import logging
import sys
# from configparser import ConfigParser
from pathlib import Path
from urllib.request import urlopen
import yaml
from gammapy import __version__

log = logging.getLogger(__name__)

# fetch params from setup.cfg
# PATH_CFG = Path(__file__).resolve().parent / ".." / ".."
# conf = ConfigParser()
# conf.read(PATH_CFG / "setup.cfg")
# setup_cfg = dict(conf.items("metadata"))
# URL_GAMMAPY_MASTER = setup_cfg["url_raw_github"]

URL_GAMMAPY_MASTER = "https://raw.githubusercontent.com/gammapy/gammapy/master/"
RELEASES_BASE_URL = "https://gammapy.org/download"
DEV_NBS_YAML_URL = f"{URL_GAMMAPY_MASTER}notebooks.yaml"
DEV_SCRIPTS_YAML_URL = f"{URL_GAMMAPY_MASTER}examples/scripts.yaml"
DEV_DATA_JSON_LOCAL = "../../dev/datasets/gammapy-data-index.json"  # CI tests
TAR_BUNDLE = "https://github.com/gammapy/gammapy-data/tarball/master"
# Curated datasets bundle


def parse_datafiles(datasearch, datasetslist, download_tests=False):
    for dataset in datasetslist:
        if dataset["name"] == "tests" and not download_tests and datasearch != "tests":
            continue
        if datasearch in [dataset["name"], ""] and dataset.get("files", ""):
            for ds in dataset["files"]:
                label = ds["path"]
                data = {"url": ds["url"], "path": ds["path"]}
                if "hashmd5" in ds:
                    data["hashmd5"] = ds["hashmd5"]
                yield label, data


def parse_imagefiles(notebookslist):
    for item in notebookslist:
        record = notebookslist[item]
        if record.get("images", ""):
            for im in record["images"]:
                label = "im: " + im
                path = "images/" + im + ".png"
                filename_img = record["url"][record["url"].rfind("/") :]
                url = record["url"].replace(filename_img, "")
                url = url + "/" + path
                data = {"url": url, "path": path}
                yield label, data


class ComputePlan:
    """Generates the whole list of files to download"""

    def __init__(
        self,
        src,
        outfolder,
        release,
        option,
        modetutorials=False,
        download_tests=False,
        all_notebooks=False,
    ):
        self.src = src
        self.outfolder = Path(outfolder)
        self.release = release
        self.option = option
        self.modetutorials = modetutorials
        self.download_tests = download_tests
        self.all_notebooks = all_notebooks
        self.listfiles = {}
        log.info(f"Looking for {self.option}...")

        if self.release == "" and "dev" not in __version__:
            self.release = __version__

    def getenvironment(self):
        try:
            from parfive import Downloader
        except ImportError:
            log.error("To use gammapy download, install the parfive package!")
            return

        dl = Downloader(progress=False, file_progress=False)
        filename_env = "gammapy-" + self.release + "-environment.yml"
        url_file_env = RELEASES_BASE_URL + "/install/" + filename_env
        filepath_env = str(self.outfolder / filename_env)
        dl.enqueue_file(url_file_env, path=filepath_env)
        try:
            log.info(f"Downloading {url_file_env}")
            Path(filepath_env).parent.mkdir(parents=True, exist_ok=True)
            dl.download()
        except Exception as ex:
            log.error(ex)
            exit()

    def getlocalfolder(self):
        suffix = f"-{self.release}"

        if self.release == "":
            suffix += __version__
        if self.option == "notebooks":
            return self.outfolder / f"notebooks{suffix}"
        if self.option == "scripts":
            return self.outfolder / f"scripts{suffix}"
        if self.option == "datasets" and self.modetutorials:
            return self.outfolder / "datasets"
        return self.outfolder

    def getonefile(self, keyrec, filetype):
        if keyrec in self.listfiles:
            record = self.listfiles[keyrec]
            self.listfiles = {}
            self.listfiles[keyrec] = record
        else:
            self.listfiles = {}
            if not self.modetutorials:
                log.warning(f"{filetype} {self.src} not found")

    def getfilelist(self):
        if self.option == "notebooks" or self.modetutorials:
            self.parse_notebooks_yaml()
            if self.src != "":
                self.getonefile("nb: " + self.src, "Notebook")
            self.listfiles.update(dict(parse_imagefiles(self.listfiles)))

        if (self.option == "scripts" or self.modetutorials) and not self.listfiles:
            self.parse_scripts_yaml()
            if self.src != "":
                self.getonefile("sc: " + self.src, "Script")

        if self.option == "datasets":
            if self.modetutorials and not self.listfiles:
                sys.exit()
            # datasets bundle
            if not self.src:
                self.listfiles = {"bundle": {"path": self.outfolder, "url": TAR_BUNDLE}}
                return self.listfiles
            # collection of files
            if self.release:
                url_datasets_json = (
                    RELEASES_BASE_URL
                    + "/data/gammapy-"
                    + self.release
                    + "-data-index.json"
                )
                log.info(f"Reading {url_datasets_json}")
                try:
                    txt = urlopen(url_datasets_json).read().decode("utf-8")
                except Exception as ex:
                    log.error(ex)
                    return False
            else:
                # for development just use the local index file
                local_datasets_json = (
                    Path(__file__).parent / DEV_DATA_JSON_LOCAL
                ).resolve()
                log.info(f"Reading {local_datasets_json}")
                txt = local_datasets_json.read_text()
            datasets = json.loads(txt)
            datafound = {}
            if not self.modetutorials:
                datafound.update(
                    dict(
                        parse_datafiles(
                            self.src, datasets, download_tests=self.download_tests
                        )
                    )
                )
            else:
                for item in self.listfiles:
                    record = self.listfiles[item]
                    if record.get("datasets", ""):
                        for ds in record["datasets"]:
                            datafound.update(dict(parse_datafiles(ds, datasets)))
            self.listfiles = datafound
            if not datafound:
                log.info("No datasets found")
                sys.exit()

        return self.listfiles

    def parse_notebooks_yaml(self):
        url = DEV_NBS_YAML_URL
        if self.release:
            filename_nbs = "gammapy-" + self.release + "-tutorials.yml"
            url = RELEASES_BASE_URL + "/tutorials/" + filename_nbs

        log.info(f"Reading {url}")
        try:
            txt = urlopen(url).read().decode("utf-8")
        except Exception as ex:
            log.error(ex)
            return False

        for nb in yaml.safe_load(txt):
            if not (nb.get("tutorial", True) or self.all_notebooks):
                continue
            path = nb["name"] + ".ipynb"
            label = "nb: " + nb["name"]
            self.listfiles[label] = {}
            self.listfiles[label]["url"] = nb["url"]
            self.listfiles[label]["path"] = path
            self.listfiles[label]["datasets"] = []
            self.listfiles[label]["images"] = []
            if nb.get("datasets", ""):
                for ds in nb["datasets"]:
                    self.listfiles[label]["datasets"].append(ds)
            if nb.get("images", ""):
                for im in nb["images"]:
                    self.listfiles[label]["images"].append(im)

    def parse_scripts_yaml(self):
        url = DEV_SCRIPTS_YAML_URL
        if self.release:
            filename_scripts = "gammapy-" + self.release + "-scripts.yml"
            url = RELEASES_BASE_URL + "/tutorials/" + filename_scripts

        log.info(f"Reading {url}")
        try:
            txt = urlopen(url).read().decode("utf-8")
        except Exception as ex:
            log.error(ex)
            return False

        for sc in yaml.safe_load(txt):
            path = sc["name"] + ".py"
            label = "sc: " + sc["name"]
            self.listfiles[label] = {}
            self.listfiles[label]["url"] = sc["url"]
            self.listfiles[label]["path"] = path
            self.listfiles[label]["datasets"] = []
            if sc.get("datasets", ""):
                for ds in sc["datasets"]:
                    self.listfiles[label]["datasets"].append(ds)


class ParallelDownload:
    """Manages the process of downloading files"""

    def __init__(self, listfiles, outfolder, release, option, modetutorials, progress):
        self.listfiles = listfiles
        self.outfolder = outfolder
        self.release = release
        self.option = option
        self.modetutorials = modetutorials
        self.progress = progress
        self.bar = 0

    def run(self):
        try:
            from parfive import Downloader
        except ImportError:
            log.error("To use gammapy download, install the parfive package!")
            return

        if self.listfiles:
            log.info(f"Content will be downloaded in {self.outfolder}")

        dl = Downloader(progress=self.progress, file_progress=False)
        for rec in self.listfiles:
            url = self.listfiles[rec]["url"]
            path = self.outfolder / self.listfiles[rec]["path"]
            md5 = ""
            if "hashmd5" in self.listfiles[rec]:
                md5 = self.listfiles[rec]["hashmd5"]
            retrieve = True
            if md5 and path.exists():
                md5local = hashlib.md5(path.read_bytes()).hexdigest()
                if md5local == md5:
                    retrieve = False
            if retrieve:
                dl.enqueue_file(url, path=str(path.parent))

        log.info(f"{dl.queued_downloads} files to download.")
        res = dl.download()
        log.info(f"{len(res)} files downloaded.")
        for err in res.errors:
            _, _, exception = err
            log.error(f"Error: {exception}")

    def show_info_datasets(self):
        print("")
        GAMMAPY_DATA = Path.cwd() / self.outfolder
        if self.modetutorials:
            GAMMAPY_DATA = Path.cwd() / self.outfolder.parent / "datasets"
            if self.release:
                print(
                    "*** Enter the following commands below to get started with this version of Gammapy"
                )
                print(f"cd {self.outfolder.parent}")
                condaname = "gammapy-" + self.release
                envfilename = condaname + "-environment.yml"
                print(f"conda env create -f {envfilename}")
                print(f"conda activate {condaname}")
                print("jupyter lab")
                print("")
        print("*** You might want to declare GAMMAPY_DATA env variable")
        print(f"export GAMMAPY_DATA={GAMMAPY_DATA}")
        print("")
