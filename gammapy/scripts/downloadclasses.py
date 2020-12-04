# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Download class for gammapy download CLI."""
import logging
from pathlib import Path
from urllib.request import urlopen
import yaml
from gammapy import __version__

log = logging.getLogger(__name__)

RELEASES_BASE_URL = "https://gammapy.org/download"
DEV_NBS_YAML_URL = "https://raw.githubusercontent.com/gammapy/gammapy/master/notebooks.yaml"
TAR_BUNDLE = "https://github.com/gammapy/gammapy-data/tarball/master"  # curated datasets bundle


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
        outfolder,
        release,
        option,
        modetutorials=False
    ):
        self.outfolder = Path(outfolder)
        self.release = release
        self.option = option
        self.modetutorials = modetutorials
        self.listfiles = {}
        log.info(f"Looking for {self.option}...")

        if self.release == "" and "dev" not in __version__:
            self.release = __version__

    def getenvironment(self):
        try:
            from parfive import Downloader
        except ImportError:
            log.error("To use gammapy download, install the parfive package.")
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
        if self.option == "datasets" and self.modetutorials:
            return self.outfolder / "datasets"
        return self.outfolder

    def getfilelist(self):
        if self.option == "notebooks" or self.modetutorials:
            self.parse_notebooks_yaml()
            self.listfiles.update(dict(parse_imagefiles(self.listfiles)))

        if self.option == "datasets":
            self.listfiles = {"bundle": {"path": self.outfolder, "url": TAR_BUNDLE}}

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


class ParallelDownload:
    """Manages the process of downloading files"""

    def __init__(self, listfiles, outfolder, release, option, modetutorials):
        self.listfiles = listfiles
        self.outfolder = outfolder
        self.release = release
        self.option = option
        self.modetutorials = modetutorials
        self.bar = 0

    def run(self):
        try:
            from parfive import Downloader
        except ImportError:
            log.error("To use gammapy download, install the parfive package.")
            return

        if self.listfiles:
            log.info(f"Content will be downloaded in {self.outfolder}")

        dl = Downloader(progress=True, file_progress=False)
        for rec in self.listfiles:
            url = self.listfiles[rec]["url"]
            path = self.outfolder / self.listfiles[rec]["path"]
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
