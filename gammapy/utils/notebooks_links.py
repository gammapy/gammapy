# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Process tutorials notebooks for publication in documentation."""
import argparse
import logging
import re
import shutil
from configparser import ConfigParser
from pathlib import Path
from gammapy import __version__
from gammapy.utils.scripts import get_notebooks_paths

log = logging.getLogger(__name__)
PATH_CFG = Path(__file__).resolve().parent / ".." / ".."

# fetch params from setup.cfg
conf = ConfigParser()
conf.read(PATH_CFG / "setup.cfg")
setup_cfg = dict(conf.items("metadata"))
URL_DOCS = setup_cfg["url_docs"]
build_docs_cfg = dict(conf.items("build_docs"))
SOURCE_DIR = Path(build_docs_cfg["source-dir"])
PATH_NBS = SOURCE_DIR / build_docs_cfg["downloadable-notebooks"]
PATH_DOC = Path(build_docs_cfg["build-dir"]) / "html"

# release number in absolute links
release_number_docs = __version__
if "dev" in __version__:
    release_number_docs = "dev"


def make_api_links(file_path, file_type):
    """Build links to automodapi documentation."""

    re_api = re.compile(r'<span class="pre">~gammapy\.(.*?)</span>')
    if file_type == "ipynb":
        start_link = URL_DOCS
        re_api = re.compile(r"`~gammapy\.(.*?)`")
    else:
        path_tail = str(file_path).split(str(PATH_DOC))[1]
        level_depth = path_tail.count("/") - 1
        start_link = level_depth * "../"

    txt = file_path.read_text(encoding="utf-8")

    for module in re_api.findall(txt):

        # end urls
        alt_links = []
        submodules = module.split(".")
        if len(submodules) == 1:
            target = submodules[0]
            alt_links.append(f"{target}/index.html")
        elif len(submodules) == 2:
            target = f"{submodules[0]}.{submodules[1]}"
            alt_links.append(f"api/gammapy.{target}.html#gammapy.{target}")
            alt_links.append(f"{submodules[0]}/index.html#module-gammapy.{target}")
            alt_links.append(
                f"{submodules[0]}/{submodules[1]}index.html#module-gammapy.{target}"
            )
        elif len(submodules) == 3:
            target = f"{submodules[0]}.{submodules[1]}"
            alt_links.append(
                f"api/gammapy.{target}.html#gammapy.{target}.{submodules[2]}"
            )
            alt_links.append(
                f"api/gammapy.{target}.{submodules[2]}.html#gammapy.{target}.{submodules[2]}"
            )
        elif len(submodules) == 4:
            target = f"{submodules[0]}.{submodules[1]}.{submodules[2]}"
            alt_links.append(
                f"api/gammapy.{target}.html#gammapy.{target}.{submodules[3]}"
            )
        else:
            continue

        # broken link
        broken = True
        for link in alt_links:
            search_file = re.sub(r"(#.*)$", "", link)
            search_path = PATH_DOC / search_file
            if search_path.is_file():
                link_api = f"{start_link}{link}"
                link_api = link_api.replace("()", "")
                broken = False
                break
        if broken:
            if file_type == "ipynb":
                log.warning(f"{str(search_path)} does not exist in {file_path}.")
            continue

        # replace syntax with link
        str_api = f'<span class="pre">~gammapy.{module}</span>'
        label_api = str_api.replace('<span class="pre">', "")
        label_api = label_api.replace("</span>", "")
        label_api = label_api.replace("~", "")
        replace_api = f"<a href='{link_api}'>{label_api}</a>"
        if file_type == "ipynb":
            str_api = f"`~gammapy.{module}`"
            label_api = str_api.replace("`", "")
            label_api = label_api.replace("~", "")
            replace_api = f"[[{label_api}]({link_api})]"
        txt = txt.replace(str_api, replace_api)

    # modif absolute links to rst/html doc files
    if file_type == "ipynb":
        url_docs_release = URL_DOCS.replace("dev", release_number_docs)
        txt = txt.replace(URL_DOCS, url_docs_release)
    else:
        repl = r"..\/\1html\2"
        txt = re.sub(
            pattern=URL_DOCS + r"(.*?)html(\)|#)",
            repl=repl,
            string=txt,
            flags=re.M | re.I,
        )

    file_path.write_text(txt, encoding="utf-8")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="Tutorial notebook to process")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    log.info("Building API links in .ipynb and Sphinx formatted notebooks.")
    log.info("Bring back stripped and clean notebooks.")

    for nb_path in get_notebooks_paths():
        if args.src and Path(args.src).resolve() != nb_path:
            continue
        downloadable_path = PATH_NBS / nb_path.absolute().name
        shutil.copyfile(downloadable_path, nb_path)
        make_api_links(downloadable_path, file_type="ipynb")
        html_path = str(nb_path).replace(f"/{SOURCE_DIR}", f"/{PATH_DOC}")
        html_path = html_path.replace("ipynb", "html")
        make_api_links(Path(html_path), file_type="html")


if __name__ == "__main__":
    main()
