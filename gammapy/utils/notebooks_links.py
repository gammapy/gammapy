# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Process tutorials notebooks for publication in documentation."""
import logging
import re
from configparser import ConfigParser
from pathlib import Path
from gammapy import __version__
from gammapy.utils.notebooks_test import get_notebooks

log = logging.getLogger(__name__)

PATH_NBS = Path("docs/_static/notebooks")
PATH_DOC = Path("docs/_build/html/")
PATH_CFG = Path(__file__).resolve().parent / ".." / ".."
URL_GAMMAPY_MASTER = "https://raw.githubusercontent.com/gammapy/gammapy/master/"

# fetch url_docs from setup.cfg
conf = ConfigParser()
conf.read(PATH_CFG / "setup.cfg")
setup_cfg = dict(conf.items("metadata"))
url_docs = setup_cfg["url_docs"]

# release number in absolute links
release_number_docs = __version__
if "dev" in __version__:
    release_number_docs = "dev"


def make_api_links(file_path, file_type):
    """Build links to automodapi documentation."""

    start_link = "../"
    re_api = re.compile(r'<span class="pre">~gammapy\.(.*?)</span>')
    if file_type == "ipynb":
        start_link = url_docs
        re_api = re.compile(r"`~gammapy\.(.*?)`")
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
        url_docs_release = url_docs.replace("dev", release_number_docs)
        txt = txt.replace(url_docs, url_docs_release)
    else:
        repl = r"..\/\1html\2"
        txt = re.sub(
            pattern=url_docs + r"(.*?)html(\)|#)",
            repl=repl,
            string=txt,
            flags=re.M | re.I,
        )

    file_path.write_text(txt, encoding="utf-8")


def main():
    logging.basicConfig(level=logging.INFO)
    log.info("Building API links in notebooks.")

    for notebook in get_notebooks():
        html_path = notebook["url"].replace(URL_GAMMAPY_MASTER, "")
        html_path = html_path.replace("ipynb", "html")
        make_api_links(Path(html_path), file_type="html")

    for nb_path in list(PATH_NBS.glob("*.ipynb")):
        make_api_links(Path(nb_path), file_type="ipynb")


if __name__ == "__main__":
    main()
