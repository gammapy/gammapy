# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Process tutorials notebooks for publication in documentation."""
import logging
import re
from configparser import ConfigParser
from pathlib import Path
from gammapy import __version__

log = logging.getLogger(__name__)

PATH_NBS = Path("docs/_static/notebooks")
PATH_DOC = Path("docs/_build/html/")
PATH_HTM = PATH_DOC / "notebooks"
PATH_CFG = Path(__file__).resolve().parent / ".." / ".."

# fetch url_docs from setup.cfg
conf = ConfigParser()
conf.read(PATH_CFG / "setup.cfg")
setup_cfg = dict(conf.items("metadata"))
url_docs = setup_cfg["url_docs"]

# release number in absolute links
release_number_docs = __version__
if "dev" in __version__:
    release_number_docs = "dev"


def make_api_links(file_path, start_link):
    """Build links to automodapi documentation."""

    re_api = re.compile(r'<span class="pre">~gammapy\.(.*?)</span>')
    if start_link == url_docs:
        re_api = re.compile(r"`~gammapy\.(.*?)`")
    txt = file_path.read_text(encoding="utf-8")

    for module in re_api.findall(txt):

        # build target file module
        submodules = module.split(".")
        if len(submodules) == 1:
            file_module = f"{module}/index.html"
        elif len(submodules) == 2:
            file_module = f"api/gammapy.{module}.html#module-gammapy.{module}"
        elif len(submodules) == 3:
            submodules[2] = submodules[2].replace("()", "")
            url_path = f"gammapy.{submodules[0]}.{submodules[1]}"
            anchor_path = f"{url_path}.{submodules[2]}"
            file_module = f"api/{url_path}.html#{anchor_path}"
        else:
            continue

        # check broken link
        search_file = re.sub(r"(#.*)$", "", file_module)
        search_path = PATH_DOC / search_file
        if not search_path.is_file():
            if start_link == url_docs:
                log.warning(f"{str(search_path)} does not exist in {file_path}.")
            continue

        # replace with link
        link_api = f"{start_link}{file_module}"
        str_api = f'<span class="pre">~gammapy.{module}</span>'
        label_api = str_api.replace('<span class="pre">', "")
        label_api = label_api.replace("</span>", "")
        label_api = label_api.replace("~", "")
        replace_api = f"<a href='{link_api}'>{label_api}</a>"
        if start_link == url_docs:
            str_api = f"`~gammapy.{module}`"
            label_api = str_api.replace("`", "")
            label_api = label_api.replace("~", "")
            replace_api = f"[[{label_api}]({link_api})]"
        txt = txt.replace(str_api, replace_api)

        # modif absolute links to rst/html doc files
        if start_link == url_docs:
            url_docs_release = url_docs.replace("dev", release_number_docs)
            txt = txt.replace(url_docs, url_docs_release)
        else:
            repl = r"..\/\1rst\2"
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
    for nb_path in list(PATH_NBS.glob("*.ipynb")):
        make_api_links(nb_path, url_docs)
    for html_path in list(PATH_HTM.glob("*.html")):
        make_api_links(html_path, "../")


if __name__ == "__main__":
    main()
