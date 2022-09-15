
"""Utility script to work with the CITATION.cff file
   If you have defined your GitHub token as global variable, duplications are removed and affiliations are provided
          e.g. export GITHUB_TOKEN=xxxxx
"""
import logging
import subprocess
import click
import os
from ruamel.yaml import YAML
from pathlib import Path
import requests
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import OrderedDict
import time
from gammapy.utils import pbar

pbar.SHOW_PROGRESS_BAR = True
log = logging.getLogger(__name__)

EXCLUDE_AUTHORS = [
    "azure-pipelines[bot]",
    "GitHub Actions"
]

PATH = Path(__file__).parent.parent
TEMPLATE_PATH = Path(__file__).parent
HEADER_CITATION = "CITATION_HEADER.cff"
FOOTER_CODEMETA = "FOOTER_CODEMETA.json"
CC_MEMBERS = "CC_MEMBERS.cff"
LD_MEMBERS = "LEAD_DEVELOPERS.cff"

affiliation_map = {"None": 0}
is_author = []

@click.group()
def cli():
    pass


def get_git_tag():
    """Get the list of releases from git"""
    tags = {}
    command = ('git for-each-ref --format="%(refname:short) | %(creatordate:short)" "refs/tags/*"')
    # command = ("git", "tag", "-l")
    result = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True).decode()
    data = result.split("\n")
    for row in data:
        parts = row.split(" | ")

        if len(parts) == 2:
            name, date = parts
            tags[name] = date

    sort_dict = {item[0] : item[1] for item in sorted(tags.items(), key=lambda val: val[1])}
    return sort_dict


def get_git_shortlog_authors(since=None):
    """Get dict of authors from git shortlog, formatted into the cff format"""
    authors = []
    # command = ["git", "shortlog", "--summary", "--numbered", "--email", "--merges"]
    command = ["git", "shortlog", "--summary", "--numbered", "--email"]
    if since is not None:
        command.extend([f"--since='{since}'"])
    result = subprocess.check_output(command, stderr=subprocess.STDOUT).decode()
    data = result.split("\n")
    for row in data:
        parts = row.split("\t")

        if len(parts) == 2:
            n_commits, author = parts
            if author not in EXCLUDE_AUTHORS:
                authors.append(author)

    authors_dict = get_user_info(authors)
    clean_authors_dict = remove_duplication(authors_dict)
    sorted_authors_dict = sort_by_alphabetical_order(clean_authors_dict)
    cff_authors_dict = author_cff_formatting(sorted_authors_dict)

    return cff_authors_dict


def get_full_name(author_data):
    """Get full name from CITATION.cff parts"""
    parts = []
    parts.append(author_data["given-names"])

    name_particle = author_data.get("name-particle", None)

    if name_particle:
        parts.append(name_particle)

    parts.append(author_data["family-names"])
    return " ".join(parts)


def get_citation_cff_authors():
    """Get list of authors from CITATION.cff"""
    authors = []
    citation_file = PATH / "CITATION.cff"

    yaml = YAML()

    with citation_file.open("r") as stream:
        data = yaml.load(stream)
    return data['authors']

    # for author_data in data["authors"]:
    #     full_name = get_full_name(author_data)
    #     authors.append(full_name)
    #
    # return authors


def decompose_name(name):
    array = name.split(" ")
    if len(array) == 1:
        print(f"Strange name [{name}]")
        return "", array[0]
    elif len(array) == 2:
        last_name = array[-1]
        first_name = array[0]
    else:
        tmp2 = array[-2]
        upper_case2 = bool(all(ll.isupper() for ll in tmp2))
        # Case of two components in the last name
        if "." not in tmp2 and upper_case2:
            last_name = array[-2] + " " + array[-1]
            first_name = array[0:-2]
        # Case of two components in the first name
        elif "." in tmp2:
            last_name = array[-1]
            first_name = array[0] + " " + array[1]
        elif len(array) > 2:
            last_name = array[-1]
            first_name = array[0] + " " + array[1]

    return first_name, last_name


def extract_orcid_from_xml(xml_text):
    doc = ET.fromstring(xml_text)
    try:
        return doc[0][0][1].text
    except:
        return ""


def get_orcid(author):
    # payload = {'client_id': 'APP-XSVZ8GMQCBFT1XCZ',
    #            'client_secret': '4ca79d0c-d971-4ab6-a9fe-3c48dce3eebf',
    #            'scope': '/read-public',
    #            'grant_type': 'client_credentials'
    #            }
    # url = "https://pub.orcid.org/oauth/token"
    # headers = {'Accept': 'application/json'}
    # response = requests.post(url, data=payload, headers=headers)
    # # response.raise_for_status()
    # new_token = response.json()['access_token']
    # refresh_token = response.json()['refresh_token']
    # # print(response.json())
    # print(f'Token={new_token}')
    #
    # url2 = \
    #     "https://orcid.org/oauth/authorize?client_id=APP-XSVZ8GMQCBFT1XCZ&response_type=token&scope=openid&redirect_uri=https://pub.orcid.org"
    # response = requests.post(url)
    # print(response.json())

    # orcid_url = "https://pub.sandbox.orcid.org/v3.0/search/?defType=lucene&q=email:khelifi@in2p3.fr"
    # orcid_url = "https://api.orcid.org/v3.0/csv-search/?q=email:khelifi@in2p3.fr"
    # orcid_url = "https://pub.orcid.org/v3.0/csv-search/?q=email:khelifi@in2p3.fr"
    # orcid_url = "https://api.sandbox.orcid.org/v3.0/csv-search/?q=email:khelifi@in2p3.fr"
    # orcid_url = "https://pub.sandbox.orcid.org/v3.0/csv-search/?q=email:khelifi@in2p3.fr"
    # One of 'lucene', 'edismax', 'dismax'
    # headers = {'Accept': 'application/orcid+json',
    #            'Authorization': f'Bearer {refresh_token}'}

    # orcid_url = 'https://pub.sandbox.orcid.org/v3.0/search/?q=family-name:Khélifi+AND+given-names:Bruno'
    # orcid_url = 'https://pub.sandbox.orcid.org/v3.0/search/?q=email:khelifi@in2p3.fr'
    # orcid_url = 'https://pub.sandbox.orcid.org/v3.0/search/?q=family-name:Khelifi'
    orcid_url = 'https://pub.orcid.org/v3.0/search/?q=given-and-family-names:"Bruno Khelifi"'

    # orcid_url = "https://pub.orcid.org/v2.0/search/?defType=lucene&q=email:khelifi@in2p3.fr"
    # # One of 'lucene', 'edismax', 'dismax'
    # headers = {'Accept': 'application/orcid+json',
    #            'Authorization': f'Bearer {new_token}'}

    # params = {"q=email:": "khelifi@in2p3.fr"}
    # q=johnson+AND+cardiology+AND+houston ,email,given-names,family-name,given-and-family-names
    # ores = requests.get(orcid_url, headers=headers, params=params)
    #ores = requests.get(orcid_url, headers=headers)
    # orcid_url = 'https://orcid.org/orcid-search/search?searchQuery="Axel Donath"'
    # ores = requests.get(orcid_url)
    # orcid = extract_orcid_from_xml(ores.text)
    # print(orcid)

    id = ""
    orcid_url = 'https://pub.orcid.org/v3.0/search/?q=given-and-family-names:"' + author['name'] + '"'
    ores = requests.get(orcid_url, timeout=5)
    if ores and len(ores.text) > 200:
        if len(ores.text) > 500:
            # print("Too many returned answers")
            log.warning(f"Too many returned answers to get a valid ORCID for [{author['name']}]")
        else:
            id = extract_orcid_from_xml(ores.text)
        test_url = 'https://pub.orcid.org/' + id
        tres = requests.get(test_url, timeout=5)
        # print("tres:", tres.text)
    # else:
    #     print("fallback")
    #     orcid_url = 'https://pub.orcid.org/v3.0/search/?q=email:"' + author['email'] + '"'
    #     print(orcid_url)
    #     ores = requests.get(orcid_url, timeout=5)
    #     print(ores.text)
    #     print(len(ores.text))
    #     if ores:
    #         print(len(ores.text))
    #         id = extract_orcid_from_xml(ores.text)

    return id


def get_user_info(contributors):
    TOKEN = os.getenv('GITHUB_TOKEN')
    if TOKEN is None:
        print("No GitHub Token provided -> the user info will be incomplete")
        return []
    headers = {'Authorization': f'token {TOKEN}'}
    github_url = "https://api.github.com/search/users"

    authors = []
    # for contributor in contributors:
    for contributor in pbar.progress_bar(contributors, desc="Contributors"):
        authorInfo = contributor.split()
        email = authorInfo[-1].replace('<', '').replace('>', '')
        name = ' '.join(authorInfo[:-1])
        if name in EXCLUDE_AUTHORS:
            continue
        fname, lname = decompose_name(name)
        # print(f"Name: {name}")

        login = None
        affiliation = None
        location = None
        params = {"q": email}
        resp = requests.get(github_url, headers=headers, params=params)
        user_query = resp.json()
        if len(user_query) == 0 or 'total_count' not in user_query or user_query['total_count'] == 0:
            params = {"q": name}
            resp = requests.get(github_url, headers=headers, params=params)
            user_query = resp.json()

        if len(user_query) > 0 and 'total_count' in user_query and user_query['total_count'] > 0:
            login = user_query['items'][0]['login']
            uresp = requests.get(user_query['items'][0]['url'], headers=headers)
            profile_query = uresp.json()
            affiliation = profile_query['company']
            location = profile_query['location']
        if login is None:
            log.info(f"Not found the GitHub account of [{name}]")

        author = {
            "family-names": lname,
            "given-names": fname,
            "login": login,
            "name": name,
            "email": email,
            "affiliation": f"{affiliation}",
            "location": f"{location}",
        }
        orcid = get_orcid(author)
        if orcid is not None and len(orcid) > 0:
            author['orcid'] = f"https://orcid.org/{orcid}"
        authors.append(author)

    return authors


def remove_duplication(authors):
    clean_authors = []
    for author in authors:
        if not any(d['family-names'] == author['family-names'] and d['given-names'] == author['given-names'] for d in clean_authors):
            if 'bsipocz@gmail.com' in author['email'] and 'Brigitta' in author['given-names'] and 'None' in author['affiliation']:
                continue
            if "Remy" in author['family-names'] and 'Quentin' in author['given-names']:
                author['affiliation'] = "MPIK"
                author['city'] = "Heidelberg, DE"
            clean_authors.append(author)

    return clean_authors


def sort_by_alphabetical_order(authors):
    sorted_authors = []
    sorted_authors = sorted(authors.copy(), key=lambda d: d['family-names'])
    return sorted_authors


def author_cff_formatting(authors):
    """See https://github.com/citation-file-format/citation-file-format/blob/main/schema-guide.md"""
    cff_authors = []
    for author in authors.copy():
        if 'name' in author:
            del author['name']
        if 'login' in author:
            del author['login']
        if 'location' in author:
            if 'None' not in author['location']:
                author['city'] = author['location']
            del author['location']
        if 'affiliation' in author and 'None' in author['affiliation']:
            del author['affiliation']
        if 'given-names' in author and 'None' in author['given-names']:
            del author['given-names']
        cff_authors.append(author)
    return cff_authors


def check_cff_format(filename):
    command = f"cffconvert -i {filename} --validate"
    result = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True).decode()
    if 'are valid' not in result:
        return False
    return True


def update_codemeta(filename, cff_file):
    """We follow the format defined here: https://codemeta.github.io/"""
    with open(TEMPLATE_PATH / FOOTER_CODEMETA, "r") as f:
        footer_data = json.load(f)
    with open(filename, "r") as f:
        cff_data = json.load(f)
    cff_data.update(footer_data)

    yaml = YAML()
    with open(cff_file, "r") as file:
        data = yaml.load(file)
    cff_data['referencePublication'] = dict(data['preferred-citation'])
    cff_data['dateModified'] = f"{data['date-released']}"
    cff_data['email'] = data['contact'][0]['email']
    cff_data['readme'] = data['message']

    runtime_platform = []
    with open(PATH / "setup.cfg", "r") as f:
        for line in f:
            if "Programming Language" in line:
                lan = line.rstrip('\n').replace("    Programming Language :: ", "").replace(" :: ", " ")
                if len(lan) > 0:
                    runtime_platform.append(lan)
    cff_data["runtimePlatform"] = runtime_platform

    yaml = YAML()
    with open(PATH / "environment-dev.yml", 'r') as f:
        read_data = yaml.load(f)
    cff_data["softwareRequirements"] = read_data['dependencies']

    data = OrderedDict(cff_data)
    data.move_to_end("author")

    with open(filename, "w") as f:
        json.dump(dict(data), f, indent=4)


def make_author_latex(author):
    import codecs
    fullname = get_full_name(author)
    if fullname in is_author:
        return "", ""
    is_author.append(fullname)

    af = ""
    if any('affiliation' in item for item in author):
        af = author['affiliation'].replace("&", "\&")
    if any('city' in item for item in author):
        af += ", " + author['city'].replace("&", "\&")
    iaf = -1
    saf = ""
    if len(af) > 0:
        af.replace("&", "\&")
        if af in affiliation_map.keys():
            iaf = affiliation_map[af]
        else:
            iaf = len(affiliation_map)
            affiliation_map[af] = iaf
            saf = [r'\affil[', str(iaf), r']{', af, r'}', '\n'] #Build affiliation only if not exists

    st = [r'\author[']
    if iaf > 0:
        st += str(iaf)
    st += " "
    if 'orcid' in author:
        st += r'\orcidaffil{' + author['orcid'].rsplit('/', 1)[1] + r'}'
    st += r']{' + author['family-names'] + r', ' + author['given-names'] + '}\n'

    return st, saf


def make_authors_paper(authors, output_file):
    """Make the latex file of authors list according to our Authorship policy"""

    # Header
    hd = [r'\usepackage{authblk}', '\n', r'\usepackage{graphicx}', '\n', r'\usepackage{hyperref}',
          '\n', r'\usepackage[symbol]{footmisc}', '\n', r'\usepackage[utf8, latin1]{inputenc}',
          '\n', r'\usepackage[T1]{fontenc}', '\n\n',
          r'\newbox{\myorcidaffilbox}', '\n', r'\sbox{\myorcidaffilbox}{\large\includegraphics[height=1.7ex]{orcid}}',
          '\n', r'\newcommand{\orcidaffil}[1]{%', '\n', r'  \href{https://orcid.org/#1}{\usebox{\myorcidaffilbox}}}',
          '\n', r'\def\correspondingauthor{\footnote{Corresponding author: Gammapy Coordination Committee, ',
          r'\href{mailto: GAMMAPY-COORDINATION-L@IN2P3.FR}{\nolinkurl{GAMMAPY-COORDINATION-L@IN2P3.FR}}}}'
           '\n\n', r'\author[ ]{The Gammapy~Team\correspondingauthor{}}', '\n\n']

    # Footer
    ft = ['\n\n', r'{\let\thefootnote\relax\footnote{{',
          r'The author list has three parts: the authors that made significant contributions to the writing',
          r' of the paper, the members of the Gammapy Project Coordination Committee, and contributors',
          r' to the Gammapy Project in alphabetical order. The position in the author list does not',
          r' correspond to contributions to the Gammapy Project as a whole. A more complete list of',
          r' contributors to the core package can be found in the package repository and at the Gammapy',
          r' team webpage.', r'}}}', '\n']

    yaml = YAML()
    with open(TEMPLATE_PATH / CC_MEMBERS, 'r') as f:
        data = yaml.load(f)
    cc_members = data['cc-members']
    with open(TEMPLATE_PATH / LD_MEMBERS, 'r') as f:
        data = yaml.load(f)
    ld_members = data['lead-dev']

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("".join(hd))

        ## Primary authors
        for member in ld_members:
            st, af = make_author_latex(member)
            f.write("".join(st))
            if len(af) > 0:
                f.write("".join(af))
        f.write(r"\collaboration{113}{(Primary Paper Contributors)}"+"\n\n")

        ## The CC
        for member in cc_members:
            st, af = make_author_latex(member)
            f.write("".join(st))
            if len(af) > 0:
                f.write("".join(af))
        f.write(r"\collaboration{113}{(Gammapy Coordination Committee)}"+"\n\n")

        ## The contributors
        for author in authors:
            st, af = make_author_latex(author)
            f.write("".join(st))
            if len(af) > 0:
                f.write("".join(af))
        f.write(r"\collaboration{113}{(Gammapy Project Contributors)}"+"\n\n")

        f.write('\n'.join(ft))


def get_latest_feature_release(tags):
    idx = -1
    for tag in reversed(list(tags.keys())):
        if tag.count('.') == 1:
            return tag
    return tag.keys()[-1]


def create_cff_file(authors, release_name, from_release=None):
    if not os.path.exists(TEMPLATE_PATH / HEADER_CITATION):
        print(f"\nHeader template of the CITATION file does NOT exist [{TEMPLATE_PATH / HEADER_CITATION}]! exit")
        log.fatal(f"Header template of the CITATION file does NOT exist [{TEMPLATE_PATH / HEADER_CITATION}]! exit")
        exit(0)

    yaml = YAML()
    yaml.preserve_quotes = True

    if from_release is not None:
        citation_file = PATH / f"NEW_CITATATION_{release_name}.cff"
    else:
        citation_file = PATH / "NEW_CITATION.cff"

    with open(TEMPLATE_PATH / HEADER_CITATION, "r") as file:
        data = yaml.load(file)

    data['version'] = f"{release_name.replace('v', '')}"
    data['date-released'] = datetime.date(datetime.now())
    data['authors'] = authors

    with open(citation_file, 'w') as file:
        yaml.dump(data, file)
    print(f"\nOutput file: [{citation_file}]\n\n")
    log.info(f"Output file: [{citation_file}]\n\n")

    if not check_cff_format(citation_file):
        log.fatal(f"The file [{citation_file}] is not compliant with the cff format! ")
        print(f"\nThe file [{citation_file}] is not compliant with the cff format! ")
        exit(0)

    return citation_file


@cli.command("update", help="Update the authors list from commits. Need `export GITHUB_TOKEN=xxxxx`.")
@click.option('--from_release', default=None,  help='The shortlog is made since this release name')
def update(from_release=None):
    """Sort CITATION.cff according to the git shortlog"""
    tags = get_git_tag()
    if from_release is None:
        latest_feature_release = get_latest_feature_release(tags)
        print("toto ", latest_feature_release)
        exit(0)
        since = tags[latest_feature_release]
        #list(tags.values())[-1]  # ToDo: use the latest release version (and not the last version)
    else:
        since = tags[from_release]
    authors = get_git_shortlog_authors(since)

    authors_cff = get_citation_cff_authors()

    for author in authors:
        if not any(d['family-names'] == author['family-names'] and d['given-names'] == author['given-names'] for d in authors_cff):
            authors_cff.append(author)

    citation_file = PATH / "CITATION.cff"
    yaml = YAML()
    yaml.preserve_quotes = True
    with citation_file.open("r") as stream:
        data = yaml.load(stream)
    data["authors"] = sort_by_alphabetical_order(authors_cff)
    with citation_file.open("w") as stream:
        yaml.dump(data, stream=stream)


@cli.command("check", help="Check git shortlog vs CITATION.cff authors. Need `export GITHUB_TOKEN=xxxxx`.")
@click.option('--from_release', default=None,  help='The shortlog is made since this release name')
def check_author_lists(from_release=None):
    """Check CITATION.cff with git shortlog"""

    # First check the cff format
    if not check_cff_format(PATH / "CITATION.cff"):
        log.fatal(f"The file [{PATH / 'CITATION.cff'}] is not compliant with the cff format! ")
        print(f"The file [{PATH / 'CITATION.cff'}] is not compliant with the cff format! ")
        exit(0)

    # Check if the file is up-to-date
    tags = get_git_tag()
    if from_release is None:
        latest_feature_release = get_latest_feature_release(tags)
        print("toto2 ", latest_feature_release)
        exit(0)
        since = tags[latest_feature_release]
        # since = list(tags.values())[-1]  # ToDo: use the latest release version (and not the last version)
    else:
        since = tags[from_release]

    authors = get_git_shortlog_authors(since)
    authors_cff = get_citation_cff_authors()

    message = "****Authors not in CITATION.cff****\n\n  "
    diff = []
    for author in authors:
        if not any(d['family-names'] == author['family-names'] and d['given-names'] == author['given-names'] for d in authors_cff):
            diff.append(get_full_name(author))
    print(message + "\n  ".join(sorted(diff)) + "\n")

    message = "****Authors not in shortlog****\n\n"
    diff = []
    for author in authors_cff:
        if not any(d['family-names'] == author['family-names'] and d['given-names'] == author['given-names'] for d in authors):
            diff.append(get_full_name(author))
    print(message + "\n  ".join(sorted(diff)) + "\n")


@cli.command("release", help="Get the list of releases. Need `export GITHUB_TOKEN=xxxxx`.")
def get_releases():
    """Get the list of releases"""
    tags = get_git_tag()
    print("\nList of releases:\n")
    print(tags)


@cli.command("make_cff", help="Creation of a new CITATION.cff file. Need `export GITHUB_TOKEN=xxxxx`.")
@click.argument('release_name', required=True)
@click.option('--from_release', default=None,  help='The shortlog is made since this release name')
def make_cff(release_name, from_release=None):
    """Create a new NEW_CITATION.cff file according to the git shortlog
    release_name: Name of the new release
    """
    tags = get_git_tag()
    if from_release not in tags:
        print(f"\nThe required from_release=[{from_release}] is NOT known")
        log.fatal(f"The required from_release=[{from_release}] is NOT known! -> exit")
        exit(0)

    cff_authors_dict = get_git_shortlog_authors(since=tags[from_release])

    _ = create_cff_file(cff_authors_dict, release_name=release_name, from_release=from_release)


@cli.command("make_codemeta", help="Creation of a new CITATION.cff file. Need `export GITHUB_TOKEN=xxxxx`.")
@click.argument('cff_file', default=PATH / 'CITATION.cff', required=False)
@click.option('--output_file', default=PATH / 'new_codemeta.json',  help='Full name (path/name.json) of the output file')
def make_codemeta(cff_file, output_file=None):
    log.info(f"Reading [{cff_file}]")
    print(f"Reading [{cff_file}]")
    command = f"cffconvert -i {cff_file} -f codemeta -o {output_file}"
    result = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True).decode()
    if len(result) > 0:
        log.fatal(f"\nFailed to make the codemeta file: [{result}]")
        exit(0)

    update_codemeta(output_file, cff_file)

    print(f"Output file: [{output_file}]\n")
    log.info(f"Output file: [{output_file}]\n")


@cli.command("paper_authors", help="Create the list of authors for papers. Need `export GITHUB_TOKEN=xxxxx`.")
@click.argument('lts_name', required=True)
@click.option('--from_release', default=None,  help='The shortlog is made since this release name')
def paper_authors(lts_name, from_release=None):
    """Create the list of authors for papers associated to a new LTS.
    Need `export GITHUB_TOKEN=xxxxx`.
    Output:
      - Bibtex file name
      - Authors list in HTML format (for the web site)
      - List of emails of authors
    """
    tags = get_git_tag()
    if from_release is None:
        latest_feature_release = get_latest_feature_release(tags)
        print("toto3 ", latest_feature_release)
        exit(0)
        since = tags[latest_feature_release]
        # since = list(tags.values())[-1]  # ToDo: use the latest release version (and not the last version)
    else:
        if from_release not in tags:
            print(f"\nThe required from_release=[{from_release}] is NOT known")
            log.fatal(f"The required from_release=[{from_release}] is NOT known! -> exit")
            exit(0)
        since = tags[from_release]

    authors = get_git_shortlog_authors(since)

    output_file = PATH / f"authors_{lts_name}.tex"
    make_authors_paper(authors, output_file)

    print(f"\nBibTex file: [{output_file}]\n")
    log.info(f"BibTex file: [{output_file}]")

# ToDo: 2 new functions
#  - fct email address
#  - list for papers
#add timer


if __name__ == "__main__":
    cli()
