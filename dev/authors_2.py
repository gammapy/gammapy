
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
import pickle
import xml.etree.ElementTree as ET
from datetime import datetime

log = logging.getLogger(__name__)

EXCLUDE_AUTHORS = [
    "azure-pipelines[bot]",
    "GitHub Actions"
]

PATH = Path(__file__).parent.parent
FILE_PATH = Path(os.path.dirname(__file__))
HEADER_CITATION = "CITATION_TEMPLATE.cff"


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

    return tags


def get_git_shortlog_authors(since=None):
    """Get list of authors from git shortlog"""
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

    return authors


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

    for author_data in data["authors"]:
        full_name = get_full_name(author_data)
        authors.append(full_name)

    return authors


def decompose_name(name):
    array = name.split(" ")
    tmp2 = array[-2]
    upper_case2 = bool(all(ll.isupper() for ll in tmp2))
    # Case of two components in the last name
    if "." not in tmp2 and upper_case2:
        last_name = array[-2]+" "+array[-1]
        first_name = array[0:-2]
    # Case of two components in the first name
    elif "." in tmp2:
        last_name = array[-1]
        first_name = array[0]+" "+array[1]
    elif len(array) > 2:
        last_name = array[-1]
        first_name = array[0] + " " + array[1]
    else:
        last_name = array[-1]
        first_name = array[0]

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

    orcid_url = 'https://pub.orcid.org/v3.0/search/?q=given-and-family-names:"'+author['name']+'"'
    ores = requests.get(orcid_url)
    if ores:
        return extract_orcid_from_xml(ores.text)
    else:
        print("fallback2")
        orcid_url = 'https://pub.orcid.org/v3.0/search/?q=email:"' + author['email'] + '"'
        ores = requests.get(orcid_url)
        if ores:
            return extract_orcid_from_xml(ores.text)

    return ""


def get_user_info(contributors):
    TOKEN = os.getenv('GITHUB_TOKEN')
    if TOKEN is None:
        print("No GitHub Token provided -> the user info will be incomplete")
        return {}
    headers = {'Authorization': f'token {TOKEN}'}
    github_url = "https://api.github.com/search/users"

    authors = []
    for contributor in contributors:
        authorInfo = contributor.split()
        email = authorInfo[-1].replace('<', '').replace('>', '')
        name = ' '.join(authorInfo[:-1])
        if name in EXCLUDE_AUTHORS:
            continue
        fname, lname = decompose_name(name)

        login = None
        affiliation = None
        location = None
        params = {"q": email}
        resp = requests.get(github_url, headers=headers, params=params)
        user_query = resp.json()
        if len(user_query) == 0 or user_query['total_count'] == 0:
            params = {"q": name}
            resp = requests.get(github_url, headers=headers, params=params)
            user_query = resp.json()

        if len(user_query) > 0 and user_query['total_count'] > 0:
            login = user_query['items'][0]['login']
            uresp = requests.get(user_query['items'][0]['url'], headers=headers)
            profile_query = uresp.json()
            affiliation = profile_query['company']
            location = profile_query['location']
        # else:
        #     print(f"Not found the GitHub account of [{name}]")

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
    clean_authors = list(map(pickle.loads, set(map(pickle.dumps, authors.copy()))))
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

@cli.command("sort", help="Sort authors by commits")
def sort_citation_cff():
    """Sort CITATION.cff according to the git shortlog"""
    authors = get_git_shortlog_authors()
    citation_file = PATH / "CITATION.cff"

    yaml = YAML()
    yaml.preserve_quotes = True

    with citation_file.open("r") as stream:
        data = yaml.load(stream)

    authors_cff = get_citation_cff_authors()

    sorted_authors = []

    for author in authors:
        idx = authors_cff.index(author)
        sorted_authors.append(data["authors"][idx])

    data["authors"] = sorted_authors

    with citation_file.open("w") as stream:
        yaml.dump(data, stream=stream)


@cli.command("check", help="Check git shortlog vs CITATION.cff authors")
def check_author_lists():
    """Check CITATION.cff with git shortlog"""
    authors = set(get_git_shortlog_authors())
    authors_cff = set(get_citation_cff_authors())

    message = "****Authors not in CITATION.cff****\n\n  "
    diff = authors.difference(authors_cff)
    print(message + "\n  ".join(sorted(diff)) + "\n")

    message = "****Authors not in shortlog****\n\n"
    diff = authors_cff.difference(authors)
    print(message + "\n  ".join(sorted(diff)) + "\n")


@cli.command("release", help="Get the list of releases")
def get_releases():
    tags = get_git_tag()
    print("List of releases:\n")
    print(tags)


def create_file(authors, release_name, from_release=None):
    if not os.path.exists(FILE_PATH / HEADER_CITATION):
        print("Header template of the CITATION file does NOT exist")
        print(FILE_PATH / HEADER_CITATION)
        exit(0)

    yaml = YAML()
    yaml.preserve_quotes = True

    if from_release is not None:
        citation_file = FILE_PATH / f"NEW_CITATION_{release_name}.cff"
    else:
        citation_file = FILE_PATH / "NEW_CITATION.cff"

    with open(FILE_PATH / HEADER_CITATION, "r") as file:
        data = yaml.load(file)

    data['version'] = f"{release_name.replace('v', '')}"
    data['date-released'] = datetime.date(datetime.now())
    data['authors'] = authors
    # preferred - citation:
    #     authors:
    #         - family-names: Druskat
    #         given-names: Stephan
    #     title: "Software paper about My Research Software"
    #     type: article

    with open(citation_file, 'w') as file:
        yaml.dump(data, file)


@cli.command("release-authors", help="Make the list of authors (optional: name of the release from which contributions are counted)")
@click.argument('release_name', required=True)
@click.argument('from_release', required=False)
def make_authors_list_cff(release_name, from_release=None):
    """Create a new NEW_CITATION.cff file according to the git shortlog
        e.g. python ../pyperso/scripts/authors_2.py release-authors 1.0 v0.20.1
        In order to check that it is OK, run cffconvert -i ${PATH}/NEW_CITATION_v{release_name}.cff --validate
    """
    tags = get_git_tag()
    if from_release not in tags:
        print(f"The required from_release=[{from_release}] is NOT known")
        exit(0)

    contributors = get_git_shortlog_authors(since=tags[from_release])
    authors_dict = get_user_info(contributors)
    clean_authors_dict = remove_duplication(authors_dict)
    sorted_authors_dict = sort_by_alphabetical_order(clean_authors_dict)
    cff_authors_dict = author_cff_formatting(sorted_authors_dict)

    create_file(cff_authors_dict, release_name=release_name, from_release=from_release)


if __name__ == "__main__":
    cli()
