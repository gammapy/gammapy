# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility script to work with the CITATION.cff file"""
import logging
import subprocess
from pathlib import Path
import click
from ruamel.yaml import YAML

log = logging.getLogger(__name__)

EXCLUDE_AUTHORS = ["azure-pipelines[bot]", "GitHub Actions"]

GAMMAPY_CC = [
    "Axel Donath",
    "Bruno Khelifi",
    "Catherine Boisson",
    "Christopher van Eldik",
    "David Berge",
    "Fabio Acero",
    "Fabio Pintore",
    "James Hinton",
    "José Luis Contreras Gonzalez",
    "Matthias Fuessling",
    "Régis Terrier",
    "Roberta Zanin",
    "Rubén López-Coto",
    "Stefan Funk",
]

# Approved authors that requested to be added to CITATION.cff
ADDITIONAL_AUTHORS = [
    "Amanda Weinstein",
    "Tim Unbehaun",
]

PATH = Path(__file__).parent.parent

LAST_LTS = "v1.0"
NOW = "HEAD"


@click.group()
def cli():
    pass


def get_git_shortlog_authors(since_last_lts=False):
    """Get list of authors from git shortlog"""
    authors = []
    command = ("git", "shortlog", "--summary", "--numbered")

    if since_last_lts:
        command += (f"{LAST_LTS}..{NOW}",)

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
@click.option("--since-last-lts", is_flag=True, help="Show authors since last LTS")
def check_author_lists(since_last_lts):
    """Check CITATION.cff with git shortlog"""
    authors = set(get_git_shortlog_authors(since_last_lts))
    authors_cff = set(get_citation_cff_authors())

    message = "****Authors not in CITATION.cff****\n\n  "
    diff = authors.difference(authors_cff)
    print(message + "\n  ".join(sorted(diff)) + "\n")

    message = "****Authors not in shortlog****\n\n  "
    diff = authors_cff.difference(authors)

    authors_annotated = []

    for author in sorted(diff):
        if author in ADDITIONAL_AUTHORS:
            author = "(AA) " + author
        elif author in GAMMAPY_CC:
            author = "(CC) " + author
        else:
            author = 5 * " " + author

        authors_annotated.append(author)

    print(message + "\n  ".join(authors_annotated) + "\n")

    print("(CC) = Coordination Committe, (AA) = Additional Authors, (OO) = Opted out\n")


if __name__ == "__main__":
    cli()
