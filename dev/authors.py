"""Utility script to work with the CITATION.cff file"""
import logging
import subprocess
from pathlib import Path
import click
from ruamel.yaml import YAML

log = logging.getLogger(__name__)

EXCLUDE_AUTHORS = ["azure-pipelines[bot]", "GitHub Actions"]

PATH = Path(__file__).parent.parent


@click.group()
def cli():
    pass


def get_git_shortlog_authors():
    """Get list of authors from git shortlog"""
    authors = []
    command = ("git", "shortlog", "--summary", "--numbered")
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


if __name__ == "__main__":
    cli()
