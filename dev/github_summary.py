# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import click
from github import Github, GithubException
from gammapy.utils.table import table_from_row_data

log = logging.getLogger(__name__)


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING"]),
)
def cli(log_level):
    logging.basicConfig(level=log_level)
    log.setLevel(level=log_level)


def login(token):
    if token:
        g = Github(token, per_page=100)
    else:
        g = Github()

    try:
        user_login = g.get_user().login
    except GithubException:
        user_login = "anonymous"

    log.info(f"Logging in GitHub as {user_login}")

    return g


def check_requests_number(g):
    remaining, total = g.rate_limiting
    log.info(f"Remaining {remaining} requests over {total} requests.")


@cli.command("dump_table", help="Dump a table of all PRs.")
@click.option("--token", default=None, type=str)
@click.option("--number_min", default=4000, type=int)
def dump_table(token=None, number_min=4000):
    g = login(token)
    repo = g.get_repo("gammapy/gammapy")

    pull_requests = repo.get_pulls(state="closed", sort="created", direction="desc")

    check_requests_number(g)

    results = []

    for pr in pull_requests:
        if pr.number <= number_min:
            break

        result = dict()
        result["number"] = pr.number
        result["title"] = pr.title
        result["milestone"] = "" if not pr.milestone else pr.milestone.title
        result["is_merged"] = pr.is_merged()
        result["user_name"] = pr.user.name
        result["user_login"] = pr.user.login
        result["user_email"] = pr.user.email
        results.append(result)

    table = table_from_row_data(results)
    table.write("test.ecsv", overwrite=True)

    check_requests_number(g)


@cli.command("merged_PR", help="Make a summary of PRs merged with a given milestone")
@click.option("--token", default=None, type=str)
@click.option("--number_min", default=4000, type=int)
@click.argument("milestone", type=str, default="1.0")
def list_merged_PRs(milestone, token=None, number_min=4000):
    g = login(token)
    repo = g.get_repo("gammapy/gammapy")

    pull_requests = repo.get_pulls(state="closed", sort="created", direction="desc")

    check_requests_number(g)

    total_number = 0
    names = set()

    for pr in pull_requests:
        if pr.number < number_min:
            break
        if pr.milestone and pr.milestone.title == milestone:
            if pr.is_merged():
                if pr.user.name:
                    name = pr.user.name
                else:
                    name = pr.user.login
                total_number += 1
                names.add(name)
                print(f"- [#{pr.number}] {pr.title} ({name})")

    print("--------------")
    print("Contributors:")
    for name in names:
        print(f"- {name}")
    log.info(f"Found {total_number} of merged pull requests for milestone {milestone}.")
    log.info(f"Found {len(names)} contributors for milestone {milestone}.")

    check_requests_number(g)


@cli.command(
    "closed_issues", help="Make a summary of closed issues with a given milestone"
)
@click.option("--token", default=None, type=str)
@click.option("--print_issue", default=False, type=bool)
@click.option("--number_min", default=4000, type=int)
@click.argument("milestone", type=str, default="1.0")
def list_closed_issues(milestone, token=None, print_issue=False, number_min=4000):
    g = login(token)
    repo = g.get_repo("gammapy/gammapy")

    issues = repo.get_issues(state="closed", sort="closed", direction="desc")

    check_requests_number(g)

    total_number = 0

    for issue in issues:
        if issue.number < number_min:
            break
        if issue.milestone and issue.milestone.title == milestone:
            total_number += 1
            if print_issue:
                print(f"- [#{issue.number}] {issue.title}")

    log.info(f"Found {total_number} closed issues with milestone {milestone}.")

    check_requests_number(g)


if __name__ == "__main__":
    cli()
