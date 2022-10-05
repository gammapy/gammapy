# Licensed under a 3-clause BSD style license - see LICENSE.rst
import click
from github import Github, GithubException
import logging

log = logging.getLogger(__name__)


@click.group
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
        g = Github(token, per_page=200)
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


@cli.command("merged_PR", help="Make a summary of PRs merged with a given milestone")
@click.option("--token", default=None, type=str)
@click.argument("milestone", type=str, default="1.0")
def list_merged_PRs(milestone, token=None, scan_last=1000):
    g = login(token)
    repo = g.get_repo("gammapy/gammapy")

    pull_requests = repo.get_pulls(state="closed", sort="created", direction="desc")

    check_requests_number(g)

    total_number = 0

    for pr in pull_requests:
        if pr.milestone and pr.milestone.title == milestone:
            if pr.is_merged():
                if pr.user.name:
                    name = pr.user.name
                else:
                    name = pr.user.login
                total_number += 1
                print(f"- [#{pr.number}] {pr.title} ({name})")

    log.info(f"Found {total_number} of merged pull requests for milestone {milestone}.")

    check_requests_number(g)


@cli.command(
    "closed_issues", help="Make a summary of closed issues with a given milestone"
)
@click.option("--token", default=None, type=str)
@click.option("--print_issue", default=False, type=bool)
@click.argument("milestone", type=str, default="1.0")
def list_closed_issues(milestone, token=None, print_issue=False):
    g = login(token)
    repo = g.get_repo("gammapy/gammapy")

    issues = repo.get_issues(state="closed", sort="closed", direction="desc")

    check_requests_number(g)

    total_number = 0

    for issue in issues:
        if issue.milestone and issue.milestone.title == milestone:
            total_number += 1
            if print_issue:
                print(f"- [#{issue.number}] {issue.title}")

    log.info(f"Found {total_number} closed issues with milestone {milestone}.")

    check_requests_number(g)


if __name__ == "__main__":
    cli()
