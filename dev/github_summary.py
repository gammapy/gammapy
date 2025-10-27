# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import re
import numpy as np
from astropy.table import Table
from astropy.time import Time
import click
from github import Github, GithubException

log = logging.getLogger(__name__)


class GitHubInfoExtractor:
    """Class to interact with GitHub and extract PR and issues info tables.

    Parameters
    ----------
    repo : str
        input repository. Default is 'gammapy/gammapy'
    token : str
        GitHub access token. Default is None
    """

    def __init__(self, repo=None, token=None):
        self.repo = repo if repo else "gammapy/gammapy"
        self.github = self.login(token)
        self.repo = self.github.get_repo(repo)

    @staticmethod
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

    def check_requests_number(self):
        remaining, total = self.github.rate_limiting
        log.info(f"Remaining {remaining} requests over {total} requests.")

    def extract_contributors(
        self, state="closed", number_min=1, include_backports=False
    ):
        """Extract list of unique contributors from PRs.

         Parameters
         ----------
        state : str ("closed", "open", "all")
             state of PRs to extract.
         number_min : int
             minimum PR number to include. Default is 0.
         include_backports : bool
             Include backport PRs in the table. Default is True.
        """
        pull_requests = self.repo.get_pulls(
            state=state, sort="created", direction="desc"
        )

        self.check_requests_number()

        unique_user = set()
        for pr in pull_requests:
            if pr.number <= number_min:
                log.info(f"Reached minimum PR number {number_min}.")
                break

            if not include_backports and "Backport" in pr.title:
                log.info(f"Pull Request {pr.number} is backport. Skipping")
                continue

            # Start to add authors
            if pr.user:
                unique_user.add(pr.user.login or pr.user.name)

            # For committers
            for commit in pr.get_commits():
                if commit.committer:
                    unique_users.add(commit.committer.login)

            # For reviewers
            for review in pr.get_reviews():
                if review.user:
                    unique_users.add(review.user.login)

        return sorted(list(unique_users))

        #     log.info(f"Extracting Pull Request {number}.")
        #     try:
        #         result = self._extract_pull_request_info(pr)
        #     except AttributeError:
        #         log.warning(f"Issue with Pull Request {number}. Skipping")
        #         continue
        #     results.append(result)
        #
        # table = Table(results)
        # return table
        # self.check_requests_number()


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING"]),
)
def cli(log_level):
    logging.basicConfig(level=log_level)
    log.setLevel(level=log_level)


@cli.command("append_contributors", help="Make a list of the contributors of PRs")
@click.option("--token", default=None, type=str)
@click.option("--repo", default="gammapy/gammapy", type=str)
@click.option("--state", default="closed", type=str)
@click.option("--number_min", default=4000, type=int)
@click.option("--include_backports", default=False, type=bool)
def append_contributors(
    repo, token, state, number_min, include_backports
):
    """Make list of contributors to PRs."""
    log.info(
        f"Make list of contributors to PRs."
    )

    extractor = GitHubContributorsExtractor(repo=repo, token=token)
    users = extractor.extract_contributors(
        state=state, number_min=number_min, include_backports=include_backports
    )

    log.info(f"Found {len(users)} unique contributors.")
    print("\nContributors\n~~~~~~~~~~~~")
    for user in users:
        print(f"- {user}")


if __name__ == "__main__":
    cli()
