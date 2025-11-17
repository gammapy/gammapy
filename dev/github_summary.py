# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import re
import numpy as np
from astropy.table import Table
from astropy.time import Time
import click
from github import Github, GithubException

log = logging.getLogger(__name__)


class GitHubContributorsExtractor:
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

    def extract_contributors_by_milestone(
        self, milestone_name, state="closed", include_backports=False
    ):
        """Extract list of unique contributors from PRs as per the milestone.

         Parameters
         ----------
         milestone_name :  str
            Milestone name i.e. 'v1.0'
         state : str ("closed", "open", "all")
            State of PRs to extract.
         include_backports : bool
            Include backport PRs in the table. Default is True.
        """
        milestones = self.repo.get_milestones(state="all")
        milestone_obj = None
        for m in milestones:
            if m.title == milestone_name:
                milestone_obj = m
                break
        if milestone_obj is None:
            log.error(f"Milestone '{milestone_name}' not found in repository '{self.repo_name}'.")
            return []

        # Get PRs filtered by milestone using issues API (GitHub returns PRs as issues)
        issues = self.repo.get_issues(state=state, milestone=milestone_obj)

        self.check_requests_number()

        unique_users = set()
        for issue in issues:
            if issue.pull_request is None:
                continue  # skip issues, only process PRs

            pr = self.repo.get_pull(issue.number)

            if not include_backports and "Backport" in pr.title:
                log.info(f"Pull Request {pr.number} is backport. Skipping")
                continue

            log.info(f"Extracting Pull Request {pr.number}.")

            # Start to add authors
            if pr.user:
                unique_users.add(pr.user.login or pr.user.name)

            # For committers
            for commit in pr.get_commits():
                if commit.committer:
                    unique_users.add(commit.committer.login)

            # For reviewers
            for review in pr.get_reviews():
                if review.user:
                    unique_users.add(review.user.login)

        return sorted(list(unique_users))


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING"]),
)
def cli(log_level):
    logging.basicConfig(level=log_level)
    log.setLevel(level=log_level)

@cli.command(
    "contributors_by_milestone",
    help="Make a list of contributors for a specific milestone"
)
@click.option("--token", default=None, type=str)
@click.option("--repo", default="gammapy/gammapy", type=str)
@click.option("--milestone", required=True, type=str, help="Comma-separated list of milestones, e.g., '2.0.1,2.1'")
@click.option("--state", default="closed", type=str)
@click.option("--include_backports", default=False, type=bool)
def contributors_by_milestone(repo, token, milestone, state, include_backports):
    """List contributors attached to a specific milestone."""
    extractor = GitHubContributorsExtractor(repo=repo, token=token)
    milestone_list = [m.strip() for m in milestone.split(",") if m.strip()]

    all_users = set()
    for m in milestone_list:
        log.info(f"Making list of contributors for milestone '{m}'.")
        users = extractor.extract_contributors_by_milestone(
            milestone_name=m,
            state=state,
            include_backports=include_backports
        )
        log.info(f"Found {len(users)} unique contributors for milestone '{m}'.")
        all_users.update(users)

    print(f"\nContributors for milestone '{milestone}'\n{'~' * 20}")
    for user in sorted(all_users):
        print(f"- {user}")


if __name__ == "__main__":
    cli()
