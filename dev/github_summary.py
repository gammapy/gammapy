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
        self, milestone_name, state="closed",
    ):
        """Extract list of unique contributors from PRs as per the milestone.

         Parameters
         ----------
         milestone_name :  str
            Milestone name i.e. 'v1.0'
         state : str ("closed", "open", "all")
            State of PRs to extract.
        """
        milestones = self.repo.get_milestones(state="all")
        milestone_obj = None
        for m in milestones:
            if m.title == milestone_name:
                milestone_obj = m
                break
        if milestone_obj is None:
            log.error(f"Milestone '{milestone_name}' not found in repository '{self.repo.full_name}'.")
            return []

        # Get PRs filtered by milestone using issues API (GitHub returns PRs as issues)
        issues = self.repo.get_issues(state=state, milestone=milestone_obj)

        self.check_requests_number()

        unique_users = set()
        num_prs = 0
        num_closed_issues = 0
        for issue in issues:
            if issue.pull_request is None:
                if issue.state == 'closed':
                    num_closed_issues += 1
                continue  # skip issues, only process PRs

            try:
                pr = self.repo.get_pull(issue.number)
            # Sometimes PRs are marked that way but cannot be found, because they no longer exist.
            except GithubException:
                continue

            # Skip PRs that are closed and not merged
            if not pr.merged:
                continue

            if "Backport" in pr.title:
                continue
            num_prs += 1
            log.info(f"Extracting Pull Request {pr.number}.")

            # It is possible that there will be 'authors' such as 'web-flow' which is just a merging action
            # The extra lines here ignore those users that are not real
            # Start to add authors
            if pr.user and pr.user.login not in {"web-flow", "dependabot[bot]", "github-actions[bot]"}:
                unique_users.add(pr.user.name or pr.user.login)

            # For committers
            for commit in pr.get_commits():
                if commit.committer and commit.committer.login not in {"web-flow", "dependabot[bot]",
                                                                       "github-actions[bot]"}:
                    # This is required for if a user deletes their account
                    try:
                        name = commit.committer.name
                        login = commit.committer.login
                    except GithubException:
                        continue
                    unique_users.add(name or login)

            # For reviewers
            for review in pr.get_reviews():
                if review.user and review.user.login not in {"web-flow", "dependabot[bot]", "github-actions[bot]"}:
                    unique_users.add(review.user.name or review.user.login)

        return {
            "contributors": sorted(list(unique_users)),
            "num_prs": num_prs,
            "num_closed_issues": num_closed_issues,
        }


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
@click.option("--token", default=None, type=str, help="Your GitHub token.")
@click.option("--repo", default="gammapy/gammapy", type=str, help="The relative repo.")
@click.option("--milestone", required=True, type=str, help="Comma-separated list of milestones, e.g., '2.0.1,2.1'")
@click.option("--state", default="closed", type=str, help="Is the issues closed or not.")
def contributors_by_milestone(repo, token, milestone, state):
    """List contributors attached to a specific milestone."""
    extractor = GitHubContributorsExtractor(repo=repo, token=token)
    milestone_list = [m.strip() for m in milestone.split(",") if m.strip()]

    all_users = set()
    for m in milestone_list:
        log.info(f"Making list of contributors for milestone '{m}'.")
        info = extractor.extract_contributors_by_milestone(
            milestone_name=m,
            state=state,
        )
        users = info['contributors']
        num_prs = info['num_prs']
        num_closed_issues = info['num_closed_issues']
        log.info(f"""
        For milestone '{m}':
          - {num_prs} pull requests
          - {num_closed_issues} closed issues
          - {len(users)} unique contributors
        """)
        all_users.update(users)

    print(f"\nContributors for milestone '{milestone}'\n{'~' * 20}")
    for user in sorted(all_users):
        print(f"- {user}")

    # Add the names directly to the changelog
    changelog_file = "docs/release-notes/CHANGELOG.rst"
    contributors_sorted = sorted(all_users)
    with open(changelog_file, "a", encoding="utf-8") as f:
        for name in contributors_sorted:
            f.write(f"- {name}\n")


if __name__ == "__main__":
    cli()
