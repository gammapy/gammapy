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

    @staticmethod
    def _get_commits_info(commits):
        """Builds a dictionary containing the number of commits and the list of unique committers."""
        result = dict()
        result["commits_number"] = commits.totalCount
        committers = set()
        for commit in commits:
            if commit.committer:
                committers.add(commit.committer.login)
        result["unique_committers"] = list(committers)
        return result

    @staticmethod
    def _get_reviews_info(reviews):
        """Builds a dictionary containing the number of reviews and the list of unique reviewers."""
        result = dict()
        result["review_number"] = reviews.totalCount
        reviewers = set()
        for review in reviews:
            if review.user:
                reviewers.add(review.user.login)
        result["unique_reviewers"] = list(reviewers)
        return result

    def _extract_pull_request_info(self, pull_request):
        """Builds a dictionary containing a list of summary informations.

        Parameters
        ----------
        pull_request :
            input pull request object

        Returns
        -------
        info : dict
            the result dictionary
        """
        result = dict()

        result["number"] = pull_request.number
        result["title"] = pull_request.title
        result["milestone"] = (
            "" if not pull_request.milestone else pull_request.milestone.title
        )
        result["is_merged"] = pull_request.is_merged()
        creation = pull_request.created_at
        result["date_creation"] = Time(creation) if creation else None
        closing = pull_request.closed_at
        result["date_closed"] = Time(closing) if closing else None
        result["user_name"] = pull_request.user.name
        result["user_login"] = pull_request.user.login
        result["user_email"] = pull_request.user.email
        result["labels"] = [label.name for label in pull_request.labels]
        result["changed_files"] = pull_request.changed_files
        result["base"] = pull_request.base.ref

        # extract commits
        commits = pull_request.get_commits()
        result.update(self._get_commits_info(commits))
        # extract reviews
        reviews = pull_request.get_reviews()
        result.update(self._get_reviews_info(reviews))

        return result

    def extract_pull_requests_table(
        self, state="closed", number_min=1, include_backports=False
    ):
        """Extract list of Pull Requests and build info table.

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

        results = []

        for pr in pull_requests:
            number = pr.number
            if number <= number_min:
                log.info(f"Reached minimum PR number {number_min}.")
                break

            title = pr.title
            if not include_backports and "Backport" in title:
                log.info(f"Pull Request {number} is backport. Skipping")
                continue

            log.info(f"Extracting Pull Request {number}.")
            try:
                result = self._extract_pull_request_info(pr)
            except AttributeError:
                log.warning(f"Issue with Pull Request {number}. Skipping")
                continue
            results.append(result)

        table = Table(results)
        return table
        self.check_requests_number()


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING"]),
)
def cli(log_level):
    logging.basicConfig(level=log_level)
    log.setLevel(level=log_level)


@cli.command("create_pull_request_table", help="Dump a table of all PRs.")
@click.option("--token", default=None, type=str)
@click.option("--repo", default="gammapy/gammapy", type=str)
@click.option("--state", default="closed", type=str)
@click.option("--number_min", default=4000, type=int)
@click.option("--filename", default="table_pr.ecsv", type=str)
@click.option("--overwrite", default=False, type=bool)
@click.option("--include_backports", default=False, type=bool)
def create_pull_request_table(
    repo, token, state, number_min, filename, overwrite, include_backports
):
    """Extract PR table and write it to dosk."""
    extractor = GitHubInfoExtractor(repo=repo, token=token)
    table = extractor.extract_pull_requests_table(
        state=state, number_min=number_min, include_backports=include_backports
    )
    table.write(filename, overwrite=overwrite)


@cli.command("merged_PR", help="Make a summary of PRs merged with a given milestone")
@click.argument("filename", type=str, default="table_pr.ecsv")
@click.argument("milestones", type=str, nargs=-1)
@click.option("--from_backports", default=False, type=bool)
def list_merged_PRs(filename, milestones, from_backports):
    """Make a list of merged PRs."""
    log.info(
        f"Make list of merged PRs from milestones {milestones} from file {filename}."
    )
    table = Table.read(filename)

    # Keep only merged PRs
    table = table[table["is_merged"] == True]

    # Keep the requested milestones
    valid = np.zeros((len(table)), dtype="bool")

    for i, pr in enumerate(table):
        milestone = milestones[0]
        if from_backports and "Backport" in pr["title"]:
            # check that the branch and milestone match
            if np.all(pr["base"].split(".")[:-1] == milestone.split(".")[:-1]):
                pattern = r"#(\d+)"
                parent_pr_number = int(re.search(pattern, pr["title"]).group(1))
                idx = np.where(table["number"] == parent_pr_number)[0]
                valid[idx] = True
        elif pr["milestone"] == milestone:
            valid[i] = True

    # filter the table and print info
    table = table[valid]
    log.info(f"Found {len(table)} merged PRs in the table.")

    unique_names = set()
    names = table["user_name"]
    logins = table["user_login"]
    for name, login in zip(names, logins):
        unique_names.add(name if name else login)

    unique_committers = set()
    unique_reviewers = set()
    for pr in table:
        for committer in pr["unique_committers"]:
            unique_committers.add(committer)
        for reviewer in pr["unique_reviewers"]:
            unique_reviewers.add(reviewer)

    contributor_names = list(unique_names)
    log.info(f"Found {len(contributor_names)} contributors in the table.")
    log.info(f"Found {len(unique_committers)} committers in the table.")
    log.info(f"namely: {unique_committers}")
    log.info(f"Found {len(unique_reviewers)} reviewers in the table.")
    log.info(f"namely: {unique_reviewers}")

    result = "Contributors\n"
    result += "~~~~~~~~~~~~\n"
    for name in contributor_names:
        result += f"- {name}\n"

    result += "\n\nPull Requests\n"
    result += "~~~~~~~~~~~~~\n\n"
    result += "This list is incomplete. Small improvements and bug fixes are not listed here.\n"

    for pr in table:
        number = pr["number"]
        title = pr["title"]
        user = pr["user_name"] if pr["user_name"] is not None else pr["user_login"]
        result += f"- [#{number}] {title} ({user})\n"

    print(result)


if __name__ == "__main__":
    cli()
