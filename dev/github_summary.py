# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
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
        result["date_creation"] = Time(pull_request.created_at)
        result["date_closed"] = Time(pull_request.closed_at)
        result["user_name"] = pull_request.user.name
        result["user_login"] = pull_request.user.login
        result["user_email"] = pull_request.user.email
        result["labels"] = [label.name for label in pull_request.labels]
        result["changed_files"] = pull_request.changed_files

        # extract commits
        commits = pull_request.get_commits()
        result.update(self._get_commits_info(commits))
        # extract reviews
        reviews = pull_request.get_reviews()
        result.update(self._get_reviews_info(reviews))

        return result

    def extract_pull_requests_table(
        self, state="closed", number_min=0, include_backports=False
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

    def extract_issue_table(self, state="closed", number_min=0):
        """Extract list of Issues and build info table.

        Parameters
        ----------
        state : str ("closed", "open", "all")
            state of issues to extract.
        number_min : int
            minimum PR number to include. Default is 0.
        """
        issues = self.repo.get_pulls(state=state, sort="created", direction="desc")

        self.check_requests_number()
        total_number = 0

        for issue in issues:
            if issue.number < number_min:
                break
            if issue.milestone:
                total_number += 1

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


@cli.command("dump_pull_request_table", help="Dump a table of all PRs.")
@click.option("--token", default=None, type=str)
@click.option("--repo", default="gammapy/gammapy", type=str)
@click.option("--state", default="closed", type=str)
@click.option("--number_min", default=4000, type=int)
@click.option("--filename", default="table_pr.ecsv", type=str)
@click.option("--overwrite", default=False, type=bool)
def dump_table(repo, token, state, number_min, filename, overwrite):
    """Extract PR table and write it to dosk."""
    extractor = GitHubInfoExtractor(repo=repo, token=token)
    table = extractor.extract_pull_requests_table(
        state=state, number_min=number_min, include_backports=False
    )
    table.write(filename, overwrite=overwrite)


@cli.command(
    "closed_issues", help="Make a summary of closed issues with a given milestone"
)
@click.option("--token", default=None, type=str)
@click.option("--repo", default="gammapy/gammapy", type=str)
@click.option("--state", default="closed", type=str)
@click.option("--number_min", default=4000, type=int)
@click.option("--filename", default="table_issues.ecsv", type=str)
@click.option("--overwrite", default=False, type=bool)
def list_closed_issues(repo, token, state, number_min, filename, overwrite):
    extractor = GitHubInfoExtractor(repo=repo, token=token)
    table = extractor.extract_issues_table(state=state, number_min=number_min)
    table.write(filename, overwrite=overwrite)


@cli.command("merged_PR", help="Make a summary of PRs merged with a given milestone")
@click.option("--token", default=None, type=str)
@click.option("--number_min", default=4000, type=int)
@click.argument("milestone", type=str, default="1.0")
def list_merged_PRs(milestone, token=None, number_min=4000):
    g = token
    repo = g.get_repo("gammapy/gammapy")

    pull_requests = repo.get_pulls(state="closed", sort="created", direction="desc")

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


if __name__ == "__main__":
    cli()
