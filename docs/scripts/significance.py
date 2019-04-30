"""Example how to write a command line tool with Click"""
import click
from gammapy.stats import significance


# You can call the callback function for the click command anything you like.
# `cli` is just a commonly used generic term for "command line interface".
@click.command()
@click.argument("n_observed")
@click.argument("mu_background")
@click.option(
    "--method",
    type=click.Choice(["lima", "simple"]),
    default="lima",
    help="Significance computation method",
)
def cli(n_observed, mu_background, method):
    """Compute significance for a Poisson count observation.

    The significance is the tail probability to observe N_OBSERVED counts
    or more, given a known background level MU_BACKGROUND."""
    s = significance(n_observed, mu_background, method)
    print(s)


if __name__ == "__main__":
    cli()
