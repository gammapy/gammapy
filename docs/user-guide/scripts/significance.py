"""Example how to write a command line tool with Click"""
import click
from gammapy.stats import CashCountsStatistic


# You can call the callback function for the click command anything you like.
# `cli` is just a commonly used generic term for "command line interface".
@click.command()
@click.argument("n_observed", type=float)
@click.argument("mu_background", type=float)
@click.option(
    "--value",
    type=click.Choice(["sqrt_ts", "p_value"]),
    default="sqrt_ts",
    help="Significance or p_value",
)
def cli(n_observed, mu_background, value):
    """Compute significance for a Poisson count observation.

    The significance is the tail probability to observe N_OBSERVED counts
    or more, given a known background level MU_BACKGROUND."""
    stat = CashCountsStatistic(n_observed, mu_background)
    if value == "sqrt_ts":
        s = stat.sqrt_ts
    else:
        s = stat.p_value

    print(s)


if __name__ == "__main__":
    cli()
