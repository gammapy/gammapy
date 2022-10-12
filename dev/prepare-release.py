import logging
from datetime import date
from pathlib import Path
import click
from ruamel.yaml import YAML

log = logging.getLogger(__name__)


CITATION_PATH = Path(__file__).parent.parent / "CITATION.cff"


def update_citation_cff(release):
    # TODO: update author list according to PIG 24
    yaml = YAML()
    yaml.preserve_quotes = True

    with CITATION_PATH.open("r") as stream:
        data = yaml.load(stream=stream)

    data["date-released"] = date.today()
    data["version"] = release

    with CITATION_PATH.open("w") as stream:
        log.info(f"Writing {CITATION_PATH}")
        yaml.dump(data, stream=stream)


@click.command()
@click.option("--release", help="Release tag")
def cli(release):
    update_citation_cff(release=release)


if __name__ == "__main__":
    cli()
