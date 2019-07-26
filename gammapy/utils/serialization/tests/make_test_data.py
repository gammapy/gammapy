"""Create example model YAML files programmatically.

(some will be also written manually)
"""
from pathlib import Path
from gammapy.image.models import SkyGaussian
from gammapy.spectrum.models import PowerLaw
from gammapy.cube.models import SkyModels, SkyModel

DATA_PATH = Path("gammapy/utils/serialization/tests/data/")


def make_example_2():
    spatial = SkyGaussian("0 deg", "0 deg", "1 deg")
    model = SkyModel(spatial, PowerLaw())
    models = SkyModels([model])
    models.to_yaml(DATA_PATH / "example2.yaml")


if __name__ == "__main__":
    make_example_2()
