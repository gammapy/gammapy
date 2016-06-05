import os
print(os.environ)

from gammapy.datasets import gammapy_extra
print(gammapy_extra.dir)
print(gammapy_extra.is_available)
print(os.listdir(str(gammapy_extra.dir)))

from gammapy.utils.scripts import make_path;
print(make_path("$GAMMAPY_EXTRA/README.rst").absolute())
