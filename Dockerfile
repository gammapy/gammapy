# Two-stage build, first build the python wheel
FROM python:3.12 AS builder

# Add necessary sources, including .git for version info
COPY gammapy/ /repo/gammapy/
COPY .git/ /repo/.git/
COPY pyproject.toml setup.py codemeta.json CITATION.cff MANIFEST.in README.md LICENSE.rst CHANGES.rst /repo/

WORKDIR /repo

# Build the Gammapy wheel
RUN python -m pip install --no-cache-dir build \
    && python -m build --wheel

# Second stage, copy and install wheel in the slim python image to reduce image size.
FROM python:3.12-slim
COPY --from=builder /repo/dist /tmp/dist

RUN python -m pip install --no-cache-dir /tmp/dist/* \
    && rm -r /tmp/dist

RUN useradd --create-home --system --user-group gammapy
USER gammapy
