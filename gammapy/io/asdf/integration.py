import importlib.resources as importlib_resources

from asdf.resource import DirectoryResourceMapping


def get_resource_mappings():
    """
    Get the resource mapping instances for the gammapy schemas
    and manifests. This method is registered with the
    asdf.resource_mappings entry point.

    Returns
    -------
    list of collections.abc.Mapping
    """
    from . import resources

    resource_root = importlib_resources.files(resources)

    return [
        DirectoryResourceMapping(
            resource_root / "schemas",
            "asdf://gammapy.org/gammapy/schemas/",
            recursive=True,
        ),
        DirectoryResourceMapping(
            resource_root / "manifests", "asdf://gammapy.org/gammapy/manifests/"
        ),
    ]


def get_extensions():
    """
    Get the extension instances for gammapy ASDF
    extension. This method is registered with the
    asdf.extensions entry point.

    Returns
    -------
    list of asdf.extension.Extension
    """
    from . import extensions

    return [
        *extensions.GAMMAPY_EXTENSIONS,
    ]
