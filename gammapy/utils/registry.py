# Licensed under a 3-clause BSD style license - see LICENSE.rst
__all__ = ["Registry"]


class Registry(list):
    """Registry class."""

    def get_cls(self, tag):
        for cls in self:
            tags = getattr(cls, "tag", [])
            tags = [tags] if isinstance(tags, str) else tags
            if tag in tags:
                return cls
        raise KeyError(f"No object found with tag: {tag!r}")

    def __str__(self):
        info = "Registry\n"
        info += "--------\n\n"
        len_max = max([len(_.__name__) for _ in self])

        for item in self:
            info += f"{item.__name__:{len_max}s}: {item.tag} \n"

        return info.expandtabs(tabsize=2)
