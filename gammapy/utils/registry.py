# Licensed under a 3-clause BSD style license - see LICENSE.rst
__all__ = ["Registry"]


class Registry(list):
    """Registry class."""

    def get_cls(self, tag):
        for cls in self:
            if hasattr(cls, "tag") and tag in cls.tag:
                return cls
        raise KeyError(f"No model found with tag: {tag!r}")

    def __str__(self):
        info = "Registry\n"
        info += "--------\n\n"
        tags = [_.tag[0] if isinstance(_.tag, list) else _.tag for _ in self]
        len_max = max([len(tag) for tag in tags])

        for item in self:
            info += f"\t{item.tag:{len_max}s}: {item.__name__}\n"

        return info.expandtabs(tabsize=2)
