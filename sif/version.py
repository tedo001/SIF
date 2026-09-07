"""Single source of truth for the application version.

The git tag is authoritative. Release builds stamp this file from the tag (see
``.github/workflows/release.yml``), so the version reported by the running
application, the version baked into the installer, and the tag it was built from
cannot drift apart.

Between releases the committed value is what a developer sees, suffixed by the
build metadata CI injects, so a support ticket can always be tied to a build.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

__all__ = ["__version__", "VERSION", "BUILD", "parse", "is_newer", "normalise",
           "describe"]

#: Semantic version, without a leading "v". CI rewrites this line on release.
__version__ = "2.0.0"

#: Build metadata CI stamps in (short commit sha), empty for a working tree.
BUILD = ""

VERSION = __version__

_SEMVER = re.compile(
    r"^v?(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?"
    r"(?:-(?P<pre>[0-9A-Za-z.-]+))?(?:\+(?P<build>[0-9A-Za-z.-]+))?$")


def normalise(value: str) -> str:
    """Strip a leading ``v`` and surrounding whitespace from a tag."""
    return (value or "").strip().lstrip("vV")


def parse(value: str) -> Optional[Tuple[int, int, int, Tuple]]:
    """Parse a version into ``(major, minor, patch, prerelease)``.

    Returns ``None`` for anything unparsable rather than raising - an update
    check must never crash the application over a malformed tag.
    """
    match = _SEMVER.match((value or "").strip())
    if not match:
        return None
    pre = match.group("pre")
    # A release outranks any pre-release of the same numbers, so an absent
    # pre-release sorts last: () is greater than any non-empty tuple below.
    prerelease: Tuple = ()
    if pre:
        parts = []
        for chunk in pre.split("."):
            parts.append((0, int(chunk)) if chunk.isdigit() else (1, chunk))
        prerelease = tuple(parts)
    return (int(match.group("major")), int(match.group("minor")),
            int(match.group("patch") or 0), prerelease)


def is_newer(candidate: str, current: str = __version__) -> bool:
    """True when ``candidate`` is a strictly newer version than ``current``."""
    left, right = parse(candidate), parse(current)
    if left is None or right is None:
        return False
    if left[:3] != right[:3]:
        return left[:3] > right[:3]
    # Same numbers: a release beats a pre-release, and pre-releases order among
    # themselves (1.0.0-rc.2 > 1.0.0-rc.1).
    if left[3] == right[3]:
        return False
    if not left[3]:
        return True
    if not right[3]:
        return False
    return left[3] > right[3]


def describe() -> str:
    """Human-readable version for the interface and the log."""
    return f"{__version__}+{BUILD}" if BUILD else __version__
