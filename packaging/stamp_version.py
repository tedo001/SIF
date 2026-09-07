"""Write the release version into the package, from the git tag.

The tag is the source of truth. CI runs this before packaging so the version the
application reports, the version in the installer's metadata, and the tag the
build came from are the same string - a mismatch is a support nightmare and a
silent update-loop risk (an app that reports 2.0.0 forever will offer 2.1.0 again
after every install).

    python packaging/stamp_version.py v2.1.0 --build 9f3a1c2

With ``--check`` it verifies instead of writing, which is what a pull request
should run.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

VERSION_FILE = pathlib.Path(__file__).resolve().parent.parent / "sif" / "version.py"
SEMVER = re.compile(r"^v?\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Stamp the release version.")
    parser.add_argument("tag", help="git tag, e.g. v2.1.0")
    parser.add_argument("--build", default="", help="short commit sha")
    parser.add_argument("--check", action="store_true",
                        help="verify the file already matches, do not write")
    args = parser.parse_args(argv)

    if not SEMVER.match(args.tag):
        print(f"error: '{args.tag}' is not a semantic version tag (expected v1.2.3)",
              file=sys.stderr)
        return 1

    version = args.tag.lstrip("vV")
    source = VERSION_FILE.read_text(encoding="utf-8")
    current = re.search(r'(?m)^__version__ = "([^"]+)"$', source)
    if current is None:
        print("error: could not find the __version__ line", file=sys.stderr)
        return 1

    if args.check:
        if current.group(1) != version:
            print(f"error: tag {args.tag} does not match sif/version.py "
                  f"({current.group(1)}). Run: python packaging/stamp_version.py {args.tag}",
                  file=sys.stderr)
            return 1
        print(f"version.py matches {args.tag}")
        return 0

    updated = re.sub(r'(?m)^__version__ = "[^"]+"$', f'__version__ = "{version}"', source)
    updated = re.sub(r'(?m)^BUILD = "[^"]*"$', f'BUILD = "{args.build}"', updated)
    VERSION_FILE.write_text(updated, encoding="utf-8")
    print(f"stamped {version}" + (f"+{args.build}" if args.build else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
