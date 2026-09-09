"""Update checking against GitHub Releases.

The console ships as an installer, so it needs to tell an operator when a newer
build exists. This module does that with the standard library only - no extra
dependency, and nothing that runs without being asked:

1. :meth:`UpdateChecker.check` reads the repository's latest release from the
   GitHub API and compares its tag with :data:`sif.version.__version__`.
2. :meth:`UpdateChecker.download` fetches the asset for this platform, reporting
   progress, and **verifies its SHA-256** against the ``SHA256SUMS.txt`` the
   release workflow publishes alongside it.
3. :meth:`UpdateChecker.launch` hands the verified file to the operating system's
   installer and asks the application to quit.

Rules this follows, deliberately:

* **Never silent.** Nothing is downloaded or installed without the operator
  agreeing; the caller drives every step.
* **Never unverified.** An asset whose checksum is missing or wrong is refused,
  because an update path is a code-execution path.
* **Never from source.** Running from a checkout, the checker reports the new
  version and tells the developer to pull, rather than installing over git.
* **Never fatal.** No network, a rate limit, a malformed tag: the check returns a
  result that says so and the console carries on.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import ssl
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .version import __version__, is_newer, normalise

__all__ = ["UpdateChecker", "UpdateInfo", "ReleaseAsset", "DEFAULT_REPOSITORY",
           "running_frozen", "platform_key"]

LOGGER = logging.getLogger(__name__)

#: ``owner/repo`` the console checks. Overridable so a fork or an internal
#: mirror can be pointed at without a rebuild.
DEFAULT_REPOSITORY = os.environ.get("SIF_UPDATE_REPO", "tedo001/SIF")
API_ROOT = os.environ.get("SIF_UPDATE_API", "https://api.github.com")
CHECKSUM_ASSET = "SHA256SUMS.txt"
USER_AGENT = f"SIF-Insight-Console/{__version__}"
CHECK_TIMEOUT = 10.0
DOWNLOAD_TIMEOUT = 900.0

#: Asset name endings that identify this platform's installer.
PLATFORM_SUFFIXES: Dict[str, tuple] = {
    "windows": ("-setup.exe", ".msi", ".exe"),
    "macos": (".dmg", ".pkg"),
    "linux": (".appimage", ".tar.gz"),
}


def platform_key() -> str:
    """``windows``, ``macos`` or ``linux`` for the running system."""
    system = platform.system().lower()
    if system.startswith("win"):
        return "windows"
    if system == "darwin":
        return "macos"
    return "linux"


def running_frozen() -> bool:
    """True when running from a packaged build rather than a source checkout."""
    return bool(getattr(sys, "frozen", False))


@dataclass(frozen=True)
class ReleaseAsset:
    """One downloadable file attached to a release."""

    name: str
    url: str
    size: int = 0

    @property
    def megabytes(self) -> float:
        return round(self.size / 1_048_576, 1) if self.size else 0.0


@dataclass
class UpdateInfo:
    """The outcome of one update check."""

    available: bool = False
    current: str = __version__
    latest: str = ""
    notes: str = ""
    published: str = ""
    asset: Optional[ReleaseAsset] = None
    checksum_url: str = ""
    html_url: str = ""
    error: str = ""
    #: True when the repository answered, but has published no release yet. This
    #: is the normal state before the first tag and is not a failure.
    no_release: bool = False
    assets: List[ReleaseAsset] = field(default_factory=list)

    @property
    def installable(self) -> bool:
        """True when there is an asset this platform can actually install."""
        return self.available and self.asset is not None and running_frozen()

    def summary(self) -> str:
        """One line for the status bar or the log."""
        if self.error:
            return f"Update check failed: {self.error}"
        if self.no_release:
            return (f"No release has been published yet - running {self.current} "
                    "from this build")
        if not self.available:
            return f"Up to date (version {self.current})"
        if not running_frozen():
            return (f"Version {self.latest} is available - this is a source checkout, "
                    "so update with: git pull")
        if self.asset is None:
            return (f"Version {self.latest} is available, but no installer was published "
                    f"for {platform_key()}")
        return f"Version {self.latest} is available ({self.asset.megabytes} MB)"


class UpdateChecker:
    """Checks GitHub Releases, downloads an installer, and verifies it."""

    def __init__(self, repository: str = DEFAULT_REPOSITORY,
                 current_version: str = __version__,
                 include_prereleases: bool = False,
                 token: Optional[str] = None) -> None:
        self.repository = repository
        self.current_version = current_version
        self.include_prereleases = include_prereleases
        # Only needed for a private repository; public releases need no auth.
        self.token = token or os.environ.get("SIF_UPDATE_TOKEN") or ""

    # -- HTTP --------------------------------------------------------------

    def _open(self, url: str, timeout: float, accept: str = "application/vnd.github+json"):
        request = urllib.request.Request(url, headers={
            "Accept": accept,
            "User-Agent": USER_AGENT,
            **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
        })
        context = ssl.create_default_context()
        return urllib.request.urlopen(request, timeout=timeout, context=context)

    def _get_json(self, url: str) -> object:
        with self._open(url, CHECK_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))

    # -- check -------------------------------------------------------------

    def check(self) -> UpdateInfo:
        """Ask GitHub for the newest release. Never raises."""
        info = UpdateInfo(current=self.current_version)
        try:
            release = self._latest_release()
        except urllib.error.HTTPError as exc:
            info.error = self._describe(exc)
            LOGGER.warning("Update check failed: %s", info.error)
            return info
        except Exception as exc:  # noqa: BLE001 - offline, DNS, TLS, malformed JSON
            info.error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("Update check failed: %s", info.error)
            return info

        if release is None:
            # The repository answered and simply has nothing published. Before the
            # first tag that is the expected state, so it is reported, not raised.
            info.no_release = True
            LOGGER.info("No release published for %s yet; running %s",
                        self.repository, self.current_version)
            return info

        tag = normalise(str(release.get("tag_name", "")))
        info.latest = tag
        info.notes = str(release.get("body", "") or "")
        info.published = str(release.get("published_at", "") or "")
        info.html_url = str(release.get("html_url", "") or "")
        info.assets = [
            ReleaseAsset(name=str(item.get("name", "")),
                         url=str(item.get("browser_download_url", "")),
                         size=int(item.get("size", 0) or 0))
            for item in release.get("assets", []) if item.get("browser_download_url")
        ]
        info.available = is_newer(tag, self.current_version)
        if info.available:
            info.asset = self._select_asset(info.assets)
            checksum = next((item for item in info.assets if item.name == CHECKSUM_ASSET), None)
            info.checksum_url = checksum.url if checksum else ""
            LOGGER.info("Update available: %s -> %s", self.current_version, tag)
        else:
            LOGGER.info("No update: running %s, latest published is %s",
                        self.current_version, tag or "unknown")
        return info

    def _latest_release(self) -> Optional[dict]:
        """Newest release, honouring the pre-release preference.

        ``releases/latest`` answers 404 both when a repository has published
        nothing and when it cannot be seen at all, so a 404 there is resolved
        against the listing endpoint, which answers ``200 []`` for the first case
        and 404 for the second.
        """
        if not self.include_prereleases:
            try:
                body = self._get_json(f"{API_ROOT}/repos/{self.repository}/releases/latest")
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
                return self._newest_listed()
            return body if isinstance(body, dict) else None
        return self._newest_listed()

    def _newest_listed(self) -> Optional[dict]:
        """Newest release from the listing endpoint, drafts always excluded."""
        body = self._get_json(f"{API_ROOT}/repos/{self.repository}/releases?per_page=10")
        if not isinstance(body, list):
            return None
        published = [item for item in body if not item.get("draft")
                     and (self.include_prereleases or not item.get("prerelease"))]
        return published[0] if published else None

    def _describe(self, exc: urllib.error.HTTPError) -> str:
        """Turn an HTTP status into something an operator can act on."""
        if exc.code == 404:
            return (f"repository {self.repository} is not visible from this machine - "
                    "check the name, or set SIF_UPDATE_TOKEN if it is private")
        if exc.code in (403, 429):
            return f"GitHub returned {exc.code} - rate limited, try later"
        if exc.code == 401:
            return "GitHub returned 401 - SIF_UPDATE_TOKEN was rejected"
        return f"GitHub returned {exc.code}"

    @staticmethod
    def _select_asset(assets: List[ReleaseAsset]) -> Optional[ReleaseAsset]:
        """Pick the asset that installs on this platform."""
        suffixes = PLATFORM_SUFFIXES[platform_key()]
        for suffix in suffixes:                      # most specific first
            for asset in assets:
                if asset.name.lower().endswith(suffix):
                    return asset
        return None

    # -- download and verify -----------------------------------------------

    def download(self, info: UpdateInfo,
                 progress: Optional[Callable[[int, int], None]] = None,
                 directory: Optional[str] = None) -> str:
        """Download the installer and verify its checksum; returns the path.

        Raises ``RuntimeError`` when the download cannot be verified - an
        unverified installer is never handed back to the caller.
        """
        if info.asset is None:
            raise RuntimeError("this release has no installer for this platform")

        expected = self._expected_checksum(info)
        target_dir = directory or tempfile.mkdtemp(prefix="sif-update-")
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, info.asset.name)

        digest = hashlib.sha256()
        downloaded = 0
        with self._open(info.asset.url, DOWNLOAD_TIMEOUT,
                        accept="application/octet-stream") as response, \
                open(path, "wb") as handle:
            total = int(response.headers.get("Content-Length") or info.asset.size or 0)
            while True:
                chunk = response.read(262_144)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                if progress is not None:
                    progress(downloaded, total)

        actual = digest.hexdigest()
        if expected and actual != expected:
            os.unlink(path)
            raise RuntimeError(
                "checksum mismatch - the download was refused. "
                f"expected {expected[:16]}..., got {actual[:16]}...")
        if not expected:
            os.unlink(path)
            raise RuntimeError(
                f"the release does not publish {CHECKSUM_ASSET}, so the installer "
                "cannot be verified; download it manually from the release page")

        LOGGER.info("Downloaded and verified %s (%d bytes)", info.asset.name, downloaded)
        return path

    def _expected_checksum(self, info: UpdateInfo) -> str:
        """The SHA-256 recorded for this asset in the release's checksum file."""
        if not info.checksum_url or info.asset is None:
            return ""
        try:
            with self._open(info.checksum_url, CHECK_TIMEOUT,
                            accept="application/octet-stream") as response:
                text = response.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - treated as "unverifiable"
            LOGGER.warning("Could not read %s: %s", CHECKSUM_ASSET, exc)
            return ""
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 2 and os.path.basename(parts[-1].lstrip("*")) == info.asset.name:
                return parts[0].strip().lower()
        return ""

    # -- install -----------------------------------------------------------

    @staticmethod
    def launch(path: str) -> bool:
        """Hand the verified installer to the operating system.

        The caller should quit immediately afterwards: an installer cannot
        replace files the running application still holds open.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        system = platform_key()
        try:
            if system == "windows":
                os.startfile(path)  # type: ignore[attr-defined]  # noqa: S606
            elif system == "macos":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:  # noqa: BLE001 - report, never crash on exit
            LOGGER.error("Could not start the installer: %s", exc)
            return False
        LOGGER.info("Installer started: %s", path)
        return True
