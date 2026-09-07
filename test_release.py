"""Tests for versioning and the GitHub-Releases updater.

The updater is a code-execution path - it downloads a file and hands it to the
operating system - so the checksum behaviour is tested from both sides: a good
download is accepted, a tampered one is refused and deleted.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from sif import updater as updater_module
from sif.updater import UpdateChecker, UpdateInfo, platform_key
from sif.version import __version__, describe, is_newer, normalise, parse

INSTALLER_BODY = b"pretend installer payload" * 64
GOOD_SHA = hashlib.sha256(INSTALLER_BODY).hexdigest()


class MockGitHub(BaseHTTPRequestHandler):
    """Serves one release with an installer for every platform."""

    tag = "v9.9.9"
    tamper = False

    def _send(self, payload: bytes, code: int = 200, content_type="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802 - http.server API
        base = f"http://{self.headers['Host']}"
        if self.path.endswith("/releases/latest") or "/releases?" in self.path:
            names = ("SIF-Console-setup.exe", "SIF-Console.dmg", "SIF-Console.AppImage")
            release = {
                "tag_name": self.tag,
                "body": "Fixes the thing.",
                "published_at": "2026-01-01T00:00:00Z",
                "html_url": "https://example.invalid/release",
                "assets": [{"name": name, "size": len(INSTALLER_BODY),
                            "browser_download_url": f"{base}/download/{name}"}
                           for name in names]
                + [{"name": "SHA256SUMS.txt", "size": 200,
                    "browser_download_url": f"{base}/download/SHA256SUMS.txt"}],
            }
            payload = json.dumps([release] if "/releases?" in self.path else release)
            self._send(payload.encode())
        elif self.path.endswith("SHA256SUMS.txt"):
            lines = "\n".join(f"{GOOD_SHA}  {name}" for name in
                              ("SIF-Console-setup.exe", "SIF-Console.dmg",
                               "SIF-Console.AppImage"))
            self._send(lines.encode(), content_type="text/plain")
        elif "/download/" in self.path:
            body = INSTALLER_BODY + (b"tampered" if self.tamper else b"")
            self._send(body, content_type="application/octet-stream")
        else:
            self._send(b'{"message":"not found"}', 404)

    def log_message(self, *args):
        pass


class TestVersioning(unittest.TestCase):
    """The version is the contract between the tag, the installer and the app."""

    def test_parse_and_normalise(self) -> None:
        self.assertEqual(normalise("v3.1.4"), "3.1.4")
        self.assertEqual(parse("2.0.0")[:3], (2, 0, 0))
        self.assertEqual(parse("v1.2")[:3], (1, 2, 0))
        self.assertIsNone(parse("not-a-version"))
        self.assertIsNone(parse(""))

    def test_ordering(self) -> None:
        self.assertTrue(is_newer("2.1.0", "2.0.9"))
        self.assertTrue(is_newer("2.0.1", "2.0.0"))
        self.assertTrue(is_newer("v10.0.0", "9.9.9"))
        self.assertFalse(is_newer("2.0.0", "2.0.0"))
        self.assertFalse(is_newer("1.0.0", "2.0.0"))

    def test_prereleases_sort_below_their_release(self) -> None:
        self.assertTrue(is_newer("2.0.0", "2.0.0-rc.1"))
        self.assertTrue(is_newer("2.0.0-rc.2", "2.0.0-rc.1"))
        self.assertFalse(is_newer("2.0.0-rc.1", "2.0.0"))

    def test_garbage_never_triggers_an_update(self) -> None:
        for value in ("", "latest", "release-2026", "v", "nightly"):
            self.assertFalse(is_newer(value, __version__))

    def test_the_package_reports_one_version(self) -> None:
        import sif

        self.assertEqual(sif.__version__, __version__)
        self.assertTrue(describe().startswith(__version__))

    def test_version_line_is_machine_editable(self) -> None:
        """CI rewrites this line from the tag - keep it matchable."""
        with open(os.path.join("sif", "version.py"), encoding="utf-8") as handle:
            source = handle.read()
        self.assertRegex(source, r'(?m)^__version__ = "\d+\.\d+\.\d+[^"]*"$')
        self.assertRegex(source, r'(?m)^BUILD = "[^"]*"$')


class TestUpdateChecker(unittest.TestCase):
    """The check, the download, and the refusal to install what it cannot verify."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = HTTPServer(("127.0.0.1", 0), MockGitHub)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.api = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls._api_root = updater_module.API_ROOT
        updater_module.API_ROOT = cls.api

    @classmethod
    def tearDownClass(cls) -> None:
        updater_module.API_ROOT = cls._api_root
        cls.server.shutdown()

    def setUp(self) -> None:
        MockGitHub.tag = "v9.9.9"
        MockGitHub.tamper = False

    def checker(self, current="2.0.0", **kwargs) -> UpdateChecker:
        return UpdateChecker(repository="tedo001/SIF", current_version=current, **kwargs)

    def test_finds_a_newer_release(self) -> None:
        info = self.checker().check()
        self.assertTrue(info.available)
        self.assertEqual(info.latest, "9.9.9")
        self.assertIn("Fixes the thing", info.notes)
        self.assertTrue(info.checksum_url.endswith("SHA256SUMS.txt"))

    def test_picks_the_asset_for_this_platform(self) -> None:
        info = self.checker().check()
        expected = {"windows": ".exe", "macos": ".dmg", "linux": ".AppImage"}[platform_key()]
        self.assertTrue(info.asset.name.endswith(expected), info.asset.name)

    def test_same_or_older_release_is_not_an_update(self) -> None:
        MockGitHub.tag = "v2.0.0"
        self.assertFalse(self.checker("2.0.0").check().available)
        MockGitHub.tag = "v1.0.0"
        self.assertFalse(self.checker("2.0.0").check().available)

    def test_download_verifies_the_checksum(self) -> None:
        info = self.checker().check()
        seen = []
        with tempfile.TemporaryDirectory() as folder:
            path = self.checker().download(info, progress=lambda done, total: seen.append(done),
                                           directory=folder)
            self.assertTrue(os.path.isfile(path))
            with open(path, "rb") as handle:
                self.assertEqual(hashlib.sha256(handle.read()).hexdigest(), GOOD_SHA)
        self.assertTrue(seen, "progress should be reported")

    def test_tampered_download_is_refused_and_deleted(self) -> None:
        MockGitHub.tamper = True
        info = self.checker().check()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RuntimeError) as caught:
                self.checker().download(info, directory=folder)
            self.assertIn("checksum mismatch", str(caught.exception))
            self.assertEqual(os.listdir(folder), [], "the bad file must not be kept")

    def test_release_without_checksums_is_refused(self) -> None:
        info = self.checker().check()
        info.checksum_url = ""                     # release published without SHA256SUMS
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RuntimeError) as caught:
                self.checker().download(info, directory=folder)
            self.assertIn("cannot be verified", str(caught.exception))
            self.assertEqual(os.listdir(folder), [])

    def test_no_asset_for_this_platform_is_reported_not_raised(self) -> None:
        # Pretend to be a packaged build, so the "source checkout" branch (which
        # takes precedence, and rightly so) does not mask this one.
        info = UpdateInfo(available=True, latest="9.9.9", asset=None)
        original = updater_module.running_frozen
        updater_module.running_frozen = lambda: True
        try:
            self.assertIn("no installer", info.summary())
        finally:
            updater_module.running_frozen = original
        with self.assertRaises(RuntimeError):
            self.checker().download(info)

    def test_a_packaged_build_is_offered_the_installer(self) -> None:
        original = updater_module.running_frozen
        updater_module.running_frozen = lambda: True
        try:
            info = self.checker().check()
            self.assertTrue(info.installable)
            self.assertIn("is available", info.summary())
        finally:
            updater_module.running_frozen = original

    def test_unreachable_api_degrades_quietly(self) -> None:
        updater_module.API_ROOT = "http://127.0.0.1:9"
        try:
            info = self.checker().check()
        finally:
            updater_module.API_ROOT = self.api
        self.assertFalse(info.available)
        self.assertTrue(info.error)
        self.assertIn("Update check failed", info.summary())

    def test_source_checkout_is_told_to_pull_not_to_install(self) -> None:
        info = self.checker().check()
        self.assertTrue(info.available)
        self.assertFalse(info.installable)         # not frozen in the test run
        self.assertIn("git pull", info.summary())


class TestReleaseWorkflow(unittest.TestCase):
    """The workflow and packaging files must stay consistent with the code."""

    WORKFLOW = os.path.join(".github", "workflows", "release.yml")

    def test_workflow_exists_and_triggers_on_version_tags(self) -> None:
        self.assertTrue(os.path.isfile(self.WORKFLOW))
        with open(self.WORKFLOW, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("v*", text)
        self.assertIn("workflow_dispatch", text)

    def test_workflow_publishes_what_the_updater_looks_for(self) -> None:
        with open(self.WORKFLOW, encoding="utf-8") as handle:
            text = handle.read()
        # The updater needs a checksum file and per-platform installers.
        self.assertIn("SHA256SUMS.txt", text)
        for fragment in ("-setup.exe", ".dmg", ".tar.gz"):
            self.assertIn(fragment, text)

    def test_workflow_stamps_the_version_from_the_tag(self) -> None:
        with open(self.WORKFLOW, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("stamp_version.py", text)
        self.assertTrue(os.path.isfile(os.path.join("packaging", "stamp_version.py")))

    def test_pyinstaller_spec_is_present(self) -> None:
        self.assertTrue(os.path.isfile(os.path.join("packaging", "sif_console.spec")))
        self.assertTrue(os.path.isfile(os.path.join("packaging", "installer.iss")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
