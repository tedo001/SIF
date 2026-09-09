# Releasing the SIF Insight Console

One command publishes a release; everything else is automatic.

```bash
git tag -a v2.1.0 -m "Tamil OCR fixes, faster hotspots"
git push origin v2.1.0
```

That tag triggers `.github/workflows/release.yml`, which runs the test suites,
builds an installer for Windows, macOS and Linux, publishes a GitHub Release with
checksums, and makes it visible to every installed copy of the app.

---

## 1. Versioning — the tag is the only source of truth

The version lives in exactly one place in the code:

```python
# sif/version.py
__version__ = "2.0.0"
BUILD = ""            # CI stamps the short commit sha here
```

Before packaging, CI rewrites those two lines from the tag:

```bash
python packaging/stamp_version.py v2.1.0 --build 9f3a1c2
```

So four things always agree: the git tag, `sif.__version__`, the installer's
version metadata (`AppVersion` in Inno Setup, `CFBundleShortVersionString` on
macOS), and what the app reports under **Help → About**.

**Why this matters more than it looks.** If the app reports an older version than
the release it was installed from, the updater offers the same update forever —
a loop the operator cannot escape. Stamping from the tag makes that impossible.

Two ways to keep the committed file honest between releases:

| Situation | Command |
| --- | --- |
| Bump the version in a PR before tagging | edit `sif/version.py`, then `git tag v2.1.0` |
| Verify a tag matches the file (CI or a hook) | `python packaging/stamp_version.py v2.1.0 --check` |

`--check` exits non-zero on a mismatch and prints the fix, so it can guard a
release branch.

**Tag format:** `vMAJOR.MINOR.PATCH`, optionally `-rc.1` for a pre-release.
Anything else is rejected by the stamper before a build starts. Pre-releases sort
below their final version (`2.1.0-rc.2` < `2.1.0`), and the updater ignores them
unless a build opts in.

---

## 2. What the workflow does

| Job | Runs on | What it does |
| --- | --- | --- |
| `test` | ubuntu | Both suites, headless, offline encoder. **Nothing is built if these fail.** |
| `build` | windows · macos · ubuntu-22.04 | Stamps the version, runs PyInstaller, then Inno Setup / `hdiutil` / `tar` |
| `publish` | ubuntu | Collects every artifact, writes `SHA256SUMS.txt`, creates the Release with `gh` |

Assets published per release:

```
SIF-Console-2.1.0-setup.exe                Windows installer (Inno Setup)
SIF-Console-2.1.0.dmg                      macOS disk image
SIF-Console-2.1.0-linux-x86_64.tar.gz      Linux tarball
SHA256SUMS.txt                             checksums for all of the above
```

**Dry run before a real tag:** Actions → *release* → *Run workflow*, enter a tag
like `v0.0.0-test`. It builds and uploads artifacts but does **not** publish a
release (the `publish` job only runs for `refs/tags/v*`).

### The slim bundle, and why

The packaged app excludes torch, sentence-transformers, xgboost, mlflow and
paddleocr — together roughly 3 GB, most of it optional at run time. The installer
is ~200 MB and ships the interface, the deterministic rule engine and PDF
text-layer ingestion.

An operator who needs the semantic encoder, the learned model or scanned-page OCR
runs `pip install -r requirements.txt` once; the console detects each component at
start-up and the **Engines** page reports what is available. For an air-gapped
site that cannot pip-install afterwards, run the workflow manually with
`variant: full`.

### Prerequisites

None beyond the repository itself. `GITHUB_TOKEN` is provided automatically, and
the `contents: write` permission in the workflow is the only grant it needs. Code
signing is *not* configured — see "Not done yet" below.

---

## 3. How the app updates itself

`sif/updater.py`, wired into `main2.py`. Standard library only — no extra
dependency.

* **On start-up**, four seconds after the window appears, it asks GitHub for the
  latest release. If there is nothing newer, or no network, the operator sees
  nothing at all.
* **Help → Check for updates** does the same thing on demand, and reports "up to
  date" because the operator asked.
* When a newer version exists, a dialog shows the release notes and the download
  size, with **Download and install**, **Later**, and **Skip this version**
  (remembered in the per-user settings file).
* The download is **verified against `SHA256SUMS.txt` before anything runs**. A
  mismatch, or a release with no checksum file, is refused and the file deleted —
  an update path is a code-execution path.
* After verification the installer is launched and the console closes, because a
  running application cannot be overwritten.

Everything degrades: from a source checkout it says "update with `git pull`"
rather than installing over your working tree; with no asset for the platform it
says so; on a rate limit or an offline machine it logs and carries on.

**Before the first tag** the repository has no release, and GitHub answers the
`releases/latest` endpoint with 404. That is not a failure: the console reports
"No release has been published yet" and carries on. A 404 is only reported as a
problem when the repository itself cannot be seen — a wrong `SIF_UPDATE_REPO`, or
a private repository with no `SIF_UPDATE_TOKEN`. Publish a release (section 4)
and the check starts finding it.

Configuration, if a fork or an internal mirror is used:

```bash
SIF_UPDATE_REPO=myorg/SIF        # default: tedo001/SIF
SIF_UPDATE_TOKEN=ghp_...         # only for a private repository
SIF_UPDATE_API=https://github.mycorp/api/v3   # GitHub Enterprise
```

To disable the start-up check entirely, set `check_updates: false` in the
settings file (`%APPDATA%\SIF Insight Console\settings.json`, or
`~/.config/SIF Insight Console/settings.json`).

---

## 4. Release checklist

1. `python -m unittest discover -p "test_*.py"` — green locally.
2. Update `sif/version.py` to the version you are about to tag.
3. Note what changed; the tag message becomes the release notes' backbone.
4. `git tag -a vX.Y.Z -m "..."` and `git push origin vX.Y.Z`.
5. Watch the run in Actions. On failure the tag stays but nothing is published —
   fix, delete the tag (`git push --delete origin vX.Y.Z`), and tag again.
6. Download the installer from the Release and run it on a clean machine before
   telling anyone it exists.

---

## 5. Not done yet — read before shipping to a plant

* **No code signing.** Windows SmartScreen will warn on first run, and macOS
  Gatekeeper will refuse the `.dmg` until it is notarised. For a pilot this is
  tolerable; for wide deployment, buy an EV certificate (Windows) and an Apple
  Developer ID, then add `signtool` and `codesign`/`notarytool` steps to the
  build job. The updater's checksum verification is *integrity*, not
  *authenticity*: it proves the file arrived intact, not who built it.
* **The workflow has never run.** It is written against the documented behaviour
  of the runners; the first tag you push is also its first execution. Do that
  with a throwaway tag (`v0.0.1-rc.1`) rather than a version anyone will install.
* **macOS builds are arm64** on `macos-latest`. Add a second matrix entry on
  `macos-13` if Intel Macs need support.
* **Private repositories** need `SIF_UPDATE_TOKEN` on every client, which is a
  secret-distribution problem. Public releases avoid it entirely.
