# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for SENTRA.

Two variants, chosen with the ``SIF_BUILD_VARIANT`` environment variable:

``slim`` (default)
    Interface, deterministic rule engine and PDF text-layer ingestion. Roughly
    200 MB packaged. This is what CI publishes, because the heavy machine
    learning stack is ~3 GB and most of it is optional at run time.

``full``
    Adds torch, sentence-transformers, xgboost, mlflow and paddleocr. Only build
    this for an air-gapped site that cannot pip-install afterwards.

Build:  pyinstaller packaging/sif_console.spec --noconfirm
"""

import os
import sys
from PyInstaller.utils.hooks import collect_submodules

VARIANT = os.environ.get("SIF_BUILD_VARIANT", "slim").lower()
APP_NAME = "SENTRA"
#: The executable and one-folder bundle keep this stem. It is wired into
#: installer.iss (AppExeName), the release workflow's tar step and the published
#: asset names, so renaming it is a pipeline change rather than a label change -
#: and the pipeline has never run yet. The name an operator sees is APP_NAME.
EXECUTABLE = "SIFConsole"
ENTRY = "app2.py"          # build 2 is the shipped interface

hidden = [
    "sif", "sif.pipeline", "sif.lexical", "sif.encoders", "sif.heads", "sif.evidence",
    "sif.scoring", "sif.patterns", "sif.review", "sif.ocr", "sif.llm", "sif.mlops",
    "sif.logging_setup", "sif.updater", "sif.version",
    "ui", "ui.theme", "ui.charts", "ui.components", "ui.views",
    "ui2", "ui2.components", "ui2.views", "ui2.workflow",
    "main", "main2",
]

# Everything optional is excluded from the slim bundle; the console already
# detects each one at run time and says so on the Engines page.
OPTIONAL = ["torch", "transformers", "sentence_transformers", "xgboost", "sklearn",
            "scipy", "mlflow", "paddle", "paddleocr", "matplotlib", "pandas"]

excludes = list(OPTIONAL) if VARIANT == "slim" else ["matplotlib"]
if VARIANT == "full":
    for package in ("torch", "sentence_transformers", "xgboost", "mlflow"):
        try:
            hidden += collect_submodules(package)
        except Exception:                                  # not installed on this runner
            pass

datas = [("../ui/assets", "ui/assets"), ("../sample_reports.csv", ".")]

a = Analysis(
    [os.path.join("..", ENTRY)],
    pathex=[os.path.abspath("..")],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=EXECUTABLE,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                      # a desktop app, not a terminal tool
    icon=os.environ.get("SIF_ICON") or None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=EXECUTABLE,
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=os.environ.get("SIF_ICON") or None,
        bundle_identifier="in.co.oilindia.sifconsole",
        info_plist={
            "CFBundleShortVersionString": os.environ.get("SIF_VERSION", "0.0.0"),
            "CFBundleVersion": os.environ.get("SIF_VERSION", "0.0.0"),
            "NSHighResolutionCapable": True,
        },
    )
