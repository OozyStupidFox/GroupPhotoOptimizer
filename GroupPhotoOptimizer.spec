# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all


root = Path(SPEC).resolve().parent
mediapipe_datas, mediapipe_binaries, mediapipe_hiddenimports = collect_all("mediapipe")
webview_datas, webview_binaries, webview_hiddenimports = collect_all("webview")
model_files = [
    (str(root / "models" / "face_detection_yunet_2023mar.onnx"), "models"),
    (str(root / "models" / "face_recognition_sface_2021dec.onnx"), "models"),
    (str(root / "models" / "face_landmarker.task"), "models"),
]

a = Analysis(
    [str(root / "launcher.py")],
    pathex=[str(root / "src")],
    binaries=mediapipe_binaries + webview_binaries,
    datas=(
        mediapipe_datas
        + webview_datas
        + model_files
        + [(str(root / "config.yaml"), "."), (str(root / "gui" / "index.html"), "gui")]
    ),
    hiddenimports=mediapipe_hiddenimports + webview_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="GroupPhotoOptimizer",
    icon=str(root / "assets" / "app_icon.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
