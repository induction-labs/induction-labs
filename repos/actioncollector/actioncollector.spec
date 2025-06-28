# actioncollector.spec
# PyInstaller ≥6.0
# Build command:  pyinstaller --clean --noconfirm actioncollector.spec

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

root         = Path(os.getcwd())
src_dir      = root / "src"
ffmpeg_src   = root / "bin" / ("ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg")

# ---- 1. Main entry-point ----------------------------------------------------
# If you prefer to start via `actioncollector.main:app`, just
# change  'src/actioncollector/controller.py'  →  '-m', 'actioncollector'
entry_script = str(src_dir / "actioncollector" / "controller.py")

# ---- 2. Extra binaries / data ----------------------------------------------
binaries = [
    # (<source>, <dest-dir-inside-dist>)
    (str(ffmpeg_src), "."),      # puts the ffmpeg executable beside your app
]

# matplotlib, gooey, etc. already have official hooks, but
# Gooey ships a “languages” folder that isn’t picked up automatically.
datas = collect_data_files("gooey") + collect_data_files("matplotlib") \
    + [(root / "service-account-key.json", "credentials")] \
    + [(root / "assets" / "*", "assets")] \

# ---- 3. Hidden imports ------------------------------------------------------
hiddenimports  = (
    collect_submodules("gooey")       # ensures Gooey’s runtime plugins are seen
    + collect_submodules("google")    # gcloud meta-packages sometimes confuse PyInstaller
)

# ---- 4. Analysis ------------------------------------------------------------
a = Analysis(
    [entry_script],
    pathex=[str(src_dir)],            # makes     import actioncollector     resolve
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],                     # add custom hooks here if you write any
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

# ---- 5. Build steps ---------------------------------------------------------
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

gui_exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,            # binaries handled by COLLECT
    name="ActionCollector",           # executable name
    console=False,                    # GUI app for proper .app bundle behavior
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                         # disable on macOS arm64 if codesigning
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    gui_exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ActionCollector",
)

app = BUNDLE(
    coll,
    name='ActionCollector.app',
    icon=root / "assets" / "logo.icns",
    bundle_identifier='com.inductionlabs.actioncollector',
    info_plist={
        'CFBundleName': 'ActionCollector',
        'CFBundleDisplayName': 'ActionCollector',
        'CFBundleIdentifier': 'com.inductionlabs.actioncollector',
        'CFBundleVersion': '0.1.0',
        'CFBundleShortVersionString': '0.1.0',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,
        "LSUIElement": True,
        'NSCameraUsageDescription': 'ActionCollector needs camera access for screen recording',
        'NSMicrophoneUsageDescription': 'ActionCollector needs microphone access for audio recording',
        'NSScreenCaptureDescription': 'ActionCollector needs screen capture permission to record screen activity',
    },
)
