# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).resolve().parents[1]
DEPLOY_DIR = ROOT / '05Deploy'
GUI_DIR = DEPLOY_DIR / 'gui'

datas = []
binaries = []
hiddenimports = []
tmp_ret = collect_all('snap7')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    [str(GUI_DIR / 'app.py')],
    pathex=[
        str(DEPLOY_DIR),
        str(GUI_DIR),
        str(DEPLOY_DIR / 'plc'),
        str(ROOT / '02predictModel' / 'src'),
        str(ROOT / '04EquipmentOptimization'),
    ],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    [],
    exclude_binaries=True,
    name='CoolingControl',
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
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CoolingControl',
)
