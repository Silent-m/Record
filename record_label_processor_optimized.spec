# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['record_label_processor_optimized.py'],
    pathex=[],
    binaries=[('D:\\Coding\\Lev projects\\Record\\dist\\tcl86t.dll', '.'), ('D:\\Coding\\Lev projects\\Record\\dist\\tk86t.dll', '.')],
    datas=[],
    hiddenimports=['tkinter'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'scipy'],
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
    name='record_label_processor_optimized',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
