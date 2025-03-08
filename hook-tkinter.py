from PyInstaller.utils.hooks import collect_dynamic_libs

# Exclude wish86t.exe from UPX compression
binaries = collect_dynamic_libs("tkinter")
binaries = [b for b in binaries if not b[0].endswith("wish86t.exe")]

# Add the filtered binaries to the build
datas = []
hiddenimports = []