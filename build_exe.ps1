$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

py -3 -m pip install "pyinstaller==6.11.1"
py -3 tools\generate_icon.py
py -3 -m PyInstaller --noconfirm --clean GroupPhotoOptimizer.spec

Write-Host "Built: $projectRoot\dist\GroupPhotoOptimizer.exe"
