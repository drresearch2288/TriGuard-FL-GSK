# One-time setup on Windows (PowerShell, from the repository root). Needs Python 3.11 and git on PATH.
$ErrorActionPreference = "Stop"
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -r requirements.txt
# FLamby (Fed-ISIC2019 split and Fed-Heart-Disease), pinned to the commit used for the reported results
if (-not (Test-Path third_party\FLamby)) { git clone https://github.com/owkin/FLamby.git third_party\FLamby }
git -C third_party\FLamby checkout edacf54d5211520583b0133d55ac39b6fda8324b
Write-Host "Setup done. Next: see README.md, section 'Reproduce'."
