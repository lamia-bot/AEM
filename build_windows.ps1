$ErrorActionPreference = "Stop"
python -m pip install -r requirements-build.txt
python -m compileall -q app.py mea_analyzer
pyinstaller --noconfirm --clean MEA_Analyzer_Build22.spec
Write-Host "Built: dist\MEA_Analyzer\MEA_Analyzer.exe"
