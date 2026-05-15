@echo off
setlocal EnableDelayedExpansion

echo === SAP Developer Guidelines MCP Server — Setup (Windows) ===
echo.

:: Python-Version pruefen
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python nicht gefunden. Bitte Python 3.10+ installieren: https://python.org
    exit /b 1
)

:: Virtuelles Environment erstellen
if not exist ".venv" (
    echo [1/3] Erstelle virtuelles Environment...
    python -m venv .venv
) else (
    echo [1/3] Virtuelles Environment bereits vorhanden.
)

:: Pakete installieren
echo [2/3] Installiere Abhaengigkeiten (kann einige Minuten dauern)...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\pip.exe install -r requirements.txt --quiet

if errorlevel 1 (
    echo [ERROR] Installation fehlgeschlagen.
    exit /b 1
)

:: Absoluten Pfad ermitteln
set "SERVER_DIR=%~dp0"
set "PYTHON_PATH=%SERVER_DIR%.venv\Scripts\python.exe"
set "SERVER_PATH=%SERVER_DIR%server.py"

echo [3/3] Installation abgeschlossen!
echo.
echo ============================================================
echo  Claude Desktop Konfiguration (claude_desktop_config.json):
echo  Pfad: %%APPDATA%%\Claude\claude_desktop_config.json
echo ============================================================
echo {
echo   "mcpServers": {
echo     "sap-dev-guidelines": {
echo       "command": "%PYTHON_PATH:\=\\%",
echo       "args": ["%SERVER_PATH:\=\\%"]
echo     }
echo   }
echo }
echo ============================================================
echo.
echo Danach Claude Desktop neu starten.
pause
