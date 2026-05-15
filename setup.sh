#!/usr/bin/env bash
set -e

echo "=== SAP Developer Guidelines MCP Server — Setup (macOS / Linux) ==="
echo

# Python pruefen
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 nicht gefunden. Bitte Python 3.10+ installieren."
    exit 1
fi

# Virtuelles Environment erstellen
if [ ! -d ".venv" ]; then
    echo "[1/3] Erstelle virtuelles Environment..."
    python3 -m venv .venv
else
    echo "[1/3] Virtuelles Environment bereits vorhanden."
fi

# Pakete installieren
echo "[2/3] Installiere Abhaengigkeiten (kann einige Minuten dauern)..."
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install -r requirements.txt --quiet

echo "[3/3] Installation abgeschlossen!"
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PATH="$SCRIPT_DIR/.venv/bin/python"
SERVER_PATH="$SCRIPT_DIR/server.py"

echo "============================================================"
echo " Claude Desktop Konfiguration (claude_desktop_config.json):"
echo " macOS: ~/Library/Application Support/Claude/claude_desktop_config.json"
echo " Windows: %APPDATA%\Claude\claude_desktop_config.json"
echo "============================================================"
cat <<EOF
{
  "mcpServers": {
    "sap-dev-guidelines": {
      "command": "$PYTHON_PATH",
      "args": ["$SERVER_PATH"]
    }
  }
}
EOF
echo "============================================================"
echo
echo "Danach Claude Desktop neu starten."
