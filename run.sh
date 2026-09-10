#!/usr/bin/env bash
# Double-click this file (or run it in a terminal) to start.
cd "$(dirname "$0")" || exit 1

echo
echo "==================================================================="
echo "  Starting up. The first time, this takes a few minutes while it"
echo "  downloads what it needs. After that it is quick."
echo "==================================================================="
echo

PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done

if [ -z "$PY" ]; then
  echo "  PYTHON IS NOT INSTALLED (or is too old)."
  echo
  echo "  Mac:   open the Terminal and type:  brew install python"
  echo "         or download it from https://www.python.org/downloads/"
  echo "  Linux: sudo apt install python3 python3-venv"
  echo
  read -r -p "Press Enter to close." _
  exit 1
fi

if [ ! -d .venv ]; then
  echo "Setting up (one time only)..."
  "$PY" -m venv .venv || { echo "Could not create the working folder."; read -r -p "Press Enter to close." _; exit 1; }
fi

echo "Installing what it needs (this is the slow part, one time only)..."
./.venv/bin/python -m pip install --quiet --upgrade pip
if ! ./.venv/bin/python -m pip install --quiet -r requirements.lock.txt; then
  echo
  echo "  The install did not finish. This is almost always the internet"
  echo "  connection. Check you are online and run this again."
  echo
  read -r -p "Press Enter to close." _
  exit 1
fi

echo
./.venv/bin/python run_all.py "$@"

echo
echo "==================================================================="
echo "  Finished. Read the message above."
echo "==================================================================="
echo
read -r -p "Press Enter to close." _
