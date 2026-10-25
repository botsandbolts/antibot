#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
# Codespaces reruns postStartCommand whenever the environment resumes.
# Only start a second server if nothing is listening on port 8501.
if python - <<'PY'
import socket
with socket.socket() as s:
    s.settimeout(.3)
    exit_code = s.connect_ex(("127.0.0.1", 8501))
raise SystemExit(0 if exit_code == 0 else 1)
PY
then
  echo "AntiBot Lab is already serving on port 8501."
  exit 0
fi
nohup python -m streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true \
  > /tmp/antibot-streamlit.log 2>&1 < /dev/null &
echo $! > /tmp/antibot-streamlit.pid
echo "Started AntiBot Lab (PID $(cat /tmp/antibot-streamlit.pid)); logs: /tmp/antibot-streamlit.log"
