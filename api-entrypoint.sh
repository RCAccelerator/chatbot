#!/bin/sh
set -e

if [ -n "$KRB_USER" ] && [ -n "$KRB_PASS" ]; then
  echo "Initializing Kerberos ticket..."
  echo "$KRB_PASS" | kinit "$KRB_USER"
fi

exec uvicorn rca_accelerator_chatbot.api:app --host 0.0.0.0 --port 8001
