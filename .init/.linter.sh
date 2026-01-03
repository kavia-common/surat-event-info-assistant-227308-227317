#!/bin/bash
cd /home/kavia/workspace/code-generation/surat-event-info-assistant-227308-227317/fastapi_langgraph_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

