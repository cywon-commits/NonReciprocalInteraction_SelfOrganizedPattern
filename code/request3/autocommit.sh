#!/bin/sh
# Commits and pushes new request-3 result files every 20 minutes while the runs go on.
cd "$(dirname "$0")/../.."
while true; do
  sleep 1200
  git add -A results_request3 >/dev/null 2>&1
  if ! git diff --cached --quiet; then
    git commit -qm "Request 3: intermediate chain results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Cr2GsgGF5EnBPWfYmErUDZ" && git push -q origin claude/loving-cray-7g4ju9
  fi
done
