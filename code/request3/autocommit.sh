#!/bin/sh
# Commits and pushes new request-3/4 result files every 20 minutes while the runs go on.
cd "$(dirname "$0")/../.."
while true; do
  sleep 1200
  git add -A results_request3 results_request4 >/dev/null 2>&1
  if ! git diff --cached --quiet; then
    git commit -qm "Request 3/4: intermediate results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Cr2GsgGF5EnBPWfYmErUDZ" && git push -q origin claude/loving-cray-7g4ju9
  fi
done
