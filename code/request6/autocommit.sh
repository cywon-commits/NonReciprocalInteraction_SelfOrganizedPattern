#!/bin/sh
# commit and push results_request6 every 20 minutes (branch b1/request6)
cd "$(dirname "$0")/../.."
while true; do
  sleep 1200
  git add results_request6 code/request6 >/dev/null 2>&1
  git commit -qm "Request 6: autosave results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Cr2GsgGF5EnBPWfYmErUDZ" >/dev/null 2>&1 && git push -q origin b1/request6 >/dev/null 2>&1
done
