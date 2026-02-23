#!/bin/bash
# Update ST 2026 Pitching Leaderboard
# Usage: ./update.sh

set -e
cd "$(dirname "$0")"

echo "=== ST 2026 Leaderboard Update ==="
echo ""

# 1. Pull latest data from Google Sheets and process
echo "→ Fetching data from Google Sheets..."
python3 process_data.py

# 2. Commit and push to GitHub Pages
echo ""
echo "→ Deploying to GitHub Pages..."
git add data/ index.html css/ js/ process_data.py update.sh
if git diff --cached --quiet; then
  echo "  No changes to deploy."
else
  git commit -m "Update leaderboard $(date '+%Y-%m-%d %H:%M')"
  git push origin main
  echo ""
  echo "=== Done! ==="
  echo "Site will update in ~30 seconds at:"
  echo "https://wjhuron.github.io/2026-Spring-Training-Pitch-Metrics-Leaderboard/"
fi
