#!/usr/bin/env bash
# README media from a recorder video (cornichon_rl.record): an animated WebP at 2x speed (shows inline on GitHub)
# and a smaller real-time mp4.
#   scripts/make_media.sh videos/cornichon_agent_d4_seed1000003.mp4 docs/media/agent_d4
set -euo pipefail
IN=$1
OUT=$2
mkdir -p "$(dirname "$OUT")"
ffmpeg -v error -y -i "$IN" -vf "setpts=PTS/2,fps=15,scale=640:-1" -c:v libwebp -q:v 50 -loop 0 -an "$OUT.webp"
ffmpeg -v error -y -i "$IN" -vf "scale=960:-1" -c:v libx264 -crf 28 -preset slow -pix_fmt yuv420p -movflags +faststart -an "$OUT.mp4"
ls -la "$OUT.webp" "$OUT.mp4"
