#!/usr/bin/env bash
# Rebuild the headless simulator and swap it in without disturbing running trainings.
# Running Java processes keep their open jars; new processes (each evaluation starts some) get the new build.
# The new build must answer both protocol formats before it replaces the live one.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export GRADLE_USER_HOME=${GRADLE_USER_HOME:-/data/local/berke/cache/gradle}
./gradlew :core:jar :headless:jar :headless:installDist --no-daemon -q -x :headless:installDist 2>&1 | grep -v "warning: \[options\]\|^Note:\|^[0-9] warning" || true
I=headless/build/install
NEW=$I/headless.new
rm -rf "$NEW"
if [ -d "$I/headless" ]; then cp -r "$I/headless" "$NEW"; else ./gradlew :headless:installDist --no-daemon -q && exit 0; fi
cp headless/build/libs/headless-1.0.jar core/build/libs/core-1.0.jar "$NEW/lib/"
for flags in "--grid-bytes=2" ""; do
  reply=$(printf '%s\n' '{"op":"reset","ids":[0],"seeds":[3],"difficulties":[1]}' '{"op":"close"}' | timeout 60 "$NEW/bin/headless" --envs=1 $flags 2>/dev/null | sed -n 2p)
  case "$reply" in *'"envs"'*) ;; *) echo "new simulator failed ($flags): $reply"; rm -rf "$NEW"; exit 1 ;; esac
done
mv "$I/headless" "$I/headless.old-$(date +%s)"
mv "$NEW" "$I/headless"
ls -d $I/headless.old-* 2>/dev/null | sort | head -n -2 | xargs -r rm -rf  # keep the two most recent old builds
echo "simulator installed: $I/headless"
