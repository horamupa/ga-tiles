#!/usr/bin/env bash
# Cloudflare R2 for the CI workflows, through a pinned official rclone. Every bucket path starts at DEST: a prefix in a
# shared bucket is DEST=r2:shared/aviation-weather.
#
#   tools/r2.sh setup NAME      install rclone, write and read back _probe/NAME: a broken upload fails in seconds,
#                               with the requests and R2's answer in the log
#   tools/r2.sh pull DIR KEY…   the objects that exist, at their keys under DIR; a missing one is no error
#   tools/r2.sh upload SITE     the tiles, then the manifests, then sky/v1/models.json of the manifests in the bucket
#
# Needs R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY: an R2 API token with Object Read & Write on the bucket.
set -euo pipefail

DEST=r2:ga-tiles
RCLONE_VERSION=v1.75.2
RCLONE_SHA256=349ac8fba6ff65d6247043f1750cdcb518ec5d500ef91463a10d37c0ccdf3702  # linux-amd64.zip, release SHA256SUMS

export RCLONE_CONFIG_R2_TYPE=s3 RCLONE_CONFIG_R2_PROVIDER=Cloudflare
export RCLONE_CONFIG_R2_ENDPOINT=https://55cc1ebb03c9f23942476a22279e6df0.r2.cloudflarestorage.com
export RCLONE_CONFIG_R2_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID:?}"
export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY:?}"
export RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true  # the token cannot create buckets
export RCLONE_CONFIG_R2_NO_HEAD=true  # no HEAD after every PUT
# No listing and no HEAD of the destination: run folders are new, and a listing would walk every tile of every run.
# A retry of the whole copy would upload everything again; each request is retried on its own instead.
COPY=(--no-check-dest --retries 1 --transfers 64 --stats 1m --stats-one-line --stats-log-level NOTICE)
PNG='Cache-Control: public, max-age=31536000, immutable'
JSON='Cache-Control: public, max-age=60'

setup() {
  local name=rclone-$RCLONE_VERSION-linux-amd64
  curl -fsSL -o "$RUNNER_TEMP/rclone.zip" "https://downloads.rclone.org/$RCLONE_VERSION/$name.zip"
  echo "$RCLONE_SHA256  $RUNNER_TEMP/rclone.zip" | sha256sum -c -
  unzip -q -o -j "$RUNNER_TEMP/rclone.zip" "$name/rclone" -d "$RUNNER_TEMP"
  sudo install -m 755 "$RUNNER_TEMP/rclone" /usr/local/bin/rclone
  rclone version
  echo "run $GITHUB_RUN_ID at $(date -u +%FT%TZ)" > "$RUNNER_TEMP/probe.txt"
  rclone copyto "$RUNNER_TEMP/probe.txt" "$DEST/_probe/$1" "${COPY[@]}" --header-upload "$PNG" \
    --low-level-retries 1 --dump headers,responses  # implies -vv; the Authorization header is left out
  rclone cat "$DEST/_probe/$1" | cmp - "$RUNNER_TEMP/probe.txt"
}

pull() {
  local dir=$1
  shift
  printf '%s\n' "$@" | rclone copy "$DEST" "$dir" --files-from - --no-traverse --retries 1
}

upload() {
  local site=$1 bucket=$RUNNER_TEMP/bucket keys
  # The app sees a run only when its manifest arrives, after all its tiles.
  rclone copy "$site" "$DEST" --include '*.png' "${COPY[@]}" --header-upload "$PNG"
  # The site's own models.json lists what this job built alone.
  rclone copy "$site" "$DEST" --filter '- /sky/v1/models.json' --filter '+ *.{json,html}' --filter '- **' "${COPY[@]}" \
    --header-upload "$JSON"
  # Every job writes it from all manifests in the bucket. A manifest stays once there (until the lifecycle rule takes
  # a model dead for days), so two jobs that finish together write the same list; a model that appears for the first
  # time can miss the other job's list, and the next job adds it.
  keys=$(python -I -c 'from windtiles.models import MODELS; print(*(f"sky/v1/{m.id}/manifest.json" for m in MODELS))')
  pull "$bucket" $keys
  python -I -m windtiles.models "$bucket" > "$bucket/models.json"
  rclone copyto "$bucket/models.json" "$DEST/sky/v1/models.json" "${COPY[@]}" --header-upload "$JSON"
  cat "$bucket/models.json"
}

"$@"
