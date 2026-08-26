#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/render-report-webkit.sh [HTML_FILE] [OUTPUT_DIRECTORY]

Renders full-page desktop and mobile PNGs with Playwright WebKit. Defaults to
the Evalmine v2 planning report and reports/browser-qa.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
browser_root="$repo_root/.evalmine-tools/ms-playwright"

input_path="${1:-$repo_root/docs/plans/evalmine-v2-planning-report.html}"
output_dir="${2:-$repo_root/reports/browser-qa}"

if ! command -v playwright >/dev/null 2>&1; then
  echo "The Playwright CLI is required. Install it, then rerun this script." >&2
  exit 1
fi

if [[ ! -f "$input_path" ]]; then
  echo "HTML report not found: $input_path" >&2
  exit 1
fi

webkit_path="$(
  PLAYWRIGHT_BROWSERS_PATH="$browser_root" playwright install --dry-run webkit \
    | awk -F ': +' '/Install location:/ { print $2; exit }'
)"

if [[ -z "$webkit_path" ]]; then
  echo "Could not determine the WebKit install location." >&2
  exit 1
fi

if [[ ! -x "$webkit_path/pw_run.sh" ]]; then
  echo "Installing WebKit in Evalmine's ignored tooling directory..."
  PLAYWRIGHT_BROWSERS_PATH="$browser_root" playwright install webkit
fi

mkdir -p "$output_dir"

input_dir="$(cd "$(dirname "$input_path")" && pwd -P)"
input_file="$input_dir/$(basename "$input_path")"
input_url="file://$input_file"

desktop_path="$output_dir/report-desktop.png"
mobile_path="$output_dir/report-mobile.png"

echo "Rendering desktop screenshot with WebKit..."
PLAYWRIGHT_BROWSERS_PATH="$browser_root" playwright screenshot \
  --browser webkit \
  --viewport-size "1440,1000" \
  --full-page \
  "$input_url" \
  "$desktop_path"

echo "Rendering mobile screenshot with WebKit..."
PLAYWRIGHT_BROWSERS_PATH="$browser_root" playwright screenshot \
  --browser webkit \
  --viewport-size "390,844" \
  --full-page \
  "$input_url" \
  "$mobile_path"

assert_width() {
  local image_path="$1"
  local expected_width="$2"
  local label="$3"
  local actual_width

  actual_width="$(sips -g pixelWidth "$image_path" | awk '/pixelWidth:/ { print $2; exit }')"
  if [[ "$actual_width" != "$expected_width" ]]; then
    echo "$label screenshot is ${actual_width}px wide; expected ${expected_width}px." >&2
    echo "The page has horizontal overflow. Inspect the report CSS before accepting it." >&2
    exit 1
  fi
}

assert_width "$desktop_path" 1440 "Desktop"
assert_width "$mobile_path" 390 "Mobile"

echo
echo "Created:"
echo "  $desktop_path"
echo "  $mobile_path"
