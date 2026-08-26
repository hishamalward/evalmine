#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/setup-browser-qa.sh

Installs a pinned Playwright MCP server in Evalmine's ignored tooling directory,
registers it with Codex in existing-browser extension mode, and opens the official
Playwright MCP Bridge extension page. Start a new Codex session after setup.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  "")
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
tool_root="$repo_root/.evalmine-tools"
mcp_root="$tool_root/playwright-mcp"
mcp_bin="$mcp_root/node_modules/.bin/playwright-mcp"
mcp_output="$tool_root/playwright-mcp-output"
extension_url="https://chromewebstore.google.com/detail/playwright-mcp-bridge/mmlmfjhmonkocbjadbfplnigmagldckm"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to install Playwright MCP." >&2
  exit 1
fi

if ! command -v codex >/dev/null 2>&1; then
  echo "The Codex CLI is required to register Playwright MCP." >&2
  exit 1
fi

mkdir -p "$tool_root/npm-cache" "$mcp_output"

echo "Installing pinned Playwright MCP 0.0.78 in Evalmine's ignored tooling directory..."
npm install \
  --prefix "$mcp_root" \
  --cache "$tool_root/npm-cache" \
  @playwright/mcp@0.0.78 \
  --save-exact

if [[ ! -x "$mcp_bin" ]]; then
  echo "Playwright MCP was installed, but its executable is missing: $mcp_bin" >&2
  exit 1
fi

if codex mcp get playwright >/dev/null 2>&1; then
  echo "Codex already has an MCP server named 'playwright'; leaving it unchanged."
  codex mcp get playwright
else
  echo "Registering Playwright MCP with Codex in existing-browser extension mode..."
  codex mcp add playwright -- \
    "$mcp_bin" \
    --extension \
    --caps vision \
    --output-dir "$mcp_output" \
    --output-max-size 104857600
fi

echo
echo "Playwright MCP is ready. Install/enable the official Playwright MCP Bridge"
echo "extension, then start a new Codex session so the new MCP tools are loaded:"
echo "$extension_url"

if command -v open >/dev/null 2>&1; then
  open "$extension_url"
fi
