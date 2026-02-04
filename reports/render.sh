#!/bin/bash
# Render a markdown file to styled HTML and open in browser
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <markdown-file> [output-file]" >&2
    exit 1
fi

input="$1"
output="${2:-/tmp/rendered.html}"
dir="$(cd "$(dirname "$0")" && pwd)"

pandoc -s \
    -H <(printf '<style>\n%s\n</style>' "$(cat "$dir/style.css")") \
    --mathjax=https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js \
    --metadata title=" " \
    "$input" -o "$output"

echo "$output"
xdg-open "$output" 2>/dev/null || open "$output" 2>/dev/null || echo "Open $output in your browser"
