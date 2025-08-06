#!/bin/bash
# Cleanup script for fullyautobob8.1
# Removes empty folders and __pycache__
# Keeps only essential files and static/

set -e

# Remove all __pycache__ folders
find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove all empty directories except .git, .github, .venv, static
find . -type d -empty \
    ! -path "./.git" \
    ! -path "./.github" \
    ! -path "./.venv" \
    ! -path "./static" \
    ! -path "." \
    -exec rmdir {} +

# Print remaining structure
if command -v tree >/dev/null; then
    tree -L 2 -I '__pycache__|*.pyc|.git'
else
    find . -maxdepth 2
fi

echo "\n✅ Cleanup complete. Only essential files and static/ remain."
