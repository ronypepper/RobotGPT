#!/usr/bin/env bash
# This script updates the RobotGPT repository from remote and subsequently the openpi repository with new files from
# the RobotGPT repository. Assumes openpi to be installed next to the RobotGPT repository.

# Change cwd to shell script directory regardless of where script was called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Update RobotGPT
echo "Updating RobotGPT from GitHub..."
cd ..
git pull

# Copy files to openpi repository
cd ..
echo "Confirm to remove \"openpi/src/robotgpt\" from local openpi repository and re-copy..."
rm -rI openpi/src/robotgpt
cp -r RobotGPT/openpi/robotgpt openpi/src
echo "Done."