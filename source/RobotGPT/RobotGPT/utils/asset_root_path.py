"""
Copyright (c) 2026 ronypepper.

License: Apache 2.0
"""

import pathlib

ROBOTGPT_ASSETS_RELATIVE_PATH = "~/RobotLearning/robotgpt_assets"

ROBOTGPT_ASSETS_PATH = pathlib.Path(ROBOTGPT_ASSETS_RELATIVE_PATH).expanduser().resolve()
