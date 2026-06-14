"""
Helper function for parsing images from a LeRobot dataset.

Sourced from the "src/openpi/policies/libero_policy.py" script of the openpi repository.

License: Apache 2.0
"""

import einops
import numpy as np


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image
