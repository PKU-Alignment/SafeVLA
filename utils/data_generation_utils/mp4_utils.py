import os
from typing import Sequence, Optional, Dict, Any

import imageio
import numpy as np


def save_frames_to_mp4(
    frames: Sequence[np.ndarray],
    file_path: str,
    fps: float,
    extra_kwargs: Optional[Dict[str, Any]] = None,
):
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    if not isinstance(frames, np.ndarray):
        frames = np.array(frames)

    # Ensure frames are contiguous in memory to avoid alignment warnings
    # This creates a properly aligned copy if needed
    frames = np.ascontiguousarray(frames, dtype=np.uint8)

    kwargs = {
        "fps": fps,
        "quality": 5,
        "codec": "libx264",  # Explicitly specify codec
        "pixelformat": "yuv420p",  # Standard pixel format
        **(extra_kwargs if extra_kwargs is not None else {}),
    }
    imageio.mimwrite(file_path, frames, macro_block_size=1, **kwargs)
