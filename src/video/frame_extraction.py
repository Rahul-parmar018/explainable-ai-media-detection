import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

def sample_video_frames(
    video_path: str,
    num_frames: int = 10
) -> Dict[str, Any]:
    """
    Uniformly samples a fixed number of frames from a video file using OpenCV.

    Args:
        video_path (str): Path to the target video file.
        num_frames (int): Number of frames to uniformly sample (default: 10).

    Returns:
        Dict[str, Any]: Dictionary containing:
            - video_path (str): Path to the video file
            - video_name (str): Basename of the video file
            - requested_frames (int): Target number of frames requested
            - total_frames (int): Total frame count of the video
            - sampled_indices (List[int]): Generated frame index positions
            - sampled_frames (List[np.ndarray]): List of frame arrays kept in memory
            - actual_sampled_count (int): Number of frames successfully read
            - frame_shape (Optional[Tuple[int, int, int]]): (Height, Width, Channels) of frames
            - readable (bool): Whether the video stream could be opened
            - error (Optional[str]): Error message if opening or reading failed
    """
    path_obj = Path(video_path)
    if not path_obj.exists():
        return {
            "video_path": str(video_path),
            "video_name": path_obj.name,
            "requested_frames": num_frames,
            "total_frames": 0,
            "sampled_indices": [],
            "sampled_frames": [],
            "actual_sampled_count": 0,
            "frame_shape": None,
            "readable": False,
            "error": "File does not exist"
        }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return {
            "video_path": str(video_path),
            "video_name": path_obj.name,
            "requested_frames": num_frames,
            "total_frames": 0,
            "sampled_indices": [],
            "sampled_frames": [],
            "actual_sampled_count": 0,
            "frame_shape": None,
            "readable": False,
            "error": "Failed to open video stream"
        }

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return {
                "video_path": str(video_path),
                "video_name": path_obj.name,
                "requested_frames": num_frames,
                "total_frames": 0,
                "sampled_indices": [],
                "sampled_frames": [],
                "actual_sampled_count": 0,
                "frame_shape": None,
                "readable": True,
                "error": "Video has 0 frames"
            }

        # Generate evenly spaced frame indices
        if total_frames <= num_frames:
            sampled_indices = list(range(total_frames))
        else:
            sampled_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

        sampled_frames = []
        frame_shape = None

        for idx in sampled_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                sampled_frames.append(frame)
                if frame_shape is None:
                    frame_shape = frame.shape
            else:
                # Handle unreadable frame gracefully
                pass

        actual_sampled_count = len(sampled_frames)

        return {
            "video_path": str(video_path),
            "video_name": path_obj.name,
            "requested_frames": num_frames,
            "total_frames": total_frames,
            "sampled_indices": sampled_indices,
            "sampled_frames": sampled_frames,
            "actual_sampled_count": actual_sampled_count,
            "frame_shape": frame_shape,
            "readable": True,
            "error": None
        }
    finally:
        # Ensure VideoCapture handle is always safely released
        cap.release()
