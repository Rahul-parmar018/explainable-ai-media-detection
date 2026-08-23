import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2

# Mapping for binary classification
CATEGORY_LABEL_MAP = {
    "original": "REAL",
    "Deepfakes": "FAKE",
    "Face2Face": "FAKE",
    "FaceShifter": "FAKE",
    "FaceSwap": "FAKE",
    "NeuralTextures": "FAKE",
    "DeepFakeDetection": "FAKE"
}

def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Opens a video using OpenCV, extracts metadata properties,
    confirms readability, and releases the video handle.

    Args:
        video_path (str): Path to the target video file.

    Returns:
        Dict[str, Any]: Dictionary containing video metadata and readability status.
    """
    path_obj = Path(video_path)
    if not path_obj.exists():
        return {
            "video_path": str(video_path),
            "video_name": path_obj.name,
            "readable": False,
            "error": "File does not exist",
            "fps": 0.0,
            "frame_count": 0,
            "width": 0,
            "height": 0,
            "duration": 0.0
        }

    cap = cv2.VideoCapture(str(video_path))
    
    is_readable = cap.isOpened()
    if not is_readable:
        cap.release()
        return {
            "video_path": str(video_path),
            "video_name": path_obj.name,
            "readable": False,
            "error": "Failed to open video stream",
            "fps": 0.0,
            "frame_count": 0,
            "width": 0,
            "height": 0,
            "duration": 0.0
        }

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = round(frame_count / fps, 2) if fps > 0 else 0.0

    # Ensure video stream handle is properly released
    cap.release()

    return {
        "video_path": str(video_path),
        "video_name": path_obj.name,
        "readable": True,
        "error": None,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }


def get_sample_videos_info(
    dataset_dir: str = "Dataset/Video",
    samples_per_folder: int = 3
) -> List[Dict[str, Any]]:
    """
    Selects sample videos from each dataset folder and returns their OpenCV metadata.

    Args:
        dataset_dir (str): Base directory containing video folders.
        samples_per_folder (int): Number of sample videos to select per folder (default: 3).

    Returns:
        List[Dict[str, Any]]: List of metadata dictionaries for each sample video.
    """
    base_dir = Path(dataset_dir)
    results = []

    for folder_name, label in CATEGORY_LABEL_MAP.items():
        folder_path = base_dir / folder_name
        if not folder_path.exists() or not folder_path.is_dir():
            continue

        video_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(".mp4")
        ])

        selected_samples = video_files[:samples_per_folder]

        for vfile in selected_samples:
            vpath = folder_path / vfile
            info = get_video_info(str(vpath))
            info["folder"] = folder_name
            info["label"] = label
            results.append(info)

    return results
