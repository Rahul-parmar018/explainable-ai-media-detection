from pathlib import Path
from typing import Dict, Any, Optional
from src.video.predict_video import predict_video, RESEARCH_PROTOTYPE_DISCLAIMER

class VideoInferenceService:
    """
    Service class wrapping the end-to-end video inference and explainability pipeline.
    Designed for seamless integration with web services and REST APIs (e.g. FastAPI / Flask).
    """

    def __init__(
        self,
        model_path: str = "data/videos/models/best_model.joblib",
        scaler_path: str = "data/videos/models/scaler.joblib",
        default_n_frames: int = 10
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.default_n_frames = default_n_frames

    def analyze(self, video_path: str, n_frames: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyzes an input video stream and returns a structured JSON-serializable result.

        Args:
            video_path (str): Path to input MP4 video file.
            n_frames (Optional[int]): Number of frames to sample (defaults to service setting).

        Returns:
            Dict[str, Any]: Structured result dictionary.
        """
        num_f = n_frames if n_frames is not None else self.default_n_frames
        return predict_video(
            video_path=video_path,
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            n_frames=num_f
        )
