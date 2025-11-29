"""
Analyze a video file to detect shots and compute simple metadata.

Usage:
    python analyze_video.py --input input.mp4 --output shots.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np


@dataclass
class ShotStats:
    start: float
    end: float
    avg_brightness: float
    motion: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect shots and compute simple metadata")
    parser.add_argument("--input", required=True, help="Path to the input video file")
    parser.add_argument("--output", required=True, help="Path to the output JSON file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Histogram difference threshold for shot changes (0-1).",
    )
    parser.add_argument(
        "--min-shot-length",
        type=float,
        default=0.5,
        help="Minimum shot duration in seconds to avoid overly frequent cuts.",
    )
    return parser.parse_args()


def ensure_video_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input video does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Input path is not a file: {path}")


def compute_histogram(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    cv2.normalize(hist, hist)
    return hist


def frame_brightness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray) / 255.0)


def frame_motion(frame: np.ndarray, prev_frame: np.ndarray | None) -> float:
    if prev_frame is None:
        return 0.0
    diff = cv2.absdiff(frame, prev_frame)
    return float(np.mean(diff) / 255.0)


def detect_shots(
    video_path: Path, threshold: float, min_shot_length: float
) -> List[ShotStats]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Could not read FPS from video.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if frame_count > 0 else None

    shots: List[ShotStats] = []
    shot_start_time = 0.0
    prev_frame: np.ndarray | None = None
    prev_hist: np.ndarray | None = None

    brightness_accumulator = 0.0
    motion_accumulator = 0.0
    frames_in_shot = 0

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_index / fps
        hist = compute_histogram(frame)
        hist_diff = 1.0 - cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) if prev_hist is not None else 0.0

        brightness_accumulator += frame_brightness(frame)
        motion_accumulator += frame_motion(frame, prev_frame)
        frames_in_shot += 1

        shot_duration = current_time - shot_start_time
        should_cut = hist_diff > threshold and shot_duration >= min_shot_length

        if should_cut:
            shots.append(
                ShotStats(
                    start=shot_start_time,
                    end=current_time,
                    avg_brightness=brightness_accumulator / max(frames_in_shot, 1),
                    motion=motion_accumulator / max(frames_in_shot, 1),
                )
            )
            shot_start_time = current_time
            brightness_accumulator = 0.0
            motion_accumulator = 0.0
            frames_in_shot = 0

        prev_frame = frame
        prev_hist = hist
        frame_index += 1

    cap.release()

    if frames_in_shot > 0:
        end_time = duration if duration is not None else shot_start_time
        shots.append(
            ShotStats(
                start=shot_start_time,
                end=end_time,
                avg_brightness=brightness_accumulator / max(frames_in_shot, 1),
                motion=motion_accumulator / max(frames_in_shot, 1),
            )
        )

    return shots


def save_shots(shots: List[ShotStats], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [shot.__dict__ for shot in shots]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def analyze_video(input_path: Path, output_path: Path, threshold: float, min_shot_length: float) -> None:
    ensure_video_exists(input_path)
    shots = detect_shots(input_path, threshold=threshold, min_shot_length=min_shot_length)
    save_shots(shots, output_path)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    try:
        analyze_video(input_path, output_path, args.threshold, args.min_shot_length)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to analyze video: {exc}") from exc


if __name__ == "__main__":
    main()
