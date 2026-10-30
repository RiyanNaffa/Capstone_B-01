#!/usr/bin/env python3
"""Extract one evenly spaced frame per second from each raw recording."""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import sys
from typing import List

import cv2

from adas.constants import RECORD_DIR

log = logging.getLogger("sample_video_frames")


def sample_video(video_path: str, output_dir: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Gagal membuka video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError(f"FPS video tidak valid: {video_path}")

    os.makedirs(output_dir, exist_ok=True)
    if os.listdir(output_dir):
        cap.release()
        raise FileExistsError(
            f"Direktori output tidak kosong: {output_dir}. "
            "Pindahkan sampel lama sebelum menjalankan ulang."
        )

    frame_index = 0
    next_sample_frame = 0
    sample_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index >= next_sample_frame:
                sample_path = os.path.join(
                    output_dir, f"frame_{sample_count:06d}.jpg"
                )
                if not cv2.imwrite(sample_path, frame):
                    raise OSError(f"Gagal menyimpan frame: {sample_path}")
                sample_count += 1
                next_sample_frame = max(frame_index + 1, round(sample_count * fps))
            frame_index += 1
    finally:
        cap.release()

    if sample_count == 0:
        raise RuntimeError(f"Tidak ada frame yang dapat dibaca: {video_path}")
    return sample_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ambil satu sampel frame per detik dari semua video raw"
    )
    parser.add_argument(
        "--recordings-dir",
        default=RECORD_DIR,
        help=f"Direktori sumber rekaman (default: {RECORD_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        help="Direktori root output (default: <recordings-dir>/frame_samples)",
    )
    args = parser.parse_args()
    output_root = args.output_dir or os.path.join(
        args.recordings_dir, "frame_samples"
    )
    if not os.path.isdir(args.recordings_dir):
        parser.error(f"Direktori rekaman tidak ditemukan: {args.recordings_dir}")

    videos: List[str] = sorted(
        os.path.join(args.recordings_dir, name)
        for name in os.listdir(args.recordings_dir)
        if name.lower().endswith("_raw.avi")
        and os.path.isfile(os.path.join(args.recordings_dir, name))
    )
    if not videos:
        parser.error(f"Tidak ada file *_raw.avi di {args.recordings_dir}")

    failures = 0
    for video_path in videos:
        stem = re.sub(r"_raw\.avi$", "", os.path.basename(video_path), flags=re.I)
        output_dir = os.path.join(output_root, stem)
        try:
            count = sample_video(video_path, output_dir)
            print(f"{os.path.basename(video_path)}: {count} frame -> {output_dir}")
        except (OSError, RuntimeError) as exc:
            log.error("%s", exc)
            failures += 1

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
