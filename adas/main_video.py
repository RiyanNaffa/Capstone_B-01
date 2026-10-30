#!/usr/bin/env python3
"""Entry point for video replay and endurance testing."""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import math
import os
import signal
import sys
from typing import Optional

from adas.constants import LOG_DIR, RECORD_DIR
from adas.pipeline_video import VideoADASPipeline


def setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(sh)

    fh = logging.handlers.RotatingFileHandler(
        filename=os.path.join(LOG_DIR, "adas_video_error.log"),
        maxBytes=1 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8",
    )
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    ))
    root.addHandler(fh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADAS Bus Kampus UGM — replay video dan endurance test"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", "-v", help="Path ke satu video raw")
    source.add_argument(
        "--all",
        action="store_true",
        help="Putar semua file *_raw.avi pada direktori recordings",
    )
    parser.add_argument(
        "--directory",
        default=RECORD_DIR,
        help=f"Direktori sumber video untuk --all (default: {RECORD_DIR})",
    )
    parser.add_argument(
        "--speed",
        "-s",
        type=float,
        default=1.0,
        help="Kecepatan playback: 1=realtime, 2=2x, 0=secepat mungkin",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        help="Ulang sumber video sampai durasi ini tercapai (mis. 4)",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Simpan video bbox (kelas + confidence); hanya untuk pemutaran satu pass",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(RECORD_DIR, "pipeline_video"),
        help="Direktori output video annotated",
    )
    args = parser.parse_args()

    if not math.isfinite(args.speed) or args.speed < 0:
        parser.error("--speed harus finite dan tidak boleh negatif")
    if args.duration_hours is not None and (
        not math.isfinite(args.duration_hours) or args.duration_hours <= 0
    ):
        parser.error("--duration-hours harus lebih besar dari 0")
    if args.annotate and args.duration_hours is not None:
        parser.error("--annotate tidak dapat digabung dengan --duration-hours")

    if args.all:
        if not os.path.isdir(args.directory):
            parser.error(f"Direktori video tidak ditemukan: {args.directory}")
        video_paths = sorted(
            os.path.join(args.directory, name)
            for name in os.listdir(args.directory)
            if name.lower().endswith("_raw.avi")
            and os.path.isfile(os.path.join(args.directory, name))
        )
        if not video_paths:
            parser.error(f"Tidak ada file *_raw.avi di {args.directory}")
    else:
        if not os.path.isfile(args.video):
            parser.error(f"Video tidak ditemukan: {args.video}")
        video_paths = [args.video]

    setup_logging()
    log = logging.getLogger("main_video")
    duration_seconds: Optional[float] = (
        args.duration_hours * 3600 if args.duration_hours is not None else None
    )
    pipeline = VideoADASPipeline(
        video_path=video_paths,
        playback_speed=args.speed,
        duration_seconds=duration_seconds,
        annotated_dir=args.output_dir if args.annotate else None,
    )

    def _handle_signal(signum, _frame) -> None:
        log.info("Signal %d diterima — menghentikan...", signum)
        pipeline.stop()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        log.info("Input: %s", ", ".join(video_paths))
        if duration_seconds is not None:
            log.info("Mode endurance: %.1f jam", args.duration_hours)
        pipeline.run()
    except Exception as exc:
        log.critical("Pipeline video gagal: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
