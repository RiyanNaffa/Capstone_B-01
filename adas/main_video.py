#!/usr/bin/env python3
"""
main_video.py — Entry point ADAS mode video file (offline/debug).

Cara pakai:
  python3 /opt/adas/main_video.py --video /opt/adas/recordings/20241020_raw.avi
  python3 /opt/adas/main_video.py --video rec.avi --speed 2.0
  python3 /opt/adas/main_video.py --video rec.avi --speed 0   # secepat mungkin
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import signal
import sys

from adas.constants import LOG_DIR
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
        filename    = os.path.join(LOG_DIR, "adas_video_error.log"),
        maxBytes    = 1 * 1024 * 1024,
        backupCount = 2,
        encoding    = "utf-8",
    )
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    ))
    root.addHandler(fh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADAS Bus Kampus UGM — Mode Video Offline"
    )
    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Path ke file video input (.avi, .mp4, dll)",
    )
    parser.add_argument(
        "--speed", "-s",
        type=float,
        default=1.0,
        help="Kecepatan playback: 1.0=realtime, 2.0=2x, 0=benchmark (default: 1.0)",
    )
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("main_video")

    log.info("=" * 50)
    log.info("ADAS Bus Kampus UGM — Mode Video")
    log.info("Input : %s", args.video)
    log.info("Speed : %.1fx", args.speed)
    log.info("=" * 50)

    pipeline = VideoADASPipeline(
        video_path     = args.video,
        playback_speed = args.speed,
    )

    def _handle_signal(signum, frame):
        log.info("Signal %d diterima — menghentikan...", signum)
        pipeline.stop()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    try:
        pipeline.run()
    except FileNotFoundError as e:
        log.critical("%s", e)
        sys.exit(1)
    except Exception as e:
        log.critical("Error fatal: %s", e, exc_info=True)
        sys.exit(1)

    log.info("Selesai.")


if __name__ == "__main__":
    main()
