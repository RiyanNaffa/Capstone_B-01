#!/usr/bin/env python3
"""Run the standard AMCAS pipeline with a laptop UDP video source."""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from adas.pipeline import ADASPipeline
from adas.udp_stream import UdpFrameCapture


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADAS pipeline input from laptop H.264/MJPEG UDP stream"
    )
    parser.add_argument("--bind", default="0.0.0.0", help="Alamat bind UDP")
    parser.add_argument("--port", type=int, default=5000, help="UDP video port")
    parser.add_argument(
        "--control-port", type=int, default=5001, help="UDP codec handshake port"
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Path executable FFmpeg")
    args = parser.parse_args()
    if not 1 <= args.port <= 65534 or not 1 <= args.control_port <= 65535:
        parser.error("Port UDP harus berada dalam rentang 1–65535")
    if args.port == args.control_port:
        parser.error("--port dan --control-port harus berbeda")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    capture = UdpFrameCapture(
        port=args.port,
        control_port=args.control_port,
        bind_host=args.bind,
        ffmpeg=args.ffmpeg,
    )
    pipeline = ADASPipeline()

    def stop_pipeline(signum, _frame) -> None:
        logging.getLogger("main_laptop").info(
            "Signal %d diterima; menghentikan pipeline.", signum
        )
        pipeline.stop()
        capture.release()

    signal.signal(signal.SIGTERM, stop_pipeline)
    signal.signal(signal.SIGINT, stop_pipeline)

    try:
        capture.start()
        pipeline.run(input_capture=capture, input_name="UDP")
    except Exception:
        logging.getLogger("main_laptop").exception("Pipeline UDP gagal.")
        capture.release()
        sys.exit(1)
    finally:
        capture.release()


if __name__ == "__main__":
    main()
