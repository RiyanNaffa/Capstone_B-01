#!/usr/bin/env python3
"""
main.py — Entry point ADAS Bus Kampus UGM.

Dipanggil saat boot oleh systemd: /opt/adas/.venv/bin/python3 /opt/adas/main.py

Tugas:
    1. Setup logging ke systemd journal (stdout) dan file error.
    2. Pastikan direktori log ada.
    3. Panggil ADASPipeline().run() yang blocking.

Logging:
    - INFO  → stdout → ditangkap oleh journald (journalctl -u adas.service)
    - ERROR + state change → file di LOG_DIR
    - Jangan log setiap frame — hanya event signifikan.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import signal
import sys

from adas.constants import LOG_DIR
from adas.pipeline  import ADASPipeline


def setup_logging() -> None:
    """
    Konfigurasi logging dua saluran:
      1. StreamHandler (stdout) → ditangkap journald, level INFO
      2. RotatingFileHandler (LOG_DIR/adas_error.log) → level ERROR
         Rotasi 1 MB × 3 file agar tidak memenuhi flash storage.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # --- stdout → journal ---
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(sh)

    # --- file → hanya error ---
    fh = logging.handlers.RotatingFileHandler(
        filename    = os.path.join(LOG_DIR, "adas_error.log"),
        maxBytes    = 1 * 1024 * 1024,  # 1 MB
        backupCount = 3,
        encoding    = "utf-8",
    )
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    ))
    root.addHandler(fh)


def main() -> None:
    setup_logging()

    log = logging.getLogger("main")
    log.info("=" * 50)
    log.info("ADAS Bus Kampus UGM — Memulai...")
    log.info("=" * 50)

    pipeline = ADASPipeline()

    # Tangkap SIGTERM (dari systemctl stop) dan SIGINT (Ctrl+C)
    # agar recorder.stop() → VideoWriter.release() selalu dipanggil
    def _handle_signal(signum, frame):
        log.info("Signal %d diterima — menghentikan pipeline...", signum)
        pipeline.stop()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    try:
        pipeline.run()
    except Exception as e:
        log.critical("Pipeline error fatal: %s", e, exc_info=True)
        sys.exit(1)

    log.info("ADAS selesai.")


if __name__ == "__main__":
    main()
