#!/usr/bin/env python3
"""
pipeline.py — Orchestrator utama AMCAS Bus Kampus UGM.

Arsitektur thread:
    ┌─────────────────────────────────────────────────────┐
    │  Main Thread (inference loop)                        │
    │    Camera → Detector → Tracker → TTC → SharedState  │
    └──────────┬──────────────────────────────────────────┘
               │ SharedState (lock)
       ┌───────┴────────┐
       ▼                ▼
  LCD Thread        Buzzer Thread
  (5 Hz update)    (pola beep non-blocking)
       ▲                ▲
       └──── GPS Thread (background, update speed) ────────┘

Frame queue:
    queue.Queue(maxsize=1) — frame lama dibuang, tidak di-buffer.
    Ini mencegah inference lag saat kamera lebih cepat dari inference.

Semua I/O device (GPS, LCD, Buzzer) berjalan di thread terpisah.
Hanya state yang di-share lewat lock — tidak ada busy-wait.
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime, timezone
import threading
import time
from typing import List, Optional

import cv2
import numpy as np

from .constants import (
    CAMERA_KEYWORD,
    CAMERA_FALLBACK,
    INPUT_W, INPUT_H,
    FRAME_BUFFER_SIZE,
    FPS_TARGET,
    LCD_UPDATE_INTERVAL,
    RECORD_ENABLED,
    RECORD_DIR,
    RECORD_ANNOTATED,
    RECORD_RAW,
    RECORD_FPS,
    RECORD_CODEC,
    COL_PRIORITY, COL_NORMAL, COL_LOST, COL_DANGER, COL_WARNING,
    USE_OPTICAL_FLOW,
)
from .detector         import YOLODetector
from .ipm              import IPMCalculator
from .tracker          import VehicleTracker, Track
from .tracker_optflow  import OptFlowTracker
from .ttc              import TTCProcessor
from .gps_reader  import GPSReader
from .buzzer      import BuzzerController
from .lcd         import ACASDisplay

log = logging.getLogger(__name__)


class TTCDataLogger:
    """Merekam output TTC per objek ke CSV selama pipeline berjalan."""

    FIELDNAMES = [
        "timestamp",
        "elapsed_s",
        "frame_id",
        "track_id",
        "class_name",
        "distance_system_m",
        "ego_speed_mps",
        "ego_speed_kmh",
        "gps_valid",
        "gps_status",
        "ipm_confidence",
        "is_priority",
        "ttc_system_s",
        "system_status",
    ]

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self._file = None
        self._writer = None
        self._frame_id = 0
        self._rows_since_flush = 0
        self._start_perf = None
        self.path: Optional[str] = None

    def start(self) -> None:
