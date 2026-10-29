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
from datetime import datetime
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


# ─────────────────────────────────────────────────────────────────────────────
# Pencarian kamera otomatis
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Anotasi frame
# ─────────────────────────────────────────────────────────────────────────────
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_STATUS_COL = {'AMAN': (50, 200, 50), 'SIAGA': (0, 165, 255), 'BAHAYA': (0, 0, 255)}

def _draw_annotations(
    frame: np.ndarray,
    tracks: List[Track],
    status: str,
    ttc: float,
    dist_m: float,
    fps: float,
    ego_speed_ms: float,
) -> np.ndarray:
    """Gambar bbox, label jarak+TTC, dan status bar di atas frame 256×256."""
    canvas = frame.copy()

    for t in tracks:
        if not t.is_active:
            col = COL_LOST
        elif t.is_priority:
            col = COL_PRIORITY if t.alert_status == 'AMAN' else (
                COL_WARNING if t.alert_status == 'SIAGA' else COL_DANGER)
        else:
            col = COL_NORMAL

        x1, y1, x2, y2 = t.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 1)

        dist_str = f"{t.distance_m:.1f}m" if t.distance_m > 0 else "---"
        ttc_str  = f"{t.ttc:.1f}s" if t.ttc < 99 else "--s"
        label    = f"{t.class_name} {dist_str} {ttc_str}"
        cv2.putText(canvas, label, (x1, max(y1 - 3, 8)),
                    _FONT, 0.28, col, 1, cv2.LINE_AA)

    # Status bar bawah
    s_col = _STATUS_COL.get(status, (200, 200, 200))
    cv2.rectangle(canvas, (0, INPUT_H - 14), (INPUT_W, INPUT_H), (0, 0, 0), cv2.FILLED)
    info = (f"{status} | TTC:{ttc:.1f}s | {dist_m:.1f}m | "
            f"{ego_speed_ms*3.6:.0f}km/h | {fps:.1f}fps")
    cv2.putText(canvas, info, (2, INPUT_H - 3),
                _FONT, 0.25, s_col, 1, cv2.LINE_AA)
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Recorder
# ─────────────────────────────────────────────────────────────────────────────
class FrameRecorder:
    """
    Merekam dua file AVI secara paralel:
      - <timestamp>_annotated.avi  — frame dengan bbox, label, status bar
      - <timestamp>_raw.avi        — frame mentah 256×256 tanpa anotasi

    Tulis frame dipanggil dari inference loop (main thread).
    Tidak menggunakan thread tambahan — VideoWriter.write() cukup cepat
    untuk MJPG 256×256 (~0.5-1 ms per frame).
    """

    def __init__(self) -> None:
        self._ann_writer: Optional[cv2.VideoWriter] = None
        self._raw_writer: Optional[cv2.VideoWriter] = None
        self._enabled = RECORD_ENABLED
        self._frame_count = 0

    def start(self) -> None:
        if not self._enabled:
            return
        os.makedirs(RECORD_DIR, exist_ok=True)
        ts    = time.strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*RECORD_CODEC)
        size   = (INPUT_W, INPUT_H)

        if RECORD_ANNOTATED:
            path = os.path.join(RECORD_DIR, f"{ts}_annotated.avi")
            self._ann_writer = cv2.VideoWriter(path, fourcc, RECORD_FPS, size)
            log.info("[recorder] Annotated → %s", path)

        if RECORD_RAW:
            path = os.path.join(RECORD_DIR, f"{ts}_raw.avi")
            self._raw_writer = cv2.VideoWriter(path, fourcc, RECORD_FPS, size)
            log.info("[recorder] Raw       → %s", path)

    def write(self, raw_frame: np.ndarray, ann_frame: np.ndarray) -> None:
        if not self._enabled:
            return
        if self._raw_writer is not None:
            self._raw_writer.write(raw_frame)
        if self._ann_writer is not None:
            self._ann_writer.write(ann_frame)
        self._frame_count += 1

    def stop(self) -> None:
        if self._ann_writer is not None:
            self._ann_writer.release()
            self._ann_writer = None
        if self._raw_writer is not None:
            self._raw_writer.release()
            self._raw_writer = None
        log.info("[recorder] Selesai. Total frame: %d", self._frame_count)


class TTCDataLogger:
    """Write per-frame, per-track TTC output to a timestamped CSV file."""

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
        self._start_perf: Optional[float] = None
        self.path: Optional[str] = None

    def start(self) -> None:
        os.makedirs(self.directory, exist_ok=True)
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.path = os.path.join(self.directory, f"{timestamp}_ttc.csv")
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._start_perf = time.perf_counter()
        log.info("[csv] TTC data → %s", self.path)

    def write_frame(
        self,
        tracks: List[Track],
        ego_speed_mps: float,
        gps_valid: bool,
        gps_status: str,
        system_status: str,
    ) -> None:
        if self._writer is None or self._file is None or self._start_perf is None:
            return

        self._frame_id += 1
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        elapsed_s = time.perf_counter() - self._start_perf
        active_tracks = [track for track in tracks if track.is_active]

        if not active_tracks:
            self._write_row(
                timestamp, elapsed_s, None, ego_speed_mps, gps_valid,
                gps_status, system_status,
            )
        else:
            for track in active_tracks:
                self._write_row(
                    timestamp, elapsed_s, track, ego_speed_mps, gps_valid,
                    gps_status, system_status,
                )

        if self._frame_id % 10 == 0:
            self._file.flush()

    def _write_row(
        self,
        timestamp: str,
        elapsed_s: float,
        track: Optional[Track],
        ego_speed_mps: float,
        gps_valid: bool,
        gps_status: str,
        system_status: str,
    ) -> None:
        row = {
            "timestamp": timestamp,
            "elapsed_s": f"{elapsed_s:.3f}",
            "frame_id": self._frame_id,
            "track_id": track.id if track else "",
            "class_name": track.class_name if track else "",
            "distance_system_m": track.distance_m if track else "",
            "ego_speed_mps": f"{ego_speed_mps:.3f}",
            "ego_speed_kmh": f"{ego_speed_mps * 3.6:.2f}",
            "gps_valid": gps_valid,
            "gps_status": gps_status,
            "ipm_confidence": track.ipm_confidence if track else "",
            "is_priority": track.is_priority if track else "",
            "ttc_system_s": track.ttc if track else "",
            "system_status": system_status,
        }
        self._writer.writerow(row)

    def stop(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None


# ─────────────────────────────────────────────────────────────────────────────
# Pencarian kamera otomatis
# ─────────────────────────────────────────────────────────────────────────────
def find_camera_device(keyword: str = CAMERA_KEYWORD) -> str:
    """
    Cari /dev/v4l/by-id/ yang mengandung keyword (case-insensitive).
    Symlink di /dev/v4l/by-id/ tidak bisa dibuka langsung oleh cv2.VideoCapture
    dengan CAP_V4L2 — resolve ke path device nyata (/dev/videoN) via os.path.realpath.
    Fallback ke CAMERA_FALLBACK (/dev/video1) jika tidak ditemukan.
    """
    by_id_dir = "/dev/v4l/by-id"
    if os.path.isdir(by_id_dir):
        for name in os.listdir(by_id_dir):
            if keyword.lower() in name.lower():
                symlink = os.path.join(by_id_dir, name)
                real    = os.path.realpath(symlink)   # resolve ke /dev/videoN
                log.info("[camera] Ditemukan: %s → %s", name, real)
                return real
    log.warning("[camera] '%s' tidak ditemukan di %s — fallback ke %s",
                keyword, by_id_dir, CAMERA_FALLBACK)
    return CAMERA_FALLBACK


# ─────────────────────────────────────────────────────────────────────────────
# State bersama antar komponen
# ─────────────────────────────────────────────────────────────────────────────
class SharedState:
    """Wadah state yang di-share antara inference loop dan thread I/O."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.ttc        : float = float('inf')
        self.dist_m     : float = -1.0
        self.status     : str   = 'AMAN'
        self.alarm_on   : bool  = False
        self.fps        : float = 0.0

    def update(self, ttc: float, dist_m: float, status: str, alarm_on: bool) -> None:
        with self._lock:
            self.ttc      = ttc
            self.dist_m   = dist_m
            self.status   = status
            self.alarm_on = alarm_on

    def get(self):
        with self._lock:
            return self.ttc, self.dist_m, self.status, self.alarm_on


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline utama
# ─────────────────────────────────────────────────────────────────────────────
class ADASPipeline:
    """
    Inisialisasi, jalankan, dan hentikan seluruh sistem AMCAS.

    Cara pakai:
        pipeline = ADASPipeline()
        pipeline.run()   # blocking hingga Ctrl+C atau error fatal
    """

    def __init__(self) -> None:
        self._state   = SharedState()
        self._stop_ev = threading.Event()

        # Komponen inference
        self._detector : Optional[YOLODetector]              = None
        self._ipm      : Optional[IPMCalculator]             = None
        self._tracker  : Optional[VehicleTracker]            = None  # atau OptFlowTracker
        self._ttc_proc : Optional[TTCProcessor]              = None

        # Komponen I/O device
        self._gps      = GPSReader()
        self._buzzer   = BuzzerController()
        self._lcd      = ACASDisplay()
        self._recorder = FrameRecorder()
        self._data_logger = TTCDataLogger(RECORD_DIR)

        # Kamera
        self._cap: cv2.VideoCapture  # assigned setelah kamera berhasil dibuka

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Jalankan pipeline. Blocking sampai stop() dipanggil atau error."""
        log.info("[pipeline] Memulai sistem AMCAS...")

        # --- Inisialisasi LCD hardware lebih awal untuk boot screen ---
        self._lcd.init_hardware()

        # Daftar komponen boot — pakai huruf kapital semua agar aman di font 5x7
        boot = [
            ("MODEL YOLO",  "wait"),
            ("KAMERA",      "wait"),
            ("GPS",         "wait"),
            ("BUZZER",      "wait"),
            ("SISTEM",      "wait"),
        ]
        self._lcd.show_boot_screen(boot)

        # --- Inisialisasi model & tracker ---
        try:
            self._detector  = YOLODetector()
            self._ipm       = IPMCalculator()
            self._tracker   = OptFlowTracker() if USE_OPTICAL_FLOW else VehicleTracker()
            self._ttc_proc  = TTCProcessor()
            tracker_name    = "OptFlow" if USE_OPTICAL_FLOW else "IoU"
            log.info("[pipeline] Model dan tracker siap. Tracker: %s", tracker_name)
            boot[0] = ("MODEL YOLO", "ok")
        except Exception as e:
            log.critical("[pipeline] Gagal inisialisasi model: %s", e)
            boot[0] = ("MODEL YOLO", "fail")
            self._lcd.show_boot_screen(boot)
            return
        self._lcd.show_boot_screen(boot)

        # --- Buka kamera ---
        cam_path = find_camera_device()
        cap = cv2.VideoCapture(cam_path, cv2.CAP_V4L2)
        if not cap.isOpened():
            log.critical("[pipeline] Gagal membuka kamera: %s", cam_path)
            cap.release()
            boot[1] = ("KAMERA", "fail")
            self._lcd.show_boot_screen(boot)
            return
        self._cap = cap

        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  INPUT_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_H)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, FRAME_BUFFER_SIZE)
        log.info("[pipeline] Kamera dibuka: %dx%d",
                 int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        boot[1] = ("KAMERA", "ok")
        self._lcd.show_boot_screen(boot)

        # --- Mulai GPS ---
        self._gps.start()
        time.sleep(1.5)   # beri waktu GPS thread membuka port
        gps_st = self._gps.gps_status
        if gps_st == 'ERROR':
            boot[2] = ("GPS", "fail")
        elif gps_st == 'NOFIX':
            boot[2] = ("GPS", "warn")   # port terbuka tapi belum fix — normal
        else:
            boot[2] = ("GPS", "ok")
        self._lcd.show_boot_screen(boot)

        # --- Mulai Buzzer ---
        self._buzzer.start()
        boot[3] = ("BUZZER", "ok")
        self._lcd.show_boot_screen(boot)

        # --- Semua komponen siap ---
        boot[4] = ("SISTEM", "ok")
        self._lcd.show_boot_screen(boot)
        log.info("[pipeline] Boot selesai — menunggu 3 detik...")
        time.sleep(3.0)   # tampilkan layar 100% selama 3 detik sebelum mulai

        try:
            self._lcd.start()
            self._recorder.start()
            if RECORD_ENABLED:
                self._data_logger.start()

            # --- Loop utama inference ---
            self._inference_loop()
        except KeyboardInterrupt:
            log.info("[pipeline] Dihentikan oleh pengguna (Ctrl+C).")
        except Exception as e:
            log.critical("[pipeline] Error fatal: %s", e, exc_info=True)
        finally:
            self._shutdown()

    def stop(self) -> None:
        """Sinyal untuk menghentikan pipeline dari thread lain."""
        self._stop_ev.set()

    # ------------------------------------------------------------------
    # Inference loop (main thread)
    # ------------------------------------------------------------------
    def _inference_loop(self) -> None:
        fps_count       = 0
        fps             = 0.0
        t_fps           = time.perf_counter()
        last_status     = ''
        last_gps_status = ''

        while not self._stop_ev.is_set():
            # Ambil frame — drop frame lama dengan grab() tanpa retrieve()
            self._cap.grab()
            ret, frame_raw = self._cap.retrieve()
            if not ret or frame_raw is None:
                log.warning("[pipeline] Frame kosong — lewati.")
                time.sleep(0.05)
                continue

            # Resize ke resolusi model
            frame_model = cv2.resize(frame_raw, (INPUT_W, INPUT_H),
                                     interpolation=cv2.INTER_LINEAR)

            # Inferensi + tracking
            # OptFlowTracker mengelola sendiri kapan deteksi dijalankan
            # (setiap DETECTION_INTERVAL frame), selebihnya pakai optical flow
            if USE_OPTICAL_FLOW:
                detections = self._detector.detect(frame_model)
                tracks = self._tracker.update(frame_model, detections, self._ipm)
            else:
                detections = self._detector.detect(frame_model)
                tracks = self._tracker.update(detections, self._ipm)

            # Baca kecepatan ego dan status GPS
            ego_speed  = self._gps.ego_speed_ms
            gps_valid  = self._gps.is_valid
            gps_status = self._gps.gps_status   # 'OK'|'STALE'|'NOFIX'|'ERROR'

            # Hitung TTC, filter macet, debounce (dengan sensor fusion GPS)
            critical, status, alarm_on = self._ttc_proc.process(
                tracks, ego_speed, gps_valid
            )

            self._data_logger.write_frame(
                tracks, ego_speed, gps_valid, gps_status, status
            )

            # Kemas hasil
            ttc    = critical.ttc        if critical else float('inf')
            dist_m = critical.distance_m if critical else -1.0

            # Perbarui state bersama
            self._state.update(ttc, dist_m, status, alarm_on)

            # Update I/O device
            self._buzzer.set_status(status if alarm_on else 'AMAN')
            self._lcd.update(ttc, dist_m, status, gps_status)

            # Record — gambar anotasi hanya jika recorder aktif
            if RECORD_ENABLED:
                ann_frame = _draw_annotations(
                    frame_model, tracks, status, ttc, dist_m, fps, ego_speed
                )
                self._recorder.write(frame_model, ann_frame)

            # Log saat status AMCAS berganti
            if status != last_status:
                log.info("[pipeline] Status: %s → %s | TTC: %.1fs | Dist: %.1fm"
                         " | GPS[%s]: %.1fkm/h",
                         last_status or '-', status, ttc, dist_m,
                         gps_status, ego_speed * 3.6)
                last_status = status

            # Log saat status GPS berganti
            if gps_status != last_gps_status:
                log.info("[pipeline] GPS status: %s → %s | speed: %.1f km/h",
                         last_gps_status or '-', gps_status, ego_speed * 3.6)
                last_gps_status = gps_status

            # FPS heartbeat tiap detik
            fps_count += 1
            elapsed = time.perf_counter() - t_fps
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                self._state.fps = fps
                fps_count = 0
                t_fps = time.perf_counter()
                log.info("[pipeline] FPS: %.1f | %s | TTC: %.1fs | Dist: %.1fm"
                         " | GPS[%s]: %.1fkm/h",
                         fps, status, ttc, dist_m, gps_status, ego_speed * 3.6)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def _shutdown(self) -> None:
        log.info("[pipeline] Membersihkan resources...")
        self._data_logger.stop()
        self._recorder.stop()
        self._gps.stop()
        self._buzzer.stop()
        self._lcd.stop()
        if self._cap is not None:
            self._cap.release()
            log.info("[pipeline] Kamera dilepas.")
        log.info("[pipeline] Sistem AMCAS dihentikan.")
