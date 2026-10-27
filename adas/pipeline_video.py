#!/usr/bin/env python3
"""
pipeline_video.py — Pipeline ADAS untuk input video file (mode offline/debug).

Perbedaan dari pipeline.py (live camera):
  - Input dari file video (.avi, .mp4, dll) bukan kamera USB
  - Frame dibaca sekuensial (tidak di-skip/grab), setiap frame diproses
  - Kecepatan playback bisa dikontrol (PLAYBACK_SPEED di constants atau argumen)
  - GPS, LCD, buzzer dinonaktifkan (tidak relevan untuk replay offline)
  - Recording tetap bisa diaktifkan (output annotated + raw dari video input)
  - Setelah video habis, pipeline berhenti sendiri

Cara pakai:
  Dipanggil dari main_video.py:
    python3 /opt/adas/main_video.py --video /opt/adas/recordings/20241020_raw.avi
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import cv2

from .constants import (
    INPUT_W, INPUT_H,
    RECORD_ENABLED,
)
from .detector    import YOLODetector
from .ipm         import IPMCalculator
from .tracker     import VehicleTracker
from .ttc         import TTCProcessor
from .pipeline    import (
    SharedState,
    FrameRecorder,
    _draw_annotations,
)

log = logging.getLogger(__name__)


class VideoADASPipeline:
    """
    Pipeline ADAS yang membaca dari file video.

    GPS tidak dipakai — kecepatan ego selalu 0.0 (gps_valid=False).
    TTC dihitung murni dari V_closing IPM.
    LCD dan buzzer tidak dijalankan.
    """

    def __init__(self, video_path: str, playback_speed: float = 1.0) -> None:
        """
        Parameters
        ----------
        video_path     : path ke file video input
        playback_speed : 1.0 = realtime, 2.0 = 2x lebih cepat, 0.5 = slow motion
                         0.0 = secepat mungkin (benchmark mode)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video tidak ditemukan: {video_path}")

        self._video_path    = video_path
        self._playback_speed = playback_speed
        self._state          = SharedState()
        self._stop_ev        = __import__('threading').Event()

        # Komponen inference
        self._detector : Optional[YOLODetector]   = None
        self._ipm      : Optional[IPMCalculator]  = None
        self._tracker  : Optional[VehicleTracker] = None
        self._ttc_proc : Optional[TTCProcessor]   = None
        self._recorder = FrameRecorder()

        self._cap: cv2.VideoCapture  # assigned di run()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self) -> None:
        log.info("[video] Memulai pipeline video: %s", self._video_path)

        # Inisialisasi model
        try:
            self._detector = YOLODetector()
            self._ipm      = IPMCalculator()
            self._tracker  = VehicleTracker()
            self._ttc_proc = TTCProcessor()
            log.info("[video] Model dan tracker siap.")
        except Exception as e:
            log.critical("[video] Gagal inisialisasi model: %s", e)
            return

        # Buka video
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            log.critical("[video] Gagal membuka video: %s", self._video_path)
            return
        self._cap = cap

        video_fps  = cap.get(cv2.CAP_PROP_FPS) or 10.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s   = total_frames / video_fps if video_fps > 0 else 0
        log.info("[video] FPS: %.1f | Frame: %d | Durasi: %.1f s",
                 video_fps, total_frames, duration_s)

        # Interval antar frame untuk kontrol kecepatan playback
        if self._playback_speed > 0:
            frame_interval = (1.0 / video_fps) / self._playback_speed
        else:
            frame_interval = 0.0   # secepat mungkin

        self._recorder.start()

        try:
            self._inference_loop(video_fps, frame_interval, total_frames)
        except KeyboardInterrupt:
            log.info("[video] Dihentikan oleh pengguna.")
        except Exception as e:
            log.critical("[video] Error fatal: %s", e, exc_info=True)
        finally:
            self._recorder.stop()
            cap.release()
            log.info("[video] Pipeline video selesai.")

    def stop(self) -> None:
        self._stop_ev.set()

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    def _inference_loop(
        self,
        video_fps: float,
        frame_interval: float,
        total_frames: int,
    ) -> None:
        fps_count   = 0
        fps         = 0.0
        frame_idx   = 0
        t_fps       = time.perf_counter()
        last_status = ''

        while not self._stop_ev.is_set():
            t_frame_start = time.perf_counter()

            ret, frame_raw = self._cap.read()
            if not ret or frame_raw is None:
                log.info("[video] Video selesai pada frame %d / %d.",
                         frame_idx, total_frames)
                break

            frame_idx += 1

            # Resize ke resolusi model
            frame_model = cv2.resize(frame_raw, (INPUT_W, INPUT_H),
                                     interpolation=cv2.INTER_LINEAR)

            # Inferensi
            detections = self._detector.detect(frame_model)

            # Tracking + estimasi jarak IPM
            tracks = self._tracker.update(detections, self._ipm)

            # GPS tidak tersedia di mode video — gps_valid=False
            # TTC dihitung murni dari V_closing IPM
            ego_speed = 0.0
            gps_valid = False

            # Hitung TTC, filter macet, debounce
            critical, status, alarm_on = self._ttc_proc.process(
                tracks, ego_speed, gps_valid
            )

            ttc    = critical.ttc        if critical else float('inf')
            dist_m = critical.distance_m if critical else -1.0

            self._state.update(ttc, dist_m, status, alarm_on)

            # Record
            if RECORD_ENABLED:
                ann_frame = _draw_annotations(
                    frame_model, tracks, status, ttc, dist_m, fps, ego_speed
                )
                self._recorder.write(frame_model, ann_frame)

            # Log saat state berganti
            if status != last_status:
                log.info("[video] Frame %d | Status: %s → %s | TTC: %.1fs | Dist: %.1fm",
                         frame_idx, last_status or '-', status, ttc, dist_m)
                last_status = status

            # FPS tracking + progress log tiap detik
            fps_count += 1
            elapsed = time.perf_counter() - t_fps
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count = 0
                t_fps = time.perf_counter()
                progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                log.info("[video] %.0f%% | Frame %d | FPS: %.1f | %s | TTC: %.1fs | Dist: %.1fm",
                         progress, frame_idx, fps, status, ttc, dist_m)

            # Speed control — tunggu sisa waktu frame agar playback sesuai kecepatan
            if frame_interval > 0:
                elapsed_frame = time.perf_counter() - t_frame_start
                sleep_s = frame_interval - elapsed_frame
                if sleep_s > 0:
                    time.sleep(sleep_s)
