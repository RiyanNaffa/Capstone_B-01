#!/usr/bin/env python3
"""
pipeline_video.py — Pipeline ADAS untuk input video file (mode offline/debug).

Perbedaan dari pipeline.py (live camera):
  - Input dari file video (.avi, .mp4, dll) bukan kamera USB
  - Frame dibaca sekuensial (tidak di-skip/grab), setiap frame diproses
  - Kecepatan playback bisa dikontrol (PLAYBACK_SPEED di constants atau argumen)
  - GPS, LCD, buzzer dinonaktifkan (tidak relevan untuk replay offline)
  - Output video annotated (kelas + confidence) opsional
  - Daftar video dapat diulang sampai durasi endurance tercapai

Jalankan melalui entry point adas.main_video; lihat README untuk contoh CLI.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Optional, Sequence, Union

import cv2

from .constants import (
    INPUT_W, INPUT_H,
    RECORD_CODEC,
)
from .detector    import YOLODetector
from .ipm         import IPMCalculator
from .tracker     import VehicleTracker
from .ttc         import TTCProcessor
from .pipeline    import SharedState

log = logging.getLogger(__name__)


class VideoADASPipeline:
    """
    Pipeline ADAS yang membaca dari file video.

    GPS tidak dipakai — kecepatan ego selalu 0.0 (gps_valid=False).
    TTC dihitung murni dari V_closing IPM.
    LCD dan buzzer tidak dijalankan.
    """

    def __init__(
        self,
        video_path: Union[str, Sequence[str]],
        playback_speed: float = 1.0,
        duration_seconds: Optional[float] = None,
        annotated_dir: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        video_path     : satu file atau daftar file video yang diputar berurutan
        playback_speed : 1.0 = realtime, 2.0 = 2x lebih cepat, 0.5 = slow motion
                         0.0 = secepat mungkin (benchmark mode)
        duration_seconds: jika diisi, ulang daftar video sampai durasi tercapai
        annotated_dir  : jika diisi, simpan video bbox berlabel class dan confidence
        """
        video_paths = (video_path,) if isinstance(video_path, str) else video_path
        if not video_paths:
            raise ValueError("Daftar video tidak boleh kosong")
        if not math.isfinite(playback_speed) or playback_speed < 0:
            raise ValueError("playback_speed harus finite dan tidak boleh negatif")
        if duration_seconds is not None and (
            not math.isfinite(duration_seconds) or duration_seconds <= 0
        ):
            raise ValueError("duration_seconds harus lebih besar dari 0")

        self._video_paths = tuple(os.path.abspath(path) for path in video_paths)
        for path in self._video_paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Video tidak ditemukan: {path}")
        self._playback_speed = playback_speed
        self._duration_seconds = duration_seconds
        self._annotated_dir = annotated_dir
        self._state = SharedState()
        self._stop_ev = threading.Event()

        # Komponen inference
        self._detector : Optional[YOLODetector]   = None
        self._ipm      : Optional[IPMCalculator]  = None
        self._tracker  : Optional[VehicleTracker] = None
        self._ttc_proc : Optional[TTCProcessor]   = None

        self._cap: cv2.VideoCapture  # assigned di run()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self) -> None:
        log.info("[video] Sumber video: %d", len(self._video_paths))

        # Inisialisasi model
        self._detector = YOLODetector()
        self._ipm = IPMCalculator()
        log.info("[video] Model siap.")
        if self._annotated_dir:
            os.makedirs(self._annotated_dir, exist_ok=True)

        deadline = (
            time.monotonic() + self._duration_seconds
            if self._duration_seconds is not None else None
        )
        completed_pass = False
        try:
            while not self._stop_ev.is_set() and (
                deadline is None or time.monotonic() < deadline
            ):
                for video_path in self._video_paths:
                    if self._stop_ev.is_set():
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    self._run_video(video_path, deadline)
                if deadline is None:
                    completed_pass = True
                    break
        except KeyboardInterrupt:
            log.info("[video] Dihentikan oleh pengguna.")
        finally:
            if deadline is not None:
                elapsed = self._duration_seconds - max(0.0, deadline - time.monotonic())
                log.info("[video] Durasi berjalan: %.1f menit.", elapsed / 60.0)
            elif completed_pass:
                log.info("[video] Semua video selesai diputar.")
            log.info("[video] Pipeline video selesai.")

    def stop(self) -> None:
        self._stop_ev.set()

    def _run_video(self, video_path: str, deadline: Optional[float]) -> None:
        log.info("[video] Membuka: %s", video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Gagal membuka video: {video_path}")
        self._cap = cap

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(video_fps) or video_fps <= 0:
            cap.release()
            raise RuntimeError(f"FPS video tidak valid: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / video_fps if total_frames > 0 else 0
        frame_interval = (
            (1.0 / video_fps) / self._playback_speed
            if self._playback_speed > 0 else 0.0
        )
        log.info("[video] FPS: %.1f | Frame: %d | Durasi: %.1f s",
                 video_fps, total_frames, duration_s)

        # Track dan histori TTC tidak boleh terbawa dari video/pass sebelumnya.
        self._tracker = VehicleTracker()
        self._ttc_proc = TTCProcessor()
        writer = None
        if self._annotated_dir:
            basename = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(self._annotated_dir, f"{basename}_annotated.avi")
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*RECORD_CODEC),
                video_fps,
                (INPUT_W, INPUT_H),
            )
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Gagal membuat video output: {output_path}")
            log.info("[video] Annotated → %s", output_path)

        try:
            frames_read = self._inference_loop(
                frame_interval, total_frames, writer, deadline
            )
            if (
                frames_read == 0
                and not self._stop_ev.is_set()
                and (deadline is None or time.monotonic() < deadline)
            ):
                raise RuntimeError(f"Tidak ada frame yang dapat dibaca: {video_path}")
        finally:
            if writer is not None:
                writer.release()
            cap.release()

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    def _inference_loop(
        self,
        frame_interval: float,
        total_frames: int,
        writer: Optional[cv2.VideoWriter],
        deadline: Optional[float],
    ) -> int:
        fps_count   = 0
        fps         = 0.0
        frame_idx   = 0
        t_fps       = time.perf_counter()
        last_status = ''

        while not self._stop_ev.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                break
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

            if writer is not None:
                _draw_detection_annotations(frame_model, detections)
                writer.write(frame_model)

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
        return frame_idx


def _draw_detection_annotations(frame, detections: list[dict]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        color = (0, 220, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        text_y = max(y1 - 4, 10)
        cv2.putText(
            frame, label, (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
        )
