#!/usr/bin/env python3
"""
tracker_optflow.py — Alternatif tracker berbasis Lucas-Kanade Sparse Optical Flow.

Motivasi
--------
Inferensi TFLite di H618 Cortex-A53 (tanpa NPU) membutuhkan ~50-100 ms per frame.
Jika kita jalankan detektor setiap frame → FPS ≈ 10.
Dengan optical flow di frame antar deteksi, FPS efektif naik 3-5× karena
LK optical flow (CPU numpy+cv2) jauh lebih cepat (~3-8 ms per frame).

Pendekatan
----------
- Setiap DETECTION_INTERVAL frame, jalankan YOLODetector penuh + reset track.
- Di frame di antaranya:
    1. Cari feature points (Shi-Tomasi) di dalam setiap bbox track.
    2. Jalankan cv2.calcOpticalFlowPyrLK → posisi baru tiap feature point.
    3. Estimasi translasi bbox dari median displacement.
    4. Estimasi scale factor dari rasio spread feature points → turunan jarak.

Trade-off & Validitas untuk Kondisi Indonesia
---------------------------------------------
(+) Kecepatan: LK ~3-8 ms vs inferensi ~50-100 ms → hemat ~85% CPU per frame.
(+) Kontinuitas: bbox bergerak mulus antar frame, TTC lebih stabil.
(-) Akurasi jarak dari scale factor TIDAK dapat diandalkan:
    - Jalan kampus UGM sering bergetar (polisi tidur, aspal rusak).
    - Getaran kendaraan menyebabkan perubahan feature spread yang bukan
      dari perubahan jarak → estimasi jarak via scale factor TIDAK VALID.
    - Pencahayaan berubah (tunnel/pohon rindang): feature bisa hilang tiba-tiba.
(!) Kesimpulan: gunakan optical flow HANYA untuk propagasi posisi bbox.
    Estimasi jarak tetap dari IPMCalculator pada frame deteksi terakhir
    yang valid, bukan dari scale factor optical flow.
(-) Drift: bbox bisa drift jauh saat kendaraan berbelok cepat atau ada oklusi.
    Batasi ke DETECTION_INTERVAL ≤ 5 frame untuk menghindari drift signifikan.

Gunakan tracker ini sebagai mode eksperimen. Untuk produksi yang membutuhkan
akurasi jarak, gunakan tracker.py (default IoU-based).
"""

from __future__ import annotations

import time
import collections
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .constants import (
    MAX_LOST_FRAMES,
    IOU_ASSIGN_THRESH,
    TTC_HISTORY_LEN,
    CAMERA_VEHICLE_WIDTH_M,
    DETECTION_INTERVAL,
    LK_PARAMS,
)
from .ipm import IPMCalculator
from .tracker import Track, _iou  # reuse dataclass dan helper IoU


# Parameter Shi-Tomasi untuk feature detection di dalam bbox
SHITOMASI_PARAMS = dict(
    maxCorners    = 20,   # maks feature per bbox — cukup untuk estimasi translasi
    qualityLevel  = 0.3,
    minDistance   = 5,
    blockSize     = 5,
)


class OptFlowTracker:
    """
    Tracker dengan Lucas-Kanade Sparse Optical Flow sebagai interpolasi
    antara frame deteksi penuh.

    API identik dengan VehicleTracker di tracker.py sehingga bisa dipakai
    sebagai drop-in replacement di pipeline.py.
    """

    def __init__(self) -> None:
        self._next_id: int = 0
        self.tracks: Dict[int, Track] = {}
        self._frame_count: int = 0

        # State untuk optical flow
        self._prev_gray:    Optional[np.ndarray]          = None
        self._prev_pts:     Dict[int, np.ndarray]         = {}  # tid → points
        self._last_det_ipm: Dict[int, float]              = {}  # tid → dist dari IPM

    # ------------------------------------------------------------------
    # Publik
    # ------------------------------------------------------------------
    def update(
        self,
        frame_bgr: np.ndarray,
        detections: List[dict],   # kosong di frame non-deteksi
        ipm: IPMCalculator,
        force_detect: bool = False,
    ) -> List[Track]:
        """
        Parameters
        ----------
        frame_bgr   : frame BGR resolusi model (256×256)
        detections  : output detector (kosong kalau frame non-deteksi)
        ipm         : IPMCalculator
        force_detect: paksa update penuh dari deteksi (abaikan frame count)

        Returns
        -------
        List[Track] — semua track aktif & lost
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        is_det_frame = force_detect or (self._frame_count % DETECTION_INTERVAL == 0)

        if is_det_frame and detections:
            # ── Frame deteksi: update track via IoU matching ──────────────
            self._update_from_detections(detections, ipm)
            # Perbarui feature points untuk semua track aktif
            self._refresh_features(gray)
        elif not is_det_frame and self._prev_gray is not None:
            # ── Frame interpolasi: propagasi bbox via optical flow ─────────
            self._propagate_optflow(gray, ipm)
        # Jika is_det_frame tapi detections kosong: biarkan lost_frames naik

        self._frame_count += 1
        self._prev_gray = gray
        return list(self.tracks.values())

    # ------------------------------------------------------------------
    # Internal — update dari deteksi
    # ------------------------------------------------------------------
    def _update_from_detections(
        self, detections: List[dict], ipm: IPMCalculator
    ) -> None:
        """IoU matching identik dengan tracker.py."""
        track_ids = list(self.tracks.keys())
        matched_t: set = set()
        matched_d: set = set()

        if track_ids and detections:
            iou_mat = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
            for ti, tid in enumerate(track_ids):
                for di, det in enumerate(detections):
                    iou_mat[ti, di] = _iou(self.tracks[tid].bbox, det["bbox"])

            while True:
                if iou_mat.size == 0:
                    break
                best = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                ti, di = int(best[0]), int(best[1])
                if iou_mat[ti, di] < IOU_ASSIGN_THRESH:
                    break

                tid = track_ids[ti]
                det = detections[di]
                t   = self.tracks[tid]
                t.bbox        = tuple(det["bbox"])  # type: ignore
                t.class_name  = det["class_name"]
                t.confidence  = float(det["confidence"])
                t.lost_frames = 0
                t.age        += 1
                self._update_distance(t, ipm)

                matched_t.add(tid)
                matched_d.add(di)
                iou_mat[ti, :] = -1.0
                iou_mat[:, di] = -1.0

        for di, det in enumerate(detections):
            if di not in matched_d:
                new_t = self._create_track(det, ipm)
                self.tracks[new_t.id] = new_t

        to_delete = []
        for tid in track_ids:
            if tid not in matched_t:
                self.tracks[tid].lost_frames += 1
                if self.tracks[tid].lost_frames > MAX_LOST_FRAMES:
                    to_delete.append(tid)
                    self._prev_pts.pop(tid, None)
                    self._last_det_ipm.pop(tid, None)
        for tid in to_delete:
            del self.tracks[tid]

    # ------------------------------------------------------------------
    # Internal — propagasi optical flow
    # ------------------------------------------------------------------
    def _propagate_optflow(self, gray: np.ndarray, ipm: IPMCalculator) -> None:
        """
        Geser bbox setiap track berdasarkan median displacement feature points.

        CATATAN VALIDITAS:
        - Translasi bbox: VALID — displacement fitur 2D dapat diandalkan
          untuk pergeseran posisi selama getaran kecil.
        - Estimasi jarak dari scale factor: TIDAK VALID untuk kondisi jalan
          Indonesia (getaran, pencahayaan berubah). Jarak tetap menggunakan
          nilai IPM dari frame deteksi terakhir + model kecepatan sederhana.
        """
        if self._prev_gray is None:
            return

        to_delete = []
        for tid, t in self.tracks.items():
            pts_prev = self._prev_pts.get(tid)
            if pts_prev is None or len(pts_prev) < 3:
                # Tidak cukup feature → lost
                t.lost_frames += 1
                if t.lost_frames > MAX_LOST_FRAMES:
                    to_delete.append(tid)
                continue

            # Jalankan Lucas-Kanade
            pts_next, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, pts_prev, None, **LK_PARAMS
            )

            if pts_next is None or status is None:
                t.lost_frames += 1
                continue

            # Filter hanya point yang berhasil dilacak
            good_prev = pts_prev[status.flatten() == 1]
            good_next = pts_next[status.flatten() == 1]

            if len(good_next) < 2:
                t.lost_frames += 1
                continue

            # Estimasi translasi dari median displacement
            # (median lebih robust terhadap outlier vs mean)
            displacement = good_next - good_prev
            dx = float(np.median(displacement[:, 0]))
            dy = float(np.median(displacement[:, 1]))

            # Geser bbox
            x1, y1, x2, y2 = t.bbox
            new_bbox = (
                int(np.clip(x1 + dx, 0, gray.shape[1] - 1)),
                int(np.clip(y1 + dy, 0, gray.shape[0] - 1)),
                int(np.clip(x2 + dx, 0, gray.shape[1] - 1)),
                int(np.clip(y2 + dy, 0, gray.shape[0] - 1)),
            )
            t.bbox        = new_bbox  # type: ignore
            t.lost_frames = 0
            t.age        += 1

            # Jarak: tetap dari IPM (bbox bergerak → jarak dihitung ulang)
            # Ini lebih akurat daripada scale factor optical flow di jalan bergelombang
            self._update_distance(t, ipm)

            # Simpan feature points baru untuk frame berikutnya
            self._prev_pts[tid] = good_next.reshape(-1, 1, 2)

        for tid in to_delete:
            del self.tracks[tid]
            self._prev_pts.pop(tid, None)

    # ------------------------------------------------------------------
    # Internal — feature detection
    # ------------------------------------------------------------------
    def _refresh_features(self, gray: np.ndarray) -> None:
        """
        Deteksi feature points Shi-Tomasi di dalam setiap bbox track.

        Shi-Tomasi lebih stabil daripada Harris di kondisi pencahayaan
        rendah (malam hari di jalan kampus) karena threshold berbasis
        eigenvalue minimum, bukan determinan.
        """
        self._prev_pts.clear()
        for tid, t in self.tracks.items():
            if not t.is_active:
                continue
            x1, y1, x2, y2 = t.bbox
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            pts = cv2.goodFeaturesToTrack(roi, mask=None, **SHITOMASI_PARAMS)
            if pts is not None and len(pts) >= 3:
                # Konversi koordinat ROI → koordinat frame penuh
                pts[:, 0, 0] += x1
                pts[:, 0, 1] += y1
                self._prev_pts[tid] = pts

    # ------------------------------------------------------------------
    # Internal — track management
    # ------------------------------------------------------------------
    def _create_track(self, det: dict, ipm: IPMCalculator) -> Track:
        t = Track(
            id         = self._next_id,
            bbox       = tuple(det["bbox"]),  # type: ignore
            class_name = det["class_name"],
            confidence = float(det["confidence"]),
        )
        self._next_id += 1
        self._update_distance(t, ipm)
        return t

    def _update_distance(self, t: Track, ipm: IPMCalculator) -> None:
        dist_m, conf = ipm.estimate(t.class_name, t.bbox)
        t.distance_m     = round(dist_m, 2) if dist_m < 9999.0 else -1.0
        t.ipm_confidence = conf

        u, v = t.bottom_center
        res  = ipm.pixel_to_world(float(u), float(v))
        if res is not None:
            X, Z          = res
            t.lateral_m   = round(X, 2)
            t.is_priority = abs(X) < (CAMERA_VEHICLE_WIDTH_M / 2.0)
        else:
            t.lateral_m   = 0.0
            t.is_priority = False

        if t.distance_m > 0:
            t.dist_history.append(t.distance_m)
            t.time_history.append(time.perf_counter())
