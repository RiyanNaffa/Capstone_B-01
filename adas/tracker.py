#!/usr/bin/env python3
"""
tracker.py — IoU-based Vehicle Tracker (default).

Diambil dari VehicleTracker di detection/vehicle_distance_ipm-pi.py dan
detection/ttc_calculate.py, lalu direfactor menjadi class mandiri yang
menerima IPMCalculator dan mengembalikan objek Track.

Perbedaan dari script asli:
  1. Distance estimation tidak lagi memakai IPM bottom-center, melainkan
     memanggil ipm.estimate(class_name, bbox) → (dist_m, confidence)
     agar per-kelas logic IPM terpusat di ipm.py.
  2. Track membawa 'dist_history' dan 'time_history' untuk TTC di ttc.py.
  3. Track membawa 'ipm_confidence' agar pipeline bisa filter 'unreliable'.
"""

from __future__ import annotations

import time
import collections
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import (
    MAX_LOST_FRAMES,
    IOU_ASSIGN_THRESH,
    TTC_HISTORY_LEN,
    CAMERA_VEHICLE_WIDTH_M,
    INPUT_W,
)
from .ipm import IPMCalculator


@dataclass
class Track:
    """Representasi satu kendaraan yang sedang dilacak."""
    id:             int
    bbox:           Tuple[int, int, int, int]
    class_name:     str
    confidence:     float
    distance_m:     float = 0.0
    lateral_m:      float = 0.0
    ipm_confidence: str   = 'unreliable'  # 'high'|'medium'|'low'|'unreliable'
    is_priority:    bool  = False
    lost_frames:    int   = 0
    age:            int   = 0
    ttc:            float = float('inf')
    alert_status:   str   = 'AMAN'        # 'AMAN'|'SIAGA'|'BAHAYA'

    # Histori untuk smoothing TTC
    dist_history: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=TTC_HISTORY_LEN)
    )
    time_history: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=TTC_HISTORY_LEN)
    )

    @property
    def bottom_center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, y2)

    @property
    def is_active(self) -> bool:
        return self.lost_frames == 0


class VehicleTracker:
    """
    Simple IoU-based tracker.

    Setiap frame:
      1. Hitung IoU antara semua track lama vs deteksi baru.
      2. Greedy matching: assign pasangan IoU terbesar berulang.
      3. Deteksi tanpa pasangan → track baru.
      4. Track tanpa pasangan → lost_frames++; hapus jika > MAX_LOST_FRAMES.
    """

    def __init__(self) -> None:
        self._next_id: int = 0
        self.tracks: Dict[int, Track] = {}

    # ------------------------------------------------------------------
    # Publik
    # ------------------------------------------------------------------
    def update(
        self,
        detections: List[dict],
        ipm: IPMCalculator,
    ) -> List[Track]:
        """
        Parameters
        ----------
        detections : output dari YOLODetector.postprocess()
        ipm        : IPMCalculator instance (sudah dikonfigurasi)

        Returns
        -------
        Semua track aktif (termasuk yang sedang lost < MAX_LOST_FRAMES).
        """
        track_ids = list(self.tracks.keys())
        matched_t: set = set()
        matched_d: set = set()

        # --- Matching ---
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

        # --- Buat track baru untuk deteksi yang tidak match ---
        for di, det in enumerate(detections):
            if di not in matched_d:
                new_t = self._create_track(det, ipm)
                self.tracks[new_t.id] = new_t

        # --- Tandai yang tidak match sebagai lost, hapus jika sudah terlalu lama ---
        to_delete = []
        for tid in track_ids:
            if tid not in matched_t:
                self.tracks[tid].lost_frames += 1
                if self.tracks[tid].lost_frames > MAX_LOST_FRAMES:
                    to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        return list(self.tracks.values())

    # ------------------------------------------------------------------
    # Internal
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

        # Lateral: tetap pakai bottom-center IPM dasar
        u, v = t.bottom_center
        res  = ipm.pixel_to_world(float(u), float(v))
        if res is not None:
            X, Z            = res
            t.lateral_m     = round(X, 2)
            t.is_priority   = abs(X) < (CAMERA_VEHICLE_WIDTH_M / 2.0)
        else:
            t.lateral_m   = 0.0
            t.is_priority = False

        # Simpan histori jarak & waktu untuk kalkulasi TTC
        if t.distance_m > 0:
            t.dist_history.append(t.distance_m)
            t.time_history.append(time.perf_counter())


# ------------------------------------------------------------------
# Helper (bebas class)
# ------------------------------------------------------------------
def _iou(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)
