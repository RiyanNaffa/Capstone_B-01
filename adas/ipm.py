#!/usr/bin/env python3
"""
ipm.py — Inverse Perspective Mapping: estimasi jarak kendaraan per kelas.

Setiap estimasi mengembalikan tuple (distance_m, confidence):
  'high'       — dimensi referensi penuh terlihat, orientasi frontal/rear
  'medium'     — estimasi campuran atau orientasi diagonal
  'low'        — bbox terpotong atau orientasi samping
  'unreliable' — jangan trigger alarm sama sekali

Logika per kelas:
  bus / truk : default → lebar (W_real); exception samping → tinggi (H_real)
  mobil      : default → lebar; aspect ratio > 1.5 → fallback tinggi, conf 'low'
  motor      : default → min(D_height, D_width); bbox terpotong → lebar, conf 'low'
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from .constants import (
    CAMERA_HEIGHT_M,
    CAMERA_PITCH_DEG,
    FOCAL_LENGTH_PX,
    INPUT_W,
    INPUT_H,
    VEHICLE_DIMS,
    CENTER_ZONE_RATIO,
    SIDE_ASPECT_RATIO,
)

DistResult = Tuple[float, str]  # (distance_m, confidence)


class IPMCalculator:
    """
    Menghitung jarak kendaraan menggunakan model Inverse Perspective Mapping
    berbasis geometri kamera pinhole sederhana.

    Model:
        Z = H / tan(α + atan((v - cy) / f))
        X = Z × (u - cx) / f

    di mana:
        H  = tinggi kamera dari tanah
        α  = sudut pitch kamera (positif = menunduk)
        f  = focal length dalam piksel
        (u, v) = koordinat piksel pada gambar
        (cx, cy) = pusat gambar
    """

    def __init__(
        self,
        camera_height: float = CAMERA_HEIGHT_M,
        pitch_deg: float = CAMERA_PITCH_DEG,
        focal_px: float = FOCAL_LENGTH_PX,
        img_w: int = INPUT_W,
        img_h: int = INPUT_H,
    ) -> None:
        self.H  = camera_height
        self.a  = math.radians(pitch_deg)
        self.f  = focal_px
        self.cx = img_w / 2.0
        self.cy = img_h / 2.0
        self.img_w = img_w
        self.img_h = img_h

    # ------------------------------------------------------------------
    # Primitif geometri
    # ------------------------------------------------------------------
    def pixel_to_world(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """Konversi piksel (u, v) ke koordinat dunia nyata (X, Z) dalam meter."""
        v_c = v - self.cy
        phi = self.a + math.atan2(v_c, self.f)
        if phi <= 1e-6:
            return None  # piksel di atas horizon → tidak terdefinisi
        Z = self.H / math.tan(phi)
        X = Z * (u - self.cx) / self.f
        return float(X), float(Z)

    def _dist_from_known_size(self, size_px: float, size_real: float) -> float:
        """D = (size_real × f) / size_px — rumus dasar pinhole dari ukuran objek."""
        if size_px < 1.0:
            return float('inf')
        return (size_real * self.f) / size_px

    # ------------------------------------------------------------------
    # Estimasi jarak per kelas
    # ------------------------------------------------------------------
    def estimate(
        self,
        class_name: str,
        bbox: Tuple[int, int, int, int],
    ) -> DistResult:
        """
        Hitung (distance_m, confidence) untuk objek terdeteksi.

        Parameters
        ----------
        class_name : 'bus' | 'mobil' | 'motor' | 'truk'
        bbox       : (x1, y1, x2, y2) dalam koordinat piksel model (256×256)
        """
        dims = VEHICLE_DIMS.get(class_name)
        if dims is None:
            return float('inf'), 'unreliable'

        W_real, H_real, _ = dims
        x1, y1, x2, y2 = bbox
        W_px = float(x2 - x1)
        H_px = float(y2 - y1)

        if W_px < 2 or H_px < 2:
            return float('inf'), 'unreliable'

        if class_name in ('bus', 'truk'):
            return self._estimate_bus_truk(W_real, H_real, W_px, H_px, x1, x2)
        elif class_name == 'mobil':
            return self._estimate_mobil(W_real, H_real, W_px, H_px)
        elif class_name == 'motor':
            return self._estimate_motor(W_real, H_real, W_px, H_px, y1, y2)
        return float('inf'), 'unreliable'

    # ------------------------------------------------------------------
    # Logika bus & truk
    # ------------------------------------------------------------------
    def _estimate_bus_truk(
        self,
        W_real: float, H_real: float,
        W_px: float, H_px: float,
        x1: float, x2: float,
    ) -> DistResult:
        """
        Default: gunakan lebar (lebih konsisten — regulasi PP No.55/2012).
        Exception: jika center_x bbox berada di luar zona tengah frame
        (≥ CENTER_ZONE_RATIO dari tepi), kendaraan kemungkinan menyamping
        → gunakan tinggi sebagai dimensi referensi, confidence 'medium'.
        """
        cx_bbox = (x1 + x2) / 2.0
        left_threshold  = self.img_w * CENTER_ZONE_RATIO
        right_threshold = self.img_w * (1.0 - CENTER_ZONE_RATIO)

        if cx_bbox < left_threshold or cx_bbox > right_threshold:
            # Kendaraan menyamping — pakai tinggi
            d = self._dist_from_known_size(H_px, H_real)
            return d, 'medium'

        d = self._dist_from_known_size(W_px, W_real)
        return d, 'high'

    # ------------------------------------------------------------------
    # Logika mobil
    # ------------------------------------------------------------------
    def _estimate_mobil(
        self,
        W_real: float, H_real: float,
        W_px: float, H_px: float,
    ) -> DistResult:
        """
        Default: lebar (lebih konsisten lintas segmen LCGC/sedan/SUV).
        Jika aspect ratio bbox > SIDE_ASPECT_RATIO (kemungkinan tampak samping),
        fallback ke tinggi dengan confidence 'low'.
        """
        aspect = H_px / W_px if W_px > 0 else 0.0

        if aspect > SIDE_ASPECT_RATIO:
            # Kemungkinan tampak samping atau terpotong horizontal
            d = self._dist_from_known_size(H_px, H_real)
            return d, 'low'

        d = self._dist_from_known_size(W_px, W_real)
        return d, 'high'

    # ------------------------------------------------------------------
    # Logika motor
    # ------------------------------------------------------------------
    def _estimate_motor(
        self,
        W_real: float, H_real: float,
        W_px: float, H_px: float,
        y1: float, y2: float,
    ) -> DistResult:
        """
        Default: ambil minimum antara estimasi via tinggi dan via lebar.
        Pendekatan konservatif (ambil yang lebih dekat) → safety.

        Tinggi motor (roda ke helm) sangat konsisten ~1.10 m dibanding lebar
        yang bervariasi tergantung posisi pengendara.

        Jika bbox terpotong atas/bawah frame → fallback ke lebar, conf 'low'.
        """
        top_clipped    = y1 <= 1.0
        bottom_clipped = y2 >= self.img_h - 2

        if top_clipped or bottom_clipped:
            # Tinggi tidak bisa diandalkan saat bbox terpotong
            d = self._dist_from_known_size(W_px, W_real)
            return d, 'low'

        d_height = self._dist_from_known_size(H_px, H_real)
        d_width  = self._dist_from_known_size(W_px, W_real)

        # Ambil yang lebih kecil (lebih konservatif / lebih dekat)
        d = min(d_height, d_width)
        return d, 'high'
