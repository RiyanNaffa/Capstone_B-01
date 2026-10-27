#!/usr/bin/env python3
"""
ttc.py — Kalkulasi Time-to-Collision, filter macet, dan debounce alarm.

Rumus TTC:
    V_closing = (D_prev - D_curr) / Δt          ← dari delta jarak histori track
    V_floor   = V_ego × ALPHA_GPS               ← fallback GPS (target dianggap diam)
    V_eff     = max(V_closing, V_floor)
    TTC       = D_curr / V_eff

Status:
    AMAN   : TTC > TTC_SIAGA  atau  confidence == 'unreliable'
    SIAGA  : TTC_BAHAYA < TTC ≤ TTC_SIAGA
    BAHAYA : TTC ≤ TTC_BAHAYA

Filter macet:
    Jika V_ego < MACET_SPEED_MS DAN V_closing < MACET_CLOSING_MS
    selama N_MACET_FRAMES berturut-turut → paksa AMAN.

Debounce alarm:
    BAHAYA hanya aktif setelah N_TRIGGER_FRAMES berturut-turut.
    Setelah alarm, cooldown N_COOLDOWN_FRAMES sebelum bisa trigger lagi.
"""

from __future__ import annotations

from typing import List, Optional

from .constants import (
    TTC_BAHAYA,
    TTC_SIAGA,
    ALPHA_GPS,
    MACET_SPEED_MS,
    MACET_CLOSING_MS,
    N_MACET_FRAMES,
    N_TRIGGER_FRAMES,
    N_COOLDOWN_FRAMES,
    GPS_FUSION_MAX_DIFF_MS,
)
from .tracker import Track


# ---------------------------------------------------------------------------
# Kalkulasi TTC untuk satu track
# ---------------------------------------------------------------------------
def compute_ttc(track: Track, ego_speed_mps: float, gps_valid: bool = True) -> float:
    """
    Hitung TTC hybrid dengan sensor fusion GPS + IPM.

    Sensor fusion logic:
        1. Hitung V_closing_ipm dari histori jarak (smoothed, window 5 frame).
        2. Jika GPS valid DAN selisih |V_gps - V_closing_ipm| < GPS_FUSION_MAX_DIFF_MS:
               V_floor = ego_speed_mps * ALPHA_GPS   (GPS dipercaya)
           Jika GPS tidak valid ATAU selisih terlalu besar (GPS outlier/drift):
               V_floor = 0.0                          (GPS diabaikan)
        3. V_eff = max(V_closing_ipm, V_floor)
        4. TTC   = D_curr / V_eff

    gps_valid : bool — dari GPSReader.is_valid (False jika STALE/NOFIX/ERROR)

    Returns
    -------
    float — TTC dalam detik; float('inf') jika aman / tidak dapat dihitung.
    """
    if track.distance_m is None or track.distance_m < 0:
        return float('inf')

    if len(track.dist_history) < 2:
        # Track baru — jika GPS valid, pakai sebagai estimasi awal
        if gps_valid and ego_speed_mps > 1.0:
            return track.distance_m / ego_speed_mps
        return float('inf')

    # V_closing dari histori jarak (smoothed window, tidak diubah)
    delta_d = track.dist_history[0] - track.dist_history[-1]   # + = mendekat
    delta_t = track.time_history[-1] - track.time_history[0]
    v_closing_ipm = (delta_d / delta_t) if delta_t > 1e-6 else 0.0

    # Sensor fusion: validasi GPS terhadap IPM sebelum dipakai sebagai floor
    v_gps_floor = ego_speed_mps * ALPHA_GPS
    if gps_valid and abs(v_gps_floor - v_closing_ipm) <= GPS_FUSION_MAX_DIFF_MS:
        # GPS konsisten dengan IPM → pakai sebagai floor
        v_floor = v_gps_floor
    else:
        # GPS tidak valid atau outlier → buang, andalkan IPM saja
        v_floor = 0.0

    v_eff = max(v_closing_ipm, v_floor)

    if v_eff <= 1e-3:
        return float('inf')

    return track.distance_m / v_eff


# ---------------------------------------------------------------------------
# Klasifikasi status
# ---------------------------------------------------------------------------
def classify_status(ttc: float, confidence: str) -> str:
    """
    Returns
    -------
    'AMAN' | 'SIAGA' | 'BAHAYA'
    """
    if confidence == 'unreliable':
        return 'AMAN'
    if ttc <= TTC_BAHAYA:
        return 'BAHAYA'
    if ttc <= TTC_SIAGA:
        return 'SIAGA'
    return 'AMAN'


# ---------------------------------------------------------------------------
# Pilih track paling kritis
# ---------------------------------------------------------------------------
def pick_critical(tracks: List[Track]) -> Optional[Track]:
    """
    Pilih satu track yang paling mengancam:
      1. is_priority=True lebih diutamakan daripada tidak.
      2. Di antara priority, pilih yang TTC terkecil.
      3. Jika tidak ada priority track, pilih yang TTC terkecil dari semua.
    """
    active = [t for t in tracks if t.is_active and t.distance_m > 0]
    if not active:
        return None

    priority = [t for t in active if t.is_priority]
    candidates = priority if priority else active

    return min(candidates, key=lambda t: t.ttc)


# ---------------------------------------------------------------------------
# Filter macet
# ---------------------------------------------------------------------------
class MacetFilter:
    """
    Mencatat berapa frame berturut-turut V_ego kecil DAN V_closing kecil.
    Jika mencapai N_MACET_FRAMES → paksa status AMAN.
    """

    def __init__(self) -> None:
        self._count: int = 0

    def update(self, v_ego_mps: float, v_closing_mps: float) -> bool:
        """
        Returns
        -------
        True jika kondisi macet terdeteksi (status harus dipaksa AMAN).
        """
        if v_ego_mps < MACET_SPEED_MS and abs(v_closing_mps) < MACET_CLOSING_MS:
            self._count += 1
        else:
            self._count = 0

        return self._count >= N_MACET_FRAMES

    def reset(self) -> None:
        self._count = 0


# ---------------------------------------------------------------------------
# Debounce alarm
# ---------------------------------------------------------------------------
class AlarmDebouncer:
    """
    Alarm BAHAYA hanya aktif setelah N_TRIGGER_FRAMES berturut-turut.
    Setelah berbunyi, cooldown N_COOLDOWN_FRAMES sebelum bisa trigger lagi.

    Output method `update()`:
        True  → alarm harus aktif (buzzer on, LCD merah)
        False → alarm tidak aktif
    """

    def __init__(self) -> None:
        self._trigger_count:  int = 0
        self._cooldown_count: int = 0
        self._alarm_active:   bool = False

    def update(self, status: str) -> bool:
        """
        Parameters
        ----------
        status : 'AMAN' | 'SIAGA' | 'BAHAYA'

        Returns
        -------
        bool — True jika alarm harus berbunyi sekarang.
        """
        if self._cooldown_count > 0:
            self._cooldown_count -= 1
            self._trigger_count = 0
            self._alarm_active  = False
            return False

        if status == 'BAHAYA':
            self._trigger_count += 1
        else:
            self._trigger_count = 0
            self._alarm_active  = False
            return False

        if self._trigger_count >= N_TRIGGER_FRAMES:
            self._alarm_active    = True
            self._cooldown_count  = N_COOLDOWN_FRAMES
            self._trigger_count   = 0
            return True

        return False

    @property
    def is_active(self) -> bool:
        return self._alarm_active


# ---------------------------------------------------------------------------
# Orchestrator TTC untuk seluruh pipeline
# ---------------------------------------------------------------------------
class TTCProcessor:
    """
    Dipakai oleh pipeline.py untuk memproses satu frame lengkap.

    Alur per frame:
        1. Hitung TTC tiap track → perbarui track.ttc dan track.alert_status.
        2. Pilih track paling kritis.
        3. Cek filter macet.
        4. Cek debounce alarm.
        5. Kembalikan (critical_track, final_status, alarm_on).
    """

    def __init__(self) -> None:
        self._macet   = MacetFilter()
        self._debounce = AlarmDebouncer()

    def process(
        self,
        tracks: List[Track],
        ego_speed_mps: float,
        gps_valid: bool = True,
    ):
        """
        Returns
        -------
        (critical_track, final_status, alarm_on)
            critical_track : Track | None
            final_status   : 'AMAN' | 'SIAGA' | 'BAHAYA'
            alarm_on       : bool
        """
        # --- Hitung TTC dan status tiap track ---
        for t in tracks:
            t.ttc          = compute_ttc(t, ego_speed_mps, gps_valid)
            t.alert_status = classify_status(t.ttc, t.ipm_confidence)

        # --- Pilih track paling kritis ---
        critical = pick_critical(tracks)

        if critical is None:
            self._macet.reset()
            self._debounce.update('AMAN')
            return None, 'AMAN', False

        raw_status = critical.alert_status

        # --- Filter macet ---
        # Hitung V_closing dari histori track kritis
        if len(critical.dist_history) >= 2:
            delta_d = critical.dist_history[0] - critical.dist_history[-1]
            delta_t = critical.time_history[-1] - critical.time_history[0]
            v_closing = (delta_d / delta_t) if delta_t > 1e-6 else 0.0
        else:
            v_closing = 0.0

        if self._macet.update(ego_speed_mps, v_closing):
            raw_status = 'AMAN'

        # --- Debounce ---
        alarm_on = self._debounce.update(raw_status)

        return critical, raw_status, alarm_on
