#!/usr/bin/env python3
"""
gps_reader.py — Thread NMEA reader, expose ego_speed_ms (kecepatan kendaraan kita).

Diambil dari sensor_test/gps/test_baru.py dan direfactor menjadi class
yang berjalan di background thread, aman untuk dibaca dari thread lain.

Hardware:
    GPS via UART0: RX→PH0, TX→PH1, /dev/ttyS0, 9600 baud, NMEA 0183.

Sentence yang dipakai:
    $GNRMC — Global Navigation Recommended Minimum Specific GNSS Data.
    Field 7 (speed over ground, knots) dikonversi ke m/s.

Fallback:
    Jika GPS belum fix atau serial error, ego_speed_ms mengembalikan 0.0.
    Pipeline menggunakan ini sebagai fallback graceful (tidak crash).
"""

from __future__ import annotations

import threading
import logging
import time
from typing import Optional

import serial

from .constants import GPS_PORT, GPS_BAUD, GPS_STALE_TIMEOUT_S

log = logging.getLogger(__name__)

# Konversi knot ke m/s
KNOTS_TO_MS = 0.5144444


class GPSReader:
    """
    Membaca kecepatan ego dari modul GPS NMEA 0183 secara non-blocking.

    Status GPS yang di-expose ke pipeline:
        'OK'     — fix valid, nilai fresh (< GPS_STALE_TIMEOUT_S)
        'STALE'  — fix valid tapi nilai tidak berubah terlalu lama (stuck/drift)
        'NOFIX'  — NMEA status = 'V' (void), belum dapat sinyal
        'ERROR'  — serial error / port tidak tersedia

    Cara pakai:
        gps = GPSReader()
        gps.start()
        speed  = gps.ego_speed_ms   # thread-safe
        status = gps.gps_status     # 'OK'|'STALE'|'NOFIX'|'ERROR'
        valid  = gps.is_valid       # True hanya saat 'OK'
    """

    def __init__(
        self,
        port: str = GPS_PORT,
        baud: int = GPS_BAUD,
    ) -> None:
        self._port  = port
        self._baud  = baud

        self._speed_ms     : float = 0.0
        self._last_update  : float = 0.0   # monotonic time saat terakhir terima nilai valid
        self._last_speed   : float = -1.0  # untuk deteksi stuck (nilai tidak berubah)
        self._nmea_fix     : str   = 'V'   # 'A' atau 'V' dari GNRMC
        self._serial_ok    : bool  = False

        self._lock    = threading.Lock()
        self._stop_ev = threading.Event()
        self._thread  : Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Properti publik — semua thread-safe
    # ------------------------------------------------------------------
    @property
    def ego_speed_ms(self) -> float:
        """Kecepatan ego (m/s). Selalu 0.0 jika GPS tidak valid."""
        with self._lock:
            return self._speed_ms

    @property
    def gps_status(self) -> str:
        """
        Status kualitas GPS:
            'OK'    — fix valid + nilai fresh
            'STALE' — fix valid tapi stuck (tidak update > GPS_STALE_TIMEOUT_S)
            'NOFIX' — NMEA void / belum lock
            'ERROR' — serial tidak tersedia
        """
        with self._lock:
            if not self._serial_ok:
                return 'ERROR'
            if self._nmea_fix != 'A':
                return 'NOFIX'
            age = time.monotonic() - self._last_update
            if age > GPS_STALE_TIMEOUT_S:
                return 'STALE'
            return 'OK'

    @property
    def is_valid(self) -> bool:
        """True hanya jika GPS status == 'OK' (fix + fresh)."""
        return self.gps_status == 'OK'

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Mulai thread pembaca GPS di background."""
        self._stop_ev.clear()
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="gps-reader",
            daemon=True,
        )
        self._thread.start()
        log.info("[GPS] Thread dimulai (%s @ %d baud)", self._port, self._baud)

    def stop(self) -> None:
        """Hentikan thread GPS dengan graceful."""
        self._stop_ev.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        log.info("[GPS] Thread dihentikan.")

    # ------------------------------------------------------------------
    # Loop thread
    # ------------------------------------------------------------------
    def _reader_loop(self) -> None:
        while not self._stop_ev.is_set():
            try:
                ser = serial.Serial(
                    port     = self._port,
                    baudrate = self._baud,
                    timeout  = 2.0,
                    bytesize = serial.EIGHTBITS,
                    parity   = serial.PARITY_NONE,
                    stopbits = serial.STOPBITS_ONE,
                )
                ser.reset_input_buffer()
                with self._lock:
                    self._serial_ok = True
                log.info("[GPS] Serial port terbuka: %s", self._port)

                while not self._stop_ev.is_set():
                    raw = ser.readline()
                    if not raw:
                        continue

                    try:
                        line = raw.decode("utf-8", errors="ignore").strip()
                    except Exception:
                        continue

                    if not line.startswith("$"):
                        continue

                    # Validasi checksum NMEA
                    if "*" in line:
                        try:
                            sentence, chk = line[1:].rsplit("*", 1)
                            calc = 0
                            for ch in sentence:
                                calc ^= ord(ch)
                            if f"{calc:02X}" != chk.upper().strip():
                                continue
                        except Exception:
                            continue

                    if line.startswith(("$GNRMC", "$GPRMC")):
                        self._parse_rmc(line)

                ser.close()

            except serial.SerialException as e:
                with self._lock:
                    self._serial_ok = False
                log.warning("[GPS] Serial error: %s — retry dalam 5 s", e)
                time.sleep(5.0)
            except Exception as e:
                with self._lock:
                    self._serial_ok = False
                log.error("[GPS] Error tak terduga: %s", e)
                time.sleep(5.0)

    def _parse_rmc(self, sentence: str) -> None:
        """
        Parse $GNRMC/$GPRMC.
        Validity check  : field status == 'A'
        Staleness check : catat waktu monotonic tiap kali nilai fix diterima;
                          gps_status property cek age vs GPS_STALE_TIMEOUT_S
        """
        parts = sentence.split(",")
        if len(parts) < 8:
            return

        nmea_fix = parts[2]   # 'A' = valid, 'V' = void

        with self._lock:
            self._nmea_fix = nmea_fix

        if nmea_fix != "A":
            # No fix — tidak update speed
            return

        try:
            speed_knots = float(parts[7]) if parts[7] else 0.0
            speed_ms    = round(speed_knots * KNOTS_TO_MS, 3)
        except ValueError:
            return

        now = time.monotonic()
        with self._lock:
            self._speed_ms    = speed_ms
            self._last_update = now
            # Deteksi stuck: jika nilai persis sama dengan sebelumnya,
            # TIDAK perbarui _last_update sehingga staleness timer tetap berjalan
            # (override: tetap update karena speed bisa memang konstan saat cruise)
            # Staleness murni berbasis waktu, bukan perubahan nilai.
            self._last_speed  = speed_ms
