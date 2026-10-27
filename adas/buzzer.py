#!/usr/bin/env python3
"""
buzzer.py — Kontrol buzzer aktif GPIO via gpiod v2 (libgpiod).

Hardware:
    Buzzer aktif PI1 → /dev/gpiochip0, line 257.
    Aktivasi: LOW → HIGH (sisi positif terhubung ke GPIO, GND ke ground).

Pola bunyi per status (sesuai lcdcapstone.py):
    SIAGA  : periode 1.0 s, on 0.1 s  → 1 beep pendek per detik
    BAHAYA : periode 0.2 s, on 0.1 s  → 5 beep cepat per detik

Berjalan di thread terpisah (non-blocking terhadap inference loop).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

try:
    import gpiod
    from gpiod.line import Direction, Value, Bias
    _GPIOD_AVAILABLE = True
except ImportError:
    _GPIOD_AVAILABLE = False
    log.warning("[buzzer] gpiod tidak tersedia — buzzer akan berjalan dalam mode dummy.")

from .constants import GPIO_CHIP, BUZZER_LINE, BUZZER_ENABLED


class BuzzerController:
    """
    Mengontrol buzzer aktif di background thread.

    Cara pakai:
        buzzer = BuzzerController()
        buzzer.start()
        buzzer.set_status('BAHAYA')   # dipanggil dari thread pipeline
        ...
        buzzer.stop()
    """

    # Pola bunyi (period_s, on_time_s)
    _PATTERNS = {
        'AMAN'  : None,
        'SIAGA' : (1.0, 0.1),
        'BAHAYA': (0.2, 0.1),
    }

    def __init__(
        self,
        chip_path: str = GPIO_CHIP,
        line_no:   int = BUZZER_LINE,
    ) -> None:
        self._chip_path = chip_path
        self._line_no   = line_no

        self._status    : str           = 'AMAN'
        self._lock       = threading.Lock()
        self._stop_ev    = threading.Event()
        self._thread     : Optional[threading.Thread] = None

        # GPIO request handle
        self._req = None
        self._gpio_ok = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if not BUZZER_ENABLED:
            log.info("[buzzer] BUZZER_ENABLED=False — buzzer dinonaktifkan.")
        self._gpio_ok = self._open_gpio() if BUZZER_ENABLED else False
        self._stop_ev.clear()
        self._thread = threading.Thread(
            target=self._buzzer_loop,
            name="buzzer",
            daemon=True,
        )
        self._thread.start()
        log.info("[buzzer] Thread dimulai (line %d, gpio_ok=%s)", self._line_no, self._gpio_ok)

    def stop(self) -> None:
        self._stop_ev.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._set_gpio(False)
        self._close_gpio()
        log.info("[buzzer] Thread dihentikan.")

    # ------------------------------------------------------------------
    # Publik — dipanggil dari thread pipeline
    # ------------------------------------------------------------------
    def set_status(self, status: str) -> None:
        """Set status alarm: 'AMAN' | 'SIAGA' | 'BAHAYA'. Thread-safe."""
        with self._lock:
            self._status = status

    # ------------------------------------------------------------------
    # Loop thread buzzer
    # ------------------------------------------------------------------
    def _buzzer_loop(self) -> None:
        t0 = time.monotonic()
        while not self._stop_ev.is_set():
            with self._lock:
                status = self._status

            pattern = self._PATTERNS.get(status)

            if pattern is None:
                # AMAN — matikan buzzer
                self._set_gpio(False)
                time.sleep(0.05)
                t0 = time.monotonic()
                continue

            period, on_time = pattern
            elapsed = (time.monotonic() - t0) % period
            should_be_on = elapsed < on_time
            self._set_gpio(should_be_on)

            # Sleep sampai transisi berikutnya
            if should_be_on:
                sleep_s = on_time - elapsed
            else:
                sleep_s = period - elapsed
            time.sleep(max(0.005, min(sleep_s, 0.05)))

    # ------------------------------------------------------------------
    # GPIO helpers
    # ------------------------------------------------------------------
    def _open_gpio(self) -> bool:
        if not _GPIOD_AVAILABLE:
            return False
        try:
            self._req = gpiod.request_lines(
                self._chip_path,
                consumer="adas-buzzer",
                config={
                    self._line_no: gpiod.LineSettings(
                        direction    = Direction.OUTPUT,
                        output_value = Value.INACTIVE,
                        bias         = Bias.DISABLED,
                    )
                },
            )
            return True
        except Exception as e:
            log.error("[buzzer] Gagal membuka GPIO line %d: %s", self._line_no, e)
            return False

    def _set_gpio(self, on: bool) -> None:
        if not self._gpio_ok or self._req is None:
            return
        try:
            self._req.set_value(
                self._line_no,
                Value.ACTIVE if on else Value.INACTIVE,
            )
        except Exception as e:
            log.debug("[buzzer] set_value error: %s", e)

    def _close_gpio(self) -> None:
        if self._req is not None:
            try:
                self._req.release()
            except Exception:
                pass
            self._req = None
