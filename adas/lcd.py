#!/usr/bin/env python3
"""
lcd.py — Driver LCD ILI9341 320×240 (LANDSCAPE) + UI AMCAS untuk ADAS Bus Kampus UGM.

Diambil dari lcdcapstone.py (root workspace) dan direfactor menjadi class
ACASDisplay yang dapat dipanggil dari pipeline.py.

Perubahan dari lcdcapstone.py:
  1. Semua state global (lcd, frame, buzzer, INPUT_MODE, dsb.) dipindahkan
     ke dalam kelas ACASDisplay — tidak ada variabel modul-level mutable.
  2. Konstanta hardware (SPI_BUS, GPIO_CHIP, PIN_DC, PIN_RST) dibaca dari
     constants.py sehingga satu tempat konfigurasi.
  3. Buzzer dipisahkan ke buzzer.py — ACASDisplay tidak kontrol buzzer.
  4. ACASDisplay.update(ttc, dist_m, status) dipanggil dari thread LCD
     di pipeline.py; tidak mengandung loop sendiri.
  5. ACASDisplay diinstansiasi di start() dan dilepas di stop() agar bisa
     di-restart tanpa perlu restart proses.

Revisi landscape:
  6. Orientasi layar diubah ke landscape 320×240 (MADCTL dari LCD_ROTATION
     di constants.py). Semua koordinat UI disusun ulang untuk 320×240.
  7. Indikator GPS dipindah ke tengah header sehingga tidak lagi menimpa
     tulisan "AMCAS" (pada versi portrait area GPS x 4–94 menimpa judul).
  8. Tabel font diganti dengan glcdfont Adafruit lengkap (ASCII 32–126),
     sehingga bentuk huruf sama dengan font TFT standar dan simbol '%' tersedia.
  9. Splash screen AMCAS (±2,3 s) ditampilkan otomatis di init_hardware()
     sebelum layar boot. Bisa dimatikan lewat LCD_SPLASH_ENABLED di constants.py.

Hardware:
    TFT ILI9341 320×240 landscape
    SPI1: MOSI=PH7, MISO=PH8, CLK=PH6, CS0=PH5
    RST → PI6 (line 262), DC → PH4 (line 228)
    LED → 3.3V (selalu on)
    /dev/spidev1.0
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

try:
    import spidev
    import gpiod
    from gpiod.line import Direction, Value, Bias
    _LCD_HW_AVAILABLE = True
except ImportError:
    _LCD_HW_AVAILABLE = False
    log.warning("[lcd] spidev/gpiod tidak tersedia — LCD berjalan dalam mode dummy.")

from .constants import (
    GPIO_CHIP,
    SPI_BUS,
    SPI_DEVICE,
    SPI_SPEED_HZ,
    SPI_CHUNK,
    LCD_DC_LINE,
    LCD_RST_LINE,
    LCD_W, LCD_H,
    LCD_ROTATION,
    LCD_SPLASH_ENABLED,
    LCD_UPDATE_INTERVAL,
    TTC_BAHAYA,
    TTC_SIAGA,
)

# ── Warna RGB565 ──────────────────────────────────────────────────────────────
def _c565(r: int, g: int, b: int) -> int:
    return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)

BG       = _c565(0x0B, 0x12, 0x20)
HEADER   = _c565(0x13, 0x1D, 0x30)
LINE     = _c565(0x26, 0x34, 0x4D)
TEXT     = _c565(0xFF, 0xFF, 0xFF)
TEXT2    = _c565(0x9C, 0xA8, 0xB8)
DIM      = _c565(0x4A, 0x58, 0x70)
CYAN     = _c565(0x56, 0xC8, 0xE0)
GREEN    = _c565(0x35, 0xD0, 0x7F)
GREEN_BG = _c565(0x0F, 0x2E, 0x26)
AMBER    = _c565(0xFF, 0xC8, 0x57)
RED      = _c565(0xFF, 0x4D, 0x4D)

# ── State styles: (name, accent, frameTTC, bracket, detectText, barFill, barText, outline) ──
AMAN, SIAGA, BAHAYA = 0, 1, 2
STYLE = (
    ("AMAN",   GREEN, LINE,  DIM,   TEXT2, GREEN_BG, GREEN, True),
    ("SIAGA",  AMBER, AMBER, AMBER, AMBER, AMBER,    BG,    False),
    ("BAHAYA", RED,   RED,   RED,   RED,   RED,      TEXT,  False),
)

# ── Layout landscape 320×240 ─────────────────────────────────────────────────
#   Header      y   0– 32   AMCAS | indikator GPS | ● ACTIVE
#   Panel TTC   x   8–155   y  40–111   (kiri, informasi utama)
#   Panel JARAK x 164–311   y  40–111   (kanan)
#   Deteksi     y 117–190   bus + bracket di kiri, "OBJEK TERDETEKSI" di kanan
#   Status bar  x   8–311   y 196–231
# ── Regions yang di-refresh parsial (x, y, w, h) ─────────────────────────────
R_TTC  = (8,   40, 148, 72)    # panel TTC lengkap
R_DIST = (164, 40, 148, 72)    # panel JARAK lengkap
R_GPS  = (76,   3, 124, 27)    # indikator GPS di tengah header

# ── Boot screen ───────────────────────────────────────────────────────────────
# Status item: (label, status) di mana status: 'wait'|'ok'|'fail'|'warn'
BOOT_COL = {
    'wait': DIM,
    'ok'  : GREEN,
    'fail': RED,
    'warn': AMBER,
}
BOOT_ICON = {
    'wait': '-',
    'ok'  : 'O',   # huruf O sebagai ceklis — font terbatas
    'fail': 'X',
    'warn': '.',
}

# ── Font 5×7 — glcdfont Adafruit_GFX (ASCII 32–126, font TFT standar) ────────
FONT = {
    ' ':(0x00,0x00,0x00,0x00,0x00), '!':(0x00,0x00,0x5F,0x00,0x00), '"':(0x00,0x07,0x00,0x07,0x00), '#':(0x14,0x7F,0x14,0x7F,0x14),
    '$':(0x24,0x2A,0x7F,0x2A,0x12), '%':(0x23,0x13,0x08,0x64,0x62), '&':(0x36,0x49,0x56,0x20,0x50), "'":(0x00,0x08,0x07,0x03,0x00),
    '(':(0x00,0x1C,0x22,0x41,0x00), ')':(0x00,0x41,0x22,0x1C,0x00), '*':(0x2A,0x1C,0x7F,0x1C,0x2A), '+':(0x08,0x08,0x3E,0x08,0x08),
    ',':(0x00,0x80,0x70,0x30,0x00), '-':(0x08,0x08,0x08,0x08,0x08), '.':(0x00,0x00,0x60,0x60,0x00), '/':(0x20,0x10,0x08,0x04,0x02),
    '0':(0x3E,0x51,0x49,0x45,0x3E), '1':(0x00,0x42,0x7F,0x40,0x00), '2':(0x72,0x49,0x49,0x49,0x46), '3':(0x21,0x41,0x49,0x4D,0x33),
    '4':(0x18,0x14,0x12,0x7F,0x10), '5':(0x27,0x45,0x45,0x45,0x39), '6':(0x3C,0x4A,0x49,0x49,0x31), '7':(0x41,0x21,0x11,0x09,0x07),
    '8':(0x36,0x49,0x49,0x49,0x36), '9':(0x46,0x49,0x49,0x29,0x1E), ':':(0x00,0x00,0x14,0x00,0x00), ';':(0x00,0x40,0x34,0x00,0x00),
    '<':(0x00,0x08,0x14,0x22,0x41), '=':(0x14,0x14,0x14,0x14,0x14), '>':(0x00,0x41,0x22,0x14,0x08), '?':(0x02,0x01,0x59,0x09,0x06),
    '@':(0x3E,0x41,0x5D,0x59,0x4E), 'A':(0x7C,0x12,0x11,0x12,0x7C), 'B':(0x7F,0x49,0x49,0x49,0x36), 'C':(0x3E,0x41,0x41,0x41,0x22),
    'D':(0x7F,0x41,0x41,0x41,0x3E), 'E':(0x7F,0x49,0x49,0x49,0x41), 'F':(0x7F,0x09,0x09,0x09,0x01), 'G':(0x3E,0x41,0x41,0x51,0x73),
    'H':(0x7F,0x08,0x08,0x08,0x7F), 'I':(0x00,0x41,0x7F,0x41,0x00), 'J':(0x20,0x40,0x41,0x3F,0x01), 'K':(0x7F,0x08,0x14,0x22,0x41),
    'L':(0x7F,0x40,0x40,0x40,0x40), 'M':(0x7F,0x02,0x1C,0x02,0x7F), 'N':(0x7F,0x04,0x08,0x10,0x7F), 'O':(0x3E,0x41,0x41,0x41,0x3E),
    'P':(0x7F,0x09,0x09,0x09,0x06), 'Q':(0x3E,0x41,0x51,0x21,0x5E), 'R':(0x7F,0x09,0x19,0x29,0x46), 'S':(0x26,0x49,0x49,0x49,0x32),
    'T':(0x03,0x01,0x7F,0x01,0x03), 'U':(0x3F,0x40,0x40,0x40,0x3F), 'V':(0x1F,0x20,0x40,0x20,0x1F), 'W':(0x3F,0x40,0x38,0x40,0x3F),
    'X':(0x63,0x14,0x08,0x14,0x63), 'Y':(0x03,0x04,0x78,0x04,0x03), 'Z':(0x61,0x59,0x49,0x4D,0x43), '[':(0x00,0x7F,0x41,0x41,0x41),
    '\\':(0x02,0x04,0x08,0x10,0x20), ']':(0x00,0x41,0x41,0x41,0x7F), '^':(0x04,0x02,0x01,0x02,0x04), '_':(0x40,0x40,0x40,0x40,0x40),
    '`':(0x00,0x03,0x07,0x08,0x00), 'a':(0x20,0x54,0x54,0x78,0x40), 'b':(0x7F,0x28,0x44,0x44,0x38), 'c':(0x38,0x44,0x44,0x44,0x28),
    'd':(0x38,0x44,0x44,0x28,0x7F), 'e':(0x38,0x54,0x54,0x54,0x18), 'f':(0x00,0x08,0x7E,0x09,0x02), 'g':(0x18,0xA4,0xA4,0x9C,0x78),
    'h':(0x7F,0x08,0x04,0x04,0x78), 'i':(0x00,0x44,0x7D,0x40,0x00), 'j':(0x20,0x40,0x40,0x3D,0x00), 'k':(0x7F,0x10,0x28,0x44,0x00),
    'l':(0x00,0x41,0x7F,0x40,0x00), 'm':(0x7C,0x04,0x78,0x04,0x78), 'n':(0x7C,0x08,0x04,0x04,0x78), 'o':(0x38,0x44,0x44,0x44,0x38),
    'p':(0xFC,0x18,0x24,0x24,0x18), 'q':(0x18,0x24,0x24,0x18,0xFC), 'r':(0x7C,0x08,0x04,0x04,0x08), 's':(0x48,0x54,0x54,0x54,0x24),
    't':(0x04,0x04,0x3F,0x44,0x24), 'u':(0x3C,0x40,0x40,0x20,0x7C), 'v':(0x1C,0x20,0x40,0x20,0x1C), 'w':(0x3C,0x40,0x30,0x40,0x3C),
    'x':(0x44,0x28,0x10,0x28,0x44), 'y':(0x4C,0x90,0x90,0x90,0x7C), 'z':(0x44,0x64,0x54,0x4C,0x44), '{':(0x00,0x08,0x36,0x41,0x00),
    '|':(0x00,0x00,0x77,0x00,0x00), '}':(0x00,0x41,0x36,0x08,0x00), '~':(0x02,0x01,0x02,0x04,0x02),
}

TTC_MAX = 99.9

# ─────────────────────────────────────────────────────────────────────────────
# Low-level driver ILI9341
# ─────────────────────────────────────────────────────────────────────────────
class _ILI9341:
    """Driver SPI minimal untuk ILI9341. Dipakai hanya oleh _Frame."""

    def __init__(self) -> None:
        self.spi = spidev.SpiDev()
        self.spi.open(SPI_BUS, SPI_DEVICE)
        self.spi.max_speed_hz = SPI_SPEED_HZ
        self.spi.mode = 0

        self._req_dc  = self._gpio_out(LCD_DC_LINE,  "lcd-dc",  False)
        self._req_rst = self._gpio_out(LCD_RST_LINE, "lcd-rst", True)
        self._reset()
        self._init_panel()

    @staticmethod
    def _gpio_out(line_no: int, consumer: str, initial: bool):
        return gpiod.request_lines(
            GPIO_CHIP,
            consumer=consumer,
            config={
                line_no: gpiod.LineSettings(
                    direction    = Direction.OUTPUT,
                    output_value = Value.ACTIVE if initial else Value.INACTIVE,
                    bias         = Bias.DISABLED,
                )
            },
        )

    def _set(self, req, line_no: int, on: bool) -> None:
        req.set_value(line_no, Value.ACTIVE if on else Value.INACTIVE)

    def _write(self, data) -> None:
        for i in range(0, len(data), SPI_CHUNK):
            self.spi.writebytes(data[i:i + SPI_CHUNK])

    def _reset(self) -> None:
        self._set(self._req_rst, LCD_RST_LINE, True);  time.sleep(0.1)
        self._set(self._req_rst, LCD_RST_LINE, False); time.sleep(0.1)
        self._set(self._req_rst, LCD_RST_LINE, True);  time.sleep(0.15)

    def cmd(self, c: int, data=None) -> None:
        self._set(self._req_dc, LCD_DC_LINE, False)
        self.spi.writebytes([c])
        if data:
            self._set(self._req_dc, LCD_DC_LINE, True)
            self._write(list(data))

    def _init_panel(self) -> None:
        # Landscape: 90° = MV|BGR (0x28), 270° = MX|MY|MV|BGR (0xE8).
        # Jika tampilan terbalik di prototipe, ganti LCD_ROTATION di constants.py.
        madctl = 0xE8 if LCD_ROTATION == 270 else 0x28
        seq = [
            (0x01, None,                                                      0.15),
            (0xEF, [0x03,0x80,0x02],                                          0),
            (0xCF, [0x00,0xC1,0x30],                                          0),
            (0xED, [0x64,0x03,0x12,0x81],                                     0),
            (0xE8, [0x85,0x00,0x78],                                          0),
            (0xCB, [0x39,0x2C,0x00,0x34,0x02],                               0),
            (0xF7, [0x20],                                                    0),
            (0xEA, [0x00,0x00],                                               0),
            (0xC0, [0x23],                                                    0),
            (0xC1, [0x10],                                                    0),
            (0xC5, [0x3E,0x28],                                               0),
            (0xC7, [0x86],                                                    0),
            (0x36, [madctl],                                                  0),
            (0x37, [0x00],                                                    0),
            (0x3A, [0x55],                                                    0),
            (0xB1, [0x00,0x18],                                               0),
            (0xB6, [0x08,0x82,0x27],                                          0),
            (0xF2, [0x00],                                                    0),
            (0x26, [0x01],                                                    0),
            (0xE0, [0x0F,0x31,0x2B,0x0C,0x0E,0x08,0x4E,0xF1,
                    0x37,0x07,0x10,0x03,0x0E,0x09,0x00],                     0),
            (0xE1, [0x00,0x0E,0x14,0x03,0x11,0x07,0x31,0xC1,
                    0x48,0x08,0x0F,0x0C,0x31,0x36,0x0F],                     0),
            (0x11, None,                                                      0.12),
            (0x29, None,                                                      0.05),
        ]
        for c, d, wait in seq:
            self.cmd(c, d)
            if wait:
                time.sleep(wait)

    def block(self, x: int, y: int, w: int, h: int, data: bytes) -> None:
        x1, y1 = x + w - 1, y + h - 1
        self.cmd(0x2A, [x >> 8, x & 0xFF, x1 >> 8, x1 & 0xFF])
        self.cmd(0x2B, [y >> 8, y & 0xFF, y1 >> 8, y1 & 0xFF])
        self.cmd(0x2C)
        self._set(self._req_dc, LCD_DC_LINE, True)
        self._write(data)

    def close(self) -> None:
        self._req_dc.release()
        self._req_rst.release()
        self.spi.close()


# ─────────────────────────────────────────────────────────────────────────────
# Framebuffer in-memory
# ─────────────────────────────────────────────────────────────────────────────
class _Frame:
    """Salinan layar LCD_W×LCD_H (320×240) di memori (2 byte/piksel, big-endian)."""

    def __init__(self, lcd: _ILI9341) -> None:
        self.lcd = lcd
        self.buf = bytearray(LCD_W * LCD_H * 2)

    def rect(self, x: int, y: int, w: int, h: int, c: int) -> None:
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, LCD_W), min(y + h, LCD_H)
        if x1 <= x0 or y1 <= y0:
            return
        row = bytes((c >> 8, c & 0xFF)) * (x1 - x0)
        for yy in range(y0, y1):
            i = (yy * LCD_W + x0) * 2
            self.buf[i:i + len(row)] = row

    def push(self, x: int = 0, y: int = 0, w: int = LCD_W, h: int = LCD_H) -> None:
        if x == 0 and w == LCD_W:
            data = bytes(self.buf[y * LCD_W * 2:(y + h) * LCD_W * 2])
        else:
            data = bytearray()
            for yy in range(y, y + h):
                i = (yy * LCD_W + x) * 2
                data += self.buf[i:i + w * 2]
            data = bytes(data)
        self.lcd.block(x, y, w, h, data)


# ─────────────────────────────────────────────────────────────────────────────
# Primitif gambar
# ─────────────────────────────────────────────────────────────────────────────
def _text(s: _Frame, t: str, x: int, y: int, size: int, c: int) -> None:
    for i, ch in enumerate(t):
        glyph = FONT.get(ch, FONT[' '])
        cx = x + i * 6 * size
        for col in range(5):
            bits = glyph[col]; row = 0
            while row < 8:
                if (bits >> row) & 1:
                    start = row
                    while row < 8 and (bits >> row) & 1: row += 1
                    s.rect(cx + col * size, y + start * size, size, (row - start) * size, c)
                else:
                    row += 1

def _fill_circle(s: _Frame, cx: int, cy: int, r: int, c: int) -> None:
    for dy in range(-r, r + 1):
        v = r * r + r * 0.6 - dy * dy
        h = int(v ** 0.5) if v >= 0 else -1
        if h >= 0:
            s.rect(cx - h, cy + dy, 2 * h + 1, 1, c)

def _in_rr(x: int, y: int, w: int, h: int, r: int, X: int, Y: int) -> bool:
    if X < x or Y < y or X >= x + w or Y >= y + h: return False
    cx = x + r if X < x + r else (x + w - 1 - r if X > x + w - 1 - r else X)
    cy = y + r if Y < y + r else (y + h - 1 - r if Y > y + h - 1 - r else Y)
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r * r + r * 0.6

def _fill_rr(s: _Frame, x: int, y: int, w: int, h: int, r: int, c: int) -> None:
    s.rect(x, y + r, w, h - 2 * r, c)
    for i in range(r):
        for Y in (y + i, y + h - 1 - i):
            inset = 0
            while inset < r and not _in_rr(x, y, w, h, r, x + inset, Y): inset += 1
            s.rect(x + inset, Y, w - 2 * inset, 1, c)

def _draw_rr(s: _Frame, x: int, y: int, w: int, h: int, r: int, c: int) -> None:
    s.rect(x + r, y, w - 2 * r, 1, c);         s.rect(x + r, y + h - 1, w - 2 * r, 1, c)
    s.rect(x, y + r, 1, h - 2 * r, c);          s.rect(x + w - 1, y + r, 1, h - 2 * r, c)
    for (ox, oy) in ((x, y), (x+w-1-r, y), (x, y+h-1-r), (x+w-1-r, y+h-1-r)):
        for Y in range(oy, oy + r + 1):
            for X in range(ox, ox + r + 1):
                if _in_rr(x,y,w,h,r,X,Y) and not (
                        _in_rr(x,y,w,h,r,X-1,Y) and _in_rr(x,y,w,h,r,X+1,Y) and
                        _in_rr(x,y,w,h,r,X,Y-1) and _in_rr(x,y,w,h,r,X,Y+1)):
                    s.rect(X, Y, 1, 1, c)

def _line(s: _Frame, x0: int, y0: int, x1: int, y1: int, c: int) -> None:
    dx, sx = abs(x1-x0), (1 if x0 < x1 else -1)
    dy, sy = -abs(y1-y0), (1 if y0 < y1 else -1)
    e = dx + dy
    while True:
        s.rect(x0, y0, 1, 1, c)
        if x0 == x1 and y0 == y1: break
        e2 = 2 * e
        if e2 >= dy: e += dy; x0 += sx
        if e2 <= dx: e += dx; y0 += sy


# ─────────────────────────────────────────────────────────────────────────────
# Layer UI — landscape 320×240
# ─────────────────────────────────────────────────────────────────────────────
# Warna indikator GPS
_GPS_COL = {
    'OK'   : _c565(0x35, 0xD0, 0x7F),   # hijau — GPS valid
    'STALE': _c565(0xFF, 0xC8, 0x57),   # amber — GPS stuck/drift
    'NOFIX': _c565(0x9C, 0xA8, 0xB8),   # abu   — belum fix
    'ERROR': _c565(0xFF, 0x4D, 0x4D),   # merah — tidak tersambung
}
_GPS_LABEL = {
    'OK'   : 'GPS',
    'STALE': 'GPS.',
    'NOFIX': 'NO FIX',
    'ERROR': 'NO GPS',
}

def _draw_gps_indicator(f: _Frame, gps_status: str) -> None:
    """Indikator GPS di tengah header (antara AMCAS dan ● ACTIVE)."""
    f.rect(*R_GPS, HEADER)   # bersihkan area dulu
    col   = _GPS_COL.get(gps_status, DIM)
    label = _GPS_LABEL.get(gps_status, '???')
    _fill_circle(f, 88, 15, 4, col)
    _text(f, label, 98, 9, 2, col)

def _draw_boot_icon(f: _Frame, cx: int, cy: int, r: int, status: str) -> None:
    """
    Ikon status boot: dua lingkaran konsentris — tanpa teks.
        - 'ok'   : cincin hijau + titik hijau di tengah
        - 'fail' : cincin merah + titik merah di tengah
        - 'wait' / 'warn' : cincin polos (abu / amber)
    """
    col = BOOT_COL.get(status, DIM)
    _fill_circle(f, cx, cy, r, col)
    _fill_circle(f, cx, cy, r - 3, BG)
    if status == 'ok':
        _fill_circle(f, cx, cy, 2, col)
    elif status == 'fail':
        _fill_circle(f, cx, cy, 2, RED)


def _draw_header(f: _Frame, right_label: str, right_col: int) -> None:
    """Header 320×33: AMCAS kiri, label status rata kanan (berakhir di x=311)."""
    f.rect(0, 0, LCD_W, 32, HEADER)
    f.rect(0, 32, LCD_W, 1, LINE)
    _text(f, "AMCAS", 8, 9, 2, TEXT)
    x = 312 - len(right_label) * 12
    _fill_circle(f, x - 10, 15, 4, right_col)
    _text(f, right_label, x, 9, 2, right_col)


def _draw_boot_screen(f: _Frame, items: list) -> None:
    """
    Gambar layar boot status (landscape 320×240).

    items : list of (label_str, status_str)
        label_str  — nama komponen (maks ~20 karakter)
        status_str — 'wait' | 'ok' | 'fail' | 'warn'

    Layout:
        Header      : "AMCAS            ● BOOTING"
        Daftar item : y 42, 70, 98, 126, 154  →  [IKON] LABEL
        Subtitle    : "Capstone B-01"  (tengah, y 188)
        Progress    : bar x 8–311, y 214–229, persentase di tengah bar
    """
    f.rect(0, 0, LCD_W, LCD_H, BG)
    _draw_header(f, "BOOTING", AMBER)

    for i, (label, status) in enumerate(items):
        y   = 42 + i * 28
        col = BOOT_COL.get(status, DIM)
        _draw_boot_icon(f, 20, y + 8, 7, status)
        _text(f, label, 36, y + 2, 2, col)

    sub = "Capstone B-01"
    _text(f, sub, 160 - (len(sub) * 12 - 2) // 2, 188, 2, TEXT2)

    n_ok    = sum(1 for _, s in items if s == 'ok')
    n_total = len(items) if items else 1
    bar_w   = int((n_ok / n_total) * 304)
    f.rect(8, 214, 304, 16, LINE)
    f.rect(8, 214, bar_w, 16, GREEN)
    pct_str = f"{int(n_ok / n_total * 100)}%"
    _text(f, pct_str, 160 - (len(pct_str) * 12 - 2) // 2, 215, 2, TEXT)


# ── Splash screen (landscape 320×240) ────────────────────────────────────────
#   Logo bracket + bus mini  x 128–191, y 24–75
#   AMCAS (size 5)           x  87–231, y 90–124, garis pemindai y 132
#   Subjudul (size 2)        y 146      Progress bar x 60–259, y 186–191
#   "INITIALIZING SYSTEM..." y 198      Footer "BUS KAMPUS UGM" y 222
SPLASH_SUBTITLE = "PERINGATAN DINI TABRAKAN"

def _splash_brackets(f: _Frame) -> None:
    x, y, w, h, L, t = 128, 24, 64, 52, 10, 2
    f.rect(x, y, L, t, CYAN);            f.rect(x, y, t, L, CYAN)
    f.rect(x+w-L, y, L, t, CYAN);        f.rect(x+w-t, y, t, L, CYAN)
    f.rect(x, y+h-t, L, t, CYAN);        f.rect(x, y+h-L, t, L, CYAN)
    f.rect(x+w-L, y+h-t, L, t, CYAN);    f.rect(x+w-t, y+h-L, t, L, CYAN)

def _splash_bus(f: _Frame) -> None:
    f.rect(136, 38, 3, 7, CYAN);  f.rect(139, 38, 1, 2, CYAN)
    f.rect(181, 38, 3, 7, CYAN);  f.rect(180, 38, 1, 2, CYAN)
    _fill_rr(f, 140, 32, 40, 34, 4, CYAN)
    f.rect(145, 37, 30, 12, BG)
    f.rect(145, 55, 6, 4, BG);    f.rect(169, 55, 6, 4, BG)
    f.rect(143, 61, 34, 2, BG)
    f.rect(146, 66, 7, 5, DIM);   f.rect(167, 66, 7, 5, DIM)

def _run_splash(f: _Frame) -> None:
    """Splash ±2,3 detik (blocking). Tiap langkah hanya mengirim area yang berubah."""
    f.rect(0, 0, LCD_W, LCD_H, BG)
    _splash_brackets(f)
    f.rect(60, 186, 200, 1, LINE); f.rect(60, 191, 200, 1, LINE)       # Progress_Track
    f.rect(60, 186, 1, 6, LINE);   f.rect(259, 186, 1, 6, LINE)
    _text(f, "INITIALIZING SYSTEM...", 94, 198, 1, TEXT2)
    _text(f, "BUS KAMPUS UGM", 118, 222, 1, DIM)
    f.push()
    time.sleep(0.3)
    _splash_bus(f);  f.push(128, 24, 64, 52)                             # Bus_Mini
    time.sleep(0.4)
    for i, ch in enumerate("AMCAS"):                                     # 1 huruf / 80 ms
        _text(f, ch, 87 + i * 30, 90, 5, TEXT)
        f.rect(87 + i * 29, 132, 29, 2, CYAN)                            # Scan_Line
        f.push(87, 90, 146, 44)
        time.sleep(0.08)
    time.sleep(0.1)
    w = len(SPLASH_SUBTITLE) * 12 - 2
    _text(f, SPLASH_SUBTITLE, 160 - w // 2, 146, 2, TEXT2)
    f.push(0, 144, LCD_W, 18)
    for k in range(1, 11):                                               # progress 0,9 s
        f.rect(61, 187, 198 * k // 10, 4, CYAN)
        f.push(60, 186, 200, 6)
        time.sleep(0.09)
    time.sleep(0.3)


def _draw_static(f: _Frame) -> None:
    f.rect(0, 0, LCD_W, LCD_H, BG)
    _draw_header(f, "ACTIVE", CYAN)

def _draw_ttc(f: _Frame, st: int, s_ttc: str) -> None:
    """Panel TTC kiri: label size 2, angka size 5, satuan size 3."""
    f.rect(*R_TTC, BG)
    _draw_rr(f, 8, 40, 148, 72, 4, STYLE[st][2])
    _text(f, "TTC", 16, 48, 2, TEXT2)
    _text(f, s_ttc, 16, 68, 5, STYLE[st][1])
    _text(f, "s", 16 + len(s_ttc) * 30 + 4, 82, 3, TEXT2)

def _draw_dist(f: _Frame, s_dist: str) -> None:
    """Panel JARAK kanan: label size 2, angka size 4, satuan size 2."""
    f.rect(*R_DIST, BG)
    _draw_rr(f, 164, 40, 148, 72, 4, LINE)
    _text(f, "JARAK", 172, 48, 2, TEXT2)
    _text(f, s_dist, 172, 70, 4, TEXT)
    _text(f, "m", 172 + len(s_dist) * 24 + 4, 84, 2, TEXT2)

def _draw_bus(f: _Frame, st: int) -> None:
    """Area deteksi kiri: jalan, bus tampak belakang, bracket target."""
    c = STYLE[st][1]
    f.rect(3, 116, 134, 76, BG)
    f.rect(8, 190, 128, 1, LINE)                              # Road_Ground
    _line(f, 8, 189, 17, 124, DIM)                            # Lane_Left
    _line(f, 131, 189, 122, 124, DIM)                         # Lane_Right
    f.rect(32, 131, 4, 12, c);  f.rect(36, 131, 2, 2, c)      # Mirror_L
    f.rect(104, 131, 4, 12, c); f.rect(102, 131, 2, 2, c)     # Mirror_R
    _fill_rr(f, 38, 123, 64, 56, 6, c)                        # Body
    f.rect(44, 129, 52, 24, BG)                               # Rear_Window
    f.rect(44, 161, 10, 6, BG); f.rect(86, 161, 10, 6, BG)    # Lamp_L / Lamp_R
    f.rect(42, 171, 56, 3, BG)                                # Bumper
    f.rect(46, 179, 12, 8, DIM); f.rect(82, 179, 12, 8, DIM)  # Wheel_L / Wheel_R
    b = STYLE[st][3]                                          # Target_Brackets
    bx, by, bw, bh, L, t = 20, 117, 100, 72, 12, 2
    f.rect(bx, by, L, t, b);            f.rect(bx, by, t, L, b)
    f.rect(bx+bw-L, by, L, t, b);       f.rect(bx+bw-t, by, t, L, b)
    f.rect(bx, by+bh-t, L, t, b);       f.rect(bx, by+bh-L, t, L, b)
    f.rect(bx+bw-L, by+bh-t, L, t, b);  f.rect(bx+bw-t, by+bh-L, t, L, b)

def _draw_detect_text(f: _Frame, st: int) -> None:
    """Teks deteksi di kanan bus, dua baris."""
    f.rect(140, 128, 172, 50, BG)
    _text(f, "OBJEK", 148, 134, 2, STYLE[st][4])
    _text(f, "TERDETEKSI", 148, 156, 2, STYLE[st][4])

def _draw_status_bar(f: _Frame, st: int) -> None:
    name, acc, _, _, _, fill, txt, outline = STYLE[st]
    f.rect(8, 196, 304, 36, BG)
    _fill_rr(f, 8, 196, 304, 36, 4, fill)
    if outline:
        _draw_rr(f, 8, 196, 304, 36, 4, acc)
    gw = 22 + len(name) * 18
    sx = 160 - gw // 2
    _fill_circle(f, sx + 6, 214, 6, txt)
    _text(f, name, sx + 22, 204, 3, txt)

def _draw_border(f: _Frame, on: bool) -> None:
    W, H = LCD_W, LCD_H
    if on:
        f.rect(0, 0, W, 3, RED); f.rect(0, H - 3, W, 3, RED)
        f.rect(0, 0, 3, H, RED); f.rect(W - 3, 0, 3, H, RED)
    else:                     # kembalikan warna asli di bawah border
        f.rect(0, 0, W, 3, HEADER)
        f.rect(0, 0, 3, 32, HEADER); f.rect(W - 3, 0, 3, 32, HEADER)
        f.rect(0, 32, 3, 1, LINE);   f.rect(W - 3, 32, 3, 1, LINE)
        f.rect(0, 33, 3, H - 33, BG); f.rect(W - 3, 33, 3, H - 33, BG)
        f.rect(0, H - 3, W, 3, BG)

def _push_border(f: _Frame) -> None:
    W, H = LCD_W, LCD_H
    f.push(0, 0, W, 3); f.push(0, H - 3, W, 3)
    f.push(0, 3, 3, H - 6); f.push(W - 3, 3, 3, H - 6)

def _fmt(ttc: float, dist: float):
    s_ttc  = "--.-" if ttc > TTC_MAX else f"{ttc:.1f}"
    s_dist = "--.-" if dist < 0 else f"{dist:.1f}"
    return s_ttc, s_dist

def _classify(ttc: float) -> int:
    if ttc < TTC_BAHAYA: return BAHAYA
    if ttc <= TTC_SIAGA: return SIAGA
    return AMAN


# ─────────────────────────────────────────────────────────────────────────────
# Kelas publik — dipakai dari pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
class ACASDisplay:
    """
    Driver LCD AMCAS yang berjalan di background thread terpisah.

    Cara pakai:
        display = ACASDisplay()
        display.start()
        display.update(ttc=3.2, dist_m=18.5, status='SIAGA')
        ...
        display.stop()
    """

    def __init__(self) -> None:
        self._lock     = threading.Lock()
        self._stop_ev  = threading.Event()
        self._thread   : Optional[threading.Thread] = None

        # State shared antara caller dan thread LCD
        self._ttc_val   : float = float('inf')
        self._dist_val  : float = -1.0   # sentinel → ditampilkan "--.-"
        self._status    : str   = 'AMAN'
        self._gps_status: str   = 'NOFIX'
        self._dirty     : bool  = True

        # Handle hardware
        self._lcd  : Optional[_ILI9341] = None
        self._frame: Optional[_Frame]   = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def init_hardware(self) -> bool:
        """
        Inisialisasi hardware LCD secara sinkron (dari main thread).
        Dipanggil sebelum start() agar boot screen bisa ditampilkan
        sebelum thread LCD dimulai.

        Returns True jika berhasil, False jika hardware tidak tersedia.
        """
        if not _LCD_HW_AVAILABLE:
            return False
        try:
            self._lcd   = _ILI9341()
            self._frame = _Frame(self._lcd)
            log.info("[lcd] Hardware diinisialisasi.")
            if LCD_SPLASH_ENABLED:
                _run_splash(self._frame)
            return True
        except Exception as e:
            log.error("[lcd] Gagal inisialisasi hardware: %s", e)
            return False

    def show_boot_screen(self, items: list) -> None:
        """
        Tampilkan boot status screen secara sinkron (blocking, main thread).
        Dipanggil berulang kali saat pipeline sedang inisialisasi komponen.

        items : list of (label, status)
            status: 'wait' | 'ok' | 'fail' | 'warn'

        Contoh:
            lcd.show_boot_screen([
                ("Model YOLO",  "ok"),
                ("Kamera",      "wait"),
                ("GPS",         "wait"),
            ])
        """
        if self._frame is None:
            return   # hardware belum diinisialisasi, skip
        try:
            _draw_boot_screen(self._frame, items)
            self._frame.push()
        except Exception as e:
            log.error("[lcd] Boot screen error: %s", e)

    def start(self) -> None:
        self._stop_ev.clear()
        self._thread = threading.Thread(
            target=self._lcd_loop,
            name="lcd",
            daemon=True,
        )
        self._thread.start()
        log.info("[lcd] Thread dimulai.")

    def stop(self) -> None:
        self._stop_ev.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        log.info("[lcd] Thread dihentikan.")

    # ------------------------------------------------------------------
    # Publik — dipanggil dari thread pipeline
    # ------------------------------------------------------------------
    def update(self, ttc: float, dist_m: float, status: str, gps_status: str = 'NOFIX') -> None:
        """Perbarui nilai yang akan ditampilkan. Thread-safe."""
        with self._lock:
            changed = (ttc != self._ttc_val or dist_m != self._dist_val
                       or status != self._status or gps_status != self._gps_status)
            if changed:
                self._ttc_val    = ttc
                self._dist_val   = dist_m
                self._status     = status
                self._gps_status = gps_status
                self._dirty      = True

    # ------------------------------------------------------------------
    # Loop thread LCD
    # ------------------------------------------------------------------
    def _lcd_loop(self) -> None:
        if not _LCD_HW_AVAILABLE:
            log.warning("[lcd] Hardware tidak tersedia — thread berjalan dalam mode dummy.")
            while not self._stop_ev.is_set():
                time.sleep(LCD_UPDATE_INTERVAL)
            return

        try:
            # Jika sudah diinisialisasi oleh init_hardware() (boot screen),
            # gunakan instance yang ada — jangan inisialisasi ulang
            if self._lcd is None or self._frame is None:
                self._lcd   = _ILI9341()
                self._frame = _Frame(self._lcd)
            _draw_static(self._frame)
            _draw_gps_indicator(self._frame, 'NOFIX')
            self._frame.push()
        except Exception as e:
            log.error("[lcd] Gagal menginisialisasi hardware: %s", e)
            return

        current_st     = -1
        last_ttc_str   = ""
        last_dist_str  = ""
        last_gps_st    = ""
        border_on      = False
        last_blink     = time.monotonic()

        while not self._stop_ev.is_set():
            now = time.monotonic()

            with self._lock:
                ttc        = self._ttc_val
                dist       = self._dist_val
                status     = self._status
                gps_status = self._gps_status
                dirty      = self._dirty
                self._dirty = False

            st            = _classify(ttc)
            s_ttc, s_dist = _fmt(ttc, dist)

            try:
                if st != current_st:
                    current_st = st
                    _draw_ttc(self._frame, st, s_ttc)
                    _draw_dist(self._frame, s_dist)
                    _draw_bus(self._frame, st)
                    _draw_detect_text(self._frame, st)
                    _draw_status_bar(self._frame, st)
                    border_on = (st == BAHAYA)
                    _draw_border(self._frame, border_on)
                    # Gambar ulang GPS indicator karena full push
                    _draw_gps_indicator(self._frame, gps_status)
                    self._frame.push()
                    last_ttc_str, last_dist_str = s_ttc, s_dist
                    last_gps_st = gps_status
                    last_blink  = now
                    log.info("[lcd] State: %s | TTC: %s s | Jarak: %s m | GPS: %s",
                             STYLE[st][0], s_ttc, s_dist, gps_status)
                elif dirty:
                    if s_ttc != last_ttc_str:
                        _draw_ttc(self._frame, st, s_ttc)
                        self._frame.push(*R_TTC)
                        last_ttc_str = s_ttc
                    if s_dist != last_dist_str:
                        _draw_dist(self._frame, s_dist)
                        self._frame.push(*R_DIST)
                        last_dist_str = s_dist
                    # Update indikator GPS jika berubah (partial refresh)
                    if gps_status != last_gps_st:
                        _draw_gps_indicator(self._frame, gps_status)
                        self._frame.push(*R_GPS)
                        last_gps_st = gps_status

                # Blink border saat BAHAYA
                if st == BAHAYA and now - last_blink >= 0.5:
                    last_blink = now
                    border_on  = not border_on
                    _draw_border(self._frame, border_on)
                    _push_border(self._frame)

            except Exception as e:
                log.error("[lcd] Error saat menggambar: %s", e)

            time.sleep(LCD_UPDATE_INTERVAL)

        # Cleanup
        if self._frame is not None:
            try:
                self._frame.rect(0, 0, LCD_W, LCD_H, 0x0000)
                self._frame.push()
            except Exception:
                pass
        if self._lcd is not None:
            try:
                self._lcd.close()
            except Exception:
                pass
