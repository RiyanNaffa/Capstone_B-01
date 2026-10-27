#!/usr/bin/env python3
"""
lcd.py — Driver LCD ILI9341 240×320 + UI AMCAS untuk ADAS Bus Kampus UGM.

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

Hardware:
    TFT ILI9341 240×320 portrait
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

# ── Regions yang di-refresh parsial ──────────────────────────────────────────
R_TTC  = (8,   40, 224,  72)   # TTC frame + value + unit
R_DIST = (100, 126, 108, 24)   # dist value saja
R_GPS  = (4,    4,  90,  24)   # indikator GPS di kiri header

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

# ── Font 5×7 ─────────────────────────────────────────────────────────────────
FONT = {
    ' ': (0,0,0,0,0), '.': (0,0x60,0x60,0,0), '-': (8,8,8,8,8),
    '0':(0x3E,0x51,0x49,0x45,0x3E),'1':(0x00,0x42,0x7F,0x40,0x00),
    '2':(0x42,0x61,0x51,0x49,0x46),'3':(0x21,0x41,0x45,0x4B,0x31),
    '4':(0x18,0x14,0x12,0x7F,0x10),'5':(0x27,0x45,0x45,0x45,0x39),
    '6':(0x3C,0x4A,0x49,0x49,0x30),'7':(0x01,0x71,0x09,0x05,0x03),
    '8':(0x36,0x49,0x49,0x49,0x36),'9':(0x06,0x49,0x49,0x29,0x1E),
    'A':(0x7C,0x12,0x11,0x12,0x7C),'B':(0x7F,0x49,0x49,0x49,0x36),
    'C':(0x3E,0x41,0x41,0x41,0x22),'D':(0x7F,0x41,0x41,0x22,0x1C),
    'E':(0x7F,0x49,0x49,0x49,0x41),'G':(0x3E,0x41,0x49,0x49,0x7A),
    'H':(0x7F,0x08,0x08,0x08,0x7F),'I':(0x00,0x41,0x7F,0x41,0x00),
    'J':(0x20,0x40,0x41,0x3F,0x01),'K':(0x7F,0x08,0x14,0x22,0x41),
    'M':(0x7F,0x02,0x0C,0x02,0x7F),'N':(0x7F,0x04,0x08,0x10,0x7F),
    'O':(0x3E,0x41,0x41,0x41,0x3E),'R':(0x7F,0x09,0x19,0x29,0x46),
    'F':(0x7F,0x09,0x09,0x09,0x01),'L':(0x7F,0x40,0x40,0x40,0x40),
    'P':(0x7F,0x09,0x09,0x09,0x06),'U':(0x3F,0x40,0x40,0x40,0x3F),
    'S':(0x46,0x49,0x49,0x49,0x31),'T':(0x01,0x01,0x7F,0x01,0x01),
    'V':(0x1F,0x20,0x40,0x20,0x1F),'Y':(0x07,0x08,0x70,0x08,0x07),
    'W':(0x3F,0x40,0x38,0x40,0x3F),'Q':(0x1E,0x21,0x25,0x22,0x1D),
    'X':(0x41,0x22,0x1C,0x22,0x41),'Z':(0x61,0x51,0x49,0x45,0x43),
    'a':(0x20,0x54,0x54,0x54,0x78),'e':(0x38,0x54,0x54,0x54,0x18),
    'r':(0x7C,0x08,0x04,0x04,0x08),'o':(0x38,0x44,0x44,0x44,0x38),
    'i':(0x00,0x44,0x7D,0x40,0x00),'u':(0x3C,0x40,0x40,0x20,0x7C),
    'z':(0x44,0x64,0x54,0x4C,0x44),'p':(0xFE,0x11,0x11,0x11,0x0E),
    'n':(0x7C,0x08,0x04,0x04,0x78),'t':(0x04,0x3F,0x44,0x40,0x20),
    'c':(0x38,0x44,0x44,0x44,0x28),'g':(0x04,0x2A,0x2A,0x2A,0x1E),
    'b':(0x7F,0x44,0x44,0x44,0x38),'h':(0x7F,0x08,0x04,0x04,0x78),
    'l':(0x00,0x41,0x7F,0x40,0x00),'d':(0x38,0x44,0x44,0x48,0x7F),
    'f':(0x08,0x7E,0x09,0x01,0x02),'k':(0x7F,0x10,0x28,0x44,0x00),
    'v':(0x1C,0x20,0x40,0x20,0x1C),'w':(0x3C,0x40,0x30,0x40,0x3C),
    'x':(0x44,0x28,0x10,0x28,0x44),'y':(0x04,0x08,0x30,0x08,0x04),
    'q':(0x18,0x24,0x24,0x14,0x7C),'j':(0x20,0x40,0x44,0x3D,0x00),
    's':(0x48,0x54,0x54,0x54,0x20),'m':(0x7C,0x04,0x18,0x04,0x78),
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
        madctl = 0x48   # portrait, BGR
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
    """Salinan layar 240×320 di memori (2 byte/piksel, big-endian)."""

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
# Layer UI (identik dengan lcdcapstone.py)
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
    """Gambar indikator GPS kecil di kiri header, sebelah kiri teks AMCAS."""
    f.rect(*R_GPS, HEADER)   # bersihkan area dulu
    col   = _GPS_COL.get(gps_status, DIM)
    label = _GPS_LABEL.get(gps_status, '???')
    _fill_circle(f, 9, 16, 4, col)
    _text(f, label, 18, 9, 2, col)

def _draw_boot_icon(f: _Frame, cx: int, cy: int, r: int, status: str) -> None:
    """
    Gambar ikon status boot sebagai dua lingkaran konsentris — tanpa teks.

    Menghindari artefak "kotak hitam" yang terjadi saat _text() dipakai di
    atas lingkaran berwarna: _text() hanya menggambar piksel '1' pada bitmap
    font, sehingga piksel '0' (background glyph) tidak dicat — tapi saat
    warna teks = BG (hitam), piksel '1' justru menjadi hitam di atas circle.

    Pendekatan baru: ikon murni berbasis dua lingkaran konsentris:
        - Luar (r)     : warna status
        - Dalam (r//2) : BG untuk 'wait'/'warn'  (menunjukkan "pending/hati-hati")
        - Dalam (r//2) : warna lebih terang untuk 'ok' (solid filled = sukses)
        - Dalam (r//2) : BG dengan inti merah gelap untuk 'fail'
    """
    col = BOOT_COL.get(status, DIM)
    _fill_circle(f, cx, cy, r, col)                   # lingkaran luar (status color)
    if status == 'ok':
        # Lingkaran dalam lebih kecil, warna BG → membentuk cincin hijau
        _fill_circle(f, cx, cy, r - 3, BG)
        # Titik pusat solid hijau kecil sebagai konfirmasi
        _fill_circle(f, cx, cy, 2, col)
    elif status == 'fail':
        # Cincin merah, isi BG, titik merah gelap di tengah
        _fill_circle(f, cx, cy, r - 3, BG)
        _fill_circle(f, cx, cy, 2, RED)
    else:
        # 'wait' / 'warn': lingkaran dalam BG → cincin polos
        _fill_circle(f, cx, cy, r - 3, BG)


def _draw_boot_screen(f: _Frame, items: list) -> None:
    """
    Gambar layar boot status.

    items : list of (label_str, status_str)
        label_str  — nama komponen, maks ~14 karakter agar muat di 240px
        status_str — 'wait' | 'ok' | 'fail' | 'warn'

    Layout (portrait 240×320):
        Header bar  : "AMCAS  BOOTING..."
        Per item    : [ICON] LABEL
        Subtitle    : "Capstone B-01"
        Footer      : progress bar dari jumlah item 'ok'
    """
    f.rect(0, 0, LCD_W, LCD_H, BG)

    # Header
    f.rect(0, 0, 240, 32, HEADER)
    f.rect(0, 32, 240, 1, LINE)
    _text(f, "AMCAS", 8, 9, 2, TEXT)
    _text(f, "BOOTING", 142, 9, 2, AMBER)

    # Item list — mulai y=48, setiap baris 26px
    for i, (label, status) in enumerate(items):
        y   = 48 + i * 26
        col = BOOT_COL.get(status, DIM)
        # Ikon dua lingkaran konsentris — tanpa teks di atas circle
        _draw_boot_icon(f, 16, y + 8, 7, status)
        # Label
        _text(f, label, 32, y + 4, 2, col)

    # Subtitle "Capstone B-01" — di atas progress bar
    _text(f, "Capstone B-01", 44, 280, 2, TEXT2)

    # Progress bar bawah
    n_ok    = sum(1 for _, s in items if s == 'ok')
    n_total = len(items) if items else 1
    bar_w   = int((n_ok / n_total) * 224)
    f.rect(8,  300, 224, 12, LINE)           # background bar
    f.rect(8,  300, bar_w, 12, GREEN)        # fill
    pct_str = f"{int(n_ok/n_total*100)}."    # misal "75."
    _text(f, pct_str, 96, 301, 2, TEXT2)


def _draw_static(f: _Frame) -> None:
    f.rect(0,0,LCD_W,LCD_H,BG)
    f.rect(0,0,240,32,HEADER); f.rect(0,32,240,1,LINE)
    _text(f,"AMCAS",8,9,2,TEXT)
    _fill_circle(f,150,15,4,CYAN); _text(f,"ACTIVE",160,9,2,CYAN)
    _draw_rr(f,8,118,224,38,4,LINE)
    _text(f,"JARAK",16,130,2,TEXT2); _text(f,"m",212,133,2,TEXT2)
    f.rect(8,244,224,1,LINE)
    _line(f,20,243,58,176,DIM); _line(f,219,243,181,176,DIM)

def _draw_ttc(f: _Frame, st: int, s_ttc: str) -> None:
    f.rect(*R_TTC, BG)
    _draw_rr(f,8,40,224,72,4,STYLE[st][2])
    _text(f,"TTC",16,48,2,TEXT2)
    _text(f,s_ttc,16,68,5,STYLE[st][1])
    _text(f,"s",16+len(s_ttc)*30+4,82,3,TEXT2)

def _draw_dist(f: _Frame, s_dist: str) -> None:
    f.rect(*R_DIST, BG)
    _text(f,s_dist,208-len(s_dist)*18,126,3,TEXT)

def _draw_bus(f: _Frame, st: int) -> None:
    c = STYLE[st][1]
    f.rect(62,164,116,80,BG)
    f.rect(75,180,4,12,c); f.rect(79,180,3,2,c)
    f.rect(161,180,4,12,c); f.rect(158,180,3,2,c)
    _fill_rr(f,82,172,76,62,6,c)
    f.rect(88,178,64,24,BG); f.rect(88,214,12,6,BG); f.rect(140,214,12,6,BG)
    f.rect(86,225,68,3,BG); f.rect(90,234,12,8,DIM); f.rect(138,234,12,8,DIM)
    b = STYLE[st][3]
    bx,by,bw,bh,L,t = 64,164,112,80,12,2
    f.rect(bx,by,L,t,b);            f.rect(bx,by,t,L,b)
    f.rect(bx+bw-L,by,L,t,b);       f.rect(bx+bw-t,by,t,L,b)
    f.rect(bx,by+bh-t,L,t,b);       f.rect(bx,by+bh-L,t,L,b)
    f.rect(bx+bw-L,by+bh-t,L,t,b);  f.rect(bx+bw-t,by+bh-L,t,L,b)

def _draw_detect_text(f: _Frame, st: int) -> None:
    f.rect(24,252,194,16,BG)
    _text(f,"OBJEK TERDETEKSI",25,252,2,STYLE[st][4])

def _draw_status_bar(f: _Frame, st: int) -> None:
    name,acc,_,_,_,fill,txt,outline = STYLE[st]
    f.rect(8,276,224,36,BG)
    _fill_rr(f,8,276,224,36,4,fill)
    if outline: _draw_rr(f,8,276,224,36,4,acc)
    gw = 22 + len(name) * 18
    sx = 120 - gw // 2
    _fill_circle(f,sx+6,294,6,txt)
    _text(f,name,sx+22,284,3,txt)

def _draw_border(f: _Frame, on: bool) -> None:
    if on:
        f.rect(0,0,240,3,RED); f.rect(0,317,240,3,RED)
        f.rect(0,0,3,320,RED); f.rect(237,0,3,320,RED)
    else:
        f.rect(0,0,240,3,HEADER)
        f.rect(0,0,3,32,HEADER); f.rect(237,0,3,32,HEADER)
        f.rect(0,32,3,1,LINE);   f.rect(237,32,3,1,LINE)
        f.rect(0,33,3,287,BG);   f.rect(237,33,3,287,BG)
        f.rect(0,317,240,3,BG)

def _push_border(f: _Frame) -> None:
    f.push(0,0,240,3); f.push(0,317,240,3)
    f.push(0,3,3,314); f.push(237,3,3,314)

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
