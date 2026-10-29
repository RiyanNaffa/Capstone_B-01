#!/usr/bin/env python3
"""
constants.py — Semua konstanta sistem AMCAS Bus Kampus UGM
Orange Pi Zero 2W (Allwinner H618, 4x Cortex-A53 @ 1.5 GHz, 2 GB RAM)

Daftar interface:
- GPS: UART0 (uart0)
    - RX    -> Pin 8:   UART0_TX    | PH0
    - TX    -> Pin 10:  UART0_RX    | PH1
- LCD TFT: SPI1 (spidev1.0)
    - MOSI  -> Pin 19:  SPI1_MOSI   | PH7
    - MISO  -> Pin 21:  SPI1_MISO   | PH8
    - CLK   -> Pin 23:  SPI1_CLK    | PH6
    - CS    -> Pin 24:  SPI1_CS0    | PH5
    - RST   -> Pin 22:  GPIO262     | PI6
    - DC    -> Pin 18:  GPIO228     | PH4
    - LED   -> Pin 17               | 3.3V
- Buzzer: GPIO
    - Base  -> Pin 12: GPIO257      | PI1 
- Camera: USB HD Camera (v4l2)
"""

import cv2  # hanya untuk LK_PARAMS

# ===========================================================================
# PATH & DEVICE
# ===========================================================================
# /opt/adas/ dipilih karena:
#   - /opt/ adalah konvensi FHS untuk paket perangkat lunak pihak ketiga;
#     lebih rapi daripada /home/orangepi/ yang bergantung pada user session.
#   - Memudahkan pemberian hak akses yang konsisten tanpa bergantung pada
#     home-directory pengguna yang bisa berganti.
MODEL_PATH   = "/opt/adas/models/best_int8.tflite"
LOG_DIR      = "/var/log/adas/"
VENV_PYTHON  = "/opt/adas/.venv/bin/python3"

GPS_PORT     = "/dev/ttyS0"        # UART0 (sesuai hardware aktual, ubah ke ttyS0 saat pindah ke UART0)
GPS_BAUD     = 9600                # Baud rate standar modul GPS NMEA 0183

SPI_BUS      = 1                   # /dev/spidev1.0 → SPI1 header 40-pin
SPI_DEVICE   = 0

GPIO_CHIP    = "/dev/gpiochip0"

# GPIO line number — diambil dari lcdcapstone.py yang sudah terbukti jalan
BUZZER_ENABLED = False  # Set True untuk aktifkan buzzer, False untuk tes tanpa bunyi
BUZZER_LINE  = 257      # PI1 (pin fisik 12) → bank I (8×32) + 1 = 257
LCD_DC_LINE  = 228   # PIN_DC  di lcdcapstone.py = 228
LCD_RST_LINE = 262   # PIN_RST di lcdcapstone.py = 262

# Kamera: gunakan keyword yang ada di nama device v4l2
# ls /dev/v4l/by-id/ menampilkan dua entry untuk kamera HD USB:
#   usb-HD_Camera_Manufacturer_HD_USB_Camera-video-index0  ← stream video utama
#   usb-HD_Camera_Manufacturer_HD_USB_Camera-video-index1  ← metadata/control, tidak bisa dibuka
# Keyword harus cukup spesifik agar match ke index0 saja.
CAMERA_KEYWORD  = "HD_USB_Camera-video-index0"  # match hanya ke stream video utama
CAMERA_FALLBACK = "/dev/video1"                 # fallback: video1 = kamera USB di sistem ini

# ===========================================================================
# MODEL & INFERENSI
# ===========================================================================
INPUT_W, INPUT_H = 256, 256        # Resolusi input model (sesuai training)
CONF_THRESHOLD   = 0.35            # Di bawah ini dianggap noise
NMS_THRESHOLD    = 0.45            # IoU threshold untuk NMS
NUM_THREADS      = 4               # Gunakan semua 4 core Cortex-A53
CLASS_NAMES      = ['bus', 'mobil', 'motor', 'truk']

# ===========================================================================
# KAMERA & IPM
# ===========================================================================
CAMERA_HEIGHT_M        = 2.10    # meter
CAMERA_PITCH_DEG       = 9.0    # derajat ke bawah
FOCAL_LENGTH_PX        = 213.7   # piksel
CAMERA_VEHICLE_WIDTH_M = 2.20    # meter (lebar bodi bus, eksklusif spion)

# ===========================================================================
# DIMENSI KENDARAAN (meter) — konteks Indonesia
# Tuple: (W_real, H_real, use_height_default)
# ===========================================================================
# Justifikasi tiap kelas:
#   bus  : Lebar bus kota Indonesia umumnya 2.40–2.55 m (PP No.55/2012 max 2.5 m).
#           Tinggi variatif (3.0–3.5 m), jadi lebar lebih konsisten.
#   mobil: Mayoritas LCGC Indonesia (Calya, Sigra, BRIO) lebar 1.65–1.73 m.
#           Sedan 1.70–1.76 m, SUV 1.75–1.85 m → rerata 1.70 m.
#           Lebar lebih konsisten antar segmen daripada tinggi (sedan 1.45 m
#           vs SUV 1.65 m).
#   motor: Tinggi roda→helm sangat konsisten ≈ 1.10 m (data Yamaha NMAX, Honda
#           Vario, Kawasaki KLX). Diambil minimum antara estimasi tinggi & lebar
#           sebagai pendekatan konservatif (safety).
#   truk : Lebar diregulasi PP No.55/2012 ≤ 2.5 m. Default 2.40 m (truk medium
#           Isuzu ELF / Mitsubishi Colt Diesel yang dominan di jalan UGM).
VEHICLE_DIMS = {
    #          W_real   H_real   use_height_default
    'bus'  : (2.50,    3.20,    False),
    'mobil': (1.70,    1.50,    False),
    'motor': (0.75,    1.10,    True ),
    'truk' : (2.40,    3.00,    False),
}

# Rasio tepi frame: bbox center di luar zona ini dianggap "kendaraan menyamping"
# 0.35 berarti 35% dari tepi kiri/kanan (70% total) → zona tengah = 30% tengah frame
CENTER_ZONE_RATIO = 0.35

# Aspect ratio bbox minimum agar fallback ke tinggi (kemungkinan tampak samping)
SIDE_ASPECT_RATIO = 1.5

# ===========================================================================
# GPS QUALITY
# ===========================================================================
# Staleness: jika nilai GPS tidak berubah selama ini → anggap stuck/drift
GPS_STALE_TIMEOUT_S  = 3.0    # detik — GPS consumer grade update 1 Hz;
                               # jika 3 detik tidak ada perubahan nilai = stuck

# Sensor fusion: jika selisih V_gps vs V_closing_ipm melebihi threshold ini,
# GPS dianggap outlier dan tidak dipakai sebagai floor.
# 5 m/s = 18 km/h — toleransi cukup longgar untuk GPS consumer grade
GPS_FUSION_MAX_DIFF_MS = 4.0  # m/s

# ===========================================================================
# TTC
# ===========================================================================
TTC_BAHAYA        = 2.0    # detik — di bawah ini status BAHAYA
TTC_SIAGA         = 5.0    # detik — di bawah ini status SIAGA
ALPHA_GPS         = 0.7    # safety factor GPS fallback (0.5–1.0):
                           #   v_floor = v_ego × 0.7 → target diasumsikan bergerak
                           #   mendekati dengan 70% kecepatan ego kita
TTC_MAX_DISPLAY   = 99.9   # untuk tampilan LCD saja

# Window histori jarak untuk smoothing closing-rate (redam noise IPM)
TTC_HISTORY_LEN   = 5      # frame

# ===========================================================================
# FILTER MACET
# ===========================================================================
# Saat macet: V_ego kecil DAN jarak stabil → paksa AMAN agar alarm tidak
# berbunyi terus-menerus saat bus berhenti di belakang kendaraan lain.
MACET_SPEED_MS    = 1.0    # m/s — threshold kecepatan ego "hampir diam"
MACET_CLOSING_MS  = 0.5    # m/s — threshold V_closing "jarak stabil"
N_MACET_FRAMES    = 30     # frame berturut-turut sebelum filter aktif
                           #   @ 10 FPS ≈ 3 detik, cukup untuk membedakan
                           #   berhenti sementara vs berhenti di macet

# ===========================================================================
# DEBOUNCE ALARM
# ===========================================================================
# Alarm BAHAYA hanya aktif setelah kondisi terpenuhi N_TRIGGER_FRAMES berturut-turut,
# lalu ada cooldown N_COOLDOWN_FRAMES sebelum bisa trigger lagi.
# Ini menghindari false alarm dari satu frame deteksi buruk.
N_TRIGGER_FRAMES  = 3      # frame — ≈0.3 s @ 10 FPS
N_COOLDOWN_FRAMES = 15     # frame — ≈1.5 s @ 10 FPS

# ===========================================================================
# TRACKER
# ===========================================================================
MAX_LOST_FRAMES   = 8      # frame — hapus track jika tidak match selama ini
IOU_ASSIGN_THRESH = 0.25   # IoU minimum agar deteksi di-assign ke track lama

# ===========================================================================
# OPTICAL FLOW (tracker_optflow.py)
# ===========================================================================
# Toggle: True = pakai OptFlowTracker, False = pakai VehicleTracker (default IoU)
USE_OPTICAL_FLOW   = False  # ganti True untuk coba optical flow tracker

# Jalankan detector penuh tiap N frame; frame di antaranya gunakan optical flow
DETECTION_INTERVAL = 5     # frame — 5 = inferensi 2× per detik @ 10 FPS

# Parameter Lucas-Kanade Sparse Optical Flow
# winSize (15,15): cukup untuk menangkap pergerakan ≤7 piksel/frame,
#   trade-off: makin besar makin smooth tapi lambat
# maxLevel=2: 3 level pyramid → tangkap gerakan besar tanpa terlalu mahal
LK_PARAMS = dict(
    winSize  = (15, 15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# ===========================================================================
# SPI LCD
# ===========================================================================
SPI_SPEED_HZ = 40_000_000  # 40 MHz; turunkan ke 16 MHz jika gambar rusak
SPI_CHUNK    = 4096         # bytes per transfer (batas buffer spidev)
LCD_W, LCD_H = 320, 240     # resolusi ILI9341 LANDSCAPE (lebar × tinggi)
LCD_ROTATION = 90           # 90 atau 270 — ganti ke 270 jika tampilan terbalik di prototipe
LCD_SPLASH_ENABLED = True   # splash AMCAS ±2,3 s saat LCD pertama menyala (False = lewati)

# ===========================================================================
# PERFORMA
# ===========================================================================
LCD_UPDATE_INTERVAL = 0.2   # detik — update LCD max 5 Hz (cukup untuk info TTC)
FRAME_BUFFER_SIZE   = 1     # queue.Queue maxsize — drop frame lama, jangan buffer
FPS_TARGET          = 10    # target inferensi (realistis untuk H618 tanpa NPU)

# ===========================================================================
# RECORDING
# ===========================================================================
# RECORD_ENABLED mengaktifkan AVI raw, AVI annotated, dan CSV TTC per frame.
RECORD_ENABLED   = True              # False = nonaktifkan semua recording
RECORD_DIR       = "/opt/adas/recordings"
# Rekam dua track sekaligus:
#   annotated — frame 256×256 dengan bbox, label, jarak, TTC, status
#   raw       — frame 256×256 tanpa anotasi (untuk analisis post-hoc)
RECORD_ANNOTATED = True
RECORD_RAW       = True
RECORD_FPS       = 10.0              # FPS VideoWriter (sesuaikan dengan FPS aktual)
RECORD_CODEC     = "MJPG"            # MJPG ringan di CPU, bisa diputar langsung

# ===========================================================================
# VISUALISASI (stream HTTP opsional)
# ===========================================================================
DISPLAY_SCALE  = 2.5
STREAM_W = int(INPUT_W * DISPLAY_SCALE)
STREAM_H = int(INPUT_H * DISPLAY_SCALE)
COL_PRIORITY = (0,   40, 230)
COL_NORMAL   = (30, 200,  50)
COL_LOST     = (110, 110, 110)
COL_DOT      = (0,  230, 230)
COL_DANGER   = (0,    0, 255)
COL_WARNING  = (0,  165, 255)
