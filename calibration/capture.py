#!/usr/bin/env python3
"""
capture.py — Ambil foto kalibrasi dari kamera yang terpasang di bus.

Cara pakai:
    python3 capture.py

Tekan SPACE untuk simpan frame, Q untuk keluar.
Target: 20-30 foto checkerboard dari berbagai sudut dan jarak.

PENTING: Kamera harus sudah terpasang di posisi FINAL di bus.
         Yang dipindah-pindah adalah checkerboard, bukan kamera.
"""

import cv2
import os
import time

# ── Konfigurasi ──────────────────────────────────────────────────────────────
CAMERA_DEVICE = "/dev/v4l/by-id/usb-HD_Camera_Manufacturer_HD_USB_Camera-video-index0"
CAMERA_FALLBACK = "/dev/video1"
OUTPUT_DIR    = "calib_images"
CAPTURE_W     = 256    # resolusi yang dipakai sistem ADAS
CAPTURE_H     = 256

def open_camera():
    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[WARN] Tidak bisa buka {CAMERA_DEVICE}, coba fallback...")
        cap = cv2.VideoCapture(CAMERA_FALLBACK, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit("[ERROR] Tidak bisa membuka kamera.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')])
    count    = existing

    cap = open_camera()
    print(f"\n{'='*55}")
    print("  CAPTURE KALIBRASI — Kamera terpasang di bus")
    print(f"{'='*55}")
    print(f"  Output    : {OUTPUT_DIR}/")
    print(f"  Sudah ada : {existing} foto")
    print(f"  Target    : 20-30 foto")
    print()
    print("  PANDUAN:")
    print("  - Pegang checkerboard di depan kamera")
    print("  - Pindahkan ke berbagai posisi (lihat README.md)")
    print("  - SPACE = simpan foto")
    print("  - Q     = selesai")
    print(f"{'='*55}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame kosong, skip...")
            continue

        # Overlay info di frame preview
        preview = frame.copy()
        cv2.putText(preview, f"Saved: {count} | SPACE=save Q=quit",
                    (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        cv2.imshow("Calibration Capture (pastikan checkerboard terlihat jelas)", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            fname = os.path.join(OUTPUT_DIR, f"calib_{count:03d}.jpg")
            cv2.imwrite(fname, frame)
            count += 1
            print(f"  [✓] Disimpan: {fname}  (total: {count})")
            if count < 20:
                remaining = 20 - count
                print(f"      Butuh minimal {remaining} foto lagi.")
            elif count == 20:
                print("  [!] Sudah mencapai minimum 20 foto. Bisa lanjut atau tambah lebih.")

        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n  Selesai. Total foto tersimpan: {count}")
    if count < 20:
        print(f"  [!] Masih kurang {20 - count} foto. Kalibrasi mungkin kurang akurat.")
    else:
        print("  [✓] Jumlah foto cukup. Lanjutkan ke: python3 intrinsic.py")

if __name__ == "__main__":
    main()
