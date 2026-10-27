#!/usr/bin/env python3
"""
intrinsic.py — Kalibrasi intrinsik kamera dari foto checkerboard.

Output: calib_intrinsic.npz berisi:
    K     — camera matrix (fx, fy, cx, cy)
    dist  — distortion coefficients
    rms   — reprojection error (< 1.0 px = bagus)

Cara pakai:
    python3 intrinsic.py

Jalankan setelah capture.py menghasilkan minimal 20 foto di calib_images/.
"""

import cv2
import numpy as np
import glob
import os
import sys

# ── Konfigurasi checkerboard ─────────────────────────────────────────────────
# Jumlah sudut DALAM (bukan jumlah kotak).
# Checkerboard 10×7 kotak → sudut dalam = (9, 6)
CHECKERBOARD = (9, 6)

# Ukuran satu kotak dalam meter.
# Cetak di A3 landscape, kotak 3.3 cm:
#   canvas = (10+2×1) × (7+2×1) = 12×9 kotak (border 1 kotak per sisi)
#   canvas px @96 DPI = 1497 × 1123 px — tepat muat di A3 landscape (1587×1123 px)
SQUARE_SIZE_M = 0.033

IMAGE_DIR  = "calib_images"
OUTPUT_NPZ = "calib_intrinsic.npz"


def main():
    print(f"\n{'='*55}")
    print("  KALIBRASI INTRINSIK KAMERA")
    print(f"{'='*55}")
    print(f"  Checkerboard  : {CHECKERBOARD[0]+1}×{CHECKERBOARD[1]+1} kotak")
    print(f"  Ukuran kotak  : {SQUARE_SIZE_M*100:.1f} cm")
    print(f"  Folder input  : {IMAGE_DIR}/")
    print(f"{'='*55}\n")

    # Titik 3D checkerboard (z=0, flat di bidang)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M

    objpoints = []   # titik 3D
    imgpoints = []   # titik 2D di gambar
    valid_images = []

    images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    if not images:
        raise SystemExit(f"[ERROR] Tidak ada gambar di {IMAGE_DIR}/. Jalankan capture.py dulu.")

    print(f"  Memproses {len(images)} gambar...")

    img_shape = None
    for fname in images:
        img  = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]   # (w, h)

        # Cari sudut checkerboard
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

        if ret:
            # Perbaiki akurasi sudut ke sub-piksel
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            valid_images.append(os.path.basename(fname))
            print(f"  [✓] {os.path.basename(fname)}")
        else:
            print(f"  [✗] {os.path.basename(fname)} — checkerboard tidak terdeteksi")

    print(f"\n  Valid: {len(valid_images)} / {len(images)} gambar")

    if len(valid_images) < 10:
        raise SystemExit(
            f"[ERROR] Hanya {len(valid_images)} gambar valid. "
            "Minimal 10 diperlukan. Tambah foto atau periksa konfigurasi CHECKERBOARD."
        )

    print("\n  Menghitung kalibrasi...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    print(f"\n{'='*55}")
    print("  HASIL KALIBRASI")
    print(f"{'='*55}")
    print(f"  RMS reprojection error : {rms:.4f} px  ", end="")
    if rms < 0.5:
        print("(✓ sangat baik)")
    elif rms < 1.0:
        print("(✓ baik)")
    else:
        print("(⚠ kurang baik, tambah foto atau periksa checkerboard)")

    print(f"\n  Camera Matrix (K):")
    print(f"    fx = {fx:.2f} px")
    print(f"    fy = {fy:.2f} px")
    print(f"    cx = {cx:.2f} px  (principal point x)")
    print(f"    cy = {cy:.2f} px  (principal point y)")
    print(f"\n  Distortion coefficients:")
    print(f"    {dist.ravel()}")

    # Simpan hasil
    np.savez(OUTPUT_NPZ, K=K, dist=dist, rms=rms,
             image_size=img_shape, valid_count=len(valid_images))
    print(f"\n  [✓] Hasil disimpan ke: {OUTPUT_NPZ}")

    print(f"\n  Salin nilai berikut ke adas/constants.py:")
    print(f"  FOCAL_LENGTH_PX = {fx:.2f}   # fx dari kalibrasi intrinsik")
    print(f"  # (fy = {fy:.2f}, gunakan fx jika lensa simetris)")

    print(f"\n  Lanjutkan ke: python3 extrinsic.py")

if __name__ == "__main__":
    main()
