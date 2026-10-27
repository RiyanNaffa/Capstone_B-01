#!/usr/bin/env python3
"""
extrinsic.py — Kalibrasi ekstrinsik: estimasi CAMERA_PITCH_DEG otomatis.

Metode:
    Letakkan checkerboard FLAT di tanah pada jarak yang diketahui dari kamera.
    Script mendeteksi checkerboard, menghitung posisi 3D sudut-sudutnya
    relatif terhadap kamera, lalu mengestimasi sudut pitch dari vektor normal
    bidang checkerboard vs bidang tanah.

    Hasil pitch divalidasi dengan menghitung error estimasi jarak IPM
    terhadap jarak ground truth yang dimasukkan pengguna.

Cara pakai:
    python3 extrinsic.py

Prerequisite:
    - calib_intrinsic.npz (output intrinsic.py)
    - Kamera terpasang di bus, posisi final
    - Checkerboard diletakkan FLAT di tanah, tegak lurus sumbu kamera
"""

import cv2
import numpy as np
import math
import os
import sys

try:
    from scipy.optimize import minimize_scalar, minimize
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("[WARN] scipy tidak terinstall — optimizer tidak tersedia, pakai estimasi geometri saja.")

# ── Konfigurasi ──────────────────────────────────────────────────────────────
CHECKERBOARD  = (9, 6)
SQUARE_SIZE_M = 0.033        # meter — kotak 3.3 cm, cetak A3 landscape (sesuai intrinsic.py)

INTRINSIC_NPZ = "calib_intrinsic.npz"
OUTPUT_NPZ    = "calib_extrinsic.npz"

CAMERA_DEVICE   = "/dev/v4l/by-id/usb-HD_Camera_Manufacturer_HD_USB_Camera-video-index0"
CAMERA_FALLBACK = "/dev/video1"

# ── Helper IPM ───────────────────────────────────────────────────────────────
def ipm_distance(v_px: float, H: float, pitch_deg: float, fy: float, cy: float) -> float:
    """Estimasi jarak dari baris piksel v menggunakan model IPM."""
    a   = math.radians(pitch_deg)
    v_c = v_px - cy
    phi = a + math.atan2(v_c, fy)
    if phi <= 1e-6:
        return float('inf')
    return H / math.tan(phi)


def open_camera():
    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_FALLBACK, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit("[ERROR] Tidak bisa membuka kamera.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  256)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def capture_checkerboard_frame(cap, K, dist):
    """
    Tampilkan live feed. Pengguna tekan SPACE saat checkerboard terdeteksi.
    Return: (frame, corners) atau None jika gagal.
    """
    print("\n  Arahkan kamera ke checkerboard yang ada di tanah.")
    print("  Tekan SPACE saat checkerboard terdeteksi (kotak hijau), Q untuk batal.\n")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

        preview = frame.copy()
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(preview, CHECKERBOARD, corners, found)
            cv2.putText(preview, "TERDETEKSI — SPACE untuk ambil",
                        (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        else:
            cv2.putText(preview, "Checkerboard tidak terdeteksi...",
                        (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 100, 255), 1)

        cv2.imshow("Extrinsic Calibration", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and found:
            cv2.destroyAllWindows()
            return frame, corners
        elif key == ord('q') or key == 27:
            cv2.destroyAllWindows()
            return None, None


def estimate_pitch_from_rvec(rvec) -> float:
    """
    Estimasi pitch kamera dari rotation vector solvePnP.

    solvePnP menghasilkan rotasi dari frame kamera ke frame checkerboard.
    Dari matriks rotasi, ekstrak sudut pitch (rotasi terhadap sumbu X).

    Catatan: tanda pitch bergantung pada orientasi pemasangan kamera.
    Nilai absolut yang penting, tanda disesuaikan dengan arah kamera (menunduk = positif).
    """
    R, _ = cv2.Rodrigues(rvec)
    # Euler angles dari rotation matrix (Rx, Ry, Rz)
    # pitch = atan2(R[2,1], R[2,2]) → rotasi sumbu X
    pitch_rad = math.atan2(R[2, 1], R[2, 2])
    pitch_deg = math.degrees(pitch_rad)

    # Konversi ke konvensi ADAS: positif = kamera menunduk ke bawah
    # Jika hasil negatif, ambil nilai absolut (kamera memang menunduk)
    return abs(pitch_deg)


def optimize_pitch(ground_truth_pts: list, H: float, fy: float, cy: float) -> float:
    """
    Optimalkan pitch menggunakan scipy minimize_scalar untuk meminimalkan
    error estimasi jarak IPM vs ground truth.

    ground_truth_pts: list of (v_pixel, distance_m)
    """
    def total_error(pitch_deg):
        err = 0.0
        for v_px, d_true in ground_truth_pts:
            d_est = ipm_distance(v_px, H, pitch_deg, fy, cy)
            err  += (d_est - d_true) ** 2
        return err

    result = minimize_scalar(total_error, bounds=(0.0, 30.0), method='bounded')
    return float(result.x)


def main():
    # ── Load hasil kalibrasi intrinsik ───────────────────────────────────────
    if not os.path.exists(INTRINSIC_NPZ):
        raise SystemExit(
            f"[ERROR] {INTRINSIC_NPZ} tidak ditemukan.\n"
            "Jalankan intrinsic.py terlebih dahulu."
        )

    data = np.load(INTRINSIC_NPZ)
    K    = data['K']
    dist = data['dist']
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    print(f"\n{'='*55}")
    print("  KALIBRASI EKSTRINSIK — Estimasi Pitch Kamera")
    print(f"{'='*55}")
    print(f"  Menggunakan intrinsik: fx={fx:.2f}, fy={fy:.2f}")

    # ── Input dari pengguna ───────────────────────────────────────────────────
    print("\n  Masukkan tinggi kamera dari tanah (CAMERA_HEIGHT_M).")
    print("  Ukur dengan meteran dari lantai ke lensa kamera.")
    try:
        H = float(input("  CAMERA_HEIGHT_M [default 1.20]: ").strip() or "1.20")
    except ValueError:
        H = 1.20
    print(f"  → Tinggi kamera: {H:.3f} m")

    print("\n  Berapa jarak checkerboard dari kamera untuk sesi ini?")
    print("  (Letakkan checkerboard FLAT di tanah, ukur dengan meteran)")
    try:
        d_gt = float(input("  Jarak ground truth (meter) [default 5.0]: ").strip() or "5.0")
    except ValueError:
        d_gt = 5.0
    print(f"  → Jarak ground truth: {d_gt:.2f} m")

    # ── Buka kamera dan ambil frame ───────────────────────────────────────────
    cap = open_camera()

    # ── Metode 1: Estimasi pitch dari solvePnP ────────────────────────────────
    print(f"\n{'─'*55}")
    print("  METODE 1: Estimasi pitch dari pose checkerboard (solvePnP)")
    print(f"{'─'*55}")

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M

    frame, corners = capture_checkerboard_frame(cap, K, dist)

    pitch_solvepnp = None
    ground_truth_pts = []

    if frame is not None and corners is not None:
        ret, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
        if ret:
            pitch_solvepnp = estimate_pitch_from_rvec(rvec)
            print(f"  Pitch dari solvePnP : {pitch_solvepnp:.2f}°")

            # Estimasi bottom center checkerboard (untuk ground truth optimizer)
            # Rata-rata baris piksel dari semua sudut
            v_mean = float(np.mean(corners[:, 0, 1]))
            ground_truth_pts.append((v_mean, d_gt))

            # Validasi: estimasi jarak dengan pitch ini
            d_est = ipm_distance(v_mean, H, pitch_solvepnp, fy, cy)
            print(f"  Validasi jarak     : estimasi={d_est:.2f}m, ground truth={d_gt:.2f}m")
            err_pct = abs(d_est - d_gt) / d_gt * 100
            print(f"  Error              : {err_pct:.1f}%")
        else:
            print("  [✗] solvePnP gagal.")
    else:
        print("  [!] Dilewati (pengguna membatalkan).")

    # ── Metode 2: Tambah ground truth dari jarak lain ─────────────────────────
    print(f"\n{'─'*55}")
    print("  METODE 2: Optimizer pitch dari multiple ground truth")
    print(f"{'─'*55}")
    print("  Opsional: tambah ground truth dari jarak lain untuk akurasi lebih baik.")
    print("  Letakkan checkerboard di jarak berbeda, tekan SPACE untuk ambil.")
    print("  Tekan Q jika sudah cukup.\n")

    more = input("  Tambah pengukuran jarak lain? (y/N): ").strip().lower()
    while more == 'y':
        try:
            d_extra = float(input("  Jarak (meter): ").strip())
        except ValueError:
            break

        frame2, corners2 = capture_checkerboard_frame(cap, K, dist)
        if frame2 is not None and corners2 is not None:
            v_mean2 = float(np.mean(corners2[:, 0, 1]))
            ground_truth_pts.append((v_mean2, d_extra))
            print(f"  [✓] Ditambahkan: v_pixel={v_mean2:.1f}, jarak={d_extra:.2f}m")

        more = input("  Tambah lagi? (y/N): ").strip().lower()

    cap.release()

    # ── Optimasi pitch ────────────────────────────────────────────────────────
    pitch_final = pitch_solvepnp or 12.0   # fallback ke default

    if SCIPY_OK and len(ground_truth_pts) >= 1:
        print("\n  Menjalankan optimizer pitch...")
        pitch_opt = optimize_pitch(ground_truth_pts, H, fy, cy)
        print(f"  Pitch dari optimizer : {pitch_opt:.2f}°")

        if pitch_solvepnp is not None:
            print(f"  Pitch dari solvePnP  : {pitch_solvepnp:.2f}°")
            print(f"  Selisih              : {abs(pitch_opt - pitch_solvepnp):.2f}°")

        # Gunakan optimizer jika ada lebih dari 1 ground truth, solvePnP jika hanya 1
        if len(ground_truth_pts) >= 2:
            pitch_final = pitch_opt
            method_used = "optimizer (multi ground truth)"
        elif pitch_solvepnp is not None:
            pitch_final = pitch_solvepnp
            method_used = "solvePnP"
        else:
            pitch_final = pitch_opt
            method_used = "optimizer (single ground truth)"
    elif pitch_solvepnp is not None:
        pitch_final = pitch_solvepnp
        method_used = "solvePnP"
    else:
        method_used = "default (kalibrasi gagal)"

    # ── Validasi akhir ────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  HASIL KALIBRASI EKSTRINSIK")
    print(f"{'='*55}")
    print(f"  Metode        : {method_used}")
    print(f"  CAMERA_HEIGHT_M   = {H:.3f}")
    print(f"  CAMERA_PITCH_DEG  = {pitch_final:.2f}")
    print()
    print("  Validasi estimasi jarak IPM:")
    for v_px, d_true in ground_truth_pts:
        d_est    = ipm_distance(v_px, H, pitch_final, fy, cy)
        err_pct  = abs(d_est - d_true) / d_true * 100
        status   = "✓" if err_pct < 10 else "⚠"
        print(f"  [{status}] Ground truth {d_true:.1f}m → estimasi {d_est:.2f}m (error {err_pct:.1f}%)")

    # ── Simpan hasil ──────────────────────────────────────────────────────────
    np.savez(OUTPUT_NPZ,
             H=H,
             pitch_deg=pitch_final,
             fx=fx, fy=fy, cx=cx, cy=cy,
             ground_truth_pts=np.array(ground_truth_pts) if ground_truth_pts else np.array([]))
    print(f"\n  [✓] Disimpan ke: {OUTPUT_NPZ}")

    print(f"\n  Salin nilai berikut ke adas/constants.py:")
    print(f"  FOCAL_LENGTH_PX   = {fx:.2f}")
    print(f"  CAMERA_HEIGHT_M   = {H:.3f}")
    print(f"  CAMERA_PITCH_DEG  = {pitch_final:.2f}")


if __name__ == "__main__":
    main()
