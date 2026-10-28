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

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
import queue
import threading
import time

import cv2
import numpy as np

try:
    from scipy.optimize import minimize_scalar, minimize
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("[WARN] scipy tidak terinstall — optimizer tidak tersedia, pakai estimasi geometri saja.")

# ── Konfigurasi ──────────────────────────────────────────────────────────────
CHECKERBOARD  = (9, 6)
SQUARE_SIZE_M = 0.032        # meter (sesuai intrinsic.py)

INTRINSIC_NPZ = "calib_intrinsic.npz"
OUTPUT_NPZ    = "calib_extrinsic.npz"

CAMERA_DEVICE   = "/dev/v4l/by-id/usb-HD_Camera_Manufacturer_HD_USB_Camera-video-index0"
CAMERA_FALLBACK = "/dev/video1"
WEB_HOST = "127.0.0.1"
WEB_PORT = 8081
JPEG_QUALITY = 80

WEB_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Extrinsic Calibration</title>
  <style>
    body { font: 16px sans-serif; max-width: 720px; margin: 24px auto; padding: 0 12px; }
    img { display: block; width: 100%; max-width: 512px; background: #222; }
    button { font-size: 1.1rem; margin: 12px 8px 12px 0; padding: 10px 18px; }
    #status { min-height: 1.5em; }
  </style>
</head>
<body>
  <h1>Extrinsic Calibration</h1>
  <p>Place the checkerboard flat on the ground at the entered distance.
  Wait for the green detected-corners overlay, then capture.</p>
  <img src="/stream.mjpg" alt="Live camera preview">
  <p><button id="capture">Capture detected checkerboard</button>
  <button id="cancel">Cancel current capture</button></p>
  <p id="status">Waiting for the checkerboard...</p>
  <p>Capture button is enabled only while the board is detected. Stop the
  calibration with Ctrl+C in the terminal.</p>
  <script>
    const status = document.getElementById("status");
    const capture = document.getElementById("capture");
    async function updateStatus() {
      try {
        const response = await fetch("/status", { cache: "no-store" });
        const result = await response.json();
        capture.disabled = !result.detected;
        status.textContent = result.detected
          ? "Checkerboard detected — ready to capture."
          : "Waiting for the checkerboard to be detected...";
      } catch (error) {
        status.textContent = `Preview status error: ${error.message}`;
      }
    }
    capture.onclick = async () => {
      capture.disabled = true;
      status.textContent = "Capturing...";
      try {
        const response = await fetch("/capture", { method: "POST" });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || response.statusText);
        status.textContent = "Checkerboard captured. Continue in the terminal.";
      } catch (error) {
        status.textContent = `Capture error: ${error.message}`;
      }
      updateStatus();
    };
    document.getElementById("cancel").onclick = async () => {
      const response = await fetch("/cancel", { method: "POST" });
      const result = await response.json();
      status.textContent = result.message;
    };
    updateStatus();
    setInterval(updateStatus, 500);
  </script>
</body>
</html>
"""

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


class CameraFeed:
    def __init__(self, cap):
        self.cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._corners = None
        self._preview_jpeg = None
        self._running = threading.Event()
        self._running.set()
        self.capture_requests = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._read_frames, daemon=True)

    def start(self):
        self._thread.start()
        deadline = time.monotonic() + 5.0
        while self.frame() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if self.frame() is None:
            raise RuntimeError("No camera frames received within 5 seconds.")

    def _read_frames(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        while self._running.is_set():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            else:
                corners = None

            preview = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(preview, CHECKERBOARD, corners, True)
                label, color = "DETECTED - capture in browser", (0, 255, 0)
            else:
                label, color = "Waiting for checkerboard...", (0, 100, 255)
            cv2.putText(preview, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            encoded, buffer = cv2.imencode(
                ".jpg",
                preview,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            if encoded:
                with self._lock:
                    self._frame = frame
                    self._corners = corners
                    self._preview_jpeg = buffer.tobytes()

    def frame(self):
        with self._lock:
            return self._frame

    def snapshot(self):
        with self._lock:
            if self._frame is None or self._corners is None:
                return None, None
            return self._frame.copy(), self._corners.copy()

    def preview(self):
        with self._lock:
            return self._preview_jpeg

    def detected(self):
        with self._lock:
            return self._corners is not None

    def stop(self):
        self._running.clear()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


class ExtrinsicHandler(BaseHTTPRequestHandler):
    server_version = "ExtrinsicCalibration/1.0"

    def log_message(self, fmt, *args):
        print(f"[HTTP] {self.address_string()} - {fmt % args}")

    def do_GET(self):
        if self.path == "/":
            body = WEB_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/status":
            self._send_json(200, {"detected": self.server.camera.detected()})
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while self.server.camera._running.is_set():
                    frame = self.server.camera.preview()
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/capture":
            frame, corners = self.server.camera.snapshot()
            if frame is None or corners is None:
                self._send_json(409, {"error": "Checkerboard is not detected yet."})
                return
            try:
                self.server.camera.capture_requests.put_nowait((frame, corners))
            except queue.Full:
                self._send_json(409, {"error": "A capture is already waiting in the terminal."})
                return
            print("[OK] Checkerboard frame captured from browser.")
            self._send_json(200, {"message": "Frame captured."})
        elif self.path == "/cancel":
            try:
                self.server.camera.capture_requests.put_nowait(None)
            except queue.Full:
                self._send_json(409, {"error": "A capture request is already waiting."})
                return
            self._send_json(200, {"message": "Capture cancelled. Continue in the terminal."})
        else:
            self.send_error(404)

    def _send_json(self, status, result):
        body = json.dumps(result).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ExtrinsicServer(ThreadingHTTPServer):
    daemon_threads = True


def start_capture_server(cap, host, port):
    camera = CameraFeed(cap)
    try:
        camera.start()
        server = ExtrinsicServer((host, port), ExtrinsicHandler)
    except Exception:
        camera.stop()
        cap.release()
        raise
    server.camera = camera
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def capture_checkerboard_frame(server):
    print("\n  Buka halaman kalibrasi ekstrinsik di browser.")
    print("  Tunggu sudut checkerboard terdeteksi, lalu tekan Capture.\n")
    result = server.camera.capture_requests.get()
    if result is None:
        return None, None
    return result


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
    parser = argparse.ArgumentParser(
        description="Estimate camera pitch using a browser-based MJPEG preview."
    )
    parser.add_argument(
        "--host",
        default=WEB_HOST,
        help="Address to bind (default: 127.0.0.1; use 0.0.0.0 for LAN access).",
    )
    parser.add_argument("--port", type=int, default=WEB_PORT, help="HTTP port (default: 8081).")
    args = parser.parse_args()

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
    capture_server, capture_thread = start_capture_server(cap, args.host, args.port)
    if args.host == "0.0.0.0":
        print("[WARN] Calibration page is accessible on this network; use a trusted network.")
        page_host = "<ORANGE_PI_IP>"
    else:
        page_host = "127.0.0.1"
    print(f"\n  Buka http://{page_host}:{args.port}/ di browser.")
    print("  Live preview akan menandai checkerboard jika pola terdeteksi.")

    # ── Metode 1: Estimasi pitch dari solvePnP ────────────────────────────────
    print(f"\n{'─'*55}")
    print("  METODE 1: Estimasi pitch dari pose checkerboard (solvePnP)")
    print(f"{'─'*55}")

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M

    frame, corners = capture_checkerboard_frame(capture_server)

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
    print("  Letakkan checkerboard di jarak berbeda, lalu ambil dari browser.")
    print("  Klik Cancel current capture jika pengukuran tambahan tidak diperlukan.\n")

    more = input("  Tambah pengukuran jarak lain? (y/N): ").strip().lower()
    while more == 'y':
        try:
            d_extra = float(input("  Jarak (meter): ").strip())
        except ValueError:
            break

        frame2, corners2 = capture_checkerboard_frame(capture_server)
        if frame2 is not None and corners2 is not None:
            v_mean2 = float(np.mean(corners2[:, 0, 1]))
            ground_truth_pts.append((v_mean2, d_extra))
            print(f"  [✓] Ditambahkan: v_pixel={v_mean2:.1f}, jarak={d_extra:.2f}m")

        more = input("  Tambah lagi? (y/N): ").strip().lower()

    capture_server.shutdown()
    capture_server.server_close()
    capture_thread.join(timeout=2.0)
    capture_server.camera.stop()
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
