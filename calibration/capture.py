#!/usr/bin/env python3
"""Capture checkerboard calibration images with a browser-based MJPEG preview."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import math
import os
import threading
import time

import cv2

# ── Camera configuration ─────────────────────────────────────────────────────
CAMERA_DEVICE = "/dev/v4l/by-id/usb-HD_Camera_Manufacturer_HD_USB_Camera-video-index0"
CAMERA_FALLBACK = "/dev/video1"
OUTPUT_DIR = "calib_images"
CAPTURE_W = 256
CAPTURE_H = 256
JPEG_QUALITY = 80

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Camera Calibration Capture</title>
  <style>
    body { font: 16px sans-serif; max-width: 720px; margin: 24px auto; padding: 0 12px; }
    img { display: block; width: 100%; max-width: 512px; background: #222; }
    button { font-size: 1.1rem; margin: 12px 8px 12px 0; padding: 10px 18px; }
    input { width: 7em; }
    #status { min-height: 1.5em; }
  </style>
</head>
<body>
  <h1>Camera Calibration Capture</h1>
  <p>Keep the camera fixed. Move and tilt the checkerboard between captures.</p>
  <img src="/stream.mjpg" alt="Live camera preview">
  <h2>Camera exposure</h2>
  <p>Camera property units and ranges depend on the driver. Reduce exposure first
  if the board is washed out; manual exposure disables auto exposure when supported.</p>
  <label><input id="manual" type="checkbox"> Use manual exposure</label><br>
  <label>Brightness <input id="brightness" type="number" step="any"></label><br>
  <label>Exposure <input id="exposure" type="number" step="any"></label><br>
  <button id="apply">Apply camera settings</button><span id="settings"></span>
  <p><button id="capture">Capture frame</button><span id="status"></span></p>
  <p>Capture 20–30 sharp images with the full checkerboard visible in varied
  positions, distances, and tilted poses. Stop the script with Ctrl+C.</p>
  <script>
    const settings = document.getElementById("settings");
    const brightness = document.getElementById("brightness");
    const exposure = document.getElementById("exposure");
    const manual = document.getElementById("manual");
    async function loadSettings() {
      try {
        const response = await fetch("/camera-settings");
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || response.statusText);
        brightness.value = result.brightness;
        exposure.value = result.exposure;
        manual.checked = result.manual_exposure;
        settings.textContent = ` Current: brightness=${result.brightness}, exposure=${result.exposure}`;
      } catch (error) {
        settings.textContent = ` Could not read settings: ${error.message}`;
      }
    }
    document.getElementById("apply").onclick = async () => {
      settings.textContent = " Applying...";
      try {
        const response = await fetch("/camera-settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            brightness: Number(brightness.value),
            exposure: Number(exposure.value),
            manual_exposure: manual.checked
          })
        });
        const result = await response.json();
        if (result.brightness !== undefined) brightness.value = result.brightness;
        if (result.exposure !== undefined) exposure.value = result.exposure;
        if (!response.ok) throw new Error(result.error || response.statusText);
        settings.textContent = ` Applied (driver readback): brightness=${result.brightness}, exposure=${result.exposure}. ${result.message}`;
      } catch (error) {
        settings.textContent = ` Error: ${error.message}`;
      }
    };
    document.getElementById("capture").onclick = async () => {
      const status = document.getElementById("status");
      status.textContent = " Saving...";
      try {
        const response = await fetch("/capture", { method: "POST" });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || response.statusText);
        status.textContent = ` Saved ${result.file} (${result.count} total)`;
      } catch (error) {
        status.textContent = ` Error: ${error.message}`;
      }
    };
    loadSettings();
  </script>
</body>
</html>
"""


def open_camera():
    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[WARN] Tidak bisa buka {CAMERA_DEVICE}, coba fallback...")
        cap = cv2.VideoCapture(CAMERA_FALLBACK, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit("[ERROR] Tidak bisa membuka kamera.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class CameraFeed:
    def __init__(self, cap):
        self.cap = cap
        self._lock = threading.Lock()
        self._camera_lock = threading.Lock()
        self._frame = None
        self._running = threading.Event()
        self._running.set()
        self._thread = threading.Thread(target=self._read_frames, daemon=True)

    def start(self):
        self._thread.start()
        deadline = time.monotonic() + 5.0
        while self.frame() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if self.frame() is None:
            self.stop()
            raise RuntimeError("No camera frames received within 5 seconds.")

    def _read_frames(self):
        while self._running.is_set():
            with self._camera_lock:
                ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue
            # Resize to target resolution in case the driver ignores the
            # CAP_PROP_FRAME_WIDTH/HEIGHT request (e.g. MJPEG-only cameras
            # that only support fixed native resolutions like 320×240).
            h, w = frame.shape[:2]
            if w != CAPTURE_W or h != CAPTURE_H:
                frame = cv2.resize(frame, (CAPTURE_W, CAPTURE_H), interpolation=cv2.INTER_LINEAR)
            encoded, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            if encoded:
                with self._lock:
                    self._frame = buffer.tobytes()

    def frame(self):
        with self._lock:
            return self._frame

    def settings(self):
        with self._camera_lock:
            return {
                "brightness": self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                "exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE),
                "auto_exposure": self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            }

    def apply_settings(self, brightness, exposure, manual_exposure):
        with self._camera_lock:
            auto_value = 0.25 if manual_exposure else 0.75
            auto_ok = self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_value)
            brightness_ok = self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            exposure_ok = self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            actual = {
                "brightness": self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                "exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE),
                "auto_exposure": self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            }
        return actual, auto_ok, brightness_ok, exposure_ok

    def stop(self):
        self._running.clear()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.cap.release()


class CaptureHandler(BaseHTTPRequestHandler):
    server_version = "CalibrationCapture/1.0"

    def log_message(self, fmt, *args):
        print(f"[HTTP] {self.address_string()} - {fmt % args}")

    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/camera-settings":
            settings = self.server.camera.settings()
            self._send_json(
                200,
                {
                    "brightness": settings["brightness"],
                    "exposure": settings["exposure"],
                    "manual_exposure": abs(settings["auto_exposure"] - 0.25) < 0.1,
                },
            )
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while self.server.camera._running.is_set():
                    frame = self.server.camera.frame()
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
        if self.path == "/camera-settings":
            self._set_camera_settings()
            return
        if self.path != "/capture":
            self.send_error(404)
            return

        with self.server.capture_lock:
            frame = self.server.camera.frame()
            if frame is None:
                self._send_json(503, {"error": "No camera frame is available."})
                return

            count = self.server.capture_count
            filename = f"calib_{count:03d}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)
            try:
                with open(filepath, "wb") as output_file:
                    output_file.write(frame)
            except OSError as exc:
                self._send_json(500, {"error": f"Could not save image: {exc}"})
                return

            self.server.capture_count += 1
            count = self.server.capture_count
        print(f"[OK] Saved {filepath} ({count} total)")
        self._send_json(200, {"file": filename, "count": count})

    def _set_camera_settings(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 1024:
                raise ValueError("Request body must be between 1 and 1024 bytes.")
            payload = json.loads(self.rfile.read(content_length))
            brightness = float(payload["brightness"])
            exposure = float(payload["exposure"])
            manual_exposure = payload["manual_exposure"]
            if not isinstance(manual_exposure, bool):
                raise ValueError("manual_exposure must be true or false.")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"Invalid camera settings: {exc}"})
            return

        if not all(math.isfinite(value) for value in (brightness, exposure)):
            self._send_json(400, {"error": "Brightness and exposure must be finite numbers."})
            return

        actual, auto_ok, brightness_ok, exposure_ok = self.server.camera.apply_settings(
            brightness,
            exposure,
            manual_exposure,
        )
        print(
            "[CAMERA] Requested brightness=%s exposure=%s manual=%s; "
            "readback brightness=%s exposure=%s auto=%s"
            % (
                brightness,
                exposure,
                manual_exposure,
                actual["brightness"],
                actual["exposure"],
                actual["auto_exposure"],
            )
        )
        results = []
        if not auto_ok:
            results.append("driver rejected auto-exposure mode")
        if not brightness_ok:
            results.append("driver rejected brightness")
        if not exposure_ok:
            results.append("driver rejected exposure")
        message = "; ".join(results) if results else "Check preview; the driver may clamp values."
        result = {
            "brightness": actual["brightness"],
            "exposure": actual["exposure"],
            "auto_exposure": actual["auto_exposure"],
            "manual_exposure": abs(actual["auto_exposure"] - 0.25) < 0.1,
        }
        if results:
            result["error"] = message
            self._send_json(422, result)
            return
        result["message"] = message
        self._send_json(200, result)

    def _send_json(self, status, result):
        body = json.dumps(result).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class CaptureServer(ThreadingHTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(
        description="Capture checkerboard images through a browser-based MJPEG preview."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Address to bind (default: 127.0.0.1; use 0.0.0.0 for LAN access).",
    )
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080).")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing = len([name for name in os.listdir(OUTPUT_DIR) if name.endswith(".jpg")])
    server = CaptureServer((args.host, args.port), CaptureHandler)
    server.capture_count = existing
    server.capture_lock = threading.Lock()
    camera = None
    try:
        camera = CameraFeed(open_camera())
        server.camera = camera
        camera.start()
    except Exception:
        server.server_close()
        if camera is not None:
            camera.stop()
        raise

    if args.host == "0.0.0.0":
        url_host = "<ORANGE_PI_IP>"
        print("[WARN] Capture page is accessible to devices on this network; use a trusted network.")
    else:
        url_host = "127.0.0.1"

    print(f"\nCalibration capture server: http://{url_host}:{args.port}/")
    print(f"Output directory: {OUTPUT_DIR}/ ({existing} existing images)")
    print("Capture 20-30 sharp images with varied board positions and tilted poses.")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping capture server.")
    finally:
        server.server_close()
        if camera is not None:
            camera.stop()


if __name__ == "__main__":
    main()
