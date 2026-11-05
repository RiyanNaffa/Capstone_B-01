#!/usr/bin/env python3
"""Crop a game/screen region and send 256x256 video to an ADAS UDP receiver."""

from __future__ import annotations

import argparse
import logging
import os
import platform
import re
import shutil
import socket
import struct
import subprocess
import threading
import time
from typing import Iterator, List, Optional, Sequence, Tuple

_MAGIC = b"AMJ1"
_HEADER = struct.Struct("!4sIHH")
_DATAGRAM_SIZE = 1200
_FRAGMENT_PAYLOAD = _DATAGRAM_SIZE - _HEADER.size
_ENCODER_PREFERENCE = (
    "libx264", "h264_mf", "h264_amf", "h264_nvenc", "h264_qsv",
)


def _run_ffmpeg(args: Sequence[str], timeout: float = 10.0) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Gagal menjalankan FFmpeg: {exc}") from exc


def _available_h264_encoder(ffmpeg: str) -> Optional[str]:
    result = _run_ffmpeg([ffmpeg, "-hide_banner", "-encoders"])
    if result.returncode != 0:
        return None
    available = set()
    for line in result.stdout.splitlines():
        match = re.search(r"\b(h264_[A-Za-z0-9_]+|libx264)\b", line)
        if match:
            available.add(match.group(1))
    return next((item for item in _ENCODER_PREFERENCE if item in available), None)


def _screen_input_args(roi: Tuple[int, int, int, int], fps: int) -> List[str]:
    x, y, width, height = roi
    system = platform.system()
    if system == "Windows":
        return [
            "-f", "gdigrab", "-framerate", str(fps),
            "-offset_x", str(x), "-offset_y", str(y),
            "-video_size", f"{width}x{height}", "-i", "desktop",
        ]
    if system == "Linux":
        display = os.environ.get("DISPLAY", ":0.0")
        return [
            "-f", "x11grab", "-framerate", str(fps),
            "-video_size", f"{width}x{height}",
            "-i", f"{display}+{x},{y}",
        ]
    if system == "Darwin":
        return [
            "-f", "avfoundation", "-framerate", str(fps),
            "-capture_cursor", "1", "-i", "1:none",
        ]
    raise RuntimeError(f"Screen capture otomatis belum didukung pada {system}")


def _base_command(ffmpeg: str, roi: Tuple[int, int, int, int], fps: int) -> List[str]:
    x, y, width, height = roi
    filters = []
    if platform.system() == "Darwin":
        filters.append(f"crop={width}:{height}:{x}:{y}")
    filters.extend(["scale=256:256:flags=bilinear", "format=yuv420p"])
    return [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-nostdin",
        *_screen_input_args(roi, fps),
        "-an", "-vf", ",".join(filters),
        "-r", str(fps),
    ]


def _h264_command(
    ffmpeg: str,
    encoder: str,
    roi: Tuple[int, int, int, int],
    fps: int,
    bitrate: str,
    receiver: str,
    port: int,
) -> List[str]:
    command = _base_command(ffmpeg, roi, fps)
    command.extend(["-c:v", encoder])
    if encoder == "libx264":
        command.extend(["-preset", "ultrafast", "-tune", "zerolatency"])
    command.extend([
        "-pix_fmt", "yuv420p", "-g", str(fps * 2),
        "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", bitrate,
        "-flush_packets", "1", "-f", "mpegts",
        f"udp://{receiver}:{port}?pkt_size=1316",
    ])
    return command


def _mjpeg_command(
    ffmpeg: str,
    roi: Tuple[int, int, int, int],
    fps: int,
    quality: int,
) -> List[str]:
    command = _base_command(ffmpeg, roi, fps)
    command.extend(["-c:v", "mjpeg", "-q:v", str(quality), "-f", "mjpeg", "pipe:1"])
    return command


def _iter_jpeg_frames(stream) -> Iterator[bytes]:
    buffer = bytearray()
    start_marker = b"\xff\xd8"
    end_marker = b"\xff\xd9"
    while True:
        chunk = stream.read(65536)
        if not chunk:
            return
        buffer.extend(chunk)
        while True:
            start = buffer.find(start_marker)
            if start < 0:
                if len(buffer) > 1:
                    del buffer[:-1]
                break
            if start:
                del buffer[:start]
            end = buffer.find(end_marker, len(start_marker))
            if end < 0:
                break
            end += len(end_marker)
            yield bytes(buffer[:end])
            del buffer[:end]


def _send_mjpeg_frames(
    process: subprocess.Popen,
    sock: socket.socket,
    receiver: str,
    port: int,
    stop: threading.Event,
) -> None:
    frame_id = 0
    for jpeg in _iter_jpeg_frames(process.stdout):
        if stop.is_set():
            break
        fragment_count = (len(jpeg) + _FRAGMENT_PAYLOAD - 1) // _FRAGMENT_PAYLOAD
        if fragment_count > 65535:
            logging.warning("MJPEG frame terlalu besar; dilewati (%d bytes).", len(jpeg))
            continue
        for fragment_index in range(fragment_count):
            start = fragment_index * _FRAGMENT_PAYLOAD
            payload = jpeg[start:start + _FRAGMENT_PAYLOAD]
            packet = _HEADER.pack(
                _MAGIC, frame_id & 0xFFFFFFFF, fragment_index, fragment_count
            ) + payload
            try:
                sock.sendto(packet, (receiver, port))
            except OSError as exc:
                logging.error("UDP send gagal: %s", exc)
                stop.set()
                break
        frame_id += 1


def _announce_codec(
    sock: socket.socket,
    receiver: str,
    control_port: int,
    codec: str,
    stop: threading.Event,
) -> None:
    packet = f"ADAS1:{codec}".encode("ascii")
    while not stop.is_set():
        try:
            sock.sendto(packet, (receiver, control_port))
        except OSError as exc:
            logging.error("Codec handshake gagal: %s", exc)
            return
        stop.wait(1.0)


def _start_process(command: Sequence[str]) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )


def _drain_stderr(process: subprocess.Popen, stop: threading.Event) -> None:
    if process.stderr is None:
        return
    for line in iter(process.stderr.readline, b""):
        text = line.decode("utf-8", "replace").strip()
        if text:
            logging.warning("[ffmpeg] %s", text)
        if stop.is_set() and process.poll() is not None:
            break


def _stop_process(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crop laptop screen and stream 256x256 video to Orange Pi"
    )
    parser.add_argument("--receiver", required=True, help="IP address Orange Pi")
    parser.add_argument("--port", type=int, default=5000, help="UDP video port")
    parser.add_argument("--control-port", type=int, default=5001)
    parser.add_argument(
        "--roi", required=True, metavar="X,Y,W,H",
        help="ROI pada layar, mis. 700,250,800,800",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--bitrate", default="700k")
    parser.add_argument("--mjpeg-quality", type=int, default=5)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    if not shutil.which(args.ffmpeg):
        parser.error(f"FFmpeg tidak ditemukan: {args.ffmpeg}")
    try:
        roi_values = tuple(int(value.strip()) for value in args.roi.split(","))
    except ValueError:
        parser.error("--roi harus berupa X,Y,W,H dalam angka integer")
    if len(roi_values) != 4 or any(value < 0 for value in roi_values[:2]) \
            or any(value <= 0 for value in roi_values[2:]):
        parser.error("--roi harus berupa X,Y,W,H; W dan H harus positif")
    if not 1 <= args.fps <= 60:
        parser.error("--fps harus dalam rentang 1 sampai 60")
    if not 1 <= args.port <= 65534 or not 1 <= args.control_port <= 65535 \
            or args.port == args.control_port:
        parser.error("Port stream/control harus valid dan berbeda")
    if not 2 <= args.mjpeg_quality <= 31:
        parser.error("--mjpeg-quality harus dalam rentang 2 sampai 31")

    try:
        socket.gethostbyname(args.receiver)
    except OSError as exc:
        parser.error(f"Host receiver tidak valid: {exc}")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    try:
        encoder = _available_h264_encoder(args.ffmpeg)
    except RuntimeError as exc:
        logging.warning("Tidak bisa memeriksa encoder H.264 (%s); memakai MJPEG.", exc)
        encoder = None
    if encoder is None:
        logging.warning("Encoder H.264 tidak tersedia; otomatis memakai MJPEG.")
    else:
        logging.info("Encoder H.264 tersedia: %s", encoder)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    stop = threading.Event()
    process: Optional[subprocess.Popen] = None
    announcement: Optional[threading.Thread] = None
    try:
        if encoder is not None:
            logging.info("Mencoba stream H.264 ke %s:%d.", args.receiver, args.port)
            sock.sendto(
                b"ADAS1:H264", (args.receiver, args.control_port)
            )
            command = _h264_command(
                args.ffmpeg, encoder, roi_values, args.fps,
                args.bitrate, args.receiver, args.port,
            )
            process = _start_process(command)
            stderr_thread = threading.Thread(
                target=_drain_stderr, args=(process, stop), daemon=True
            )
            stderr_thread.start()
            time.sleep(2.0)
            if process.poll() is None:
                announcement = threading.Thread(
                    target=_announce_codec,
                    args=(sock, args.receiver, args.control_port, "H264", stop),
                    daemon=True,
                )
                announcement.start()
                logging.info("Stream H.264 berjalan. Ctrl+C untuk berhenti.")
                while process.poll() is None:
                    time.sleep(0.5)
                if not stop.is_set():
                    logging.error(
                        "Encoder H.264 berhenti unexpectedly; beralih ke MJPEG."
                    )
            else:
                logging.warning(
                    "H.264 gagal saat startup (code %s); beralih ke MJPEG.",
                    process.returncode,
                )
            stop.set()
            _stop_process(process)
            process = None
            stop.clear()
            if announcement is not None:
                announcement.join(timeout=1.5)
                announcement = None

        logging.info("Memulai fallback MJPEG ke %s:%d.", args.receiver, args.port)
        command = _mjpeg_command(
            args.ffmpeg, roi_values, args.fps, args.mjpeg_quality
        )
        process = _start_process(command)
        stderr_thread = threading.Thread(
            target=_drain_stderr, args=(process, stop), daemon=True
        )
        stderr_thread.start()
        sock.sendto(b"ADAS1:MJPG", (args.receiver, args.control_port))
        announcement = threading.Thread(
            target=_announce_codec,
            args=(sock, args.receiver, args.control_port, "MJPG", stop),
            daemon=True,
        )
        announcement.start()
        logging.info("Stream MJPEG berjalan. Ctrl+C untuk berhenti.")
        _send_mjpeg_frames(process, sock, args.receiver, args.port, stop)
        if not stop.is_set():
            logging.error("Encoder MJPEG berhenti sebelum diminta.")
            return 1
        return 0
    except KeyboardInterrupt:
        logging.info("Stream dihentikan pengguna.")
        return 0
    except (OSError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    finally:
        stop.set()
        _stop_process(process)
        if announcement is not None:
            announcement.join(timeout=1.5)
        sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
