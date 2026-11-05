#!/usr/bin/env python3
"""Receive low-latency H.264 or packetized MJPEG frames over UDP."""

from __future__ import annotations

import logging
import queue
import socket
import struct
import subprocess
import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .constants import INPUT_H, INPUT_W

log = logging.getLogger(__name__)

_MJPEG_HEADER = struct.Struct("!4sIHH")
_MJPEG_MAGIC = b"AMJ1"
_CONTROL_PREFIX = b"ADAS1:"
_FRAGMENT_TIMEOUT_S = 0.75
_MAX_PENDING_FRAMES = 8
_UDP_READ_SIZE = 65507


class UdpFrameCapture:
    """VideoCapture-compatible source selected by the laptop's codec handshake."""

    def __init__(
        self,
        port: int = 5000,
        control_port: int = 5001,
        bind_host: str = "0.0.0.0",
        ffmpeg: str = "ffmpeg",
    ) -> None:
        self._port = port
        self._control_port = control_port
        self._bind_host = bind_host
        self._ffmpeg = ffmpeg
        self._frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._mode_stop = threading.Event()
        self._control_thread: Optional[threading.Thread] = None
        self._control_socket: Optional[socket.socket] = None
        self._mode_thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None
        self._mode_lock = threading.Lock()
        self._mode: Optional[str] = None
        self._latest: Optional[np.ndarray] = None

    def start(self) -> None:
        control = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            control.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            control.bind((self._bind_host, self._control_port))
            control.settimeout(0.5)
        except OSError:
            control.close()
            raise
        self._control_socket = control
        self._control_thread = threading.Thread(
            target=self._control_loop, name="udp-codec-control", daemon=True
        )
        self._control_thread.start()
        log.info(
            "[udp] Menunggu stream pada %s:%d (control %d).",
            self._bind_host, self._port, self._control_port,
        )

    def isOpened(self) -> bool:
        return not self._stop.is_set()

    def grab(self) -> bool:
        try:
            self._latest = self._frames.get(timeout=0.25)
            return True
        except queue.Empty:
            self._latest = None
            return False

    def retrieve(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self._latest is not None, self._latest

    def release(self) -> None:
        self._stop.set()
        with self._mode_lock:
            self._stop_mode()
        if self._control_thread is not None:
            self._control_thread.join(timeout=2.0)
            self._control_thread = None
        log.info("[udp] Receiver dihentikan.")

    def _control_loop(self) -> None:
        control = self._control_socket
        if control is None:
            return
        try:
            while not self._stop.is_set():
                try:
                    message = control.recvfrom(128)[0]
                except socket.timeout:
                    continue
                if not message.startswith(_CONTROL_PREFIX):
                    continue
                mode = message[len(_CONTROL_PREFIX):].decode("ascii", "ignore").strip()
                if mode not in ("H264", "MJPG"):
                    log.warning("[udp] Codec handshake tidak dikenal: %s", mode)
                    continue
                self._select_mode(mode)
        except OSError:
            if not self._stop.is_set():
                log.exception("[udp] Control socket gagal.")
                self._stop.set()
        finally:
            control.close()
            self._control_socket = None

    def _select_mode(self, mode: str) -> None:
        with self._mode_lock:
            if self._mode == mode and self._mode_thread is not None \
                    and self._mode_thread.is_alive():
                return
            self._stop_mode()
            self._mode = mode
            self._mode_stop = threading.Event()
            worker = self._receive_h264 if mode == "H264" else self._receive_mjpeg
            self._mode_thread = threading.Thread(
                target=worker,
                args=(self._mode_stop,),
                name=f"udp-{mode.lower()}-receiver",
                daemon=True,
            )
            self._mode_thread.start()
            log.info("[udp] Codec input: %s", mode)

    def _stop_mode(self) -> None:
        mode_stop = self._mode_stop
        mode_stop.set()
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        worker = self._mode_thread
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)
        self._mode_thread = None
        self._mode = None

    def _receive_h264(self, mode_stop: threading.Event) -> None:
        url = (
            f"udp://{self._bind_host}:{self._port}"
            "?fifo_size=1000000&overrun_nonfatal=1&reuse=1&timeout=1000000"
        )
        command = [
            self._ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "warning",
            "-fflags", "nobuffer", "-flags", "low_delay",
            "-probesize", "32768", "-analyzeduration", "0",
            "-f", "mpegts", "-i", url,
            "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except OSError:
            log.exception("[udp] Tidak dapat menjalankan FFmpeg decoder.")
            return
        self._process = process
        frame_bytes = INPUT_W * INPUT_H * 3
        try:
            while not self._stop.is_set() and not mode_stop.is_set():
                raw_frame = self._read_exact(process.stdout, frame_bytes)
                if raw_frame is None:
                    if process.poll() is not None and not mode_stop.is_set():
                        log.error("[udp] FFmpeg H.264 decoder berhenti (code %s).",
                                  process.returncode)
                    break
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                    (INPUT_H, INPUT_W, 3)
                )
                self._publish(frame)
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait()
            if self._process is process:
                self._process = None

    @staticmethod
    def _read_exact(stream, size: int) -> Optional[bytes]:
        data = bytearray()
        while len(data) < size:
            chunk = stream.read(size - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)

    def _receive_mjpeg(self, mode_stop: threading.Event) -> None:
        fragments: Dict[int, Dict[int, bytes]] = {}
        fragment_counts: Dict[int, int] = {}
        last_seen: Dict[int, float] = {}
        stream = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            stream.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            stream.bind((self._bind_host, self._port))
            stream.settimeout(0.5)
            while not self._stop.is_set() and not mode_stop.is_set():
                try:
                    packet = stream.recvfrom(_UDP_READ_SIZE)[0]
                except socket.timeout:
                    self._expire_fragments(fragments, fragment_counts, last_seen)
                    continue
                if len(packet) <= _MJPEG_HEADER.size:
                    continue
                magic, frame_id, fragment_index, fragment_count = \
                    _MJPEG_HEADER.unpack_from(packet)
                if magic != _MJPEG_MAGIC or fragment_count == 0 \
                        or fragment_index >= fragment_count:
                    continue

                frame_parts = fragments.setdefault(frame_id, {})
                frame_parts[fragment_index] = packet[_MJPEG_HEADER.size:]
                fragment_counts[frame_id] = fragment_count
                last_seen[frame_id] = time.monotonic()
                self._trim_fragments(fragments, fragment_counts, last_seen)
                if len(frame_parts) == fragment_count:
                    encoded = b"".join(
                        frame_parts[index] for index in range(fragment_count)
                    )
                    fragments.pop(frame_id, None)
                    fragment_counts.pop(frame_id, None)
                    last_seen.pop(frame_id, None)
                    image = np.frombuffer(encoded, dtype=np.uint8)
                    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    if frame is not None:
                        if frame.shape[1] != INPUT_W or frame.shape[0] != INPUT_H:
                            frame = cv2.resize(frame, (INPUT_W, INPUT_H))
                        self._publish(frame)
        except OSError:
            if not self._stop.is_set() and not mode_stop.is_set():
                log.exception("[udp] MJPEG socket gagal.")
        finally:
            stream.close()

    @staticmethod
    def _expire_fragments(
        fragments: Dict[int, Dict[int, bytes]],
        fragment_counts: Dict[int, int],
        last_seen: Dict[int, float],
    ) -> None:
        now = time.monotonic()
        expired = [
            frame_id for frame_id, stamp in last_seen.items()
            if now - stamp > _FRAGMENT_TIMEOUT_S
        ]
        for frame_id in expired:
            fragments.pop(frame_id, None)
            fragment_counts.pop(frame_id, None)
            last_seen.pop(frame_id, None)

    @staticmethod
    def _trim_fragments(
        fragments: Dict[int, Dict[int, bytes]],
        fragment_counts: Dict[int, int],
        last_seen: Dict[int, float],
    ) -> None:
        while len(fragments) > _MAX_PENDING_FRAMES:
            oldest = min(last_seen, key=last_seen.get)
            fragments.pop(oldest, None)
            fragment_counts.pop(oldest, None)
            last_seen.pop(oldest, None)

    def _publish(self, frame: np.ndarray) -> None:
        if self._frames.full():
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            pass
