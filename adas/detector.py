#!/usr/bin/env python3
"""
detector.py — Wrapper TFLite YOLO untuk inferensi objek kendaraan.

Mendukung model int8-quantized (best_int8.tflite) maupun float32.
Output model diasumsikan berformat end-to-end NMS: [1, 300, 6]
di mana setiap baris adalah [x1_norm, y1_norm, x2_norm, y2_norm, conf, class_id].
"""

from __future__ import annotations

import os
from typing import List

import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite  # type: ignore
    except ImportError:
        raise SystemExit(
            "[ERROR] tflite_runtime maupun tensorflow tidak ditemukan. "
            "Install dengan: pip install tflite-runtime"
        )

from .constants import (
    MODEL_PATH,
    INPUT_W,
    INPUT_H,
    CONF_THRESHOLD,
    NUM_THREADS,
    CLASS_NAMES,
)


class YOLODetector:
    """
    Wrapper TFLite untuk model YOLO.

    Proses:
        1. preprocess(bgr_frame) → tensor
        2. infer(tensor) → raw output
        3. postprocess(raw) → List[dict]  # {'bbox', 'class_name', 'confidence'}

    Atau gunakan shortcut detect(bgr_frame) → List[dict] sekaligus.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        confidence_threshold: float = CONF_THRESHOLD,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold harus berada antara 0 dan 1")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[detector] Model tidak ditemukan: {model_path}\n"
                f"Pastikan file tflite tersedia di {model_path}"
            )

        self.interp = tflite.Interpreter(
            model_path=model_path,
            num_threads=NUM_THREADS,
        )
        self.interp.allocate_tensors()

        inp_det = self.interp.get_input_details()[0]
        out_det = self.interp.get_output_details()[0]

        self._in_idx    = inp_det["index"]
        self._out_idx   = out_det["index"]
        self._in_dtype  = inp_det["dtype"]
        self._in_quant  = inp_det["quantization"]   # (scale, zero_point)
        self._out_quant = out_det["quantization"]
        self._out_dtype = out_det["dtype"]
        self._confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Publik
    # ------------------------------------------------------------------
    def detect(self, bgr_frame: np.ndarray) -> List[dict]:
        """Shortcut: preprocess + infer + postprocess dalam satu panggilan."""
        tensor = self.preprocess(bgr_frame)
        raw    = self.infer(tensor)
        return self.postprocess(raw)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def preprocess(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Konversi frame BGR → tensor [1, H, W, 3].
        Untuk model int8: dequantize float→int8 menggunakan skema quantization.
        """
        import cv2  # import lokal agar modul bisa diuji tanpa cv2 di sistem lain
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rsz = cv2.resize(rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        flt = rsz.astype(np.float32) / 255.0

        if self._in_dtype in (np.int8, np.uint8):
            scale, zp = self._in_quant
            # scale=0 terjadi pada beberapa export tool; fallback ke 1/255
            scale = scale if scale != 0.0 else (1.0 / 255.0)
            lo, hi = (-128, 127) if self._in_dtype == np.int8 else (0, 255)
            flt = np.clip(np.round(flt / scale + zp), lo, hi).astype(self._in_dtype)

        return flt[np.newaxis, ...]  # shape: [1, INPUT_H, INPUT_W, 3]

    # ------------------------------------------------------------------
    # Inferensi
    # ------------------------------------------------------------------
    def infer(self, tensor: np.ndarray) -> np.ndarray:
        """Jalankan inferensi dan kembalikan output tensor (dequantize jika perlu)."""
        self.interp.set_tensor(self._in_idx, tensor)
        self.interp.invoke()
        out = self.interp.get_tensor(self._out_idx)
        if out.dtype in (np.int8, np.uint8):
            scale, zp = self._out_quant
            scale = scale if scale != 0.0 else (1.0 / 255.0)
            out = (out.astype(np.float32) - zp) * scale
        return out

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------
    def postprocess(self, raw: np.ndarray) -> List[dict]:
        """
        Parse output format [1, N, 6]:
            [:, 0:4] = bbox ternormalisasi (x1,y1,x2,y2)
            [:, 4]   = confidence
            [:, 5]   = class_id

        NMS diasumsikan sudah dilakukan oleh model (end-to-end export).
        Output dalam koordinat piksel skala INPUT_W×INPUT_H.
        """
        data        = raw[0]                        # shape [N, 6]
        boxes_norm  = data[:, :4]
        confidences = data[:, 4]
        class_ids   = data[:, 5].astype(int)

        mask        = confidences >= self._confidence_threshold
        boxes_norm  = boxes_norm[mask]
        confidences = confidences[mask]
        class_ids   = class_ids[mask]

        if len(boxes_norm) == 0:
            return []

        results: List[dict] = []
        for i in range(len(boxes_norm)):
            cls_id = int(class_ids[i])
            if cls_id >= len(CLASS_NAMES):
                continue

            x1 = int(np.clip(boxes_norm[i, 0] * INPUT_W, 0, INPUT_W - 1))
            y1 = int(np.clip(boxes_norm[i, 1] * INPUT_H, 0, INPUT_H - 1))
            x2 = int(np.clip(boxes_norm[i, 2] * INPUT_W, 0, INPUT_W - 1))
            y2 = int(np.clip(boxes_norm[i, 3] * INPUT_H, 0, INPUT_H - 1))

            results.append({
                "bbox"      : (x1, y1, x2, y2),
                "class_name": CLASS_NAMES[cls_id],
                "confidence": float(confidences[i]),
            })
        return results
