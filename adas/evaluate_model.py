#!/usr/bin/env python3
"""Evaluate model detections against YOLO-format labels on sampled frames."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from adas.constants import (
    CLASS_NAMES,
    CONF_THRESHOLD,
    INPUT_H,
    INPUT_W,
    NUM_THREADS,
)

Box = Tuple[float, float, float, float]
log = logging.getLogger("evaluate_model")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset" / "test1.v1-full-testing.yolo26"
DEFAULT_SAMPLES_DIR = DEFAULT_DATASET_DIR / "test" / "images"
DEFAULT_MODEL_PATH =  PROJECT_ROOT / "dataset" / "models" / "ugm-augment" / "best_int8.tflite"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "models" / "ugm-augment"


class TFLiteYoloDetector:
    """Evaluator-local YOLO TFLite runner, independent of adas.detector."""

    def __init__(self, model_path: str, confidence_threshold: float) -> None:
        interpreter_class = self._load_interpreter()
        self._interpreter = interpreter_class(
            model_path=model_path,
            num_threads=NUM_THREADS,
        )
        self._interpreter.allocate_tensors()

        input_detail = self._interpreter.get_input_details()[0]
        output_detail = self._interpreter.get_output_details()[0]
        self._input_index = input_detail["index"]
        self._output_index = output_detail["index"]
        self._input_dtype = input_detail["dtype"]
        self._input_quantization = input_detail["quantization"]
        self._output_quantization = output_detail["quantization"]
        self._output_dtype = output_detail["dtype"]
        self._confidence_threshold = confidence_threshold

    @staticmethod
    def _load_interpreter():
        candidates = (
            ("ai_edge_litert.interpreter", "Interpreter"),
            ("tflite_runtime.interpreter", "Interpreter"),
        )
        errors = []
        for module_name, attribute in candidates:
            try:
                module = importlib.import_module(module_name)
                return getattr(module, attribute)
            except (ImportError, OSError, RuntimeError) as exc:
                errors.append(f"{module_name}: {exc}")
        details = "; ".join(errors)
        raise RuntimeError(
            "Tidak ada runtime TFLite yang dapat dimuat. "
            "Di WSL Ubuntu 26/Python 3.14, aktifkan virtualenv dan install "
            "LiteRT dengan `python -m pip install ai-edge-litert`. "
            "Untuk versi Python yang didukung, `tflite-runtime` juga dapat digunakan. "
            f"Detail: {details}"
        )

    def detect(self, bgr_frame: np.ndarray) -> List[dict]:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR
        )
        tensor = resized.astype(np.float32) / 255.0
        if self._input_dtype in (np.int8, np.uint8):
            scale, zero_point = self._input_quantization
            scale = scale if scale != 0.0 else (1.0 / 255.0)
            low, high = (-128, 127) if self._input_dtype == np.int8 else (0, 255)
            tensor = np.clip(
                np.round(tensor / scale + zero_point), low, high
            ).astype(self._input_dtype)

        self._interpreter.set_tensor(self._input_index, tensor[np.newaxis, ...])
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_index)
        if self._output_dtype in (np.int8, np.uint8):
            scale, zero_point = self._output_quantization
            scale = scale if scale != 0.0 else (1.0 / 255.0)
            output = (output.astype(np.float32) - zero_point) * scale

        data = output[0]
        boxes = data[:, :4]
        confidences = data[:, 4]
        class_ids = data[:, 5].astype(int)
        mask = confidences >= self._confidence_threshold

        detections = []
        for box, confidence, class_id in zip(
            boxes[mask], confidences[mask], class_ids[mask]
        ):
            if class_id < 0 or class_id >= len(CLASS_NAMES):
                continue
            detections.append({
                "bbox": (
                    int(np.clip(box[0] * INPUT_W, 0, INPUT_W - 1)),
                    int(np.clip(box[1] * INPUT_H, 0, INPUT_H - 1)),
                    int(np.clip(box[2] * INPUT_W, 0, INPUT_W - 1)),
                    int(np.clip(box[3] * INPUT_H, 0, INPUT_H - 1)),
                ),
                "class_name": CLASS_NAMES[class_id],
                "confidence": float(confidence),
            })
        return detections


def _read_yolo_labels(path: str, class_count: int) -> List[Tuple[int, Box]]:
    labels: List[Tuple[int, Box]] = []
    with open(path, "r", encoding="utf-8") as label_file:
        for line_number, line in enumerate(label_file, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(f"{path}:{line_number}: format harus class x y w h")
            class_id = int(fields[0])
            cx, cy, width, height = (float(value) for value in fields[1:])
            if not 0 <= class_id < class_count:
                raise ValueError(
                    f"{path}:{line_number}: class id {class_id} di luar rentang"
                )
            valid_coordinates = (
                math.isfinite(cx)
                and math.isfinite(cy)
                and math.isfinite(width)
                and math.isfinite(height)
                and 0 <= cx <= 1
                and 0 <= cy <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            )
            if not valid_coordinates:
                raise ValueError(
                    f"{path}:{line_number}: koordinat YOLO harus ternormalisasi"
                )
            labels.append((
                class_id,
                (
                    (cx - width / 2) * INPUT_W,
                    (cy - height / 2) * INPUT_H,
                    (cx + width / 2) * INPUT_W,
                    (cy + height / 2) * INPUT_H,
                ),
            ))
    return labels


def _label_path(image_path: str, samples_dir: str, labels_dir: Optional[str]) -> str:
    if labels_dir:
        candidate = os.path.join(labels_dir, os.path.relpath(image_path, samples_dir))
        candidate = os.path.splitext(candidate)[0] + ".txt"
        if os.path.isfile(candidate):
            return candidate

        # Roboflow export may flatten labels or place them beside images.
        candidate = os.path.join(labels_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
        if os.path.isfile(candidate):
            return candidate
    else:
        candidate = os.path.splitext(image_path)[0] + ".txt"
        if os.path.isfile(candidate):
            return candidate

        parts = image_path.split(os.sep)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            candidate = os.path.splitext(os.sep.join(parts))[0] + ".txt"
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError(f"Label YOLO tidak ditemukan untuk {image_path}")


def _iou(first: Box, second: Box) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def _metric_counts(tp: int, fp: int, fn: int) -> Dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _find_images(samples_dir: str) -> List[str]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(
        os.path.join(root, name)
        for root, _, filenames in os.walk(samples_dir)
        for name in filenames
        if os.path.splitext(name)[1].lower() in extensions
    )


def _render_metrics(path: str, class_metrics: Dict[str, dict], overall: dict) -> None:
    width, height = 1000, 620
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "Object Detection Evaluation", (35, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"Overall  Precision {overall['precision']:.1%}   "
        f"Recall {overall['recall']:.1%}   F1 {overall['f1']:.1%}",
        (35, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 2, cv2.LINE_AA,
    )
    target_status = "MET" if overall["precision"] > 0.8 else "NOT MET"
    cv2.putText(
        canvas, f"Precision target > 80%: {target_status}", (35, 118),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA,
    )

    chart_top, chart_bottom = 165, 485
    chart_left, chart_right = 90, 950
    cv2.line(canvas, (chart_left, chart_bottom), (chart_right, chart_bottom),
             (80, 80, 80), 1)
    for percent in (0, 20, 40, 60, 80, 100):
        y = chart_bottom - int((chart_bottom - chart_top) * percent / 100)
        cv2.line(canvas, (chart_left, y), (chart_right, y), (225, 225, 225), 1)
        cv2.putText(canvas, str(percent), (35, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1, cv2.LINE_AA)

    colors = ((210, 105, 30), (60, 160, 60), (40, 130, 220))
    metrics = ("precision", "recall", "f1")
    names = list(class_metrics)
    slot = (chart_right - chart_left) // max(1, len(names))
    bar_width = max(10, min(32, slot // 5))
    for class_index, name in enumerate(names):
        center = chart_left + slot * class_index + slot // 2
        for metric_index, metric in enumerate(metrics):
            value = class_metrics[name][metric]
            bar_height = int((chart_bottom - chart_top) * value)
            x = center + (metric_index - 1) * (bar_width + 3)
            cv2.rectangle(canvas, (x, chart_bottom - bar_height),
                          (x + bar_width, chart_bottom), colors[metric_index], -1)
        cv2.putText(canvas, name, (center - 35, chart_bottom + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (25, 25, 25), 1, cv2.LINE_AA)

    legend_y = 550
    for index, (name, color) in enumerate(zip(("Precision", "Recall", "F1"), colors)):
        x = 330 + index * 150
        cv2.rectangle(canvas, (x, legend_y - 13), (x + 16, legend_y + 3), color, -1)
        cv2.putText(canvas, name, (x + 23, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (25, 25, 25), 1, cv2.LINE_AA)
    if not cv2.imwrite(path, canvas):
        raise OSError(f"Gagal menyimpan grafik metrik: {path}")


def evaluate(
    samples_dir: str,
    output_dir: str,
    model_path: str,
    confidence: float,
    iou_threshold: float,
    labels_dir: Optional[str],
    class_names: Sequence[str],
) -> dict:
    image_paths = _find_images(samples_dir)
    if not image_paths:
        raise ValueError(f"Tidak ada gambar sampel di {samples_dir}")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence harus berada antara 0 dan 1")
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold harus lebih dari 0 dan maksimal 1")
    if set(class_names) != set(CLASS_NAMES):
        raise ValueError(
            "Nama kelas label harus sama dengan kelas model "
            f"{', '.join(CLASS_NAMES)}; urutannya boleh disesuaikan."
        )

    detector = TFLiteYoloDetector(
        model_path=model_path,
        confidence_threshold=confidence,
    )
    totals = defaultdict(lambda: [0, 0, 0])
    confidence_sum = defaultdict(float)
    confidence_count = defaultdict(int)

    for image_path in image_paths:
        frame = cv2.imread(image_path)
        if frame is None:
            raise RuntimeError(f"Gagal membaca gambar: {image_path}")
        label_file = _label_path(image_path, samples_dir, labels_dir)
        ground_truth = _read_yolo_labels(label_file, len(class_names))
        detections = detector.detect(frame)

        predicted: Dict[int, List[dict]] = defaultdict(list)
        for detection in detections:
            try:
                class_id = class_names.index(detection["class_name"])
            except ValueError:
                continue
            predicted[class_id].append(detection)

        for class_id in range(len(class_names)):
            gt_boxes = [box for label_id, box in ground_truth if label_id == class_id]
            predictions = sorted(
                predicted[class_id], key=lambda item: item["confidence"], reverse=True
            )
            matched_gt: set[int] = set()
            for prediction in predictions:
                x1, y1, x2, y2 = prediction["bbox"]
                pred_box: Box = (float(x1), float(y1), float(x2), float(y2))
                candidates = [
                    (index, _iou(pred_box, gt_box))
                    for index, gt_box in enumerate(gt_boxes)
                    if index not in matched_gt
                ]
                best = max(candidates, key=lambda item: item[1], default=None)
                if best is not None and best[1] >= iou_threshold:
                    matched_gt.add(best[0])
                    totals[class_id][0] += 1
                    confidence_sum[class_id] += prediction["confidence"]
                    confidence_count[class_id] += 1
                else:
                    totals[class_id][1] += 1
            totals[class_id][2] += len(gt_boxes) - len(matched_gt)

    class_metrics = {}
    for class_id, class_name in enumerate(class_names):
        tp, fp, fn = totals[class_id]
        result = _metric_counts(tp, fp, fn)
        result["matched_mean_confidence"] = (
            confidence_sum[class_id] / confidence_count[class_id]
            if confidence_count[class_id] else 0.0
        )
        class_metrics[class_name] = result

    overall_counts = [
        sum(totals[class_id][column] for class_id in range(len(class_names)))
        for column in range(3)
    ]
    overall = _metric_counts(*overall_counts)
    report = {
        "model": model_path,
        "samples_directory": os.path.abspath(samples_dir),
        "image_count": len(image_paths),
        "confidence_threshold": confidence,
        "iou_threshold": iou_threshold,
        "precision_target": 0.8,
        "precision_target_operator": ">",
        "precision_target_met": overall["precision"] > 0.8,
        "class_names": list(class_names),
        "overall": overall,
        "per_class": class_metrics,
        "note": "Precision/recall/F1 are detection metrics, not classification accuracy.",
    }

    os.makedirs(output_dir, exist_ok=True)
    sample_path = Path(samples_dir)
    sample_name = (
        sample_path.parent.name
        if sample_path.name.lower() == "images"
        else sample_path.name
    ) or "samples"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_name)
    report_stem = f"metrics_{safe_name}"
    json_path = os.path.join(output_dir, f"{report_stem}.json")
    csv_path = os.path.join(output_dir, f"{report_stem}.csv")
    txt_path = os.path.join(output_dir, f"{report_stem}.txt")
    image_path = os.path.join(output_dir, f"{report_stem}.jpg")
    report["output_files"] = {
        "json": json_path,
        "csv": csv_path,
        "text": txt_path,
        "chart": image_path,
    }
    with open(json_path, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)

    fieldnames = [
        "class", "true_positive", "false_positive", "false_negative",
        "precision", "recall", "f1", "matched_mean_confidence",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"class": "overall", **overall, "matched_mean_confidence": ""})
        for class_name, metrics in class_metrics.items():
            writer.writerow({"class": class_name, **metrics})

    with open(txt_path, "w", encoding="utf-8") as output:
        output.write(f"Images evaluated: {len(image_paths)}\n")
        output.write(f"Confidence threshold: {confidence:.3f}\n")
        output.write(f"IoU threshold: {iou_threshold:.3f}\n")
        output.write(
            "Overall: precision={:.2%}, recall={:.2%}, F1={:.2%}\n".format(
                overall["precision"], overall["recall"], overall["f1"]
            )
        )
        output.write(
            f"Precision target > 80%: "
            f"{'MET' if overall['precision'] > 0.8 else 'NOT MET'}\n"
        )
        output.write("Per class:\n")
        for class_name, metrics in class_metrics.items():
            output.write(
                "  {}: P={:.2%} R={:.2%} F1={:.2%} "
                "TP={} FP={} FN={} mean_conf(TP)={:.2%}\n".format(
                    class_name,
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                    metrics["true_positive"],
                    metrics["false_positive"],
                    metrics["false_negative"],
                    metrics["matched_mean_confidence"],
                )
            )
        output.write(
            "\nCatatan: confidence model bukan akurasi; "
            "precision/recall/F1 dihitung terhadap label ground truth.\n"
        )

    _render_metrics(image_path, class_metrics, overall)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluasi deteksi model terhadap label ground truth format YOLO"
    )
    parser.add_argument(
        "--samples",
        default=str(DEFAULT_SAMPLES_DIR),
        help=f"Folder gambar (default: {DEFAULT_SAMPLES_DIR})",
    )
    parser.add_argument(
        "--labels-dir",
        help="Folder label .txt YOLO jika tidak berada di samping gambar",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Path model TFLite (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Folder laporan (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONF_THRESHOLD,
        help=f"Ambang confidence deteksi (default: {CONF_THRESHOLD})",
    )
    parser.add_argument("--iou", type=float, default=0.5, help="Ambang IoU (default: 0.5)")
    parser.add_argument(
        "--class-names",
        default=",".join(CLASS_NAMES),
        help=(
            "Urutan kelas dalam label YOLO; gunakan nama kelas model "
            "(bus,mobil,motor,truk) sesuai data.yaml Roboflow"
        ),
    )
    args = parser.parse_args()
    class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    if not class_names or len(set(class_names)) != len(class_names):
        parser.error("--class-names harus berisi nama kelas unik")
    if not os.path.isdir(args.samples):
        parser.error(f"Folder gambar tidak ditemukan: {args.samples}")
    if not os.path.isfile(args.model):
        parser.error(f"Model tidak ditemukan: {args.model}")
    if args.labels_dir and not os.path.isdir(args.labels_dir):
        parser.error(f"Folder label tidak ditemukan: {args.labels_dir}")

    try:
        report = evaluate(
            args.samples,
            args.output_dir,
            args.model,
            args.confidence,
            args.iou,
            args.labels_dir,
            class_names,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)

    print(
        "Evaluasi selesai: P={:.2%} R={:.2%} F1={:.2%}".format(
            report["overall"]["precision"],
            report["overall"]["recall"],
            report["overall"]["f1"],
        )
    )
    print(f"Laporan JSON: {report['output_files']['json']}")
    print(f"Grafik JPG: {report['output_files']['chart']}")


if __name__ == "__main__":
    main()
