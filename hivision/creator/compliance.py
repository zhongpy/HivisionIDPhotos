#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Offline image compliance checks.
"""
from typing import Dict, Tuple, Optional
import json
import os
import cv2
import numpy as np
import onnxruntime

from .context import Context
from hivision.error import ComplianceError

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - handled at runtime
    mp = None


DEFAULT_CONFIG = {
    "thresholds": {
        "brightness_min": 80.0,
        "brightness_max": 200.0,
        "sharpness_min": 80.0,
        "eye_ear_min": 0.18,
        "occlusion_ratio_min": 0.55,
        "skin_ratio_min": 0.40,
        "glasses_ratio_min": 0.01,
        "matting_head_coverage_min": 0.85,
        "matting_top_coverage_min": 0.60,
    },
    "models": {
        "face_parsing": {
            "enabled": True,
            "model_path": "hivision/creator/weights/face_parsing_bisenet.onnx",
            "input_size": 512,
            "color_order": "RGB",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "labels": {
                "skin": [1],
                "eyeglass": [4],
            },
        }
    },
}

_FACE_MESH = None
_FACE_PARSING = None
_CONFIG_CACHE = None

CONFIG_PATH_ENV = "HIVISION_COMPLIANCE_CONFIG"


def _deep_update(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            dst[key] = _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _load_config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_path = os.path.join(root_dir, "config", "compliance.json")
    config_path = os.getenv(CONFIG_PATH_ENV, default_path)
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                user_config = json.load(handle)
            config = _deep_update(config, user_config)
        except (OSError, json.JSONDecodeError):
            pass
    _CONFIG_CACHE = config
    return config


def _resolve_path(path_value: str) -> str:
    if not path_value:
        return ""
    if os.path.isabs(path_value):
        return path_value
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(root_dir, path_value)


def _get_face_mesh():
    global _FACE_MESH
    if _FACE_MESH is None:
        if mp is None:
            raise ImportError(
                "mediapipe is required for eye and occlusion checks. "
                "Install with `pip install mediapipe`."
            )
        _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
        )
    return _FACE_MESH


class FaceParsingModel:
    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", False))
        self.model_path = _resolve_path(config.get("model_path", ""))
        self.input_size = int(config.get("input_size", 512))
        self.color_order = config.get("color_order", "RGB").upper()
        self.mean = np.array(config.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.array(config.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        self.labels = config.get("labels", {})
        self.session = None
        self.input_name = None

    def available(self) -> bool:
        return self.enabled and bool(self.model_path) and os.path.exists(self.model_path)

    def load(self):
        if self.session is not None:
            return
        if not self.available():
            return
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self.session = onnxruntime.InferenceSession(self.model_path, providers=providers)
        except Exception:
            self.session = onnxruntime.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        if face_roi is None:
            return None
        self.load()
        if self.session is None:
            return None
        bgr = _ensure_bgr(face_roi)
        resized = cv2.resize(bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        if self.color_order == "RGB":
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = resized.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)[None, ...]
        outputs = self.session.run(None, {self.input_name: img})
        if not outputs:
            return None
        logits = outputs[0]
        if logits.ndim == 4:
            mask = np.argmax(logits, axis=1)[0]
        elif logits.ndim == 3:
            mask = np.argmax(logits, axis=0)
        else:
            return None
        return mask


def _get_face_parsing() -> FaceParsingModel:
    global _FACE_PARSING
    if _FACE_PARSING is None:
        config = _load_config()
        _FACE_PARSING = FaceParsingModel(config.get("models", {}).get("face_parsing", {}))
    return _FACE_PARSING


def _clamp_box(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0 = max(0, min(int(x0), width - 1))
    y0 = max(0, min(int(y0), height - 1))
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    return x0, y0, x1, y1


def _face_roi(image: np.ndarray, face_rect: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    if image is None or face_rect is None:
        return None
    x, y, w, h = face_rect
    x0, y0, x1, y1 = _clamp_box((x, y, x + w, y + h), image.shape[1], image.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    return image[y0:y1, x0:x1]


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image is None:
        return None
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _brightness_value(face_roi: np.ndarray) -> Optional[float]:
    if face_roi is None:
        return None
    if face_roi.shape[2] == 4:
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGRA2BGR)
    ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
    return float(np.mean(ycrcb[:, :, 0]))


def _sharpness_value(face_roi: np.ndarray) -> Optional[float]:
    if face_roi is None:
        return None
    gray = _to_gray(face_roi)
    if gray is None:
        return None
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _landmarks_from_mesh(image: np.ndarray):
    mesh = _get_face_mesh()
    image = _ensure_bgr(image)
    if image is None:
        return None
    candidates = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB), image]
    for candidate in candidates:
        results = mesh.process(candidate)
        if not results.multi_face_landmarks:
            continue
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = image.shape[:2]
        points = []
        for lm in landmarks:
            points.append((lm.x * w, lm.y * h))
        return points
    return None


def _eye_aspect_ratio(pts: list, idx: list) -> float:
    p1 = np.array(pts[idx[0]])
    p2 = np.array(pts[idx[1]])
    p3 = np.array(pts[idx[2]])
    p4 = np.array(pts[idx[3]])
    p5 = np.array(pts[idx[4]])
    p6 = np.array(pts[idx[5]])
    num = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    den = 2.0 * np.linalg.norm(p1 - p4)
    if den <= 1e-6:
        return 0.0
    return float(num / den)


def _eyes_open_value(image: np.ndarray) -> Optional[float]:
    pts = _landmarks_from_mesh(image)
    if pts is None:
        return None
    left_idx = [33, 160, 158, 133, 153, 144]
    right_idx = [362, 385, 387, 263, 373, 380]
    left_ear = _eye_aspect_ratio(pts, left_idx)
    right_ear = _eye_aspect_ratio(pts, right_idx)
    return float((left_ear + right_ear) / 2.0)


def _gradient_density(gray: np.ndarray, threshold: float = 20.0) -> float:
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    return float(np.mean(mag > threshold))


def _occlusion_ratio_value(image: np.ndarray) -> Optional[float]:
    pts = _landmarks_from_mesh(image)
    if pts is None:
        return None
    gray = _to_gray(image)
    if gray is None:
        return None
    face_density = _gradient_density(gray)
    if face_density <= 1e-6:
        return 0.0
    # Nose tip and mouth center regions.
    keypoints = [1, 13, 14, 33, 263]
    ratios = []
    h, w = gray.shape[:2]
    patch = int(min(h, w) * 0.04)
    for idx in keypoints:
        x, y = pts[idx]
        x0, y0, x1, y1 = _clamp_box((x - patch, y - patch, x + patch, y + patch), w, h)
        if x1 <= x0 or y1 <= y0:
            continue
        region = gray[y0:y1, x0:x1]
        ratios.append(_gradient_density(region) / face_density)
    if not ratios:
        return None
    return float(min(ratios))


def _glasses_count(face_roi: np.ndarray) -> Optional[int]:
    if face_roi is None:
        return None
    gray = _to_gray(face_roi)
    if gray is None:
        return None
    cascade_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    classifier = cv2.CascadeClassifier(cascade_path)
    if classifier.empty():
        cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        classifier = cv2.CascadeClassifier(cascade_path)
    if classifier.empty():
        return None
    eyes = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
    return int(len(eyes))


def _face_parsing_metrics(face_roi: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    parser = _get_face_parsing()
    if not parser.available():
        return None, None
    mask = parser.predict(face_roi)
    if mask is None:
        return None, None
    h, w = mask.shape[:2]
    total = float(h * w)
    labels = parser.labels or {}
    skin_labels = labels.get("skin", [])
    eyeglass_labels = labels.get("eyeglass", [])

    def _ratio_for(label_list):
        if not label_list:
            return 0.0
        counts = 0
        for label in label_list:
            counts += int(np.sum(mask == int(label)))
        return float(counts) / total if total > 0 else 0.0

    skin_ratio = _ratio_for(skin_labels) if skin_labels else None
    glasses_ratio = _ratio_for(eyeglass_labels) if eyeglass_labels else None
    return skin_ratio, glasses_ratio

def _matting_head_coverage(matting_image: np.ndarray, face_rect: Tuple[float, float, float, float]) -> Tuple[Optional[float], Optional[float]]:
    if (
        matting_image is None
        or matting_image.ndim != 3
        or matting_image.shape[2] < 4
        or face_rect is None
    ):
        return None, None
    x, y, w, h = face_rect
    h_img, w_img = matting_image.shape[:2]
    head_x0 = x - 0.2 * w
    head_x1 = x + 1.2 * w
    head_y0 = y - 0.6 * h
    head_y1 = y + 1.2 * h
    x0, y0, x1, y1 = _clamp_box((head_x0, head_y0, head_x1, head_y1), w_img, h_img)
    if x1 <= x0 or y1 <= y0:
        return None, None
    alpha = matting_image[:, :, 3]
    head = alpha[y0:y1, x0:x1]
    head_coverage = float(np.mean(head > 0))
    top_h = max(1, int((y1 - y0) * 0.25))
    top = alpha[y0:y0 + top_h, x0:x1]
    top_coverage = float(np.mean(top > 0))
    return head_coverage, top_coverage


def check_compliance(ctx: Context) -> Dict:
    config = _load_config()
    thresholds = config.get("thresholds", {})
    report = {"status": True, "items": {}, "reasons": []}
    face_rect = ctx.face.get("rectangle") if ctx.face else None
    face_roi = _face_roi(ctx.origin_image, face_rect)

    brightness = _brightness_value(face_roi)
    brightness_min = thresholds.get("brightness_min", DEFAULT_CONFIG["thresholds"]["brightness_min"])
    brightness_max = thresholds.get("brightness_max", DEFAULT_CONFIG["thresholds"]["brightness_max"])
    brightness_ok = brightness is not None and brightness_min <= brightness <= brightness_max
    report["items"]["brightness"] = {
        "value": brightness,
        "ok": brightness_ok,
        "thresholds": {
            "min": brightness_min,
            "max": brightness_max,
        },
    }
    if not brightness_ok:
        report["reasons"].append("brightness_out_of_range")

    sharpness = _sharpness_value(face_roi)
    sharpness_min = thresholds.get("sharpness_min", DEFAULT_CONFIG["thresholds"]["sharpness_min"])
    sharpness_ok = sharpness is not None and sharpness >= sharpness_min
    report["items"]["sharpness"] = {
        "value": sharpness,
        "ok": sharpness_ok,
        "thresholds": {
            "min": sharpness_min,
        },
    }
    if not sharpness_ok:
        report["reasons"].append("sharpness_low")

    try:
        eyes_ear = _eyes_open_value(ctx.origin_image)
    except ImportError:
        eyes_ear = None
    eye_ear_min = thresholds.get("eye_ear_min", DEFAULT_CONFIG["thresholds"]["eye_ear_min"])
    eyes_ok = eyes_ear is not None and eyes_ear >= eye_ear_min
    report["items"]["eyes_open"] = {
        "value": eyes_ear,
        "ok": eyes_ok,
        "thresholds": {
            "min": eye_ear_min,
        },
    }
    if not eyes_ok:
        report["reasons"].append("eyes_closed_or_unknown")

    parsing_skin_ratio, parsing_glasses_ratio = _face_parsing_metrics(face_roi)
    glasses_ratio_min = thresholds.get("glasses_ratio_min", DEFAULT_CONFIG["thresholds"]["glasses_ratio_min"])
    glasses_ok = None
    glasses_count = None
    if parsing_glasses_ratio is not None:
        glasses_ok = parsing_glasses_ratio < glasses_ratio_min
    else:
        glasses_count = _glasses_count(face_roi)
        glasses_ok = glasses_count is not None and glasses_count == 0
    report["items"]["glasses"] = {
        "value": parsing_glasses_ratio if parsing_glasses_ratio is not None else glasses_count,
        "ok": glasses_ok,
        "thresholds": {
            "max": 0 if parsing_glasses_ratio is None else glasses_ratio_min,
        },
        "source": "face_parsing" if parsing_glasses_ratio is not None else "haar",
    }
    if not glasses_ok:
        report["reasons"].append("glasses_detected_or_unknown")

    try:
        occlusion_ratio = _occlusion_ratio_value(ctx.origin_image)
    except ImportError:
        occlusion_ratio = None
    occlusion_ratio_min = thresholds.get("occlusion_ratio_min", DEFAULT_CONFIG["thresholds"]["occlusion_ratio_min"])
    skin_ratio_min = thresholds.get("skin_ratio_min", DEFAULT_CONFIG["thresholds"]["skin_ratio_min"])
    occlusion_ok = None
    if parsing_skin_ratio is not None:
        occlusion_ok = parsing_skin_ratio >= skin_ratio_min
        occlusion_value = parsing_skin_ratio
        occlusion_source = "face_parsing"
        occlusion_threshold = skin_ratio_min
    else:
        occlusion_ok = occlusion_ratio is not None and occlusion_ratio >= occlusion_ratio_min
        occlusion_value = occlusion_ratio
        occlusion_source = "gradient"
        occlusion_threshold = occlusion_ratio_min
    report["items"]["occlusion"] = {
        "value": occlusion_value,
        "ok": occlusion_ok,
        "thresholds": {
            "min": occlusion_threshold,
        },
        "source": occlusion_source,
    }
    if not occlusion_ok:
        report["reasons"].append("facial_features_occluded_or_unknown")

    head_coverage, top_coverage = _matting_head_coverage(ctx.matting_image, face_rect)
    matting_head_min = thresholds.get("matting_head_coverage_min", DEFAULT_CONFIG["thresholds"]["matting_head_coverage_min"])
    matting_top_min = thresholds.get("matting_top_coverage_min", DEFAULT_CONFIG["thresholds"]["matting_top_coverage_min"])
    head_ok = head_coverage is not None and head_coverage >= matting_head_min
    top_ok = top_coverage is not None and top_coverage >= matting_top_min
    report["items"]["matting_head_coverage"] = {
        "value": head_coverage,
        "ok": head_ok,
        "thresholds": {
            "min": matting_head_min,
        },
    }
    report["items"]["matting_top_coverage"] = {
        "value": top_coverage,
        "ok": top_ok,
        "thresholds": {
            "min": matting_top_min,
        },
    }
    if not head_ok or not top_ok:
        report["reasons"].append("matting_head_incomplete")

    report["status"] = len(report["reasons"]) == 0
    ctx.compliance = report
    if not report["status"]:
        raise ComplianceError(report)
    return report
