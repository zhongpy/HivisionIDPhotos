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
    _MP_SOLUTIONS = getattr(mp, "solutions", None)
except ImportError:  # pragma: no cover - handled at runtime
    mp = None
    _MP_SOLUTIONS = None


DEFAULT_CONFIG = {
    "debug": False,
    "thresholds": {
        "brightness_min": 80.0,
        "brightness_max": 200.0,
        "sharpness_min": 80.0,
        "sharpness_min_tenengrad": 1000.0,
        "eye_ear_min": 0.18,
        "eye_area_ratio_min": 0.003,
        "occlusion_ratio_min": 0.55,
        "skin_ratio_min": 0.40,
        "glasses_ratio_min": 0.001,
        "mouth_total_ratio_max": 0.04,
        "mouth_open_ratio_max": 0.35,
        "hat_ratio_max": 0.001,
        "earring_ratio_max": 0.0005,
        "neck_ratio_min": 0.01,
        "cloth_ratio_min": 0.02,
        "hair_ratio_min": 0.05,
        "hat_hair_ratio_max": 0.02,
        "ear_ratio_min": 0.002,
        "ear_side_strip_ratio_min": 0.10,
        "ear_side_strip_width_ratio": 0.18,
        "matting_head_coverage_min": 0.80,
        "matting_top_coverage_min": 0.60,
        "matting_parsing_missing_max": 0.10,
        "matting_parsing_missing_hair_max": None,
        "matting_parsing_missing_skin_max": None,
        "matting_parsing_missing_cloth_max": None,
        "face_fill_target_ratio": 0.80,
        "face_fill_min_scale": 0.60
    },
    "models": {
        "face_parsing": {
            "enabled": True,
            "model_path": "hivision/creator/weights/face_parsing.onnx",
            "input_size": 224,
            "color_order": "RGB",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "labels": {
                "skin": [1],
                "eyeglass": [3],
                "left_eye": [4],
                "right_eye": [5],
                "mouth_inner": [10],
                "mouth_lip": [11, 12],
                "hair": [13],
                "hat": [14],
                "earring": [15],
                "left_ear": [8],
                "right_ear": [9],
                "neck": [16, 17],
                "cloth": [18]
            }
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

def _debug_log(config: dict, message: str):
    if config.get("debug"):
        print(f"[Compliance] {message}")


def _get_face_mesh():
    global _FACE_MESH
    if _FACE_MESH is None:
        if _MP_SOLUTIONS is not None and hasattr(_MP_SOLUTIONS, "face_mesh"):
            face_mesh_cls = _MP_SOLUTIONS.face_mesh.FaceMesh
        else:
            try:
                from mediapipe.python.solutions.face_mesh import FaceMesh as face_mesh_cls
            except Exception as exc:  # pragma: no cover - handled at runtime
                raise ImportError(
                    "mediapipe is required for eye and occlusion checks. "
                    "Install with `pip install mediapipe`."
                ) from exc
        config = _load_config()
        mesh_cfg = config.get("models", {}).get("face_mesh", {})
        _FACE_MESH = face_mesh_cls(
            static_image_mode=True,
            refine_landmarks=bool(mesh_cfg.get("refine_landmarks", True)),
            max_num_faces=1,
            min_detection_confidence=float(mesh_cfg.get("min_detection_confidence", 0.3)),
            min_tracking_confidence=float(mesh_cfg.get("min_tracking_confidence", 0.3)),
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

def _expanded_face_roi(
    image: np.ndarray,
    face_rect: Tuple[float, float, float, float],
    box_factors: Optional[dict] = None,
) -> Optional[np.ndarray]:
    if image is None or face_rect is None:
        return None
    x, y, w, h = face_rect
    factors = box_factors or {"left": -0.2, "right": 1.2, "top": -0.4, "bottom": 1.2}
    x0 = x + factors.get("left", -0.2) * w
    x1 = x + factors.get("right", 1.2) * w
    y0 = y + factors.get("top", -0.4) * h
    y1 = y + factors.get("bottom", 1.2) * h
    x0, y0, x1, y1 = _clamp_box((x0, y0, x1, y1), image.shape[1], image.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    return image[y0:y1, x0:x1]

def _expanded_face_box(
    image: np.ndarray,
    face_rect: Tuple[float, float, float, float],
    box_factors: Optional[dict] = None,
) -> Optional[Tuple[int, int, int, int]]:
    if image is None or face_rect is None:
        return None
    x, y, w, h = face_rect
    factors = box_factors or {"left": -0.2, "right": 1.2, "top": -0.4, "bottom": 1.2}
    x0 = x + factors.get("left", -0.2) * w
    x1 = x + factors.get("right", 1.2) * w
    y0 = y + factors.get("top", -0.4) * h
    y1 = y + factors.get("bottom", 1.2) * h
    x0, y0, x1, y1 = _clamp_box((x0, y0, x1, y1), image.shape[1], image.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    return (int(x0), int(y0), int(x1), int(y1))

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


def _sharpness_tenengrad(face_roi: np.ndarray) -> Optional[float]:
    if face_roi is None:
        return None
    gray = _to_gray(face_roi)
    if gray is None:
        return None
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    g = gx * gx + gy * gy
    return float(np.mean(g))


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None:
        return None
    if image.dtype != np.uint8:
        max_value = float(np.max(image)) if image.size else 0.0
        if max_value <= 1.0:
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _is_check_enabled(ctx: Context, key: str, default: bool = True) -> bool:
    switches = getattr(ctx, "compliance_switches", None)
    if not switches or key not in switches:
        return default
    return bool(switches.get(key))


def _landmarks_from_mesh(image: np.ndarray):
    mesh = _get_face_mesh()
    image = _ensure_bgr(image)
    if image is None:
        return None
    image = np.ascontiguousarray(image)
    candidates = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB), image]
    h, w = image.shape[:2]
    if min(h, w) < 256:
        scale = 256.0 / min(h, w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        candidates.extend([cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), resized])
    for candidate in candidates:
        results = mesh.process(candidate)
        if not results.multi_face_landmarks:
            continue
        landmarks = results.multi_face_landmarks[0].landmark
        ch, cw = candidate.shape[:2]
        points = []
        for lm in landmarks:
            points.append((lm.x * cw, lm.y * ch))
        # Map back to original scale if resized.
        if (ch, cw) != (h, w):
            scale_x = w / float(cw)
            scale_y = h / float(ch)
            points = [(x * scale_x, y * scale_y) for x, y in points]
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


def _eyes_open_value(image: np.ndarray, fallback_image: Optional[np.ndarray] = None) -> Optional[float]:
    pts = _landmarks_from_mesh(image)
    if pts is None and fallback_image is not None:
        pts = _landmarks_from_mesh(fallback_image)
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


def _occlusion_ratio_value(image: np.ndarray, fallback_image: Optional[np.ndarray] = None) -> Optional[float]:
    pts = _landmarks_from_mesh(image)
    if pts is None and fallback_image is not None:
        pts = _landmarks_from_mesh(fallback_image)
        image = fallback_image if pts is not None else image
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


def _face_parsing_metrics(
    face_roi: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float], dict]:
    parser = _get_face_parsing()
    if not parser.available():
        return None, None, None, {}
    mask = parser.predict(face_roi)
    if mask is None:
        return None, None, None, {}
    h, w = mask.shape[:2]
    total = float(h * w)
    labels = parser.labels or {}
    skin_labels = labels.get("skin", [])
    eyeglass_labels = labels.get("eyeglass", [])
    left_eye_labels = labels.get("left_eye", [])
    right_eye_labels = labels.get("right_eye", [])
    mouth_inner_labels = labels.get("mouth_inner", [])
    mouth_lip_labels = labels.get("mouth_lip", [])
    hair_labels = labels.get("hair", [])
    hat_labels = labels.get("hat", [])
    earring_labels = labels.get("earring", [])
    left_ear_labels = labels.get("left_ear", [])
    right_ear_labels = labels.get("right_ear", [])
    neck_labels = labels.get("neck", [])
    cloth_labels = labels.get("cloth", [])

    def _ratio_for(label_list):
        if not label_list:
            return 0.0
        counts = 0
        for label in label_list:
            counts += int(np.sum(mask == int(label)))
        return float(counts) / total if total > 0 else 0.0

    skin_ratio = _ratio_for(skin_labels) if skin_labels else None
    glasses_ratio = _ratio_for(eyeglass_labels) if eyeglass_labels else None
    eye_ratio = None
    if left_eye_labels or right_eye_labels:
        eye_ratio = _ratio_for(left_eye_labels + right_eye_labels)
    mouth_inner_ratio = _ratio_for(mouth_inner_labels) if mouth_inner_labels else None
    mouth_lip_ratio = _ratio_for(mouth_lip_labels) if mouth_lip_labels else None
    extra = {
        "mouth_inner_ratio": mouth_inner_ratio,
        "mouth_lip_ratio": mouth_lip_ratio,
        "hair_ratio": _ratio_for(hair_labels) if hair_labels else None,
        "hat_ratio": _ratio_for(hat_labels) if hat_labels else None,
        "earring_ratio": _ratio_for(earring_labels) if earring_labels else None,
        "left_ear_ratio": _ratio_for(left_ear_labels) if left_ear_labels else None,
        "right_ear_ratio": _ratio_for(right_ear_labels) if right_ear_labels else None,
        "neck_ratio": _ratio_for(neck_labels) if neck_labels else None,
        "cloth_ratio": _ratio_for(cloth_labels) if cloth_labels else None,
        "_mask": mask,
        "_skin_labels": skin_labels,
    }
    return skin_ratio, glasses_ratio, eye_ratio, extra

def _matting_head_coverage(
    matting_image: np.ndarray,
    face_rect: Tuple[float, float, float, float],
    box_factors: Optional[dict] = None,
    top_fraction: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    if (
        matting_image is None
        or matting_image.ndim != 3
        or matting_image.shape[2] < 4
        or face_rect is None
    ):
        return None, None
    x, y, w, h = face_rect
    h_img, w_img = matting_image.shape[:2]
    factors = box_factors or {"left": -0.2, "right": 1.2, "top": -0.6, "bottom": 1.2}
    head_x0 = x + factors.get("left", -0.2) * w
    head_x1 = x + factors.get("right", 1.2) * w
    head_y0 = y + factors.get("top", -0.6) * h
    head_y1 = y + factors.get("bottom", 1.2) * h
    x0, y0, x1, y1 = _clamp_box((head_x0, head_y0, head_x1, head_y1), w_img, h_img)
    if x1 <= x0 or y1 <= y0:
        return None, None
    alpha = matting_image[:, :, 3]
    head = alpha[y0:y1, x0:x1]
    head_coverage = float(np.mean(head > 0))
    fraction = 0.25 if top_fraction is None else float(top_fraction)
    top_h = max(1, int((y1 - y0) * fraction))
    top = alpha[y0:y0 + top_h, x0:x1]
    top_coverage = float(np.mean(top > 0))
    return head_coverage, top_coverage


def check_compliance(ctx: Context, stage: str = "full") -> Dict:
    config = _load_config()
    thresholds = config.get("thresholds", {})
    report = {"status": True, "items": {}, "reasons": [], "stage": stage}
    face_rect = ctx.face.get("rectangle") if ctx.face else None
    face_roi = _face_roi(ctx.origin_image, face_rect)
    parsing_box = config.get("models", {}).get("face_parsing", {}).get("box")
    parsing_roi = _expanded_face_roi(ctx.origin_image, face_rect, parsing_box)
    face_box = _expanded_face_box(ctx.origin_image, face_rect, None)
    parsing_box_rect = _expanded_face_box(ctx.origin_image, face_rect, parsing_box)
    report["debug"] = {
        "face_box": face_box,
        "parsing_box": parsing_box_rect,
        "face_rect": face_rect,
        "origin_shape": None if ctx.origin_image is None else list(ctx.origin_image.shape),
    }
    _debug_log(config, f"origin_image={None if ctx.origin_image is None else ctx.origin_image.shape}")
    _debug_log(config, f"matting_image={None if ctx.matting_image is None else ctx.matting_image.shape}")
    _debug_log(config, f"face_rect={face_rect}")
    _debug_log(config, f"face_roi={None if face_roi is None else face_roi.shape}")
    _debug_log(config, f"parsing_roi={None if parsing_roi is None else parsing_roi.shape}")

    run_pre = stage in ("pre", "full")
    run_post = stage in ("post", "full")

    if run_pre:
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
        _debug_log(config, f"brightness={brightness} ok={brightness_ok} min={brightness_min} max={brightness_max}")

        sharpness = _sharpness_value(face_roi)
        sharpness_t = _sharpness_tenengrad(face_roi)
        sharpness_min = thresholds.get("sharpness_min", DEFAULT_CONFIG["thresholds"]["sharpness_min"])
        sharpness_min_t = thresholds.get("sharpness_min_tenengrad", DEFAULT_CONFIG["thresholds"]["sharpness_min_tenengrad"])
        # Pass if either metric meets its threshold
        sharpness_ok = (
            (sharpness is not None and sharpness >= sharpness_min)
            or (sharpness_t is not None and sharpness_t >= sharpness_min_t)
        )
        report["items"]["sharpness"] = {
            "value": {
                "laplacian": sharpness,
                "tenengrad": sharpness_t,
            },
            "ok": sharpness_ok,
            "thresholds": {
                "laplacian_min": sharpness_min,
                "tenengrad_min": sharpness_min_t,
            },
        }
        if not sharpness_ok:
            report["reasons"].append("sharpness_low")
        _debug_log(
            config,
            f"sharpness_laplacian={sharpness} min={sharpness_min} "
            f"sharpness_tenengrad={sharpness_t} min={sharpness_min_t} ok={sharpness_ok}",
        )

        # Use tight face ROI for facial features (occlusion/glasses/eyes/mouth)
        parsing_skin_ratio, parsing_glasses_ratio, parsing_eye_ratio, parsing_extra = _face_parsing_metrics(
            face_roi
        )
        # Use expanded ROI for hats/ears/neck/cloth when available
        _, _, _, parsing_extra_wide = _face_parsing_metrics(
            parsing_roi if parsing_roi is not None else face_roi
        )
        hair_ratio = parsing_extra.get("hair_ratio") if parsing_extra else None
        face_fill_ratio = None
        if parsing_skin_ratio is not None or hair_ratio is not None:
            face_fill_ratio = float((parsing_skin_ratio or 0.0) + (hair_ratio or 0.0))
        report["items"]["face_fill_ratio"] = {
            "value": face_fill_ratio,
            "ok": True,
            "thresholds": {
                "target": thresholds.get(
                    "face_fill_target_ratio",
                    DEFAULT_CONFIG["thresholds"]["face_fill_target_ratio"],
                ),
            },
            "source": "face_parsing",
        }
        ctx.face_fill_ratio = face_fill_ratio
        _debug_log(config, f"face_fill_ratio={face_fill_ratio}")

        try:
            eyes_ear = _eyes_open_value(
                face_roi if face_roi is not None else ctx.origin_image,
                ctx.origin_image,
            )
        except ImportError:
            eyes_ear = None
        eye_ear_min = thresholds.get("eye_ear_min", DEFAULT_CONFIG["thresholds"]["eye_ear_min"])
        eye_area_min = thresholds.get("eye_area_ratio_min")
        eyes_ok = eyes_ear is not None and eyes_ear >= eye_ear_min
        if eyes_ear is None and parsing_eye_ratio is not None and eye_area_min is not None:
            eyes_ear = parsing_eye_ratio
            eyes_ok = eyes_ear >= eye_area_min
            report["items"]["eyes_open"] = {
                "value": eyes_ear,
                "ok": eyes_ok,
                "thresholds": {
                    "min": eye_area_min,
                },
                "source": "face_parsing",
            }
        else:
            report["items"]["eyes_open"] = {
                "value": eyes_ear,
                "ok": eyes_ok,
                "thresholds": {
                    "min": eye_ear_min,
                },
                "source": "mesh",
            }
        if not eyes_ok:
            report["reasons"].append("eyes_closed_or_unknown")
        _debug_log(
            config,
            f"eyes_open={eyes_ear} ok={eyes_ok} min={eye_area_min if report['items']['eyes_open']['source'] == 'face_parsing' else eye_ear_min} "
            f"source={report['items']['eyes_open']['source']}",
        )
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
        _debug_log(
            config,
            f"glasses_value={report['items']['glasses']['value']} ok={glasses_ok} "
            f"threshold={report['items']['glasses']['thresholds']['max']} source={report['items']['glasses']['source']}",
        )

        try:
            occlusion_ratio = _occlusion_ratio_value(
                face_roi if face_roi is not None else ctx.origin_image,
                ctx.origin_image,
            )
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
        _debug_log(
            config,
            f"occlusion_value={occlusion_value} ok={occlusion_ok} min={occlusion_threshold} source={occlusion_source}",
        )

        if _is_check_enabled(ctx, "mouth"):
            mouth_inner_ratio = parsing_extra.get("mouth_inner_ratio") if parsing_extra else None
            mouth_lip_ratio = parsing_extra.get("mouth_lip_ratio") if parsing_extra else None
            mouth_total_ratio = None
            mouth_open_ratio = None
            if mouth_inner_ratio is not None and mouth_lip_ratio is not None:
                mouth_total_ratio = mouth_inner_ratio + mouth_lip_ratio
                denom = mouth_total_ratio if mouth_total_ratio > 1e-6 else 1e-6
                mouth_open_ratio = mouth_inner_ratio / denom
            mouth_total_ratio_max = thresholds.get(
                "mouth_total_ratio_max", DEFAULT_CONFIG["thresholds"]["mouth_total_ratio_max"]
            )
            mouth_open_ratio_max = thresholds.get(
                "mouth_open_ratio_max", DEFAULT_CONFIG["thresholds"]["mouth_open_ratio_max"]
            )
            mouth_ok = (
                mouth_total_ratio is not None
                and mouth_open_ratio is not None
                and mouth_total_ratio <= mouth_total_ratio_max
                and mouth_open_ratio <= mouth_open_ratio_max
            )
            report["items"]["mouth"] = {
                "value": {
                    "inner": mouth_inner_ratio,
                    "lip": mouth_lip_ratio,
                    "total": mouth_total_ratio,
                    "open_ratio": mouth_open_ratio,
                },
                "ok": mouth_ok,
                "thresholds": {
                    "total_max": mouth_total_ratio_max,
                    "open_ratio_max": mouth_open_ratio_max,
                },
                "source": "face_parsing",
            }
            if not mouth_ok:
                report["reasons"].append("mouth_open_or_unknown")
            _debug_log(
                config,
                f"mouth_inner={mouth_inner_ratio} lip={mouth_lip_ratio} total={mouth_total_ratio} "
                f"open_ratio={mouth_open_ratio} ok={mouth_ok} total_max={mouth_total_ratio_max} "
                f"open_ratio_max={mouth_open_ratio_max}",
            )
        else:
            report["items"]["mouth"] = {
                "value": None,
                "ok": True,
                "thresholds": {},
                "source": "disabled",
            }

        if _is_check_enabled(ctx, "hat"):
            hat_ratio = parsing_extra_wide.get("hat_ratio") if parsing_extra_wide else None
            hair_ratio = parsing_extra_wide.get("hair_ratio") if parsing_extra_wide else None
            hat_ratio_max = thresholds.get("hat_ratio_max", DEFAULT_CONFIG["thresholds"]["hat_ratio_max"])
            hair_ratio_min = thresholds.get("hair_ratio_min", DEFAULT_CONFIG["thresholds"]["hair_ratio_min"])
            hat_hair_ratio_max = thresholds.get("hat_hair_ratio_max", DEFAULT_CONFIG["thresholds"]["hat_hair_ratio_max"])
            if hat_ratio is None:
                hat_ok = False
            else:
                # If hair is strong, allow a tiny hat ratio without failing.
                if hair_ratio is not None and hair_ratio >= hair_ratio_min:
                    hat_ok = hat_ratio <= hat_hair_ratio_max
                else:
                    hat_ok = hat_ratio <= hat_ratio_max
            report["items"]["hat"] = {
                "value": hat_ratio,
                "ok": hat_ok,
                "thresholds": {
                    "max": hat_hair_ratio_max if (hair_ratio is not None and hair_ratio >= hair_ratio_min) else hat_ratio_max,
                    "hair_min": hair_ratio_min,
                },
                "source": "face_parsing",
            }
            if not hat_ok:
                report["reasons"].append("hat_detected_or_unknown")
            _debug_log(
                config,
                f"hat_ratio={hat_ratio} ok={hat_ok} max={hat_ratio_max} hair_ratio={hair_ratio} "
                f"hair_min={hair_ratio_min} hat_hair_max={hat_hair_ratio_max}",
            )
        else:
            report["items"]["hat"] = {
                "value": None,
                "ok": True,
                "thresholds": {},
                "source": "disabled",
            }

        if _is_check_enabled(ctx, "earring"):
            earring_ratio = parsing_extra_wide.get("earring_ratio") if parsing_extra_wide else None
            earring_ratio_max = thresholds.get("earring_ratio_max", DEFAULT_CONFIG["thresholds"]["earring_ratio_max"])
            earring_ok = earring_ratio is not None and earring_ratio <= earring_ratio_max
            report["items"]["earring"] = {
                "value": earring_ratio,
                "ok": earring_ok,
                "thresholds": {"max": earring_ratio_max},
                "source": "face_parsing",
            }
            if not earring_ok:
                report["reasons"].append("earring_detected_or_unknown")
            _debug_log(config, f"earring_ratio={earring_ratio} ok={earring_ok} max={earring_ratio_max}")
        else:
            report["items"]["earring"] = {
                "value": None,
                "ok": True,
                "thresholds": {},
                "source": "disabled",
            }

        if _is_check_enabled(ctx, "ears"):
            left_ear_ratio = parsing_extra_wide.get("left_ear_ratio") if parsing_extra_wide else None
            right_ear_ratio = parsing_extra_wide.get("right_ear_ratio") if parsing_extra_wide else None
            ear_ratio_min = thresholds.get("ear_ratio_min", DEFAULT_CONFIG["thresholds"]["ear_ratio_min"])
            left_ear_ok = left_ear_ratio is not None and left_ear_ratio >= ear_ratio_min
            right_ear_ok = right_ear_ratio is not None and right_ear_ratio >= ear_ratio_min

            # Fallback: use skin in side strips if ear labels are weak.
            side_strip_min = thresholds.get(
                "ear_side_strip_ratio_min", DEFAULT_CONFIG["thresholds"]["ear_side_strip_ratio_min"]
            )
            side_strip_width_ratio = thresholds.get(
                "ear_side_strip_width_ratio", DEFAULT_CONFIG["thresholds"]["ear_side_strip_width_ratio"]
            )
            side_left_ratio = None
            side_right_ratio = None
            mask = parsing_extra_wide.get("_mask") if parsing_extra_wide else None
            skin_labels = parsing_extra_wide.get("_skin_labels") if parsing_extra_wide else None
            if mask is not None and skin_labels:
                h, w = mask.shape[:2]
                strip_w = max(1, int(w * side_strip_width_ratio))
                left_strip = mask[:, :strip_w]
                right_strip = mask[:, w - strip_w :]
                skin_left = 0
                skin_right = 0
                for label in skin_labels:
                    skin_left += int(np.sum(left_strip == int(label)))
                    skin_right += int(np.sum(right_strip == int(label)))
                side_left_ratio = skin_left / float(left_strip.size)
                side_right_ratio = skin_right / float(right_strip.size)

                if not left_ear_ok and side_left_ratio >= side_strip_min:
                    left_ear_ok = True
                if not right_ear_ok and side_right_ratio >= side_strip_min:
                    right_ear_ok = True

            # Rule: one ear must be >= min, the other just > 0.
            left_any = left_ear_ratio is not None and left_ear_ratio > 0
            right_any = right_ear_ratio is not None and right_ear_ratio > 0
            ears_ok = (left_ear_ok and right_any) or (right_ear_ok and left_any)
            report["items"]["ears"] = {
                "value": {
                    "left": left_ear_ratio,
                    "right": right_ear_ratio,
                    "side_left_skin_ratio": side_left_ratio,
                    "side_right_skin_ratio": side_right_ratio,
                },
                "ok": ears_ok,
                "thresholds": {
                    "min": ear_ratio_min,
                    "side_strip_min": side_strip_min,
                    "side_strip_width_ratio": side_strip_width_ratio,
                },
                "source": "face_parsing",
            }
            if not ears_ok:
                report["reasons"].append("ears_not_visible_or_unknown")
            _debug_log(
                config,
                f"ears_left={left_ear_ratio} right={right_ear_ratio} ok={ears_ok} min={ear_ratio_min}",
            )
        else:
            report["items"]["ears"] = {
                "value": None,
                "ok": True,
                "thresholds": {},
                "source": "disabled",
            }

        neck_ratio = parsing_extra_wide.get("neck_ratio") if parsing_extra_wide else None
        neck_ratio_min = thresholds.get("neck_ratio_min", DEFAULT_CONFIG["thresholds"]["neck_ratio_min"])
        neck_ok = neck_ratio is not None and neck_ratio >= neck_ratio_min
        report["items"]["neck"] = {
            "value": neck_ratio,
            "ok": neck_ok,
            "thresholds": {"min": neck_ratio_min},
            "source": "face_parsing",
        }
        if not neck_ok:
            report["reasons"].append("neck_missing_or_unknown")
        _debug_log(config, f"neck_ratio={neck_ratio} ok={neck_ok} min={neck_ratio_min}")

        cloth_ratio = parsing_extra_wide.get("cloth_ratio") if parsing_extra_wide else None
        cloth_ratio_min = thresholds.get("cloth_ratio_min", DEFAULT_CONFIG["thresholds"]["cloth_ratio_min"])
        cloth_ok = cloth_ratio is not None and cloth_ratio >= cloth_ratio_min
        report["items"]["cloth"] = {
            "value": cloth_ratio,
            "ok": cloth_ok,
            "thresholds": {"min": cloth_ratio_min},
            "source": "face_parsing",
        }
        if not cloth_ok:
            report["reasons"].append("cloth_missing_or_unknown")
        _debug_log(config, f"cloth_ratio={cloth_ratio} ok={cloth_ok} min={cloth_ratio_min}")

    if run_post:
        face_fill_ratio = getattr(ctx, "face_fill_ratio", None)
        if face_fill_ratio is None:
            fill_skin, _, _, fill_extra = _face_parsing_metrics(face_roi)
            fill_hair = fill_extra.get("hair_ratio") if fill_extra else None
            if fill_skin is not None or fill_hair is not None:
                face_fill_ratio = float((fill_skin or 0.0) + (fill_hair or 0.0))
        target_ratio = thresholds.get(
            "face_fill_target_ratio",
            DEFAULT_CONFIG["thresholds"]["face_fill_target_ratio"],
        )
        min_scale = thresholds.get(
            "face_fill_min_scale",
            DEFAULT_CONFIG["thresholds"]["face_fill_min_scale"],
        )
        scale = 1.0
        if face_fill_ratio is not None and target_ratio:
            scale = max(min_scale, min(1.0, face_fill_ratio / target_ratio))

        head_box = config.get("matting", {}).get("head_box")
        top_fraction = config.get("matting", {}).get("top_fraction")
        head_coverage, top_coverage = _matting_head_coverage(
            ctx.matting_image,
            face_rect,
            box_factors=head_box,
            top_fraction=top_fraction,
        )
        matting_head_min = thresholds.get("matting_head_coverage_min", DEFAULT_CONFIG["thresholds"]["matting_head_coverage_min"])
        matting_top_min = thresholds.get("matting_top_coverage_min", DEFAULT_CONFIG["thresholds"]["matting_top_coverage_min"])
        head_min_dynamic = matting_head_min * scale
        top_min_dynamic = matting_top_min * scale
        head_ok = head_coverage is not None and head_coverage >= head_min_dynamic
        top_ok = top_coverage is not None and top_coverage >= top_min_dynamic
        report["items"]["matting_head_coverage"] = {
            "value": head_coverage,
            "ok": head_ok,
            "thresholds": {
                "min": head_min_dynamic,
                "base_min": matting_head_min,
                "scale": scale,
                "face_fill_ratio": face_fill_ratio,
                "target_ratio": target_ratio,
            },
        }
        report["items"]["matting_top_coverage"] = {
            "value": top_coverage,
            "ok": top_ok,
            "thresholds": {
                "min": top_min_dynamic,
                "base_min": matting_top_min,
                "scale": scale,
                "face_fill_ratio": face_fill_ratio,
                "target_ratio": target_ratio,
            },
        }
        if not head_ok or not top_ok:
            report["reasons"].append("matting_head_incomplete")
        _debug_log(
            config,
            f"matting_head_coverage={head_coverage} ok={head_ok} min={head_min_dynamic} "
            f"matting_top_coverage={top_coverage} ok={top_ok} min={top_min_dynamic} "
            f"scale={scale} face_fill_ratio={face_fill_ratio} target={target_ratio}",
        )

    report["status"] = len(report["reasons"]) == 0
    if not report["status"] and run_pre:
        raise ComplianceError(report)
    return report


def check_pre_compliance(ctx: Context) -> Dict:
    return check_compliance(ctx, stage="pre")


def check_post_compliance(ctx: Context) -> Dict:
    return check_compliance(ctx, stage="post")
