import math
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class IdleBehaviorConfig:
    idle_after_s: float = 6.0
    tick_hz: float = 10.0

    # scan when idle and no face
    sweep_period_s: float = 7.0
    sweep_yaw_deg: float = 16.0

    # pitch sweep: alternate up/down slowly (cylinder wall expectation)
    sweep_pitch_period_s: float = 10.0
    pitch_min_deg: float = -6.0
    pitch_max_deg: float = 10.0

    # YuNet detector (OpenCV FaceDetectorYN) - provide ONNX file
    yunet_score_thresh: float = 0.85
    yunet_nms_thresh: float = 0.30
    yunet_top_k: int = 5000
    detect_every_s: float = 0.25

    # face state stability
    face_hold_s: float = 1.2
    smooth_alpha: float = 0.18  # lower = smoother
    min_center_move_px: float = 6.0  # ignore tiny jitter in center

    # centering control
    deadzone_px: int = 18
    k_yaw: float = 0.10
    k_pitch: float = 0.10
    max_step_deg: float = 3.5

    # micro motions when centered: use paired impulses that auto-revert
    micro_interval_s: Tuple[float, float] = (0.9, 2.2)
    micro_yaw_deg: float = 1.4
    micro_pitch_deg: float = 0.7
    micro_revert_gain: float = 1.0  # 1.0 = full revert next tick


def _clip(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


def _triangle_wave01(t: float) -> float:
    x = t % 1.0
    return 2.0 * x if x < 0.5 else 2.0 * (1.0 - x)


def _clamp_box_to_image(
    x: float, y: float, w: float, h: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    x = max(0.0, min(float(img_w - 1), x))
    y = max(0.0, min(float(img_h - 1), y))
    w = max(1.0, min(float(img_w) - x, w))
    h = max(1.0, min(float(img_h) - y, h))
    return x, y, w, h


class IdleBehaviorController:
    def __init__(self, cfg: IdleBehaviorConfig):
        self.cfg = cfg
        self._lock = threading.Lock()

        self._last_user_activity = time.time()
        self._last_image_ts = 0.0
        self._last_face_detect_ts = 0.0

        self._img_bgr: Optional[np.ndarray] = None

        # Smoothed state exposed to control
        self._face_center: Optional[Tuple[float, float]] = None
        self._face_box: Optional[Tuple[int, int, int, int]] = None  # x,y,w,h
        self._last_face_seen_ts: float = 0.0
        self._smoothed_box_xywh: Optional[Tuple[float, float, float, float]] = None

        # micro impulse bookkeeping (paired move + revert)
        self._last_micro_ts = 0.0
        self._next_micro_in = random.uniform(*cfg.micro_interval_s)
        self._pending_revert: Optional[Tuple[float, float]] = None  # (dpitch, dyaw)

        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError(
                "cv2.FaceDetectorYN not available. Need OpenCV 4.6+ with dnn_objdetect "
                "(typically opencv-contrib-python or an OpenCV build with that module)."
            )

        model_path = str(Path(__file__).parent / "face_detection_yunet_2023mar.onnx")
        self._fd = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            self.cfg.yunet_score_thresh,
            self.cfg.yunet_nms_thresh,
            self.cfg.yunet_top_k,
        )

    def on_user_activity(self) -> None:
        with self._lock:
            self._last_user_activity = time.time()
            self._face_center = None
            self._face_box = None
            self._smoothed_box_xywh = None
            self._last_face_seen_ts = 0.0
            self._pending_revert = None

    def update_image(self, bgr_image: np.ndarray) -> None:
        with self._lock:
            self._img_bgr = bgr_image
            self._last_image_ts = time.time()

    def get_face_box(self) -> Optional[Tuple[int, int, int, int]]:
        with self._lock:
            return None if self._face_box is None else tuple(self._face_box)

    def _is_idle(self, now: float) -> bool:
        return (now - self._last_user_activity) >= self.cfg.idle_after_s

    def _face_fresh(self, now: float) -> bool:
        with self._lock:
            return (now - self._last_face_seen_ts) <= self.cfg.face_hold_s

    def _sweep(self, now: float) -> Tuple[float, float]:
        phase = (now % self.cfg.sweep_period_s) / self.cfg.sweep_period_s
        yaw = math.sin(phase * 2.0 * math.pi) * self.cfg.sweep_yaw_deg

        pphase = (now % self.cfg.sweep_pitch_period_s) / self.cfg.sweep_pitch_period_s
        u = _triangle_wave01(pphase)
        pitch = (
            self.cfg.pitch_min_deg
            + (self.cfg.pitch_max_deg - self.cfg.pitch_min_deg) * u
        )
        return pitch, yaw

    def _center_control(
        self, img_shape: Tuple[int, int], face_center: Tuple[float, float]
    ) -> Tuple[float, float]:
        h, w = img_shape[:2]
        cx, cy = face_center
        ex = cx - (w * 0.5)
        ey = cy - (h * 0.5)

        if abs(ex) < self.cfg.deadzone_px and abs(ey) < self.cfg.deadzone_px:
            return 0.0, 0.0

        nx = ex / (w * 0.5)
        ny = ey / (h * 0.5)

        dyaw = _clip(nx * self.cfg.k_yaw * 30.0, self.cfg.max_step_deg)
        dpitch = _clip(-ny * self.cfg.k_pitch * 30.0, self.cfg.max_step_deg)
        return dpitch, dyaw

    def _maybe_micro_impulse(self, now: float) -> Tuple[float, float]:
        if self._pending_revert is not None:
            dp, dy = self._pending_revert
            self._pending_revert = None
            return dp, dy

        if (now - self._last_micro_ts) < self._next_micro_in:
            return 0.0, 0.0

        self._last_micro_ts = now
        self._next_micro_in = random.uniform(*self.cfg.micro_interval_s)

        dpitch = random.uniform(-self.cfg.micro_pitch_deg, self.cfg.micro_pitch_deg)
        dyaw = random.uniform(-self.cfg.micro_yaw_deg, self.cfg.micro_yaw_deg)

        self._pending_revert = (
            -dpitch * self.cfg.micro_revert_gain,
            -dyaw * self.cfg.micro_revert_gain,
        )
        return dpitch, dyaw

    def _detect_face_yunet(
        self, img_bgr: np.ndarray
    ) -> Optional[Tuple[float, float, float, float]]:
        now = time.time()
        if (now - self._last_face_detect_ts) < self.cfg.detect_every_s:
            return None
        self._last_face_detect_ts = now

        h, w = img_bgr.shape[:2]
        self._fd.setInputSize((w, h))

        ok, faces = self._fd.detect(img_bgr)
        if not ok or faces is None or len(faces) == 0:
            return None

        best = None
        best_key = -1.0
        for f in faces:
            x, y, bw, bh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
            score = float(f[4])
            area = max(0.0, bw) * max(0.0, bh)
            key = score * area
            if key > best_key:
                best_key = key
                best = (x, y, bw, bh)

        return best

    def _update_smoothed_box(
        self, det_xywh: Tuple[float, float, float, float], img_shape
    ) -> None:
        img_h, img_w = img_shape[:2]
        x, y, w, h = _clamp_box_to_image(*det_xywh, img_w=img_w, img_h=img_h)

        alpha = float(self.cfg.smooth_alpha)

        with self._lock:
            if self._smoothed_box_xywh is None:
                sx, sy, sw, sh = x, y, w, h
            else:
                ox, oy, ow, oh = self._smoothed_box_xywh
                sx = (1.0 - alpha) * ox + alpha * x
                sy = (1.0 - alpha) * oy + alpha * y
                sw = (1.0 - alpha) * ow + alpha * w
                sh = (1.0 - alpha) * oh + alpha * h

            sx, sy, sw, sh = _clamp_box_to_image(
                sx, sy, sw, sh, img_w=img_w, img_h=img_h
            )

            new_cx = sx + sw * 0.5
            new_cy = sy + sh * 0.5
            if self._face_center is not None:
                old_cx, old_cy = self._face_center
                if (
                    abs(new_cx - old_cx) + abs(new_cy - old_cy)
                ) < self.cfg.min_center_move_px:
                    self._last_face_seen_ts = time.time()
                    return

            self._smoothed_box_xywh = (sx, sy, sw, sh)
            self._face_center = (new_cx, new_cy)
            self._face_box = (int(sx), int(sy), int(sw), int(sh))
            self._last_face_seen_ts = time.time()

    def _maintain_face_state(self, img_bgr: np.ndarray) -> None:
        det = self._detect_face_yunet(img_bgr)
        if det is None:
            return
        self._update_smoothed_box(det, img_bgr.shape)

    def tick(self) -> dict:
        now = time.time()
        if not self._is_idle(now):
            return {
                "delta_x": 0.0,
                "delta_y": 0.0,
                "delta_z": 0.0,
                "gripper": 1.0,
                "delta_pitch": 0.0,
                "delta_yaw": 0.0,
                "delta_roll": 0.0,
            }

        with self._lock:
            img = None if self._img_bgr is None else self._img_bgr.copy()

        if img is None:
            pitch, yaw = self._sweep(now)
            return {
                "delta_x": 0.0,
                "delta_y": 0.0,
                "delta_z": 0.0,
                "gripper": 1.0,
                "delta_pitch": pitch,
                "delta_yaw": yaw,
                "delta_roll": 0.0,
            }

        self._maintain_face_state(img)

        with self._lock:
            face_center = self._face_center

        if face_center is None and not self._face_fresh(now):
            pitch, yaw = self._sweep(now)
            return {
                "delta_x": 0.0,
                "delta_y": 0.0,
                "delta_z": 0.0,
                "gripper": 1.0,
                "delta_pitch": pitch,
                "delta_yaw": yaw,
                "delta_roll": 0.0,
            }

        if face_center is None:
            return {
                "delta_x": 0.0,
                "delta_y": 0.0,
                "delta_z": 0.0,
                "gripper": 1.0,
                "delta_pitch": 0.0,
                "delta_yaw": 0.0,
                "delta_roll": 0.0,
            }

        dpitch, dyaw = self._center_control(img.shape, face_center)
        centered = dpitch == 0.0 and dyaw == 0.0

        if not centered:
            self._pending_revert = None
            return {
                "delta_x": 0.0,
                "delta_y": 0.0,
                "delta_z": 0.0,
                "gripper": 1.0,
                "delta_pitch": dpitch,
                "delta_yaw": dyaw,
                "delta_roll": 0.0,
            }

        mp, my = self._maybe_micro_impulse(now)
        return {
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_z": 0.0,
            "gripper": 1.0,
            "delta_pitch": mp,
            "delta_yaw": my,
            "delta_roll": 0.0,
        }
