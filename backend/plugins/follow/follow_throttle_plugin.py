import threading
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import logging

from ..base import Plugin
from .follow_controller import FollowController
from control.strategies import DirectStrategy

logger = logging.getLogger(__name__)


class FollowThrottlePlugin(Plugin):
    """
    Follow plugin with configurable base throttle via FOLLOW_THROTTLE env var.
    Useful for drones like the E88 where center throttle causes descent.

    Key difference from FollowPlugin: throttle is set to FOLLOW_THROTTLE (default 0.15)
    instead of 0, so the drone maintains altitude while following.
    Bounding box is cyan to distinguish it from the original plugin.
    """

    def _on_start(self):
        try:
            import torch
            torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")

        self.frame_rate = int(os.getenv("FOLLOW_FPS", "20"))
        self.img_size = int(os.getenv("YOLO_IMG_SIZE", "320"))
        self.confidence = float(os.getenv("YOLO_CONFIDENCE", "0.65"))
        self.log_interval = float(os.getenv("FOLLOW_LOG_INTERVAL", "2.0"))
        self.base_throttle = float(os.getenv("FOLLOW_THROTTLE", "0.15"))

        weights_env = os.getenv("YOLO_WEIGHTS")
        if weights_env and os.path.exists(weights_env):
            weights_path = weights_env
        else:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            default_weights = os.path.join(repo_root, "yolov10n.pt")
            weights_path = default_weights if os.path.exists(default_weights) else "yolov10n.pt"

        self.yolo = YOLO(weights_path)

        self.ctrl = FollowController(
            yaw_deadzone=float(os.getenv("FOLLOW_CENTER_DEADZONE", "0.15")),
            pitch_deadzone=float(os.getenv("FOLLOW_PITCH_DEADZONE", "0.02")),
            min_box_width=float(os.getenv("FOLLOW_MIN_BOX_WIDTH", "0.30")),
            max_box_width=float(os.getenv("FOLLOW_MAX_BOX_WIDTH", "0.80")),
            invert_yaw=os.getenv("FOLLOW_INVERT_YAW", "false").lower() in ("1", "true", "yes"),
            invert_pitch=os.getenv("FOLLOW_INVERT_PITCH", "false").lower() in ("1", "true", "yes"),
            yaw_speed=float(os.getenv("FOLLOW_YAW_SPEED", "20.0")),
            pitch_speed=float(os.getenv("FOLLOW_PITCH_SPEED", "20.0")),
        )

        self._prev_strategy = getattr(self.fc.model, "strategy", None)
        self._prev_expo = getattr(self.fc.model, "expo_factor", None)

        try:
            self.fc.model.set_strategy(DirectStrategy())
            self.fc.model.expo_factor = 0.0
            logger.info("[FollowThrottlePlugin] Started — throttle_base=%.2f", self.base_throttle)
        except Exception as e:
            logger.warning("[FollowThrottlePlugin] Strategy warning: %s", e)

        self.loop_thread = threading.Thread(target=self._loop, daemon=True)
        self.loop_thread.start()

    def _on_stop(self):
        try:
            if self._prev_strategy is not None:
                self.fc.model.set_strategy(self._prev_strategy)
            if self._prev_expo is not None:
                self.fc.model.expo_factor = self._prev_expo
        except Exception:
            pass

        if self.loop_thread:
            self.loop_thread.join(timeout=1.0)

    def _loop(self):
        logger.info("[FollowThrottlePlugin] Loop started. Waiting for frames...")
        frame_interval = 1.0 / self.frame_rate
        last_frame_time = 0
        last_log_time = 0

        for frame in self.frames:
            now = time.time()
            if now - last_frame_time < frame_interval:
                continue
            last_frame_time = now

            if not self.running:
                break

            if hasattr(frame, "format") and frame.format == "jpeg":
                img = cv2.imdecode(np.frombuffer(frame.data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
            elif isinstance(frame, np.ndarray):
                img = frame
            else:
                continue

            persons = []
            try:
                results = self.yolo(
                    img,
                    stream=True,
                    verbose=False,
                    classes=[0],
                    imgsz=self.img_size,
                    conf=self.confidence,
                )
                for r in results:
                    for box in r.boxes or []:
                        xyxy = box.xyxy[0].tolist()
                        persons.append(xyxy)
            except Exception:
                continue

            if not persons:
                self.fc.set_axes(throttle=self.base_throttle, yaw=0, pitch=0, roll=0)
                self.send_overlay([])
                continue

            x1, y1, x2, y2 = max(persons, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            box_w = x2 - x1
            frame_h, frame_w = img.shape[:2]

            self.send_overlay([{
                "type": "rect",
                "coords": [x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h],
                "color": "cyan",
            }])

            yaw, pitch = self.ctrl.compute(
                box_center_x=(x1 + box_w / 2) / frame_w,
                box_width=box_w / frame_w,
            )
            self.fc.set_axes_from(
                "follow_t",
                throttle=self.base_throttle,
                yaw=yaw / 100.0,
                pitch=pitch / 100.0,
                roll=0,
            )

            if now - last_log_time >= self.log_interval:
                center_offset = ((x1 + box_w / 2) / frame_w - 0.5) * 100
                logger.debug(
                    "[FollowThrottlePlugin] offset=%+.1f%% box=%.1f%% yaw=%.0f pitch=%.0f thr=%.2f",
                    center_offset, box_w / frame_w * 100, yaw, pitch, self.base_throttle,
                )
                last_log_time = now
