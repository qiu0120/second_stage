#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from second_stage.my_gait import Robot_Ctrl
from second_stage.robot_control_cmd_lcmt import robot_control_cmd_lcmt

# ===== Integrated fourth-stage additions =====
import os
from dataclasses import dataclass
from typing import Any
from threading import Thread, Lock
import lcm
from cyberdog_msg.msg import YamlParam, ApplyForce


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class Detection:
    det_type: str
    center_img: Tuple[int, int]
    bbox_img: Tuple[int, int, int, int]
    score: float
    extra: Dict[str, Any]


class ControlParameterValueKind:
    kDOUBLE = 1
    kS64 = 2
    kVEC_X_DOUBLE = 3
    kMAT_X_DOUBLE = 4


class VoicePlayer:
    """
    比赛语音播报：提前准备 wav 文件，识别到对应目标时异步 aplay 播放。
    重点：播放在单独线程里执行，不阻塞 ROS2 状态机和速度控制循环。
    """

    def __init__(self, voice_dir: str = '/home/cyberdog_sim/voice', enabled: bool = True):
        self.voice_dir = voice_dir
        self.enabled = bool(enabled)
        self.lock = Lock()
        self.playing = False

        self.voice_files = {
            'bar': 'bar.wav',  # 识别到限高杆
            'obstacle': 'obstacle.wav',  # 识别到无法跨越障碍
            'cola': 'cola.wav',  # 识别到可乐瓶
            'orange_ball': 'orange_ball.wav',  # 识别到橙色小球
            'football': 'football.wav',  # 识别到足球
            # 兼容当前代码里的检测类型命名
            'blue_ball': 'orange_ball.wav',
            'white_ball': 'football.wav',
        }

    def play_async(self, key: str) -> bool:
        if not self.enabled:
            return False
        if key not in self.voice_files:
            print(f'[VOICE] unknown key: {key}')
            return False

        with self.lock:
            # 避免多个音频叠在一起播。如果正在播报，就跳过本次播报。
            if self.playing:
                print(f'[VOICE] busy, skip key={key}')
                return False
            self.playing = True

        def _run():
            try:
                path = os.path.join(self.voice_dir, self.voice_files[key])
                if os.path.exists(path):
                    os.system(f'aplay "{path}" >/dev/null 2>&1')
                else:
                    print(f'[VOICE] file not found: {path}')
            finally:
                with self.lock:
                    self.playing = False

        Thread(target=_run, daemon=True).start()
        return True


class BaseDetector:
    def __init__(self, cfg: Dict[str, Any]):
        self.roi_x_ratio_min = cfg['roi_x_ratio_min']
        self.roi_x_ratio_max = cfg['roi_x_ratio_max']
        self.roi_y_ratio_min = cfg['roi_y_ratio_min']
        self.roi_y_ratio_max = cfg['roi_y_ratio_max']

    def _roi(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        x1 = int(w * self.roi_x_ratio_min)
        x2 = int(w * self.roi_x_ratio_max)
        y1 = int(h * self.roi_y_ratio_min)
        y2 = int(h * self.roi_y_ratio_max)
        return (x1, y1, x2, y2), frame_bgr[y1:y2, x1:x2].copy()


class BarColorDetector(BaseDetector):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.lower_bar = np.array([cfg['h_min'], cfg['s_min'], cfg['v_min']], dtype=np.uint8)
        self.upper_bar = np.array([cfg['h_max'], cfg['s_max'], cfg['v_max']], dtype=np.uint8)
        self.kernel_open = np.ones((cfg['open_kernel'], cfg['open_kernel']), np.uint8)
        self.kernel_close = np.ones((cfg['close_kernel_h'], cfg['close_kernel_w']), np.uint8)
        self.min_area = cfg['min_area']
        self.min_width = cfg['min_width']
        self.max_height = cfg['max_height']
        self.min_aspect_ratio = cfg['min_aspect_ratio']
        self.max_aspect_ratio = cfg['max_aspect_ratio']
        self.max_center_y_ratio_in_roi = cfg['max_center_y_ratio_in_roi']
        self.center_weight_base = cfg['center_weight_base']
        self.center_weight_gain = cfg['center_weight_gain']

    def detect(self, frame_bgr) -> Optional[Detection]:
        (x1, y1, x2, y2), roi = self._roi(frame_bgr)
        roi_h, roi_w = roi.shape[:2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_bar, self.upper_bar)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_center_x = roi_w / 2.0
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            if rw <= 0 or rh <= 0:
                continue
            aspect_ratio = rw / float(rh)
            center_y_ratio = (ry + rh * 0.5) / float(max(roi_h, 1))
            center_x = rx + rw / 2.0
            x_dist_norm = abs(center_x - roi_center_x) / max(roi_w / 2.0, 1.0)
            center_bonus = 1.0 - x_dist_norm
            if area < self.min_area:
                continue
            if rw < self.min_width:
                continue
            if rh > self.max_height:
                continue
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            if center_y_ratio > self.max_center_y_ratio_in_roi:
                continue

            center_score = max(center_bonus, 0.0)
            area_score = math.sqrt(max(area, 1.0))
            shape_score = min(aspect_ratio, 10.0)

            score = center_score

            # 限高杆角度：用当前轮廓拟合直线，得到相对图像水平线的有符号角度。
            # angle_deg > 0 表示图像中从左到右向下倾斜；angle_deg < 0 表示从左到右向上倾斜。
            angle_deg = 0.0
            if cnt is not None and len(cnt) >= 2:
                vx, vy, _, _ = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                vx = float(vx)
                vy = float(vy)
                angle_deg = math.degrees(math.atan2(vy, vx))
                while angle_deg > 90.0:
                    angle_deg -= 180.0
                while angle_deg < -90.0:
                    angle_deg += 180.0

            candidates.append((score, rx, ry, rw, rh, aspect_ratio, angle_deg))
        if not candidates:
            return None
        score, rx, ry, rw, rh, aspect_ratio, angle_deg = max(candidates, key=lambda x: x[0])
        bx1, by1 = x1 + rx, y1 + ry
        bx2, by2 = bx1 + rw, by1 + rh
        cx = bx1 + rw // 2
        cy = by1 + rh // 2
        return Detection('bar', (cx, cy), (bx1, by1, bx2, by2), float(score), {
            'aspect_ratio': float(aspect_ratio),
            'angle_deg': float(angle_deg),
            'abs_tilt_deg': float(abs(angle_deg)),
        })


class BallDetector(BaseDetector):
    def __init__(self, cfg: Dict[str, Any], det_type: str):
        super().__init__(cfg)
        self.det_type = det_type
        self.lower = np.array([cfg['h_min'], cfg['s_min'], cfg['v_min']], dtype=np.uint8)
        self.upper = np.array([cfg['h_max'], cfg['s_max'], cfg['v_max']], dtype=np.uint8)
        self.kernel_open = np.ones((cfg['open_kernel'], cfg['open_kernel']), np.uint8)
        self.kernel_close = np.ones((cfg['close_kernel'], cfg['close_kernel']), np.uint8)
        self.min_area = cfg['min_area']
        self.max_area = cfg['max_area']
        self.min_radius = cfg['min_radius']
        self.max_radius = cfg['max_radius']
        self.min_circularity = cfg['min_circularity']
        self.min_wh_ratio = cfg['min_wh_ratio']
        self.max_wh_ratio = cfg['max_wh_ratio']
        self.max_center_y_ratio_in_roi = cfg['max_center_y_ratio_in_roi']
        self.center_weight_base = cfg['center_weight_base']
        self.center_weight_gain = cfg['center_weight_gain']
        self.radius_score_gain = cfg['radius_score_gain']

    def detect(self, frame_bgr) -> Optional[Detection]:
        (x1, y1, x2, y2), roi = self._roi(frame_bgr)
        roi_h, roi_w = roi.shape[:2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_center_x = roi_w / 2.0
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4.0 * math.pi * area) / (perimeter * perimeter) if perimeter > 1e-6 else 0.0
            wh_ratio = bw / float(bh)
            center_y_ratio = cy / float(max(roi_h, 1))
            x_dist_norm = abs(cx - roi_center_x) / max(roi_w / 2.0, 1.0)
            center_bonus = 1.0 - x_dist_norm
            if area < self.min_area or area > self.max_area:
                continue
            if radius < self.min_radius or radius > self.max_radius:
                continue
            if circularity < self.min_circularity:
                continue
            if wh_ratio < self.min_wh_ratio or wh_ratio > self.max_wh_ratio:
                continue
            if center_y_ratio > self.max_center_y_ratio_in_roi:
                continue
            score = (radius * self.radius_score_gain) * max(circularity, 0.0) * (
                        self.center_weight_base + self.center_weight_gain * center_bonus)
            candidates.append((score, x, y, bw, bh, cx, cy, radius, circularity))
        if not candidates:
            return None
        score, x, y, bw, bh, cx, cy, radius, circularity = max(candidates, key=lambda c: c[0])
        bx1, by1 = x1 + x, y1 + y
        bx2, by2 = bx1 + bw, by1 + bh
        cx_img = x1 + int(round(cx))
        cy_img = y1 + int(round(cy))
        return Detection(self.det_type, (cx_img, cy_img), (bx1, by1, bx2, by2), float(score), {
            'radius': float(radius),
            'circularity': float(circularity),
        })


class ColaDetector(BaseDetector):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.lower = np.array([cfg['h_min'], cfg['s_min'], cfg['v_min']], dtype=np.uint8)
        self.upper = np.array([cfg['h_max'], cfg['s_max'], cfg['v_max']], dtype=np.uint8)
        self.kernel_open = np.ones((cfg['open_kernel'], cfg['open_kernel']), np.uint8)
        self.kernel_close = np.ones((cfg['close_kernel'], cfg['close_kernel']), np.uint8)
        self.min_area = cfg['min_area']
        self.max_area = cfg['max_area']
        self.min_width = cfg['min_width']
        self.max_width = cfg['max_width']
        self.min_height = cfg['min_height']
        self.max_height = cfg['max_height']
        self.min_hw_ratio = cfg['min_hw_ratio']
        self.max_hw_ratio = cfg['max_hw_ratio']
        self.max_center_y_ratio_in_roi = cfg['max_center_y_ratio_in_roi']
        self.center_weight_base = cfg['center_weight_base']
        self.center_weight_gain = cfg['center_weight_gain']

    def detect(self, frame_bgr) -> Optional[Detection]:
        (x1, y1, x2, y2), roi = self._roi(frame_bgr)
        roi_h, roi_w = roi.shape[:2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_center_x = roi_w / 2.0
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                continue
            hw_ratio = bh / float(bw)
            center_y_ratio = (y + 0.5 * bh) / float(max(roi_h, 1))
            center_x = x + 0.5 * bw
            x_dist_norm = abs(center_x - roi_center_x) / max(roi_w / 2.0, 1.0)
            center_bonus = 1.0 - x_dist_norm
            if area < self.min_area or area > self.max_area:
                continue
            if bw < self.min_width or bw > self.max_width:
                continue
            if bh < self.min_height or bh > self.max_height:
                continue
            if hw_ratio < self.min_hw_ratio or hw_ratio > self.max_hw_ratio:
                continue
            if center_y_ratio > self.max_center_y_ratio_in_roi:
                continue
            score = area * min(hw_ratio, 8.0) * (self.center_weight_base + self.center_weight_gain * center_bonus)
            candidates.append((score, x, y, bw, bh, hw_ratio))
        if not candidates:
            return None
        score, x, y, bw, bh, hw_ratio = max(candidates, key=lambda c: c[0])
        bx1, by1 = x1 + x, y1 + y
        bx2, by2 = bx1 + bw, by1 + bh
        cx = bx1 + bw // 2
        cy = by1 + bh // 2
        return Detection('cola', (cx, cy), (bx1, by1, bx2, by2), float(score), {'hw_ratio': float(hw_ratio)})


class ObstacleBlueDepthDetector:
    def __init__(self, cfg: Dict[str, Any]):
        self.roi_x_ratio_min = cfg['roi_x_ratio_min']
        self.roi_x_ratio_max = cfg['roi_x_ratio_max']
        self.roi_y_ratio_min = cfg['roi_y_ratio_min']
        self.roi_y_ratio_max = cfg['roi_y_ratio_max']

        self.lower_blue = np.array(
            [cfg['h_min'], cfg['s_min'], cfg['v_min']],
            dtype=np.uint8
        )
        self.upper_blue = np.array(
            [cfg['h_max'], cfg['s_max'], cfg['v_max']],
            dtype=np.uint8
        )

        self.depth_min_m = cfg['depth_min_m']
        self.depth_max_m = cfg['depth_max_m']

        self.kernel_open = np.ones((cfg['open_kernel'], cfg['open_kernel']), np.uint8)
        self.kernel_close = np.ones((cfg['close_kernel'], cfg['close_kernel']), np.uint8)

        self.min_area = cfg['min_area']
        self.min_width = cfg['min_width']
        self.min_height = cfg['min_height']
        self.max_aspect_ratio = cfg['max_aspect_ratio']
        self.min_bottom_y_ratio_in_roi = cfg['min_bottom_y_ratio_in_roi']

        self.min_valid_depth_ratio = cfg['min_valid_depth_ratio']
        self.min_near_depth_ratio = cfg['min_near_depth_ratio']

    def depth_to_meters(self, depth_img):
        if depth_img is None:
            return None

        if depth_img.dtype == np.float32:
            depth_m = depth_img.copy()
        elif depth_img.dtype == np.uint16:
            depth_m = depth_img.astype(np.float32) / 1000.0
        else:
            depth_m = depth_img.astype(np.float32)

        depth_m[~np.isfinite(depth_m)] = 0.0
        return depth_m

    def detect(self, frame_bgr, depth_img) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        depth_m = self.depth_to_meters(depth_img)

        if depth_m is None:
            return {
                'detected': False,
                'candidates': [],
                'debug_infos': [],
                'frame_vis': frame_bgr.copy(),
                'mask': None,
            }

        x1 = int(w * self.roi_x_ratio_min)
        x2 = int(w * self.roi_x_ratio_max)
        y1 = int(h * self.roi_y_ratio_min)
        y2 = int(h * self.roi_y_ratio_max)

        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))

        roi_bgr = frame_bgr[y1:y2, x1:x2].copy()
        roi_depth_m = depth_m[y1:y2, x1:x2].copy()
        roi_h, roi_w = roi_bgr.shape[:2]

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        debug_infos = []

        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            rx, ry, rw, rh = cv2.boundingRect(cnt)

            reasons = []

            if rw <= 0 or rh <= 0:
                reasons.append('invalid_bbox')
                continue

            aspect_ratio = rw / float(max(rh, 1))
            bottom_y_ratio = (ry + rh) / float(max(roi_h, 1))

            if area < self.min_area:
                reasons.append(f'area<{self.min_area}')
            if rw < self.min_width:
                reasons.append(f'width<{self.min_width}')
            if rh < self.min_height:
                reasons.append(f'height<{self.min_height}')
            if aspect_ratio > self.max_aspect_ratio:
                reasons.append(f'aspect>{self.max_aspect_ratio}')
            if bottom_y_ratio < self.min_bottom_y_ratio_in_roi:
                reasons.append(f'bottom_y_ratio<{self.min_bottom_y_ratio_in_roi:.2f}')

            depth_patch = roi_depth_m[ry:ry + rh, rx:rx + rw]
            valid_mask = np.isfinite(depth_patch) & (depth_patch > 0.0)
            near_mask = (
                    valid_mask
                    & (depth_patch >= self.depth_min_m)
                    & (depth_patch <= self.depth_max_m)
            )

            total_pixels = max(rw * rh, 1)
            valid_depth_ratio = float(np.count_nonzero(valid_mask)) / float(total_pixels)
            near_depth_ratio = float(np.count_nonzero(near_mask)) / float(total_pixels)

            if np.any(near_mask):
                median_depth = float(np.median(depth_patch[near_mask]))
            else:
                median_depth = None

            if valid_depth_ratio < self.min_valid_depth_ratio:
                reasons.append(f'valid_depth_ratio<{self.min_valid_depth_ratio:.2f}')
            if near_depth_ratio < self.min_near_depth_ratio:
                reasons.append(f'near_depth_ratio<{self.min_near_depth_ratio:.2f}')

            passed = len(reasons) == 0

            bx1 = x1 + rx
            by1 = y1 + ry
            bx2 = bx1 + rw
            by2 = by1 + rh
            cx = bx1 + rw // 2
            cy = by1 + rh // 2

            info = {
                'idx': idx,
                'bbox_roi': (rx, ry, rw, rh),
                'bbox_img': (bx1, by1, bx2, by2),
                'center_img': (cx, cy),
                'area': float(area),
                'aspect_ratio': float(aspect_ratio),
                'bottom_y_ratio': float(bottom_y_ratio),
                'valid_depth_ratio': float(valid_depth_ratio),
                'near_depth_ratio': float(near_depth_ratio),
                'median_depth': median_depth,
                'passed': passed,
                'reasons': reasons,
            }
            debug_infos.append(info)

            if passed:
                score = area + 100.0 * bottom_y_ratio
                candidates.append(
                    Detection(
                        det_type='blue_obstacle',
                        center_img=(cx, cy),
                        bbox_img=(bx1, by1, bx2, by2),
                        score=float(score),
                        extra={
                            'median_depth': median_depth,
                            'area': float(area),
                            'aspect_ratio': float(aspect_ratio),
                            'bottom_y_ratio': float(bottom_y_ratio),
                            'valid_depth_ratio': float(valid_depth_ratio),
                            'near_depth_ratio': float(near_depth_ratio),
                        }
                    )
                )

        candidates.sort(
            key=lambda d: (
                d.extra.get('median_depth') if d.extra.get('median_depth') is not None else 999.0,
                -d.score
            )
        )

        frame_vis = frame_bgr.copy()
        cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        for i, det in enumerate(candidates):
            bx1, by1, bx2, by2 = det.bbox_img
            cx, cy = det.center_img
            d = det.extra.get('median_depth')

            cv2.rectangle(frame_vis, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
            cv2.circle(frame_vis, (cx, cy), 4, (255, 0, 0), -1)

            depth_text = 'None' if d is None else f'{d:.2f}'
            cv2.putText(
                frame_vis,
                f'OBS{i} d={depth_text}',
                (bx1, max(20, by1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

        return {
            'detected': len(candidates) > 0,
            'candidates': candidates,
            'debug_infos': debug_infos,
            'frame_vis': frame_vis,
            'mask': mask,
        }


class YellowDashedLineDetector:
    def __init__(self, cfg: Dict[str, Any]):
        self.roi_x_ratio_min = cfg['roi_x_ratio_min']
        self.roi_x_ratio_max = cfg['roi_x_ratio_max']
        self.roi_y_ratio_min = cfg['roi_y_ratio_min']
        self.roi_y_ratio_max = cfg['roi_y_ratio_max']

        self.lower_yellow = np.array(
            [cfg['h_min'], cfg['s_min'], cfg['v_min']],
            dtype=np.uint8
        )
        self.upper_yellow = np.array(
            [cfg['h_max'], cfg['s_max'], cfg['v_max']],
            dtype=np.uint8
        )

        self.open_kernel = np.ones((cfg['open_kernel'], cfg['open_kernel']), np.uint8)
        self.close_kernel = np.ones(
            (cfg['dash_close_kernel_h'], cfg['dash_close_kernel_w']),
            np.uint8
        )

        self.min_area = cfg['min_area']
        self.max_area = cfg['max_area']
        self.min_width = cfg['min_width']
        self.min_height = cfg['min_height']

        self.dash_min_segments = cfg['dash_min_segments']
        self.dash_min_total_span_y = cfg['dash_min_total_span_y']
        self.dash_max_adjacent_x_diff = cfg['dash_max_adjacent_x_diff']
        self.dash_max_gap_y = cfg['dash_max_gap_y']
        self.dash_min_gap_y = cfg['dash_min_gap_y']
        self.dash_max_total_x_range = cfg['dash_max_total_x_range']

        self.dash_segment_max_aspect_ratio = cfg['dash_segment_max_aspect_ratio']
        self.dash_segment_max_long_side = cfg['dash_segment_max_long_side']

        self.dash_duplicate_iou_thresh = cfg['dash_duplicate_iou_thresh']
        self.dash_duplicate_center_x_thresh = cfg['dash_duplicate_center_x_thresh']

        self.max_dashed_lines = cfg['max_dashed_lines']

    def _roi(self, frame_bgr):
        h, w = frame_bgr.shape[:2]

        x1 = int(w * self.roi_x_ratio_min)
        x2 = int(w * self.roi_x_ratio_max)
        y1 = int(h * self.roi_y_ratio_min)
        y2 = int(h * self.roi_y_ratio_max)

        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))

        roi = frame_bgr[y1:y2, x1:x2].copy()
        return (x1, y1, x2, y2), roi

    def _make_mask(self, roi_bgr):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)

        return mask

    def _is_valid_dash_segment_blob(self, b) -> bool:
        if b['long_side'] > self.dash_segment_max_long_side:
            return False
        if b['aspect_ratio'] > self.dash_segment_max_aspect_ratio:
            return False
        return True

    def _get_all_yellow_blobs(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blobs = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            if w < self.min_width or h < self.min_height:
                continue

            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect

            long_side = max(rw, rh)
            short_side = max(1.0, min(rw, rh))
            aspect_ratio = long_side / short_side

            blob = {
                'cnt': cnt,
                'area': float(area),
                'x': int(x),
                'y': int(y),
                'w': int(w),
                'h': int(h),
                'cx': float(cx),
                'cy': float(cy),
                'long_side': float(long_side),
                'short_side': float(short_side),
                'aspect_ratio': float(aspect_ratio),
                'angle': float(angle),
                'valid_dash_segment': False,
            }

            blob['valid_dash_segment'] = self._is_valid_dash_segment_blob(blob)
            blobs.append(blob)

        return blobs

    def _get_dash_blobs(self, mask):
        all_blobs = self._get_all_yellow_blobs(mask)
        return [b for b in all_blobs if b['valid_dash_segment']]

    def _build_group_from_start(self, start_idx: int, blobs_sorted: List[Dict[str, Any]]):
        base = blobs_sorted[start_idx]
        group = [base]
        last = base

        for j in range(start_idx + 1, len(blobs_sorted)):
            b = blobs_sorted[j]

            x_diff = abs(b['cx'] - last['cx'])
            gap_y = b['y'] - (last['y'] + last['h'])

            if gap_y < self.dash_min_gap_y:
                continue

            if x_diff <= self.dash_max_adjacent_x_diff and gap_y <= self.dash_max_gap_y:
                group.append(b)
                last = b

        return group

    def _group_to_detection(self, group, rx1: int, ry1: int, roi_h: int) -> Optional[Detection]:
        if len(group) < self.dash_min_segments:
            return None

        min_x = min(b['x'] for b in group)
        min_y = min(b['y'] for b in group)
        max_x = max(b['x'] + b['w'] for b in group)
        max_y = max(b['y'] + b['h'] for b in group)

        total_span_y = max_y - min_y
        total_x_range = max(b['cx'] for b in group) - min(b['cx'] for b in group)
        total_area = sum(b['area'] for b in group)

        if total_span_y < self.dash_min_total_span_y:
            return None

        if total_x_range > self.dash_max_total_x_range:
            return None

        x1 = rx1 + min_x
        y1 = ry1 + min_y
        x2 = rx1 + max_x
        y2 = ry1 + max_y

        cx = rx1 + int((min_x + max_x) / 2)
        cy = ry1 + int((min_y + max_y) / 2)

        bottom_ratio = max_y / float(max(roi_h, 1))

        score = (
                300.0 * len(group)
                + 2.0 * total_span_y
                + 100.0 * bottom_ratio
                + 0.01 * total_area
                - 0.5 * total_x_range
        )

        return Detection(
            det_type='yellow_vertical_dashed_line',
            center_img=(cx, cy),
            bbox_img=(x1, y1, x2, y2),
            score=float(score),
            extra={
                'segments': len(group),
                'total_span_y': float(total_span_y),
                'total_x_range': float(total_x_range),
                'total_area': float(total_area),
                'bottom_ratio': float(bottom_ratio),
                'group_centers': [
                    (float(rx1 + b['cx']), float(ry1 + b['cy']))
                    for b in group
                ],
            }
        )

    def _bbox_iou(self, box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))

        return inter_area / float(area_a + area_b - inter_area + 1e-6)

    def _remove_duplicate_dashed(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return []

        detections = sorted(
            detections,
            key=lambda d: (
                d.extra.get('total_span_y', 0.0),
                d.extra.get('segments', 0),
                d.score
            ),
            reverse=True
        )

        kept = []

        for det in detections:
            keep = True
            cx = det.center_img[0]

            for old in kept:
                old_cx = old.center_img[0]
                iou = self._bbox_iou(det.bbox_img, old.bbox_img)
                center_x_close = abs(cx - old_cx) <= self.dash_duplicate_center_x_thresh

                if iou >= self.dash_duplicate_iou_thresh or center_x_close:
                    keep = False
                    break

            if keep:
                kept.append(det)

        return kept

    def detect_dashed_lines(self, frame_bgr) -> List[Detection]:
        (rx1, ry1, rx2, ry2), roi = self._roi(frame_bgr)
        roi_h = ry2 - ry1

        mask = self._make_mask(roi)
        blobs = self._get_dash_blobs(mask)

        if len(blobs) < self.dash_min_segments:
            return []

        blobs_sorted = sorted(blobs, key=lambda b: b['cy'])

        raw_detections = []

        for i in range(len(blobs_sorted)):
            group = self._build_group_from_start(i, blobs_sorted)
            det = self._group_to_detection(group, rx1, ry1, roi_h)

            if det is not None:
                raw_detections.append(det)

        if not raw_detections:
            return []

        detections = self._remove_duplicate_dashed(raw_detections)

        detections.sort(
            key=lambda d: (
                d.extra.get('total_span_y', 0.0),
                d.extra.get('segments', 0),
                d.score
            ),
            reverse=True
        )

        return detections

    def detect_top_dashed_lines(self, frame_bgr) -> List[Detection]:
        dashed = self.detect_dashed_lines(frame_bgr)
        return dashed[:self.max_dashed_lines]


class YellowHorizontalLineDetector:
    def __init__(self, cfg: Dict[str, Any]):
        self.roi_x_ratio_min = cfg['roi_x_ratio_min']
        self.roi_x_ratio_max = cfg['roi_x_ratio_max']
        self.roi_y_ratio_min = cfg['roi_y_ratio_min']
        self.roi_y_ratio_max = cfg['roi_y_ratio_max']

        self.lower_yellow = np.array(
            [cfg['h_min'], cfg['s_min'], cfg['v_min']],
            dtype=np.uint8
        )
        self.upper_yellow = np.array(
            [cfg['h_max'], cfg['s_max'], cfg['v_max']],
            dtype=np.uint8
        )

        self.open_kernel = np.ones((cfg['open_kernel'], cfg['open_kernel']), np.uint8)
        self.close_kernel = np.ones((cfg['close_kernel_h'], cfg['close_kernel_w']), np.uint8)

        self.min_area = cfg['min_area']
        self.min_width = cfg['min_width']
        self.min_height = cfg['min_height']
        self.min_width_ratio = cfg['min_width_ratio']
        self.min_wh_ratio = cfg['min_wh_ratio']
        self.max_tilt_deg = cfg['max_tilt_deg']
        self.center_tolerance_ratio = cfg['center_tolerance_ratio']

    def _roi(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        x1 = int(w * self.roi_x_ratio_min)
        x2 = int(w * self.roi_x_ratio_max)
        y1 = int(h * self.roi_y_ratio_min)
        y2 = int(h * self.roi_y_ratio_max)
        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))
        return (x1, y1, x2, y2), frame_bgr[y1:y2, x1:x2].copy()

    def _signed_line_angle_deg(self, cnt) -> float:
        """
        Use RGB contour fitLine to estimate signed tilt angle relative to image horizontal.
        0 deg means horizontal. The sign is only used by wz correction.
        """
        if cnt is None or len(cnt) < 2:
            return 0.0
        vx, vy, _, _ = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        vx = float(vx)
        vy = float(vy)
        angle = math.degrees(math.atan2(vy, vx))
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        return float(angle)

    def detect(self, frame_bgr) -> Optional[Detection]:
        """
        RGB-only horizontal yellow-line detector.
        Distance-to-line is represented by bottom_ratio = line_bottom_y / image_height,
        matching the previous second-stage logic: line_bottom_y >= h * ratio.
        """
        h, w = frame_bgr.shape[:2]
        (rx1, ry1, rx2, ry2), roi = self._roi(frame_bgr)
        roi_h, roi_w = roi.shape[:2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < self.min_width or bh < self.min_height:
                continue

            wh_ratio = bw / float(max(bh, 1))
            if wh_ratio < self.min_wh_ratio:
                continue

            width_ratio = bw / float(max(roi_w, 1))
            if width_ratio < self.min_width_ratio:
                continue

            cx_roi = x + bw / 2.0
            roi_cx = roi_w / 2.0
            center_offset_ratio = abs(cx_roi - roi_cx) / float(max(roi_w, 1))
            if center_offset_ratio > self.center_tolerance_ratio:
                continue

            angle_deg = self._signed_line_angle_deg(cnt)
            abs_tilt_deg = abs(angle_deg)
            if abs_tilt_deg > self.max_tilt_deg:
                continue

            bx1 = rx1 + x
            by1 = ry1 + y
            bx2 = bx1 + bw
            by2 = by1 + bh
            cx = bx1 + bw // 2
            cy = by1 + bh // 2
            bottom_y = by2
            bottom_ratio = bottom_y / float(max(h, 1))

            # Prefer the bottom-most line, same idea as previous yellow stop-line logic.
            score = 3.0 * bottom_y + 0.02 * area + 100.0 * width_ratio - 2.0 * abs_tilt_deg

            candidates.append(
                Detection(
                    det_type='yellow_horizontal_line',
                    center_img=(int(cx), int(cy)),
                    bbox_img=(int(bx1), int(by1), int(bx2), int(by2)),
                    score=float(score),
                    extra={
                        'area': float(area),
                        'angle_deg': float(angle_deg),
                        'abs_tilt_deg': float(abs_tilt_deg),
                        'width_ratio': float(width_ratio),
                        'wh_ratio': float(wh_ratio),
                        'center_offset_ratio': float(center_offset_ratio),
                        'bottom_y': int(bottom_y),
                        'bottom_ratio': float(bottom_ratio),
                    }
                )
            )

        if not candidates:
            return None

        return max(candidates, key=lambda d: d.score)


class FourthStageMixin:
    # ============================================================
    # 全局总调度状态：启动预左移 -> 横向搜索 -> 居中 -> 子流程 -> 完成后左移
    # ============================================================
    # 程序启动后先固定向左移动一段距离，这段时间不识别目标；完成后才进入全局搜索。
    GLOBAL_INITIAL_LATERAL_SHIFT = 'GLOBAL_INITIAL_LATERAL_SHIFT'

    # 全局搜索：向左移动，同时按剩余任务数选择性检测限高杆/障碍物。
    GLOBAL_LATERAL_SEARCH = 'GLOBAL_LATERAL_SEARCH'
    GLOBAL_CENTER_BAR = 'GLOBAL_CENTER_BAR'
    GLOBAL_CENTER_OBSTACLE = 'GLOBAL_CENTER_OBSTACLE'
    GLOBAL_SHIFT_AFTER_SUBTASK = 'GLOBAL_SHIFT_AFTER_SUBTASK'

    # 限高杆子流程状态
    BAR_FORWARD_UNDER = 'BAR_FORWARD_UNDER'
    BAR_SEARCH_TARGET = 'BAR_SEARCH_TARGET'
    BAR_APPROACH_TARGET = 'BAR_APPROACH_TARGET'
    BAR_HIT_TARGET = 'BAR_HIT_TARGET'
    BAR_BACKOFF_TO_BAR = 'BAR_BACKOFF_TO_BAR'

    # 障碍物流程真正结束后的中转状态：用于给全局计数，不直接 DONE
    OBSTACLE_FLOW_DONE = 'OBSTACLE_FLOW_DONE'

    APPROACH_OBSTACLES = 'APPROACH_OBSTACLES'

    # 新增：刚开始识别到黄线后，先朝虚线所在方向横移一小段
    DASH_PRE_SIDE_SHIFT = 'DASH_PRE_SIDE_SHIFT'

    ALIGN_DASHED_LINE = 'ALIGN_DASHED_LINE'
    FOLLOW_DASHED_UNTIL_LOST = 'FOLLOW_DASHED_UNTIL_LOST'

    # 新增：虚线消失后的后续任务
    POST_DASH_FORWARD = 'POST_DASH_FORWARD'
    POST_DASH_TURN_1 = 'POST_DASH_TURN_1'
    POST_TURN_FORWARD = 'POST_TURN_FORWARD'
    POST_DASH_TURN_2 = 'POST_DASH_TURN_2'

    # 第二次转向后：复用限高杆代码里的目标检测、对齐、撞击逻辑
    SEARCH_TARGET_AFTER_TURNS = 'SEARCH_TARGET_AFTER_TURNS'
    APPROACH_AND_ALIGN_TARGET = 'APPROACH_AND_ALIGN_TARGET'
    HIT_TARGET = 'HIT_TARGET'

    # 撞击完成后的后续动作：后退 -> 两次左跳 -> 前进识别障碍物并按虚线侧选择单个障碍物对齐
    HIT_BACKOFF_AFTER_HIT = 'HIT_BACKOFF_AFTER_HIT'
    POST_HIT_LEFT_JUMP = 'POST_HIT_LEFT_JUMP'
    APPROACH_SELECTED_OBSTACLE_AFTER_HIT = 'APPROACH_SELECTED_OBSTACLE_AFTER_HIT'

    # 新增：撞击后对齐障碍物到达指定距离后的后续动作
    # 逻辑：按当前对齐的障碍物侧转向 -> 前进一段 -> 反向转回 -> 最后向前走一段
    POST_HIT_OBS_TURN_1 = 'POST_HIT_OBS_TURN_1'
    POST_HIT_OBS_FORWARD = 'POST_HIT_OBS_FORWARD'
    POST_HIT_OBS_TURN_2 = 'POST_HIT_OBS_TURN_2'

    # 新增：第二次转回后，先不识别黄线，按 TF 向前走一段固定距离
    POST_HIT_PRE_FINAL_FORWARD = 'POST_HIT_PRE_FINAL_FORWARD'

    # 前进固定距离后，再进入最终横向黄线识别和朝向修正
    POST_HIT_FINAL_FORWARD = 'POST_HIT_FINAL_FORWARD'

    # 新增：最终对准横向黄线后，再执行两次左跳/等效 180 度掉头动作。
    FINAL_LEFT_JUMP = 'FINAL_LEFT_JUMP'

    # 新增：障碍物流程最终黄线 + 180 度掉头完成后，再恢复 normal 姿态。
    # 注意：不再在 POST_HIT_OBS_TURN_2 后恢复 normal。
    OBSTACLE_RESTORE_NORMAL_AFTER_FINAL_TURN = 'OBSTACLE_RESTORE_NORMAL_AFTER_FINAL_TURN'

    # 全部 2 次限高杆 + 1 次障碍物完成后的最终收尾动作：
    # 右跳一次 -> 前进识别前方横向黄线并矫正朝向 -> 黄线到达图像下方阈值 -> 左跳一次 -> DONE
    GLOBAL_FINAL_RIGHT_JUMP = 'GLOBAL_FINAL_RIGHT_JUMP'
    GLOBAL_FINAL_YELLOW_FORWARD = 'GLOBAL_FINAL_YELLOW_FORWARD'
    GLOBAL_FINAL_LEFT_JUMP = 'GLOBAL_FINAL_LEFT_JUMP'
    GLOBAL_FINAL_RIGHT_SHIFT_AFTER_LEFT_JUMP = 'GLOBAL_FINAL_RIGHT_SHIFT_AFTER_LEFT_JUMP'

    # 第四赛段结束后的最终视觉矫正：复用第三赛段结束时的 P3_ALIGN_TRACK 逻辑。
    # 位置与第三赛段结束位置一致，所以同样用 p3_process_yellow_track 得到 p3_s4_lat / p3_s4_yaw。
    GLOBAL_FINAL_P3_ALIGN = 'GLOBAL_FINAL_P3_ALIGN'

    DONE = 'DONE'

    def fourth_stage_init(self):
        # Fourth stage is merged into this same ROS2 node.
        # Reuse the existing bridge, TF listener, subscriptions, Robot_Ctrl and timer.
        self.yaml_node = self
        self.declare_parameter('voice_enabled', True)
        self.declare_parameter('voice_dir', '/home/cyberdog_sim/voice')

        # ============================================================
        # 调试用：指定程序启动后的初始状态
        # 默认从 GLOBAL_INITIAL_LATERAL_SHIFT 开始完整第四赛段流程。
        #
        # 注意：
        # 1. 现在大多数“固定走一段 / 转一段”的动作已经改成按仿真时间 duration_s 判断，
        #    不再依赖 TF 距离或 TF yaw。
        # 2. TF 仍然保留在代码中作为兼容/调试，但主动作段应尽量以仿真时间为准。
        # 3. 如果单独从中间状态启动，某些状态依赖之前保存的变量，
        #    例如 dashed_side、locked_target、selected_obstacle_after_hit_side，
        #    这些变量可能为空，需要配合 debug 参数或手动初始化。
        #
        # 可选状态及含义：
        #
        # ------------------------------------------------------------
        # 一、全局任务调度状态
        # ------------------------------------------------------------
        #
        # GLOBAL_INITIAL_LATERAL_SHIFT:
        #   第四赛段启动后的预横移状态。
        #   先按 global_initial_lateral_shift_vy 横向移动
        #   global_initial_lateral_shift_duration_s 秒。
        #   这段时间不检测限高杆和障碍物，目的是避免一启动就在原地误触发。
        #   完成后进入 GLOBAL_LATERAL_SEARCH。
        #
        # GLOBAL_LATERAL_SEARCH:
        #   全局搜索状态。
        #   机器狗一边按 global_lateral_search_vy 横向移动，一边根据任务完成情况选择性检测目标：
        #     - completed_bar_count < required_bar_count 时，检测限高杆；
        #     - completed_obstacle_count < required_obstacle_count 时，检测蓝色障碍物；
        #     - 已完成的任务类型不再检测，避免重复触发。
        #   如果检测到限高杆，进入 GLOBAL_CENTER_BAR。
        #   如果检测到蓝色障碍物，进入 GLOBAL_CENTER_OBSTACLE。
        #   如果所有任务已经完成，则进入 GLOBAL_FINAL_RIGHT_JUMP，开始全局最终收尾。
        #
        # GLOBAL_CENTER_BAR:
        #   全局搜索中识别到限高杆后，先横向居中限高杆。
        #   根据限高杆中心和图像中心的误差控制 vy。
        #   连续稳定 global_center_stable_frames 帧后，进入 BAR_FORWARD_UNDER。
        #
        # GLOBAL_CENTER_OBSTACLE:
        #   全局搜索中识别到蓝色障碍物后，先横向居中障碍物。
        #   根据障碍物中心和图像中心的误差控制 vy。
        #   连续稳定 global_center_stable_frames 帧后，进入 APPROACH_OBSTACLES。
        #
        # GLOBAL_SHIFT_AFTER_SUBTASK:
        #   完成一次子任务后的全局横移状态。
        #   如果限高杆/障碍物任务还没有全部完成，就继续横移一段时间后回到 GLOBAL_LATERAL_SEARCH。
        #   横移时间通常使用：
        #     - global_after_task_shift_duration_s
        #   如果刚完成的是障碍物流程，会根据 dashed_side 选择不同横移时间：
        #     - global_after_obstacle_shift_duration_left_dash_s
        #     - global_after_obstacle_shift_duration_right_dash_s
        #   如果全部任务已经完成，则不再继续搜索，而是进入 GLOBAL_FINAL_RIGHT_JUMP。
        #
        #
        # ------------------------------------------------------------
        # 二、限高杆流程
        # ------------------------------------------------------------
        #
        # BAR_FORWARD_UNDER:
        #   限高杆居中后，机器狗向前低身通过限高杆。
        #   主要根据限高杆深度/触发距离判断是否进入后续目标搜索。
        #   到达 bar_trigger_distance_m 或满足对应条件后，进入 BAR_SEARCH_TARGET。
        #
        # BAR_SEARCH_TARGET:
        #   通过限高杆后开始搜索目标物体。
        #   目标包括：
        #     - blue_ball
        #     - white_ball
        #     - cola
        #   检测到目标后锁定 locked_target，并播放对应语音，然后进入 BAR_APPROACH_TARGET。
        #
        # BAR_APPROACH_TARGET:
        #   锁定目标后，边向前走边根据目标中心做横向对齐。
        #   使用目标中心 x 与图像中心误差控制 vy。
        #   当目标距离小于 hit_trigger_distance_m 且对齐稳定后，进入 BAR_HIT_TARGET。
        #
        # BAR_HIT_TARGET:
        #   按目标类型执行撞击动作。
        #   不同目标可以有不同撞击速度和持续时间：
        #     - hit_blue_ball_speed / hit_blue_ball_duration_s
        #     - hit_white_ball_speed / hit_white_ball_duration_s
        #     - hit_cola_speed / hit_cola_duration_s
        #   撞击完成后进入 BAR_BACKOFF_TO_BAR。
        #
        # BAR_BACKOFF_TO_BAR:
        #   撞击目标后后退，回到限高杆附近或达到后退条件。
        #   现在一般按深度/时间混合条件判断，避免卡死。
        #   完成后认为一次限高杆流程结束，增加 completed_bar_count。
        #   然后进入 GLOBAL_SHIFT_AFTER_SUBTASK。
        #
        #
        # ------------------------------------------------------------
        # 三、障碍物 + 黄色虚线流程
        # ------------------------------------------------------------
        #
        # APPROACH_OBSTACLES:
        #   进入障碍物流程后，向前移动并识别两个蓝色障碍物。
        #   同时根据两个障碍物的中心进行横向对齐。
        #   当障碍物距离小于 obstacle_trigger_distance_m 后，开始寻找黄色竖直虚线。
        #   首次识别到虚线时记录 dashed_side：
        #     - dashed_side == 'left'  表示虚线在左边；
        #     - dashed_side == 'right' 表示虚线在右边。
        #   然后进入 DASH_PRE_SIDE_SHIFT。
        #
        # DASH_PRE_SIDE_SHIFT:
        #   第一次识别到黄色竖直虚线后，不马上开始精对齐，
        #   而是先朝虚线所在方向横移 dashed_pre_shift_duration_s 秒。
        #   横移速度由 dashed_pre_shift_speed 控制。
        #   这样可以让机器狗更接近后续对齐位置。
        #   单独从该状态启动时，需要通过 debug_dashed_side 指定虚线方向。
        #
        # ALIGN_DASHED_LINE:
        #   对黄色竖直虚线做偏置对齐。
        #   不是把虚线对到图像正中心，而是根据 dashed_target_offset_px 做偏置：
        #     - 虚线在左边：目标点 = 图像中心 + offset，让虚线位于中间偏右；
        #     - 虚线在右边：目标点 = 图像中心 - offset，让虚线位于中间偏左。
        #   对齐稳定 dashed_center_stable_frames 帧后，进入 FOLLOW_DASHED_UNTIL_LOST。
        #
        # FOLLOW_DASHED_UNTIL_LOST:
        #   沿黄色竖直虚线继续向前走。
        #   行走过程中持续检测虚线，并用较小 vy 进行横向修正。
        #   当虚线连续丢失 dashed_lost_stop_frames 帧后，认为已经通过虚线区域，
        #   进入 POST_DASH_FORWARD。
        #
        # POST_DASH_FORWARD:
        #   虚线消失后继续向前走 post_dash_forward_duration_s 秒。
        #   速度由 post_dash_forward_speed 控制。
        #   完成后进入 POST_DASH_TURN_1。
        #
        # POST_DASH_TURN_1:
        #   虚线消失后的第一次转向，按仿真时间执行。
        #   转向持续 post_dash_turn_duration_s 秒，角速度为 post_dash_turn_wz。
        #   转向方向由 dashed_side 决定：
        #     - dashed_side == 'left'  时右转；
        #     - dashed_side == 'right' 时左转。
        #   完成后进入 POST_TURN_FORWARD。
        #
        # POST_TURN_FORWARD:
        #   第一次转向后继续向前走 post_turn_forward_duration_s 秒。
        #   这段前进分成两段：
        #     - 前 post_turn_forward_fast_duration_s 秒使用 post_turn_forward_fast_speed 快速前进；
        #     - 剩余时间使用 post_turn_forward_slow_speed 慢速前进，
        #       让第二次转向前的姿态和位置更稳定。
        #   完成后进入 POST_DASH_TURN_2。
        #
        # POST_DASH_TURN_2:
        #   第二次转向，方向与 POST_DASH_TURN_1 相反。
        #   转向持续 post_second_turn_duration_s 秒，角速度为 post_second_turn_wz。
        #   完成后进入 SEARCH_TARGET_AFTER_TURNS。
        #
        #
        # ------------------------------------------------------------
        # 四、虚线后目标识别、撞击、回退、再绕障碍物
        # ------------------------------------------------------------
        #
        # SEARCH_TARGET_AFTER_TURNS:
        #   第二次转向后开始搜索目标物体：
        #     - blue_ball
        #     - white_ball
        #     - cola
        #   检测到目标后锁定 locked_target，并播放语音。
        #   然后进入 APPROACH_AND_ALIGN_TARGET。
        #
        # APPROACH_AND_ALIGN_TARGET:
        #   锁定目标后，边向前走边根据目标中心做横向对齐。
        #   当目标距离小于 hit_trigger_distance_m 且连续稳定 target_stable_frames 帧后，
        #   进入 HIT_TARGET。
        #
        # HIT_TARGET:
        #   根据 locked_target 类型执行对应撞击动作。
        #   撞击速度和持续时间由目标类型决定：
        #     - blue_ball 使用 hit_blue_ball_speed / hit_blue_ball_duration_s；
        #     - white_ball 使用 hit_white_ball_speed / hit_white_ball_duration_s；
        #     - cola 使用 hit_cola_speed / hit_cola_duration_s。
        #   撞击完成后进入 HIT_BACKOFF_AFTER_HIT。
        #
        # HIT_BACKOFF_AFTER_HIT:
        #   撞击完成后，按 after_hit_backoff_speed 后退
        #   after_hit_backoff_duration_s 秒。
        #   完成后进入 POST_HIT_LEFT_JUMP。
        #
        # POST_HIT_LEFT_JUMP:
        #   后退完成后执行 after_hit_left_jump_count 次原地左跳。
        #   通常是两次左跳，用于调整朝向。
        #   完成后进入 APPROACH_SELECTED_OBSTACLE_AFTER_HIT。
        #
        # APPROACH_SELECTED_OBSTACLE_AFTER_HIT:
        #   左跳完成后继续向前走，并重新识别蓝色障碍物。
        #   根据之前记录的 dashed_side 选择一个障碍物进行对齐：
        #     - dashed_side == 'left'  时，对齐右边障碍物；
        #     - dashed_side == 'right' 时，对齐左边障碍物。
        #   当选中障碍物距离小于 post_hit_obstacle_trigger_distance_m 后，
        #   进入 POST_HIT_OBS_TURN_1。
        #
        # POST_HIT_OBS_TURN_1:
        #   对齐选中障碍物后进行第一次转向，按仿真时间执行。
        #   如果当前对齐的是左边障碍物，则左转；
        #   如果当前对齐的是右边障碍物，则右转。
        #   持续 post_hit_obs_turn_duration_s 秒，角速度为 post_hit_obs_turn_wz。
        #   完成后进入 POST_HIT_OBS_FORWARD。
        #
        # POST_HIT_OBS_FORWARD:
        #   第一次转向完成后向前走 post_hit_obs_forward_duration_s 秒。
        #   速度由 post_hit_obs_forward_speed 控制。
        #   完成后进入 POST_HIT_OBS_TURN_2。
        #
        # POST_HIT_OBS_TURN_2:
        #   第二次转向，方向与 POST_HIT_OBS_TURN_1 相反。
        #   持续 post_hit_obs_turn_duration_s 秒。
        #   完成后进入 POST_HIT_PRE_FINAL_FORWARD。
        #
        # POST_HIT_PRE_FINAL_FORWARD:
        #   第二次转回后，先不识别横向黄线，只按仿真时间向前走一小段。
        #   持续 post_hit_final_forward_duration_s 秒，
        #   速度由 post_hit_final_forward_speed 控制。
        #   完成后进入 POST_HIT_FINAL_FORWARD。
        #
        # POST_HIT_FINAL_FORWARD:
        #   障碍物流程内部的横向黄线收尾状态。
        #   开始边前进边识别前方横向黄线，并根据黄线 angle_deg 修正 wz，
        #   让机器狗尽量正对黄线。
        #   当黄线底部 bottom_ratio 到达 final_yellow_stop_line_y_ratio，
        #   且角度误差小于 final_yellow_done_tilt_deg 后，
        #   进入 FINAL_LEFT_JUMP。
        #   注意：这个状态是障碍物流程内部收尾，不是全局最终收尾。
        #
        # FINAL_LEFT_JUMP:
        #   障碍物流程内部黄线收尾完成后，执行原地左跳。
        #   当前逻辑通常执行两次左跳。
        #   完成后进入 OBSTACLE_FLOW_DONE，而不是直接 DONE。
        #
        # OBSTACLE_FLOW_DONE:
        #   障碍物流程真正完成的计数状态。
        #   在这里 completed_obstacle_count 加 1。
        #   然后根据总任务是否完成决定：
        #     - 如果限高杆和障碍物任务都完成，进入 GLOBAL_FINAL_RIGHT_JUMP；
        #     - 否则进入 GLOBAL_SHIFT_AFTER_SUBTASK，继续搜索剩余任务。
        #
        #
        # ------------------------------------------------------------
        # 五、全局最终收尾流程
        # ------------------------------------------------------------
        #
        # GLOBAL_FINAL_RIGHT_JUMP:
        #   当 required_bar_count 次限高杆流程和 required_obstacle_count 次障碍物流程全部完成后，
        #   进入全局最终收尾。
        #   该状态先执行一次右跳，用于调整进入最终出口的方向。
        #   完成后进入 GLOBAL_FINAL_YELLOW_FORWARD。
        #
        # GLOBAL_FINAL_YELLOW_FORWARD:
        #   全局最后的前方横向黄线识别和对正状态。
        #   机器狗按 global_final_yellow_forward_speed 向前走，
        #   同时识别前方横向黄线，并根据黄线 angle_deg 修正 wz。
        #   当黄线底部 bottom_ratio 到达 global_final_yellow_stop_line_y_ratio 后，
        #   不立刻停止，而是继续向前走，等待黄线从图像中消失。
        #   当黄线连续消失 global_final_yellow_disappear_confirm_count 帧后，
        #   认为机器狗已经越过最终黄线，进入 GLOBAL_FINAL_LEFT_JUMP。
        #
        # GLOBAL_FINAL_LEFT_JUMP:
        #   全局最后的左跳状态。
        #   执行一次左跳后进入 DONE。
        #
        # DONE:
        #   第四赛段全部任务结束。
        #   持续发送 STOP。
        # ============================================================
        self.declare_parameter('p4_initial_state', 'GLOBAL_INITIAL_LATERAL_SHIFT')
        # self.declare_parameter('p4_initial_state', 'GLOBAL_FINAL_YELLOW_FORWARD')


        # 调试用：当 initial_state 需要依赖 dashed_side 时，可手动指定。
        # 可选值：'auto' / 'left' / 'right'
        # auto 表示正常流程里由视觉第一次看到虚线时自动记录。
        self.declare_parameter('debug_dashed_side', 'auto')

        # ============================================================
        # 全局整合流程参数
        # required_bar_count=2：限高杆流程需要完成两次
        # required_obstacle_count=1：障碍物流程只需要完成一次；完成后后续不再触发障碍物流程
        # ============================================================
        self.declare_parameter('required_bar_count', 2)
        self.declare_parameter('required_obstacle_count', 1)

        # 程序开始先固定向左移动一段仿真时间，不检测限高杆/障碍物，避免一启动就在原地误触发。
        self.declare_parameter('global_initial_lateral_shift_duration_s', 1.5)
        self.declare_parameter('global_initial_lateral_shift_vy', 0.30)

        # 启动预左移完成后，进入全局搜索；搜索阶段会按完成计数决定是否检测限高杆/障碍物。
        self.declare_parameter('global_lateral_search_vy', 0.30)
        self.declare_parameter('global_center_stable_frames', 1)
        self.declare_parameter('global_after_task_shift_vy', 0.30)
        self.declare_parameter('global_after_task_shift_duration_s', 1.5)
        # 障碍物流程完成后，如果虚线在右边，左移持续时间更长
        self.declare_parameter('global_after_obstacle_shift_duration_left_dash_s', 0.1)
        self.declare_parameter('global_after_obstacle_shift_duration_right_dash_s', 3.0)

        # 完成一次限高杆流程后，下一轮全局搜索只在图像左半边寻找目标。
        # 这样可以避免刚完成的限高杆仍出现在右半边时被重复选中。
        self.declare_parameter('search_left_half_after_bar_done', True)
        self.declare_parameter('after_bar_search_x_ratio_max', 0.50)
        # 完成障碍物流程后，下一轮全局搜索也只在图像左半边寻找目标。
        # 这样可以避免刚完成的障碍物/绕行区域仍出现在右侧时，影响下一个目标选择。
        self.declare_parameter('search_left_half_after_obstacle_done', True)
        self.declare_parameter('after_obstacle_search_x_ratio_max', 0.70)

        # 限高杆流程参数
        self.declare_parameter('global_bar_center_fixed_vy', 0.20)
        self.declare_parameter('global_bar_center_px_deadband', 7)
        self.declare_parameter('bar_search_forward_speed', 1.0)
        self.declare_parameter('bar_trigger_distance_m', 0.50)
        self.declare_parameter('bar_align_vy_k', 0.35)
        self.declare_parameter('bar_align_vy_max', 0.30)
        self.declare_parameter('bar_align_vy_min', 0.10)
        self.declare_parameter('bar_center_px_deadband', 7)
        self.declare_parameter('bar_center_stable_frames', 3)
        # 限高杆朝向矫正：居中和穿杆时，根据限高杆左右两侧深度差给一个小 wz。
        # left_depth / right_depth 差值越明显，说明机器狗相对限高杆越斜。
        self.declare_parameter('bar_depth_yaw_align_enabled', True)
        self.declare_parameter('bar_depth_yaw_fixed_wz', 0.12)
        self.declare_parameter('bar_depth_yaw_deadband_m', 0.03)
        self.declare_parameter('bar_depth_yaw_sample_x_ratio', 0.30)
        self.declare_parameter('bar_depth_yaw_sample_y_ratio', 0.10)
        self.declare_parameter('bar_depth_yaw_sample_half_size', 4)
        # 如果实测发现越修越歪，把这个参数改成 -1。
        self.declare_parameter('bar_depth_yaw_sign', -1.0)
        self.declare_parameter('backoff_after_hit_speed', 1.0)
        self.declare_parameter('backoff_bar_depth_tolerance_m', 0.05)
        self.declare_parameter('backoff_min_time_s', 0.30)
        # 回退阶段看到限高杆时，用限高杆深度误差闭环修正 vx。
        # d < target_depth 时继续后退；d > target_depth 时允许小幅向前修回来。
        self.declare_parameter('bar_backoff_depth_vx_align_enabled', True)
        self.declare_parameter('bar_backoff_depth_vx_k', 0.60)
        self.declare_parameter('bar_backoff_depth_vx_max', 1.0)
        self.declare_parameter('bar_backoff_depth_vx_min', 0.50)
        self.declare_parameter('bar_backoff_allow_forward_correction', True)

        # =========================
        # 第四赛段机身高度策略
        # =========================
        # 默认第四赛段搜索/全局移动保持 normal；
        # 限高杆到达 bar_trigger_distance_m 触发播报时再切 low；
        # 障碍物流程在 GLOBAL_CENTER_OBSTACLE 对齐完成后切 low；
        # 不是在 POST_HIT_OBS_TURN_2 后恢复 normal，
        # 而是在后续前进识别横向黄线，并完成 FINAL_LEFT_JUMP/等效 180 度掉头后恢复 normal。
        self.declare_parameter('p4_normal_body_height', 0.25)
        self.declare_parameter('p4_low_body_height', 0.17)
        self.declare_parameter('bar_body_low_enabled', True)
        self.declare_parameter('bar_body_low_do_stop', True)
        self.declare_parameter('restore_normal_after_bar_flow', True)
        self.declare_parameter('obstacle_flow_low_enabled', True)
        self.declare_parameter('obstacle_body_low_do_stop', True)
        self.declare_parameter('obstacle_restore_normal_after_final_turn', True)
        self.declare_parameter('p4_force_refresh_body_at_start', True)

        # 运动参数
        self.declare_parameter('obstacle_forward_speed', 0.40)
        self.declare_parameter('obstacle_search_forward_speed', 0.40)
        self.declare_parameter('obstacle_trigger_distance_m', 0.30)

        self.declare_parameter('obstacle_align_vy_k', 0.35)
        self.declare_parameter('obstacle_align_vy_max', 0.30)
        self.declare_parameter('obstacle_align_vy_min', 0.20)
        self.declare_parameter('obstacle_center_px_deadband', 7)

        self.declare_parameter('dashed_align_vy_k', 0.35)
        self.declare_parameter('dashed_align_vy_max', 0.30)
        self.declare_parameter('dashed_align_vy_min', 0.20)
        self.declare_parameter('dashed_center_px_deadband', 7)
        self.declare_parameter('dashed_center_stable_frames', 1)

        # 黄线开始对齐前，先朝虚线所在方向横移一小段仿真时间。
        self.declare_parameter('dashed_pre_shift_speed', 0.2)
        self.declare_parameter('dashed_pre_shift_duration_s', 1.0)

        # 偏置对齐目标，单位：像素
        # 左边有虚线：目标点 = 图像中心 + offset，也就是中间偏右
        # 右边有虚线：目标点 = 图像中心 - offset，也就是中间偏左
        self.declare_parameter('dashed_target_offset_px', 30)

        self.declare_parameter('follow_forward_speed', 0.50)
        self.declare_parameter('follow_align_vy_k', 0.18)
        # FOLLOW_DASHED_UNTIL_LOST 阶段单独使用的横向速度上下限，
        # 不再和 ALIGN_DASHED_LINE 共用 dashed_align_vy_min / dashed_align_vy_max。
        self.declare_parameter('follow_align_vy_max', 0.15)
        self.declare_parameter('follow_align_vy_min', 0.05)
        self.declare_parameter('dashed_lost_stop_frames', 2)

        # 沿虚线前进阶段的有效识别范围。
        # 只有虚线中心 x 落在 get_dashed_target_x() 附近这个范围内，才认为虚线仍然有效。
        # 超过范围，即使视觉检测到了虚线，也按虚线消失处理。
        self.declare_parameter('follow_dashed_valid_x_range_px',50)

        # =========================
        # 虚线消失后的后续任务参数
        # =========================
        self.declare_parameter('tf_parent_frame', 'vodom')
        self.declare_parameter('tf_child_frame', 'base_link')

        # 虚线识别不到后，继续向前走一小段仿真时间
        self.declare_parameter('post_dash_forward_duration_s', 0.8)
        self.declare_parameter('post_dash_forward_speed', 0.20)

        # 第一次转向持续时间。左边虚线 -> 右转；右边虚线 -> 左转
        self.declare_parameter('post_dash_turn_duration_s', 1.5)
        self.declare_parameter('post_dash_turn_wz', 0.50)
        self.declare_parameter('post_dash_turn_tolerance_deg', 1.5)

        # 第一次转向完成后，继续前进一段仿真时间。
        # 这段前进分成两段：先快走，再慢走。
        self.declare_parameter('post_turn_forward_duration_s', 2.0)
        self.declare_parameter('post_turn_forward_fast_duration_s', 1.6)
        self.declare_parameter('post_turn_forward_fast_speed', 0.60)
        self.declare_parameter('post_turn_forward_slow_speed', 0.20)

        # 第二次转向持续时间，方向与第一次相反
        self.declare_parameter('post_second_turn_duration_s', 1.5)
        self.declare_parameter('post_second_turn_wz', 0.50)
        self.declare_parameter('post_second_turn_tolerance_deg', 1.5)

        # =========================
        # 第二次转向后的目标检测 / 对齐 / 撞击参数
        # 复用限高杆任务代码里的目标检测逻辑：蓝球、白球、可乐
        # =========================
        self.declare_parameter('target_search_forward_speed', 0.60)
        self.declare_parameter('align_forward_speed_far', 0.60)
        self.declare_parameter('align_forward_speed_near', 0.60)
        self.declare_parameter('align_vy_k', 0.35)
        self.declare_parameter('align_vy_max', 0.30)
        self.declare_parameter('align_vy_min', 0.15)
        self.declare_parameter('target_stable_frames', 3)
        self.declare_parameter('hit_trigger_distance_m', 0.20)
        self.declare_parameter('center_px_deadband', 7)

        self.declare_parameter('hit_blue_ball_speed', 0.50)
        self.declare_parameter('hit_blue_ball_duration_s', 0.35)
        self.declare_parameter('hit_white_ball_speed', 0.50)
        self.declare_parameter('hit_white_ball_duration_s', 0.75)
        self.declare_parameter('hit_cola_speed', 0.50)
        self.declare_parameter('hit_cola_duration_s', 0.35)
        self.declare_parameter('hit_timeout_s', 10.0)

        # 撞击完成后：先后退一小段仿真时间，再按固定角速度转固定时间代替原来的连续左跳
        self.declare_parameter('after_hit_backoff_duration_s', 1.0)
        self.declare_parameter('after_hit_backoff_speed', 0.40)
        self.declare_parameter('after_hit_left_jump_count', 2)

        # 用固定角速度 + 固定仿真时间代替原来的旋转跳。
        self.declare_parameter('p4_timed_turn_wz_90', 0.60)
        self.declare_parameter('p4_timed_turn_duration_90_s', 3.25)
        self.declare_parameter('p4_timed_turn_wz_180', 0.60)
        self.declare_parameter('p4_timed_turn_duration_180_s', 6.6)

        # 左跳完成后：前进并识别两个蓝色障碍物，按虚线侧选择其中一个居中
        # 之前虚线在左边 -> 对齐右边障碍物；之前虚线在右边 -> 对齐左边障碍物
        self.declare_parameter('post_hit_obstacle_forward_speed', 0.50)
        self.declare_parameter('post_hit_obstacle_search_forward_speed', 0.50)
        self.declare_parameter('post_hit_obstacle_trigger_distance_m', 0.25)
        self.declare_parameter('post_hit_obstacle_align_vy_k', 0.35)
        self.declare_parameter('post_hit_obstacle_align_vy_max', 0.30)
        self.declare_parameter('post_hit_obstacle_align_vy_min', 0.15)
        self.declare_parameter('post_hit_obstacle_center_px_deadband', 7)

        # 对齐选中障碍物并到达距离后：转向 -> 前进 -> 反向转向 -> 最后前进
        # 如果对齐左边障碍物：第一次左转；如果对齐右边障碍物：第一次右转
        self.declare_parameter('post_hit_obs_turn_duration_s', 1.75)
        self.declare_parameter('post_hit_obs_turn_wz', 0.50)
        self.declare_parameter('post_hit_obs_turn_tolerance_deg', 1.5)
        self.declare_parameter('post_hit_obs_forward_duration_s', 1.6)
        self.declare_parameter('post_hit_obs_forward_speed', 0.40)
        # 第二次转回后，先按仿真时间向前走一段，再进入最终横向黄线识别对正
        self.declare_parameter('post_hit_final_forward_duration_s',0.50)
        self.declare_parameter('post_hit_final_forward_speed', 0.40)
        # 绕过障碍物第二次转回后的预前进阶段：如果提前看到前方横向黄线，
        # 只用它的角度修正 wz，不用它提前结束该状态。
        self.declare_parameter('post_hit_pre_final_angle_align_enabled', True)

        # 最后前进阶段：前方横向黄线检测 + 朝向修正
        self.declare_parameter('final_yellow_stop_line_y_ratio', 0.95)
        self.declare_parameter('final_yellow_align_wz_k', 1.20)
        self.declare_parameter('final_yellow_align_wz_max', 0.30)
        self.declare_parameter('final_yellow_align_wz_min', 0.20)
        self.declare_parameter('final_yellow_tilt_deadband_deg', 0.5)
        self.declare_parameter('final_yellow_done_tilt_deg', 1.0)
        self.declare_parameter('final_yellow_confirm_count', 1)
        # 黄线到达图像下方后，不立刻停；继续前进，等横向黄线从画面中消失后再结束。
        self.declare_parameter('final_yellow_disappear_confirm_count', 2)

        # 全局最终收尾阶段：所有流程完成后，右跳 -> 前进识别横向黄线并矫正 -> 左跳一次。
        # 没接近最终横向黄线前，用这个速度快速前进
        self.declare_parameter('global_final_yellow_forward_speed', 0.80)
        # 横向黄线接近后，切换成这个慢速
        self.declare_parameter('global_final_yellow_slow_forward_speed', 0.40)
        # 当横向黄线底部达到图像高度的这个比例后，开始减速
        self.declare_parameter('global_final_yellow_slow_start_ratio', 0.90)
        # 当横向黄线底部达到这个比例后，认为已经到达下方区域
        # 到达后不马上左跳，而是继续前进，等黄线从画面中消失
        self.declare_parameter('global_final_yellow_stop_line_y_ratio', 1.0)
        # 黄线到达下方区域需要连续确认几帧
        self.declare_parameter('global_final_yellow_confirm_count', 1)
        # 黄线到达下方区域后，连续消失几帧才进入最终左跳
        self.declare_parameter('global_final_yellow_disappear_confirm_count', 2)

        self.declare_parameter('global_final_after_left_jump_right_shift_vy', -0.20)
        self.declare_parameter('global_final_after_left_jump_right_shift_duration_s', 0.2)

        # 限高杆参数
        self._declare_bar_params()

        # 蓝色障碍物参数
        self._declare_obstacle_params()

        # 黄色虚线参数
        self._declare_yellow_params()

        # 最后阶段前方横向黄线参数
        self._declare_final_yellow_params()

        # 第二次转向后的目标检测参数
        self._declare_ball_params('blue_ball', defaults={
            'h_min': 90, 'h_max': 135, 's_min': 80, 's_max': 255, 'v_min': 40, 'v_max': 255,
            'roi_x_ratio_min': 0.20, 'roi_x_ratio_max': 0.80,
            'roi_y_ratio_min': 0.15, 'roi_y_ratio_max': 0.95,
            'open_kernel': 3, 'close_kernel': 5,
            'min_area': 80, 'max_area': 5000000,
            'min_radius': 5.0, 'max_radius': 200.0,
            'min_circularity': 0.50,
            'min_wh_ratio': 0.50, 'max_wh_ratio': 2.0,
            'max_center_y_ratio_in_roi': 1.0,
            'center_weight_base': 0.3, 'center_weight_gain': 0.7,
            'radius_score_gain': 10.0,
        })
        self._declare_ball_params('white_ball', defaults={
            'h_min': 0, 'h_max': 20, 's_min': 0, 's_max': 20, 'v_min': 95, 'v_max': 255,
            'roi_x_ratio_min': 0.20, 'roi_x_ratio_max': 0.80,
            'roi_y_ratio_min': 0.15, 'roi_y_ratio_max': 0.95,
            'open_kernel': 3, 'close_kernel': 5,
            'min_area': 80, 'max_area': 50000,
            'min_radius': 10.0, 'max_radius': 150.0,
            'min_circularity': 0.55,
            'min_wh_ratio': 0.60, 'max_wh_ratio': 1.40,
            'max_center_y_ratio_in_roi': 1.0,
            'center_weight_base': 0.3, 'center_weight_gain': 0.7,
            'radius_score_gain': 10.0,
        })
        self._declare_cola_params()

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.control_hz = float(self.get_parameter('control_hz').value)
        self.show_debug_vis = bool(self.get_parameter('show_debug_vis').value)
        self.voice_enabled = bool(self.get_parameter('voice_enabled').value)
        self.voice_dir = str(self.get_parameter('voice_dir').value)
        self.p4_initial_state = str(self.get_parameter('p4_initial_state').value)
        self.debug_dashed_side = str(self.get_parameter('debug_dashed_side').value).lower().strip()

        self.required_bar_count = int(self.get_parameter('required_bar_count').value)
        self.required_obstacle_count = int(self.get_parameter('required_obstacle_count').value)
        self.global_initial_lateral_shift_duration_s = float(
            self.get_parameter('global_initial_lateral_shift_duration_s').value)
        self.global_initial_lateral_shift_vy = float(self.get_parameter('global_initial_lateral_shift_vy').value)
        self.global_lateral_search_vy = float(self.get_parameter('global_lateral_search_vy').value)
        self.global_center_stable_frames = int(self.get_parameter('global_center_stable_frames').value)
        self.global_bar_center_fixed_vy = abs(float(self.get_parameter('global_bar_center_fixed_vy').value))
        self.global_bar_center_px_deadband = int(self.get_parameter('global_bar_center_px_deadband').value)
        self.global_after_task_shift_vy = float(self.get_parameter('global_after_task_shift_vy').value)
        self.global_after_task_shift_duration_s = float(self.get_parameter('global_after_task_shift_duration_s').value)
        self.global_after_obstacle_shift_duration_left_dash_s = float(
            self.get_parameter('global_after_obstacle_shift_duration_left_dash_s').value)
        self.global_after_obstacle_shift_duration_right_dash_s = float(
            self.get_parameter('global_after_obstacle_shift_duration_right_dash_s').value)
        self.search_left_half_after_bar_done = bool(self.get_parameter('search_left_half_after_bar_done').value)
        self.after_bar_search_x_ratio_max = float(self.get_parameter('after_bar_search_x_ratio_max').value)
        self.search_left_half_after_obstacle_done = bool(
            self.get_parameter('search_left_half_after_obstacle_done').value)
        self.after_obstacle_search_x_ratio_max = float(self.get_parameter('after_obstacle_search_x_ratio_max').value)

        self.bar_search_forward_speed = float(self.get_parameter('bar_search_forward_speed').value)
        self.bar_trigger_distance_m = float(self.get_parameter('bar_trigger_distance_m').value)
        self.bar_align_vy_k = float(self.get_parameter('bar_align_vy_k').value)
        self.bar_align_vy_max = float(self.get_parameter('bar_align_vy_max').value)
        self.bar_align_vy_min = float(self.get_parameter('bar_align_vy_min').value)
        self.bar_center_px_deadband = int(self.get_parameter('bar_center_px_deadband').value)
        self.bar_center_stable_frames = int(self.get_parameter('bar_center_stable_frames').value)
        self.bar_depth_yaw_align_enabled = bool(self.get_parameter('bar_depth_yaw_align_enabled').value)
        self.bar_depth_yaw_fixed_wz = abs(float(self.get_parameter('bar_depth_yaw_fixed_wz').value))
        self.bar_depth_yaw_deadband_m = float(self.get_parameter('bar_depth_yaw_deadband_m').value)
        self.bar_depth_yaw_sample_x_ratio = float(self.get_parameter('bar_depth_yaw_sample_x_ratio').value)
        self.bar_depth_yaw_sample_y_ratio = float(self.get_parameter('bar_depth_yaw_sample_y_ratio').value)
        self.bar_depth_yaw_sample_half_size = int(self.get_parameter('bar_depth_yaw_sample_half_size').value)
        self.bar_depth_yaw_sign = 1.0 if float(self.get_parameter('bar_depth_yaw_sign').value) >= 0.0 else -1.0
        self.latest_bar_depth_yaw_info = {
            'left_depth': None,
            'right_depth': None,
            'depth_error': None,
            'wz': 0.0,
        }
        self.backoff_after_hit_speed = abs(float(self.get_parameter('backoff_after_hit_speed').value))
        self.backoff_bar_depth_tolerance_m = float(self.get_parameter('backoff_bar_depth_tolerance_m').value)
        self.backoff_min_time_s = float(self.get_parameter('backoff_min_time_s').value)
        self.bar_backoff_depth_vx_align_enabled = bool(self.get_parameter('bar_backoff_depth_vx_align_enabled').value)
        self.bar_backoff_depth_vx_k = abs(float(self.get_parameter('bar_backoff_depth_vx_k').value))
        self.bar_backoff_depth_vx_max = abs(float(self.get_parameter('bar_backoff_depth_vx_max').value))
        self.bar_backoff_depth_vx_min = abs(float(self.get_parameter('bar_backoff_depth_vx_min').value))
        self.bar_backoff_allow_forward_correction = bool(
            self.get_parameter('bar_backoff_allow_forward_correction').value)

        self.p4_normal_body_height = float(self.get_parameter('p4_normal_body_height').value)
        self.p4_low_body_height = float(self.get_parameter('p4_low_body_height').value)
        self.bar_body_low_enabled = bool(self.get_parameter('bar_body_low_enabled').value)
        self.bar_body_low_do_stop = bool(self.get_parameter('bar_body_low_do_stop').value)
        self.restore_normal_after_bar_flow = bool(self.get_parameter('restore_normal_after_bar_flow').value)
        self.obstacle_flow_low_enabled = bool(self.get_parameter('obstacle_flow_low_enabled').value)
        self.obstacle_body_low_do_stop = bool(self.get_parameter('obstacle_body_low_do_stop').value)
        self.obstacle_restore_normal_after_final_turn = bool(
            self.get_parameter('obstacle_restore_normal_after_final_turn').value
        )
        self.p4_force_refresh_body_at_start = bool(
            self.get_parameter('p4_force_refresh_body_at_start').value
        )

        self.obstacle_forward_speed = float(self.get_parameter('obstacle_forward_speed').value)
        self.obstacle_search_forward_speed = float(self.get_parameter('obstacle_search_forward_speed').value)
        self.obstacle_trigger_distance_m = float(self.get_parameter('obstacle_trigger_distance_m').value)

        self.obstacle_align_vy_k = float(self.get_parameter('obstacle_align_vy_k').value)
        self.obstacle_align_vy_max = float(self.get_parameter('obstacle_align_vy_max').value)
        self.obstacle_align_vy_min = float(self.get_parameter('obstacle_align_vy_min').value)
        self.obstacle_center_px_deadband = int(self.get_parameter('obstacle_center_px_deadband').value)

        self.dashed_align_vy_k = float(self.get_parameter('dashed_align_vy_k').value)
        self.dashed_align_vy_max = float(self.get_parameter('dashed_align_vy_max').value)
        self.dashed_align_vy_min = float(self.get_parameter('dashed_align_vy_min').value)
        self.dashed_center_px_deadband = int(self.get_parameter('dashed_center_px_deadband').value)
        self.dashed_center_stable_frames = int(self.get_parameter('dashed_center_stable_frames').value)

        self.dashed_pre_shift_speed = float(self.get_parameter('dashed_pre_shift_speed').value)
        self.dashed_pre_shift_duration_s = float(self.get_parameter('dashed_pre_shift_duration_s').value)
        self.dashed_target_offset_px = int(self.get_parameter('dashed_target_offset_px').value)

        self.follow_forward_speed = float(self.get_parameter('follow_forward_speed').value)
        self.follow_align_vy_k = float(self.get_parameter('follow_align_vy_k').value)
        self.follow_align_vy_max = float(self.get_parameter('follow_align_vy_max').value)
        self.follow_align_vy_min = float(self.get_parameter('follow_align_vy_min').value)
        self.dashed_lost_stop_frames = int(self.get_parameter('dashed_lost_stop_frames').value)
        self.follow_dashed_valid_x_range_px = int(self.get_parameter('follow_dashed_valid_x_range_px').value)

        self.tf_parent_frame = str(self.get_parameter('tf_parent_frame').value)
        self.tf_child_frame = str(self.get_parameter('tf_child_frame').value)

        self.post_dash_forward_duration_s = float(self.get_parameter('post_dash_forward_duration_s').value)
        self.post_dash_forward_speed = float(self.get_parameter('post_dash_forward_speed').value)

        self.post_dash_turn_duration_s = float(self.get_parameter('post_dash_turn_duration_s').value)
        self.post_dash_turn_wz = float(self.get_parameter('post_dash_turn_wz').value)
        self.post_dash_turn_tolerance_rad = math.radians(
            float(self.get_parameter('post_dash_turn_tolerance_deg').value))

        self.post_turn_forward_duration_s = float(self.get_parameter('post_turn_forward_duration_s').value)
        self.post_turn_forward_fast_duration_s = float(self.get_parameter('post_turn_forward_fast_duration_s').value)
        self.post_turn_forward_fast_speed = float(self.get_parameter('post_turn_forward_fast_speed').value)
        self.post_turn_forward_slow_speed = float(self.get_parameter('post_turn_forward_slow_speed').value)

        self.post_second_turn_duration_s = float(self.get_parameter('post_second_turn_duration_s').value)
        self.post_second_turn_wz = float(self.get_parameter('post_second_turn_wz').value)
        self.post_second_turn_tolerance_rad = math.radians(
            float(self.get_parameter('post_second_turn_tolerance_deg').value))

        self.target_search_forward_speed = float(self.get_parameter('target_search_forward_speed').value)
        self.align_forward_speed_far = float(self.get_parameter('align_forward_speed_far').value)
        self.align_forward_speed_near = float(self.get_parameter('align_forward_speed_near').value)
        self.align_vy_k = float(self.get_parameter('align_vy_k').value)
        self.align_vy_max = float(self.get_parameter('align_vy_max').value)
        self.align_vy_min = float(self.get_parameter('align_vy_min').value)
        self.target_stable_frames = int(self.get_parameter('target_stable_frames').value)
        self.hit_trigger_distance_m = float(self.get_parameter('hit_trigger_distance_m').value)
        self.center_px_deadband = int(self.get_parameter('center_px_deadband').value)

        self.hit_timeout_s = float(self.get_parameter('hit_timeout_s').value)

        self.after_hit_backoff_duration_s = float(self.get_parameter('after_hit_backoff_duration_s').value)
        self.after_hit_backoff_speed = abs(float(self.get_parameter('after_hit_backoff_speed').value))
        self.after_hit_left_jump_count = int(self.get_parameter('after_hit_left_jump_count').value)
        self.p4_timed_turn_wz_90 = abs(float(self.get_parameter('p4_timed_turn_wz_90').value))
        self.p4_timed_turn_duration_90_s = float(self.get_parameter('p4_timed_turn_duration_90_s').value)
        self.p4_timed_turn_wz_180 = abs(float(self.get_parameter('p4_timed_turn_wz_180').value))
        self.p4_timed_turn_duration_180_s = float(self.get_parameter('p4_timed_turn_duration_180_s').value)

        self.post_hit_obstacle_forward_speed = float(self.get_parameter('post_hit_obstacle_forward_speed').value)
        self.post_hit_obstacle_search_forward_speed = float(
            self.get_parameter('post_hit_obstacle_search_forward_speed').value)
        self.post_hit_obstacle_trigger_distance_m = float(
            self.get_parameter('post_hit_obstacle_trigger_distance_m').value)
        self.post_hit_obstacle_align_vy_k = float(self.get_parameter('post_hit_obstacle_align_vy_k').value)
        self.post_hit_obstacle_align_vy_max = float(self.get_parameter('post_hit_obstacle_align_vy_max').value)
        self.post_hit_obstacle_align_vy_min = float(self.get_parameter('post_hit_obstacle_align_vy_min').value)
        self.post_hit_obstacle_center_px_deadband = int(
            self.get_parameter('post_hit_obstacle_center_px_deadband').value)

        self.post_hit_obs_turn_duration_s = float(self.get_parameter('post_hit_obs_turn_duration_s').value)
        self.post_hit_obs_turn_wz = float(self.get_parameter('post_hit_obs_turn_wz').value)
        self.post_hit_obs_turn_tolerance_rad = math.radians(
            float(self.get_parameter('post_hit_obs_turn_tolerance_deg').value))
        self.post_hit_obs_forward_duration_s = float(self.get_parameter('post_hit_obs_forward_duration_s').value)
        self.post_hit_obs_forward_speed = float(self.get_parameter('post_hit_obs_forward_speed').value)
        self.post_hit_final_forward_duration_s = float(self.get_parameter('post_hit_final_forward_duration_s').value)
        self.post_hit_final_forward_speed = float(self.get_parameter('post_hit_final_forward_speed').value)
        self.post_hit_pre_final_angle_align_enabled = bool(
            self.get_parameter('post_hit_pre_final_angle_align_enabled').value
        )

        self.final_yellow_stop_line_y_ratio = float(self.get_parameter('final_yellow_stop_line_y_ratio').value)
        self.final_yellow_align_wz_k = float(self.get_parameter('final_yellow_align_wz_k').value)
        self.final_yellow_align_wz_max = float(self.get_parameter('final_yellow_align_wz_max').value)
        self.final_yellow_align_wz_min = float(self.get_parameter('final_yellow_align_wz_min').value)
        self.final_yellow_tilt_deadband_deg = float(self.get_parameter('final_yellow_tilt_deadband_deg').value)
        self.final_yellow_done_tilt_deg = float(self.get_parameter('final_yellow_done_tilt_deg').value)
        self.final_yellow_confirm_count = int(self.get_parameter('final_yellow_confirm_count').value)
        self.final_yellow_disappear_confirm_count = int(
            self.get_parameter('final_yellow_disappear_confirm_count').value)

        self.global_final_yellow_forward_speed = float(
            self.get_parameter('global_final_yellow_forward_speed').value
        )

        self.global_final_yellow_slow_forward_speed = float(
            self.get_parameter('global_final_yellow_slow_forward_speed').value
        )

        self.global_final_yellow_slow_start_ratio = float(
            self.get_parameter('global_final_yellow_slow_start_ratio').value
        )

        self.global_final_yellow_stop_line_y_ratio = float(
            self.get_parameter('global_final_yellow_stop_line_y_ratio').value
        )

        self.global_final_yellow_confirm_count = int(
            self.get_parameter('global_final_yellow_confirm_count').value
        )

        self.global_final_yellow_disappear_confirm_count = int(
            self.get_parameter('global_final_yellow_disappear_confirm_count').value
        )

        self.global_final_after_left_jump_right_shift_vy = float(
            self.get_parameter('global_final_after_left_jump_right_shift_vy').value
        )

        self.global_final_after_left_jump_right_shift_duration_s = float(
            self.get_parameter('global_final_after_left_jump_right_shift_duration_s').value
        )

        self.hit_params = {
            'blue_ball': {
                'speed': float(self.get_parameter('hit_blue_ball_speed').value),
                'duration_s': float(self.get_parameter('hit_blue_ball_duration_s').value),
            },
            'white_ball': {
                'speed': float(self.get_parameter('hit_white_ball_speed').value),
                'duration_s': float(self.get_parameter('hit_white_ball_duration_s').value),
            },
            'cola': {
                'speed': float(self.get_parameter('hit_cola_speed').value),
                'duration_s': float(self.get_parameter('hit_cola_duration_s').value),
            },
        }

        self.bar_detector = BarColorDetector(self._read_bar_cfg())
        self.obstacle_detector = ObstacleBlueDepthDetector(self._read_obstacle_cfg())
        self.dashed_detector = YellowDashedLineDetector(self._read_yellow_cfg())
        self.final_yellow_detector = YellowHorizontalLineDetector(self._read_final_yellow_cfg())

        self.blue_ball_detector = BallDetector(self._read_ball_cfg('blue_ball'), 'blue_ball')
        self.white_ball_detector = BallDetector(self._read_ball_cfg('white_ball'), 'white_ball')
        self.cola_detector = ColaDetector(self._read_cola_cfg())

        # 语音播报：用事件 ID 防止同一个触发点重复播报。
        # bar_1 / bar_2 可以分别播报；同一个目标类型只在准备撞击时播一次。
        self.voice = VoicePlayer(self.voice_dir, enabled=self.voice_enabled)
        self.voice_events_spoken = set()

        self.motion_cmd = (0.0, 0.0, 0.0)
        self.body_height_cmd = self.p4_normal_body_height
        self.body_is_low = False
        # 第四赛段单独调试/程序重启时，控制器可能残留上一次 low 姿态。
        # 用 STOP -> LOW -> STOP -> NORMAL 强制刷新一次，确保初始是 normal。
        if getattr(self, 'p4_force_refresh_body_at_start', True):
            self.reset_body_pose_to_normal_at_start()

        # Fourth-stage state is entered explicitly after P3 finishes.
        # 状态计时起点延迟到 control_loop 第一次真正运行时再记录，
        # 避免节点初始化阶段 /clock 还没就绪导致 state_enter_time=0，
        # 从而让 GLOBAL_INITIAL_LATERAL_SHIFT 等固定时间状态被瞬间跳过。
        self.state_enter_time = None

        self.dashed_center_count = 0
        self.dashed_lost_count = 0

        # 第一次看到虚线时，记录它在图像左边还是右边。
        # 后续预横移和偏置对齐都会使用这个方向，不在 ALIGN 状态里清空。
        self.dashed_side = None  # None / 'left' / 'right'
        # DASH_PRE_SIDE_SHIFT 不再按时间结束，而是记录 TF 起点，按横向位移结束。
        self.dashed_pre_shift_start_pose = None
        self.dashed_pre_shift_dir_sign = 0.0

        # 后续任务使用的 TF 起点
        self.post_forward_start_pose = None
        self.post_turn_forward_start_pose = None
        self.turn_start_yaw = None
        self.current_turn_dir = 0  # +1 左转，-1 右转
        self.current_turn_angle_rad = 0.0
        self.current_turn_tolerance_rad = 0.0
        self.current_turn_wz = 0.0

        # 第二次转向后目标检测/撞击使用
        self.latest_target: Optional[Detection] = None
        self.locked_target: Optional[Detection] = None
        self.target_stable_count = 0
        self.stable_target_type = None
        self.hit_start_pose = None

        # 撞击完成后的后退 / 左跳 / 障碍物选择对齐使用
        self.after_hit_backoff_start_pose = None
        self.selected_obstacle_after_hit: Optional[Detection] = None
        self.selected_obstacle_after_hit_side = None  # 'left' / 'right'
        self.post_hit_obs_forward_start_pose = None
        self.post_hit_pre_final_forward_start_pose = None
        self.post_hit_final_forward_start_pose = None  # 保留旧变量名，避免外部引用出错

        # 最后阶段前方横向黄线对正 / 到达判定
        self.final_yellow_done_counter = 0
        self.final_yellow_reached_lower_area = False
        self.final_yellow_disappear_counter = 0
        self.latest_final_yellow_line: Optional[Detection] = None

        # 全局最终收尾黄线确认计数器
        self.global_final_yellow_done_counter = 0
        self.global_final_yellow_reached_lower_area = False
        self.global_final_yellow_disappear_counter = 0

        # 全局整合流程变量
        self.completed_bar_count = 0
        self.completed_obstacle_count = 0

        # 记录障碍物流程是不是作为全局第 3 个子流程启动。
        # 如果障碍物是最后一个要完成的物体：
        #   POST_HIT_FINAL_FORWARD 识别到横向黄线并靠近后，
        #   FINAL_LEFT_JUMP 只执行 1 次左跳，
        #   然后直接进入 GLOBAL_FINAL_YELLOW_FORWARD，
        #   跳过 GLOBAL_FINAL_RIGHT_JUMP。
        self.obstacle_flow_is_third_object = False

        # 完成限高杆流程后的下一轮搜索过滤标志。
        # True 时，GLOBAL_LATERAL_SEARCH 只允许 center_x 位于图像左半边的目标参与选择；
        # 一旦真正选中下一个目标，就自动关闭。
        self.only_search_left_half_after_bar = False
        self.left_half_filter_reason: Optional[str] = None
        self.current_left_half_search_x_ratio_max = self.after_bar_search_x_ratio_max

        self.global_center_stable_count = 0
        self.current_global_target: Optional[Detection] = None
        self.global_initial_lateral_shift_start_pose = None
        self.global_after_task_shift_start_pose = None
        self.current_after_task_shift_duration_s = self.global_after_task_shift_duration_s
        self.global_after_task_shift_reason = 'init'

        # 限高杆子流程变量
        self.current_bar_det: Optional[Detection] = None
        self.bar_return_target_depth_m: Optional[float] = None
        self.bar_center_stable_count = 0
        self.bar_hit_start_pose = None
        self.bar_backoff_start_time = None

        self.task_done_stop_sent = False
        self.last_log_time = self.now_s()

        self.get_logger().info('Fourth-stage mixin initialized; waiting for P3 to hand off.')
        self.set_body_normal(do_stop=False)
        # self.set_body_low(do_stop=False)


    # ---------- 全局整合 / 限高杆辅助 ----------
    def _declare_bar_params(self):
        p = self.declare_parameter
        p('bar.h_min', 85);
        p('bar.h_max', 100)
        p('bar.s_min', 15);
        p('bar.s_max', 45)
        p('bar.v_min', 35);
        p('bar.v_max', 80)
        p('bar.roi_x_ratio_min', 0.20);
        p('bar.roi_x_ratio_max', 0.80)
        p('bar.roi_y_ratio_min', 0.10);
        p('bar.roi_y_ratio_max', 0.90)
        p('bar.open_kernel', 3)
        p('bar.close_kernel_h', 7);
        p('bar.close_kernel_w', 11)
        p('bar.min_area', 300)
        p('bar.min_width', 15)
        p('bar.max_height', 1000)
        p('bar.min_aspect_ratio', 1.5)
        p('bar.max_aspect_ratio', 50.0)
        p('bar.max_center_y_ratio_in_roi', 1.0)
        p('bar.center_weight_base', 0.3)
        p('bar.center_weight_gain', 0.7)

    def _read_bar_cfg(self):
        gp = self.get_parameter
        return {
            'h_min': int(gp('bar.h_min').value), 'h_max': int(gp('bar.h_max').value),
            's_min': int(gp('bar.s_min').value), 's_max': int(gp('bar.s_max').value),
            'v_min': int(gp('bar.v_min').value), 'v_max': int(gp('bar.v_max').value),
            'roi_x_ratio_min': float(gp('bar.roi_x_ratio_min').value),
            'roi_x_ratio_max': float(gp('bar.roi_x_ratio_max').value),
            'roi_y_ratio_min': float(gp('bar.roi_y_ratio_min').value),
            'roi_y_ratio_max': float(gp('bar.roi_y_ratio_max').value),
            'open_kernel': int(gp('bar.open_kernel').value),
            'close_kernel_h': int(gp('bar.close_kernel_h').value),
            'close_kernel_w': int(gp('bar.close_kernel_w').value),
            'min_area': int(gp('bar.min_area').value),
            'min_width': int(gp('bar.min_width').value),
            'max_height': int(gp('bar.max_height').value),
            'min_aspect_ratio': float(gp('bar.min_aspect_ratio').value),
            'max_aspect_ratio': float(gp('bar.max_aspect_ratio').value),
            'max_center_y_ratio_in_roi': float(gp('bar.max_center_y_ratio_in_roi').value),
            'center_weight_base': float(gp('bar.center_weight_base').value),
            'center_weight_gain': float(gp('bar.center_weight_gain').value),
        }

    def all_global_tasks_done(self) -> bool:
        return (
                self.completed_bar_count >= self.required_bar_count
                and self.completed_obstacle_count >= self.required_obstacle_count
        )

    def enter_global_final_sequence(self):
        """全部子流程完成后，不直接 DONE，而是执行最终收尾动作。"""
        self.get_logger().info(
            '[GLOBAL_FINAL] all required flows completed, start final sequence: '
            'right jump -> yellow forward align -> one left jump -> DONE'
        )
        self.enter_state(self.GLOBAL_FINAL_RIGHT_JUMP)

    def speak_event_once(self, event_id: str, voice_key: str):
        """
        在指定事件第一次发生时播报一次。
        event_id 用来防重复，例如 bar_1、bar_2、obstacle_1、target_cola。
        voice_key 对应 VoicePlayer.voice_files 里的音频 key。
        """
        if not getattr(self, 'voice_enabled', True):
            return
        if event_id in self.voice_events_spoken:
            return
        played = self.voice.play_async(voice_key)
        self.voice_events_spoken.add(event_id)
        self.get_logger().info(f'[VOICE] event={event_id}, key={voice_key}, played={played}')

    def speak_bar_at_trigger(self):
        # 两次限高杆分别播报：bar_1、bar_2
        event_id = f'bar_{self.completed_bar_count + 1}'
        self.speak_event_once(event_id, 'bar')

    def speak_obstacle_at_trigger(self):
        event_id = f'obstacle_{self.completed_obstacle_count + 1}'
        self.speak_event_once(event_id, 'obstacle')

    def target_voice_key(self, det_type: str) -> Optional[str]:
        """
        当前代码里的目标类型与赛题播报类型映射。
        cola       -> 识别到可乐瓶
        blue_ball  -> 识别到橙色小球（如果你后续改成 orange_ball，也兼容）
        white_ball -> 识别到足球（如果你后续改成 football，也兼容）
        """
        if det_type == 'cola':
            return 'cola'
        if det_type in ('orange_ball', 'blue_ball'):
            return 'orange_ball'
        if det_type in ('football', 'white_ball'):
            return 'football'
        return None

    def speak_target_at_hit_trigger(self, det_type: str):
        key = self.target_voice_key(det_type)
        if key is None:
            self.get_logger().warn(f'[VOICE] no voice mapping for target det_type={det_type}')
            return
        # 三个目标物体每类只在真正准备撞击时播一次
        event_id = f'target_{key}'
        self.speak_event_once(event_id, key)

    def is_obstacle_flow_state(self) -> bool:
        return self.state in {
            self.APPROACH_OBSTACLES,
            self.DASH_PRE_SIDE_SHIFT,
            self.ALIGN_DASHED_LINE,
            self.FOLLOW_DASHED_UNTIL_LOST,
            self.POST_DASH_FORWARD,
            self.POST_DASH_TURN_1,
            self.POST_TURN_FORWARD,
            self.POST_DASH_TURN_2,
            self.SEARCH_TARGET_AFTER_TURNS,
            self.APPROACH_AND_ALIGN_TARGET,
            self.HIT_TARGET,
            self.HIT_BACKOFF_AFTER_HIT,
            self.POST_HIT_LEFT_JUMP,
            self.APPROACH_SELECTED_OBSTACLE_AFTER_HIT,
            self.POST_HIT_OBS_TURN_1,
            self.POST_HIT_OBS_FORWARD,
            self.POST_HIT_OBS_TURN_2,
            self.POST_HIT_PRE_FINAL_FORWARD,
            self.POST_HIT_FINAL_FORWARD,
            self.FINAL_LEFT_JUMP,
            self.OBSTACLE_RESTORE_NORMAL_AFTER_FINAL_TURN,
            self.OBSTACLE_FLOW_DONE,
        }

    def is_in_after_bar_search_region(self, det: Optional[Detection]) -> bool:
        """
        完成限高杆流程后的下一轮全局搜索，只允许图像左半边目标参与选择。

        默认 after_bar_search_x_ratio_max=0.50：
            det.center_img[0] < image_width * 0.50 才有效。

        如果 only_search_left_half_after_bar=False，则不过滤。
        """
        if det is None:
            return False

        if self.latest_bgr is None:
            return True

        if not self.only_search_left_half_after_bar:
            return True

        img_w = self.latest_bgr.shape[1]
        x_limit = img_w * self.current_left_half_search_x_ratio_max
        return det.center_img[0] < x_limit

    def choose_global_object(self, bar: Optional[Detection], obs_candidates: List[Detection]):
        if self.latest_bgr is None:
            return None, None

        img_w = self.latest_bgr.shape[1]
        img_center_x = img_w / 2.0
        choices = []

        if self.only_search_left_half_after_bar:
            x_limit = img_w * self.current_left_half_search_x_ratio_max
            self.get_logger().info(
                f'[GLOBAL_FILTER] after subtask done: only search left region, '
                f'center_x < {x_limit:.1f} ({self.current_left_half_search_x_ratio_max:.2f} * image_width)',
                throttle_duration_sec=0.8
            )

        if bar is not None and self.completed_bar_count < self.required_bar_count:
            if self.is_in_after_bar_search_region(bar):
                choices.append(('bar', bar, abs(bar.center_img[0] - img_center_x)))
            else:
                self.get_logger().info(
                    f'[GLOBAL_FILTER] ignore BAR outside left-search region: center={bar.center_img}',
                    throttle_duration_sec=0.5
                )

        if obs_candidates and self.completed_obstacle_count < self.required_obstacle_count:
            valid_obs = [
                obs for obs in obs_candidates
                if self.is_in_after_bar_search_region(obs)
            ]

            if valid_obs:
                obs = min(valid_obs, key=lambda d: abs(d.center_img[0] - img_center_x))
                choices.append(('obstacle', obs, abs(obs.center_img[0] - img_center_x)))
            elif self.only_search_left_half_after_bar:
                self.get_logger().info(
                    f'[GLOBAL_FILTER] ignore all OBSTACLES outside left-search region, '
                    f'raw_count={len(obs_candidates)}',
                    throttle_duration_sec=0.5
                )

        if not choices:
            return None, None

        obj_type, det, _ = min(choices, key=lambda x: x[2])

        # 完成限高杆后的左半边限制只用于“寻找下一个目标”。
        # 一旦真的选中了目标，就关闭限制，避免进入居中/子流程后继续影响后续逻辑。
        if self.only_search_left_half_after_bar:
            self.get_logger().info(
                f'[GLOBAL_FILTER] selected {obj_type} in left-search region, '
                f'center={det.center_img}, disable left-half-only filter'
            )
            self.only_search_left_half_after_bar = False
            self.left_half_filter_reason = None

        return obj_type, det

    def finish_bar_flow(self):
        self.completed_bar_count += 1
        self.get_logger().info(
            f'[GLOBAL] bar flow finished: bar={self.completed_bar_count}/{self.required_bar_count}, '
            f'obstacle={self.completed_obstacle_count}/{self.required_obstacle_count}'
        )
        # 限高杆流程结束后恢复 normal，避免后续全局搜索/最终收尾继续低身。
        self.restore_body_normal_after_bar_flow()

        if self.all_global_tasks_done():
            self.get_logger().info('[GLOBAL] all flows done after bar flow, start final sequence')
            self.enter_global_final_sequence()
            return

        # 新增：完成一次限高杆后，下一轮 GLOBAL_LATERAL_SEARCH 只在左半边找目标。
        # 目的：避免刚刚通过的限高杆还在右半边画面里，被重复当成下一个目标。
        if self.search_left_half_after_bar_done:
            self.only_search_left_half_after_bar = True
            self.left_half_filter_reason = 'bar_done'
            self.current_left_half_search_x_ratio_max = self.after_bar_search_x_ratio_max
            self.get_logger().info(
                f'[GLOBAL_FILTER] bar done: next global search only uses left region, '
                f'x_ratio_max={self.current_left_half_search_x_ratio_max:.2f}'
            )

        self.current_after_task_shift_duration_s = self.global_after_task_shift_duration_s
        self.global_after_task_shift_reason = 'bar_done'
        self.enter_state(self.GLOBAL_SHIFT_AFTER_SUBTASK)

    def finish_obstacle_flow(self):
        self.completed_obstacle_count += 1
        self.get_logger().info(
            f'[GLOBAL] obstacle flow finished: bar={self.completed_bar_count}/{self.required_bar_count}, '
            f'obstacle={self.completed_obstacle_count}/{self.required_obstacle_count}, dashed_side={self.dashed_side}'
        )
        if self.all_global_tasks_done():
            self.get_logger().info('[GLOBAL] all flows done after obstacle flow, start final sequence')
            self.enter_global_final_sequence()
            return

        # 新增：完成障碍物流程后，下一轮 GLOBAL_LATERAL_SEARCH 也只在左半边找目标。
        # 目的：和限高杆完成后的逻辑一致，避免刚完成的障碍物流程相关目标仍在右侧画面中干扰下一目标选择。
        if self.search_left_half_after_obstacle_done:
            self.only_search_left_half_after_bar = True
            self.left_half_filter_reason = 'obstacle_done'
            self.current_left_half_search_x_ratio_max = self.after_obstacle_search_x_ratio_max
            self.get_logger().info(
                f'[GLOBAL_FILTER] obstacle done: next global search only uses left region, '
                f'x_ratio_max={self.current_left_half_search_x_ratio_max:.2f}'
            )

        if self.dashed_side == 'right':
            self.current_after_task_shift_duration_s = self.global_after_obstacle_shift_duration_right_dash_s
        elif self.dashed_side == 'left':
            self.current_after_task_shift_duration_s = self.global_after_obstacle_shift_duration_left_dash_s
        else:
            self.current_after_task_shift_duration_s = self.global_after_task_shift_duration_s
            self.get_logger().warn('[GLOBAL] obstacle finished but dashed_side is None, use default shift distance')
        self.global_after_task_shift_reason = f'obstacle_done_dash_{self.dashed_side}'
        self.get_logger().info(
            f'[GLOBAL] after obstacle shift duration={self.current_after_task_shift_duration_s:.3f}s, '
            f'reason={self.global_after_task_shift_reason}'
        )
        self.enter_state(self.GLOBAL_SHIFT_AFTER_SUBTASK)

    def estimate_bar_depth(self, bar_det: Detection) -> Optional[float]:
        if self.latest_depth is None or self.latest_bgr is None or bar_det is None:
            return None
        depth_m = self.depth_to_meters(self.latest_depth)
        if depth_m is None:
            return None
        ih, iw = self.latest_bgr.shape[:2]
        dh, dw = depth_m.shape[:2]
        x1, y1, x2, y2 = bar_det.bbox_img
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        sx1 = x1 + int(0.15 * bw)
        sx2 = x2 - int(0.15 * bw)
        sy1 = y1 + int(0.05 * bh)
        sy2 = y1 + int(0.15 * bh)
        sx1 = max(0, min(iw - 1, sx1));
        sx2 = max(sx1 + 1, min(iw, sx2))
        sy1 = max(0, min(ih - 1, sy1));
        sy2 = max(sy1 + 1, min(ih, sy2))
        dx1 = int(sx1 * dw / max(iw, 1));
        dx2 = int(sx2 * dw / max(iw, 1))
        dy1 = int(sy1 * dh / max(ih, 1));
        dy2 = int(sy2 * dh / max(ih, 1))
        dx1 = max(0, min(dw - 1, dx1));
        dx2 = max(dx1 + 1, min(dw, dx2))
        dy1 = max(0, min(dh - 1, dy1));
        dy2 = max(dy1 + 1, min(dh, dy2))
        patch = depth_m[dy1:dy2, dx1:dx2]
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > 0.05) & (valid < 10.0)]
        if valid.size == 0:
            return None
        return float(np.percentile(valid, 20))

    def is_bar_centered(self, bar: Detection, deadband_px: Optional[int] = None) -> bool:
        if self.latest_bgr is None or bar is None:
            return False

        if deadband_px is None:
            deadband_px = self.bar_center_px_deadband

        img_center_x = self.latest_bgr.shape[1] / 2.0
        err_px = bar.center_img[0] - img_center_x

        return abs(err_px) <= int(deadband_px)

    def compute_bar_align_vy(self, bar: Detection) -> float:
        if self.latest_bgr is None or bar is None:
            return 0.0
        img_center_x = self.latest_bgr.shape[1] / 2.0
        err_px = bar.center_img[0] - img_center_x
        if abs(err_px) <= self.bar_center_px_deadband:
            return 0.0
        err_norm = err_px / max(self.latest_bgr.shape[1] / 2.0, 1.0)
        vy = -self.bar_align_vy_k * err_norm
        vy = float(np.clip(vy, -self.bar_align_vy_max, self.bar_align_vy_max))
        if 0.0 < abs(vy) < self.bar_align_vy_min:
            vy = math.copysign(self.bar_align_vy_min, vy)
        return vy

    def compute_global_bar_center_fixed_vy(self, bar: Detection) -> float:
        if self.latest_bgr is None or bar is None:
            return 0.0

        img_center_x = self.latest_bgr.shape[1] / 2.0
        err_px = bar.center_img[0] - img_center_x

        if abs(err_px) <= self.global_bar_center_px_deadband:
            return 0.0

        vy = -math.copysign(self.global_bar_center_fixed_vy, err_px)
        return float(vy)

    def sample_depth_patch_by_rgb(self, rgb_x: int, rgb_y: int, half_size: int) -> Optional[float]:
        """
        在 RGB 像素点附近采样深度，返回较近的 20 分位数。
        用于限高杆左右两侧深度差判断朝向。
        """
        if self.latest_depth is None or self.latest_bgr is None:
            return None

        depth_m = self.depth_to_meters(self.latest_depth)
        if depth_m is None:
            return None

        ih, iw = self.latest_bgr.shape[:2]
        dh, dw = depth_m.shape[:2]

        dx = int(rgb_x * dw / max(iw, 1))
        dy = int(rgb_y * dh / max(ih, 1))

        half = max(1, int(half_size))
        x1 = max(0, dx - half)
        x2 = min(dw, dx + half + 1)
        y1 = max(0, dy - half)
        y2 = min(dh, dy + half + 1)

        patch = depth_m[y1:y2, x1:x2]
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > 0.05) & (valid < 10.0)]
        if valid.size == 0:
            return None
        return float(np.percentile(valid, 20))

    def compute_bar_depth_yaw_align_wz(self, bar: Detection) -> float:
        """
        限高杆朝向矫正：根据限高杆左右两侧深度差，给一个固定小角速度 wz。

        left_depth / right_depth 基本相等：说明机器狗大致正对限高杆，不转。
        两边深度差超过 deadband：说明一侧更近，机器狗相对限高杆有偏航，给固定 wz 修正。

        注意：wz 正负号和相机/机器人坐标有关。
        如果实测发现越修越歪，把参数 bar_depth_yaw_sign 从 1.0 改成 -1.0。
        """
        self.latest_bar_depth_yaw_info = {
            'left_depth': None,
            'right_depth': None,
            'depth_error': None,
            'wz': 0.0,
        }

        if not self.bar_depth_yaw_align_enabled:
            return 0.0
        if bar is None or self.latest_bgr is None or self.latest_depth is None:
            return 0.0

        x1, y1, x2, y2 = bar.bbox_img
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        # 在限高杆 bbox 内，左右各取一个采样点。
        # y 取靠近横杆上沿的位置，避免采到地面或背景。
        x_ratio = min(max(float(self.bar_depth_yaw_sample_x_ratio), 0.05), 0.45)
        y_ratio = min(max(float(self.bar_depth_yaw_sample_y_ratio), 0.02), 0.80)

        left_x = int(x1 + bw * x_ratio)
        right_x = int(x2 - bw * x_ratio)
        sample_y = int(y1 + bh * y_ratio)

        left_depth = self.sample_depth_patch_by_rgb(
            left_x,
            sample_y,
            self.bar_depth_yaw_sample_half_size
        )
        right_depth = self.sample_depth_patch_by_rgb(
            right_x,
            sample_y,
            self.bar_depth_yaw_sample_half_size
        )

        if left_depth is None or right_depth is None:
            self.get_logger().info(
                f'[BAR_DEPTH_YAW] invalid sample: left={left_depth}, right={right_depth}',
                throttle_duration_sec=0.5
            )
            return 0.0

        # 正值：左侧比右侧更远；负值：左侧比右侧更近。
        depth_error = float(left_depth - right_depth)
        if abs(depth_error) <= self.bar_depth_yaw_deadband_m:
            wz = 0.0
        else:
            # 默认符号：depth_error > 0 时给正 wz；如果实测反了，改 bar_depth_yaw_sign=-1。
            wz = math.copysign(self.bar_depth_yaw_fixed_wz, depth_error) * self.bar_depth_yaw_sign

        self.latest_bar_depth_yaw_info = {
            'left_depth': float(left_depth),
            'right_depth': float(right_depth),
            'depth_error': float(depth_error),
            'wz': float(wz),
            'left_point': (left_x, sample_y),
            'right_point': (right_x, sample_y),
        }

        self.get_logger().info(
            f'[BAR_DEPTH_YAW] left={left_depth:.3f}, right={right_depth:.3f}, '
            f'err=L-R={depth_error:.3f}m, deadband={self.bar_depth_yaw_deadband_m:.3f}m, '
            f'wz={wz:.3f}',
            throttle_duration_sec=0.3
        )
        return float(wz)

    def compute_bar_backoff_vx_by_depth(self, depth_m: Optional[float], target_depth_m: Optional[float]) -> float:
        """
        BAR_BACKOFF_TO_BAR 回退阶段的 vx 闭环。

        目标：让当前限高杆深度 depth_m 回到 GLOBAL_CENTER_BAR 时记录的 target_depth_m。
        - depth_m < target_depth_m：说明还没退够，继续给负 vx 后退。
        - depth_m > target_depth_m：说明退过了，如果允许则给小正 vx 往前修一点。
        - 接近目标深度：vx=0，外层状态机负责结束流程。
        """
        if not self.bar_backoff_depth_vx_align_enabled:
            return -self.backoff_after_hit_speed
        if depth_m is None or target_depth_m is None:
            return -self.backoff_after_hit_speed

        depth_err = float(depth_m - target_depth_m)
        if abs(depth_err) <= self.backoff_bar_depth_tolerance_m:
            return 0.0

        vx = self.bar_backoff_depth_vx_k * depth_err
        max_vx = self.bar_backoff_depth_vx_max if self.bar_backoff_depth_vx_max > 0.0 else self.backoff_after_hit_speed
        vx = float(np.clip(vx, -max_vx, max_vx))

        # 默认允许小幅向前修正，避免退过头后只能停不回来。
        if vx > 0.0 and not self.bar_backoff_allow_forward_correction:
            vx = 0.0

        # 不在目标附近时给一个最小速度，避免速度太小退不动。
        min_vx = self.bar_backoff_depth_vx_min
        if 0.0 < abs(vx) < min_vx:
            vx = math.copysign(min_vx, vx)

        return vx

    # ---------- 参数 ----------
    def _declare_obstacle_params(self):
        p = self.declare_parameter
        p('obstacle.roi_x_ratio_min', 0.15)
        p('obstacle.roi_x_ratio_max', 0.85)
        p('obstacle.roi_y_ratio_min', 0.20)
        p('obstacle.roi_y_ratio_max', 0.95)

        p('obstacle.h_min', 90)
        p('obstacle.h_max', 140)
        p('obstacle.s_min', 60)
        p('obstacle.s_max', 255)
        p('obstacle.v_min', 40)
        p('obstacle.v_max', 255)

        p('obstacle.depth_min_m', 0.05)
        p('obstacle.depth_max_m', 1.50)

        p('obstacle.open_kernel', 3)
        p('obstacle.close_kernel', 5)

        p('obstacle.min_area', 150)
        p('obstacle.min_width', 10)
        p('obstacle.min_height', 10)
        p('obstacle.max_aspect_ratio', 4.5)
        p('obstacle.min_bottom_y_ratio_in_roi', 0.2)

        p('obstacle.min_valid_depth_ratio', 0.20)
        p('obstacle.min_near_depth_ratio', 0.35)

    def _read_obstacle_cfg(self):
        gp = self.get_parameter
        return {
            'roi_x_ratio_min': float(gp('obstacle.roi_x_ratio_min').value),
            'roi_x_ratio_max': float(gp('obstacle.roi_x_ratio_max').value),
            'roi_y_ratio_min': float(gp('obstacle.roi_y_ratio_min').value),
            'roi_y_ratio_max': float(gp('obstacle.roi_y_ratio_max').value),

            'h_min': int(gp('obstacle.h_min').value),
            'h_max': int(gp('obstacle.h_max').value),
            's_min': int(gp('obstacle.s_min').value),
            's_max': int(gp('obstacle.s_max').value),
            'v_min': int(gp('obstacle.v_min').value),
            'v_max': int(gp('obstacle.v_max').value),

            'depth_min_m': float(gp('obstacle.depth_min_m').value),
            'depth_max_m': float(gp('obstacle.depth_max_m').value),

            'open_kernel': int(gp('obstacle.open_kernel').value),
            'close_kernel': int(gp('obstacle.close_kernel').value),

            'min_area': int(gp('obstacle.min_area').value),
            'min_width': int(gp('obstacle.min_width').value),
            'min_height': int(gp('obstacle.min_height').value),
            'max_aspect_ratio': float(gp('obstacle.max_aspect_ratio').value),
            'min_bottom_y_ratio_in_roi': float(gp('obstacle.min_bottom_y_ratio_in_roi').value),

            'min_valid_depth_ratio': float(gp('obstacle.min_valid_depth_ratio').value),
            'min_near_depth_ratio': float(gp('obstacle.min_near_depth_ratio').value),
        }

    def _declare_yellow_params(self):
        p = self.declare_parameter

        # =========================
        # 黄色 HSV 参数
        # =========================
        p('yellow.h_min', 18)
        p('yellow.h_max', 45)
        p('yellow.s_min', 70)
        p('yellow.s_max', 255)
        p('yellow.v_min', 70)
        p('yellow.v_max', 255)

        # =========================
        # ROI 参数
        # 只看图像下方 60%~100% 区域
        # =========================
        p('yellow.roi_x_ratio_min', 0.00)
        p('yellow.roi_x_ratio_max', 1.00)
        p('yellow.roi_y_ratio_min', 0.60)
        p('yellow.roi_y_ratio_max', 1.00)

        # =========================
        # 基础黄色块过滤
        # =========================
        p('yellow.open_kernel', 3)
        p('yellow.min_area', 50)
        p('yellow.max_area', 4000)
        p('yellow.min_width', 3)
        p('yellow.min_height', 5)

        # =========================
        # 虚线形态学参数
        # =========================
        p('yellow.dash_close_kernel_h', 3)
        p('yellow.dash_close_kernel_w', 5)

        # =========================
        # 虚线组合参数
        # =========================
        p('yellow.dash_min_segments', 2)
        p('yellow.dash_min_total_span_y', 20)
        p('yellow.dash_max_adjacent_x_diff', 110)
        p('yellow.dash_max_gap_y', 3000)
        p('yellow.dash_min_gap_y', -10)
        p('yellow.dash_max_total_x_range', 5000)

        # =========================
        # 避免长条参与虚线组合
        # =========================
        p('yellow.dash_segment_max_aspect_ratio', 10.0)
        p('yellow.dash_segment_max_long_side', 200)

        # =========================
        # 多条虚线去重参数
        # =========================
        p('yellow.dash_duplicate_iou_thresh', 0.35)
        p('yellow.dash_duplicate_center_x_thresh', 30)

        # =========================
        # 最多显示 / 使用最长几条虚线
        # =========================
        p('yellow.max_dashed_lines', 2)

    def _read_yellow_cfg(self):
        gp = self.get_parameter
        return {
            'h_min': int(gp('yellow.h_min').value),
            'h_max': int(gp('yellow.h_max').value),
            's_min': int(gp('yellow.s_min').value),
            's_max': int(gp('yellow.s_max').value),
            'v_min': int(gp('yellow.v_min').value),
            'v_max': int(gp('yellow.v_max').value),

            'roi_x_ratio_min': float(gp('yellow.roi_x_ratio_min').value),
            'roi_x_ratio_max': float(gp('yellow.roi_x_ratio_max').value),
            'roi_y_ratio_min': float(gp('yellow.roi_y_ratio_min').value),
            'roi_y_ratio_max': float(gp('yellow.roi_y_ratio_max').value),

            'open_kernel': int(gp('yellow.open_kernel').value),
            'min_area': int(gp('yellow.min_area').value),
            'max_area': int(gp('yellow.max_area').value),
            'min_width': int(gp('yellow.min_width').value),
            'min_height': int(gp('yellow.min_height').value),

            'dash_close_kernel_h': int(gp('yellow.dash_close_kernel_h').value),
            'dash_close_kernel_w': int(gp('yellow.dash_close_kernel_w').value),

            'dash_min_segments': int(gp('yellow.dash_min_segments').value),
            'dash_min_total_span_y': int(gp('yellow.dash_min_total_span_y').value),
            'dash_max_adjacent_x_diff': int(gp('yellow.dash_max_adjacent_x_diff').value),
            'dash_max_gap_y': int(gp('yellow.dash_max_gap_y').value),
            'dash_min_gap_y': int(gp('yellow.dash_min_gap_y').value),
            'dash_max_total_x_range': int(gp('yellow.dash_max_total_x_range').value),

            'dash_segment_max_aspect_ratio': float(gp('yellow.dash_segment_max_aspect_ratio').value),
            'dash_segment_max_long_side': int(gp('yellow.dash_segment_max_long_side').value),

            'dash_duplicate_iou_thresh': float(gp('yellow.dash_duplicate_iou_thresh').value),
            'dash_duplicate_center_x_thresh': int(gp('yellow.dash_duplicate_center_x_thresh').value),

            'max_dashed_lines': int(gp('yellow.max_dashed_lines').value),
        }

    def _declare_final_yellow_params(self):
        p = self.declare_parameter

        # 最后阶段前方横向黄线：HSV 默认沿用黄色阈值，但单独开放参数便于调试
        p('final_yellow.h_min', 18)
        p('final_yellow.h_max', 45)
        p('final_yellow.s_min', 70)
        p('final_yellow.s_max', 255)
        p('final_yellow.v_min', 70)
        p('final_yellow.v_max', 255)

        # 看前方中下区域；比虚线 ROI 更靠上，方便提前看到横线
        p('final_yellow.roi_x_ratio_min', 0.30)
        p('final_yellow.roi_x_ratio_max', 0.70)
        p('final_yellow.roi_y_ratio_min', 0.50)
        p('final_yellow.roi_y_ratio_max', 1.00)

        p('final_yellow.open_kernel', 3)
        p('final_yellow.close_kernel_h', 5)
        p('final_yellow.close_kernel_w', 11)

        p('final_yellow.min_area', 3000)
        p('final_yellow.min_width', 20)
        p('final_yellow.min_height', 3)
        p('final_yellow.min_width_ratio', 0.70)
        p('final_yellow.min_wh_ratio', 1.5)
        p('final_yellow.max_tilt_deg', 35.0)

        # RGB-only: use bottom_y/image_height to judge distance to line
        p('final_yellow.center_tolerance_ratio', 0.60)

    def _read_final_yellow_cfg(self):
        gp = self.get_parameter
        return {
            'h_min': int(gp('final_yellow.h_min').value),
            'h_max': int(gp('final_yellow.h_max').value),
            's_min': int(gp('final_yellow.s_min').value),
            's_max': int(gp('final_yellow.s_max').value),
            'v_min': int(gp('final_yellow.v_min').value),
            'v_max': int(gp('final_yellow.v_max').value),

            'roi_x_ratio_min': float(gp('final_yellow.roi_x_ratio_min').value),
            'roi_x_ratio_max': float(gp('final_yellow.roi_x_ratio_max').value),
            'roi_y_ratio_min': float(gp('final_yellow.roi_y_ratio_min').value),
            'roi_y_ratio_max': float(gp('final_yellow.roi_y_ratio_max').value),

            'open_kernel': int(gp('final_yellow.open_kernel').value),
            'close_kernel_h': int(gp('final_yellow.close_kernel_h').value),
            'close_kernel_w': int(gp('final_yellow.close_kernel_w').value),

            'min_area': int(gp('final_yellow.min_area').value),
            'min_width': int(gp('final_yellow.min_width').value),
            'min_height': int(gp('final_yellow.min_height').value),
            'min_width_ratio': float(gp('final_yellow.min_width_ratio').value),
            'min_wh_ratio': float(gp('final_yellow.min_wh_ratio').value),
            'max_tilt_deg': float(gp('final_yellow.max_tilt_deg').value),
            'center_tolerance_ratio': float(gp('final_yellow.center_tolerance_ratio').value),
        }

    # ---------- 第二次转向后的目标检测参数 ----------
    def _declare_ball_params(self, prefix: str, defaults: Dict[str, Any]):
        for k, v in defaults.items():
            self.declare_parameter(f'{prefix}.{k}', v)

    def _read_ball_cfg(self, prefix: str):
        keys = [
            'h_min', 'h_max', 's_min', 's_max', 'v_min', 'v_max',
            'roi_x_ratio_min', 'roi_x_ratio_max', 'roi_y_ratio_min', 'roi_y_ratio_max',
            'open_kernel', 'close_kernel',
            'min_area', 'max_area',
            'min_radius', 'max_radius',
            'min_circularity',
            'min_wh_ratio', 'max_wh_ratio',
            'max_center_y_ratio_in_roi',
            'center_weight_base', 'center_weight_gain',
            'radius_score_gain',
        ]
        cfg = {}
        for k in keys:
            val = self.get_parameter(f'{prefix}.{k}').value
            cfg[k] = float(val) if isinstance(val, float) else int(val) if isinstance(val, int) else val
        return cfg

    def _declare_cola_params(self):
        p = self.declare_parameter
        p('cola.h_min', 0);
        p('cola.h_max', 20)
        p('cola.s_min', 0);
        p('cola.s_max', 20)
        p('cola.v_min', 0);
        p('cola.v_max', 20)
        p('cola.roi_x_ratio_min', 0.20);
        p('cola.roi_x_ratio_max', 0.80)
        p('cola.roi_y_ratio_min', 0.00);
        p('cola.roi_y_ratio_max', 1.00)
        p('cola.open_kernel', 3);
        p('cola.close_kernel', 5)
        p('cola.min_area', 250);
        p('cola.max_area', 80000)
        p('cola.min_width', 8);
        p('cola.max_width', 5000)
        p('cola.min_height', 20);
        p('cola.max_height', 10000)
        p('cola.min_hw_ratio', 1.5);
        p('cola.max_hw_ratio', 20.0)
        p('cola.max_center_y_ratio_in_roi', 1.0)
        p('cola.center_weight_base', 0.3);
        p('cola.center_weight_gain', 0.7)

    def _read_cola_cfg(self):
        gp = self.get_parameter
        return {
            'h_min': int(gp('cola.h_min').value), 'h_max': int(gp('cola.h_max').value),
            's_min': int(gp('cola.s_min').value), 's_max': int(gp('cola.s_max').value),
            'v_min': int(gp('cola.v_min').value), 'v_max': int(gp('cola.v_max').value),
            'roi_x_ratio_min': float(gp('cola.roi_x_ratio_min').value),
            'roi_x_ratio_max': float(gp('cola.roi_x_ratio_max').value),
            'roi_y_ratio_min': float(gp('cola.roi_y_ratio_min').value),
            'roi_y_ratio_max': float(gp('cola.roi_y_ratio_max').value),
            'open_kernel': int(gp('cola.open_kernel').value),
            'close_kernel': int(gp('cola.close_kernel').value),
            'min_area': int(gp('cola.min_area').value),
            'max_area': int(gp('cola.max_area').value),
            'min_width': int(gp('cola.min_width').value),
            'max_width': int(gp('cola.max_width').value),
            'min_height': int(gp('cola.min_height').value),
            'max_height': int(gp('cola.max_height').value),
            'min_hw_ratio': float(gp('cola.min_hw_ratio').value),
            'max_hw_ratio': float(gp('cola.max_hw_ratio').value),
            'max_center_y_ratio_in_roi': float(gp('cola.max_center_y_ratio_in_roi').value),
            'center_weight_base': float(gp('cola.center_weight_base').value),
            'center_weight_gain': float(gp('cola.center_weight_gain').value),
        }

    # ---------- ROS ----------
    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def state_elapsed_s(self) -> float:
        """
        返回当前状态已经运行的仿真时间。

        注意：使用 use_sim_time=True 时，节点刚初始化时 /clock 可能还没就绪，
        如果在 enter_state() 里直接记录 now_s()，可能会记录到 0.0。
        后续第一帧 control_loop 看到的 /clock 可能已经是几百秒，导致 elapsed 很大，
        固定时间状态会被瞬间跳过。

        所以这里采用延迟启动计时：
        第一次进入控制循环且 now_s()>0 时，才把当前仿真时间作为状态起点。
        """
        now = self.now_s()
        if now <= 0.0:
            return 0.0

        if self.state_enter_time is None or self.state_enter_time <= 0.0:
            self.state_enter_time = now
            self.get_logger().info(
                f'[STATE_TIMER] start {self.state} at sim_time={now:.3f}',
                throttle_duration_sec=1.0
            )
            return 0.0

        return max(0.0, now - self.state_enter_time)

    def fourth_rgb_callback(self, msg: Image):
        try:
            self.latest_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB convert failed: {e}')

    def fourth_depth_callback(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'DEPTH convert failed: {e}')

    def depth_to_meters(self, depth_img):
        if depth_img is None:
            return None

        if depth_img.dtype == np.float32:
            depth_m = depth_img.copy()
        elif depth_img.dtype == np.uint16:
            depth_m = depth_img.astype(np.float32) / 1000.0
        else:
            depth_m = depth_img.astype(np.float32)

        depth_m[~np.isfinite(depth_m)] = 0.0
        return depth_m

    def estimate_depth_at_center(self, center_img: Tuple[int, int]) -> Optional[float]:
        """
        复用限高杆代码的目标深度估计：
        在目标中心附近取一个 7x7 深度窗口，取较近的 20 分位数。
        """
        if self.latest_depth is None or self.latest_bgr is None:
            return None

        depth_m = self.depth_to_meters(self.latest_depth)
        if depth_m is None:
            return None

        dh, dw = depth_m.shape[:2]
        ih, iw = self.latest_bgr.shape[:2]
        cx, cy = center_img

        dx = int(cx * dw / max(iw, 1))
        dy = int(cy * dh / max(ih, 1))

        half = 3
        x1 = max(0, dx - half)
        x2 = min(dw, dx + half + 1)
        y1 = max(0, dy - half)
        y2 = min(dh, dy + half + 1)

        patch = depth_m[y1:y2, x1:x2]
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > 0.05) & (valid < 10.0)]

        if valid.size == 0:
            return None

        return float(np.percentile(valid, 20))

    # ---------- 控制 ----------
    def _publish_body_height(self, height: float, label: str = ''):
        """
        只发布机身高度 yaml，不在这里 STOP。
        所有姿态切换前的 STOP 统一由 set_body_low / set_body_normal / reset_body_pose_to_normal_at_start 负责。
        """
        values = [0.0] * 12
        values[0] = 0.0  # roll
        values[2] = float(height)  # height

        self.yaml_node.publish_yaml_vecxd(
            "des_roll_pitch_height_motion",
            values,
            is_user=1
        )
        self.yaml_node.publish_yaml_vecxd(
            "des_roll_pitch_height",
            values,
            is_user=1
        )

        suffix = f' ({label})' if label else ''
        self.get_logger().warn(f'[BODY] publish height={float(height):.3f}{suffix}')

    def set_body_low(self, do_stop: bool = True, reason: str = '', force: bool = False):
        """
        设置第四赛段 low 姿态。

        重要：姿态切换前必须先 STOP，所以 do_stop=True 时先调用 self.stop()，
        等 mode=12/gait_id=0 完成后，再发布 body height yaml。
        force=True 时即使内部变量认为已经是 low，也强制重新发布一次。
        """
        low_height = float(getattr(self, 'p4_low_body_height', 0.17))
        already_low = (
            getattr(self, 'body_is_low', False)
            and abs(float(getattr(self, 'body_height_cmd', low_height)) - low_height) < 1e-6
        )
        if already_low and not force:
            return

        self.body_height_cmd = low_height
        self.body_is_low = True
        self._publish_body_height(low_height, label='LOW')

        if do_stop:
            self.stop()

        suffix = f', reason={reason}' if reason else ''
        self.get_logger().warn(
            f'[BODY] set LOW height={low_height:.3f}, pre_stop={do_stop}, force={force}{suffix}'
        )

    def set_body_normal(self, do_stop: bool = True, reason: str = '', force: bool = False):
        """
        恢复第四赛段 normal 姿态。

        重要：姿态切换前必须先 STOP，所以 do_stop=True 时先调用 self.stop()，
        等 mode=12/gait_id=0 完成后，再发布 body height yaml。
        force=True 时即使内部变量认为已经是 normal，也强制重新发布一次。
        """
        normal_height = float(getattr(self, 'p4_normal_body_height', 0.25))
        already_normal = (
            not getattr(self, 'body_is_low', False)
            and abs(float(getattr(self, 'body_height_cmd', normal_height)) - normal_height) < 1e-6
        )
        if already_normal and not force:
            return

        self.body_height_cmd = normal_height
        self.body_is_low = False
        self._publish_body_height(normal_height, label='NORMAL')

        if do_stop:
            self.stop()
            
        suffix = f', reason={reason}' if reason else ''
        self.get_logger().warn(
            f'[BODY] set NORMAL height={normal_height:.3f}, pre_stop={do_stop}, force={force}{suffix}'
        )

    def reset_body_pose_to_normal_at_start(self):
        """
        启动/第四赛段单独调试时强制刷新身体高度：
        STOP -> LOW -> STOP -> NORMAL。

        这样可以避免控制器残留上一次 low 姿态，也满足“切姿态前必须先发 STOP”的要求。
        """
        self.get_logger().warn('[BODY_INIT] force refresh body pose: STOP -> LOW -> STOP -> NORMAL')

        # 切到 low 前先 STOP
        self.stop()
        self.set_body_low(
            do_stop=False,
            reason='body_init_refresh_low',
            force=True
        )

        # 从 low 切回 normal 前也先 STOP
        self.stop()
        self.set_body_normal(
            do_stop=False,
            reason='body_init_refresh_normal',
            force=True
        )

        self.get_logger().warn(
            f'[BODY_INIT] body pose reset done: normal_height={self.p4_normal_body_height:.3f}'
        )

    def set_body_low_for_bar_trigger(self):
        """限高杆播报触发距离处同步切 low。"""
        if not getattr(self, 'bar_body_low_enabled', True):
            return
        self.set_body_low(
            do_stop=getattr(self, 'bar_body_low_do_stop', True),
            reason='bar_trigger_distance'
        )

    def set_body_low_for_obstacle_flow(self):
        """障碍物全局居中完成、进入障碍物流程前切 low。"""
        if not getattr(self, 'obstacle_flow_low_enabled', True):
            return
        self.set_body_low(
            do_stop=getattr(self, 'obstacle_body_low_do_stop', True),
            reason='obstacle_centered_enter_flow'
        )

    def restore_body_normal_after_bar_flow(self):
        """限高杆流程结束后恢复 normal。"""
        if not getattr(self, 'restore_normal_after_bar_flow', True):
            return
        self.set_body_normal(
            do_stop=True,
            reason='bar_flow_finished',
            force=True
        )

    def restore_body_normal_after_obstacle_final_turn(self):
        """障碍物流程最终横向黄线 + 180 度掉头完成后恢复 normal。"""
        if not getattr(self, 'obstacle_restore_normal_after_final_turn', True):
            return
        self.set_body_normal(
            do_stop=True,
            reason='obstacle_final_180_turn_finished',
            force=True
        )

    def _inc_life_count(self):
        self.msg.life_count += 1
        if self.msg.life_count > 127:
            self.msg.life_count = 0

    def send_motion_cmd(self, vx: float, vy: float, wz: float):
        self.motion_cmd = (vx, vy, wz)

        self.msg.mode = 11
        self.msg.gait_id = 3
        self._inc_life_count()
        self.msg.vel_des = [float(vx), float(vy), float(wz)]
        self.msg.step_height = [0.02, 0.02]
        self.msg.rpy_des = [0.0, 0.0, 0.0]

        self.Ctrl.Send_cmd(self.msg)

    def stop(self):
        self.motion_cmd = (0.0, 0.0, 0.0)

        self.msg.mode = 12
        self.msg.gait_id = 0
        self._inc_life_count()
        self.Ctrl.Send_cmd(self.msg)
        self.get_logger().info('[CMD] STOP', throttle_duration_sec=1.0)
        self.Ctrl.Wait_finish(12, 0)

    def send_left_jump_action_once(self):
        """执行一次原地左跳，然后 recovery stand。"""
        self.motion_cmd = (0.0, 0.0, 0.0)

        self.msg.mode = 16
        self.msg.gait_id = 0
        self._inc_life_count()
        self.Ctrl.Send_cmd(self.msg)
        self.get_logger().info('[LEFT_JUMP] send mode=16 gait=0')
        self.Ctrl.Wait_finish(16, 0)

        self.msg.mode = 12
        self.msg.gait_id = 0
        self._inc_life_count()
        self.Ctrl.Send_cmd(self.msg)
        self.get_logger().info('[LEFT_JUMP] recovery stand')
        self.Ctrl.Wait_finish(12, 0)

    def send_right_jump_action_once(self):
        """执行一次原地右跳，然后 recovery stand。

        注意：这里按你给出的右跳控制代码实现，mode=16/gait_id=0。
        如果实测发现它仍然是左跳，需要在底层 gait/action 参数里替换成真正右跳动作。
        """
        self.motion_cmd = (0.0, 0.0, 0.0)

        self.msg.mode = 16
        self.msg.gait_id = 3
        self._inc_life_count()
        self.Ctrl.Send_cmd(self.msg)
        self.get_logger().info('[RIGHT_JUMP] send mode=16 gait=0')
        self.Ctrl.Wait_finish(16, 3)

        self.msg.mode = 12
        self.msg.gait_id = 0
        self._inc_life_count()
        self.Ctrl.Send_cmd(self.msg)
        self.get_logger().info('[RIGHT_JUMP] recovery stand')
        self.Ctrl.Wait_finish(12, 0)

    def execute_timed_turn_by_jump_count(self, jump_count: int, next_state: str, direction: int, label: str):
        """
        用固定角速度 + 固定仿真时间代替原来的旋转跳。

        direction: +1 表示左转，-1 表示右转。
        jump_count=1 使用 90 度参数；jump_count=2 使用 180 度参数。
        其他次数作为兜底：按 jump_count * 90 度时间执行。
        """
        count = max(0, int(jump_count))
        if count <= 0:
            self.get_logger().warn(f'[{label}] jump_count<=0, directly enter {next_state}')
            self.enter_state(next_state)
            return

        if count == 1:
            turn_name = '90'
            duration = self.p4_timed_turn_duration_90_s
            base_wz = self.p4_timed_turn_wz_90
        elif count == 2:
            turn_name = '180'
            duration = self.p4_timed_turn_duration_180_s
            base_wz = self.p4_timed_turn_wz_180
        else:
            turn_name = f'{count}x90'
            duration = count * self.p4_timed_turn_duration_90_s
            base_wz = self.p4_timed_turn_wz_90

        wz = float(direction) * abs(base_wz)
        elapsed = self.state_elapsed_s()

        if elapsed >= duration:
            self.get_logger().info(
                f'[{label}] timed turn done: count={count}, turn={turn_name}, '
                f'elapsed={elapsed:.3f}/{duration:.3f}s, next={next_state}'
            )
            self.enter_state(next_state)
            return

        self.send_motion_cmd(0.0, 0.0, wz)
        self.get_logger().info(
            f'[{label}] timed turn running: count={count}, turn={turn_name}, '
            f'wz={wz:.3f}, elapsed={elapsed:.3f}/{duration:.3f}s',
            throttle_duration_sec=0.2
        )

    def execute_left_jump_turn(self, jump_count: int, next_state: str):
        """原来的左跳转向入口：现在改为固定角速度左转固定仿真时间。"""
        self.execute_timed_turn_by_jump_count(
            jump_count=jump_count,
            next_state=next_state,
            direction=+1,
            label='TIMED_LEFT_TURN'
        )

    def execute_right_jump_turn(self, jump_count: int, next_state: str):
        """原来的右跳转向入口：现在改为固定角速度右转固定仿真时间。"""
        self.execute_timed_turn_by_jump_count(
            jump_count=jump_count,
            next_state=next_state,
            direction=-1,
            label='TIMED_RIGHT_TURN'
        )

    def get_all_state_names(self) -> List[str]:
        """返回所有允许作为 initial_state 的状态名。"""
        return [
            self.GLOBAL_INITIAL_LATERAL_SHIFT,
            self.GLOBAL_LATERAL_SEARCH,
            self.GLOBAL_CENTER_BAR,
            self.GLOBAL_CENTER_OBSTACLE,
            self.GLOBAL_SHIFT_AFTER_SUBTASK,
            self.BAR_FORWARD_UNDER,
            self.BAR_SEARCH_TARGET,
            self.BAR_APPROACH_TARGET,
            self.BAR_HIT_TARGET,
            self.BAR_BACKOFF_TO_BAR,
            self.OBSTACLE_FLOW_DONE,
            self.APPROACH_OBSTACLES,
            self.DASH_PRE_SIDE_SHIFT,
            self.ALIGN_DASHED_LINE,
            self.FOLLOW_DASHED_UNTIL_LOST,
            self.POST_DASH_FORWARD,
            self.POST_DASH_TURN_1,
            self.POST_TURN_FORWARD,
            self.POST_DASH_TURN_2,
            self.SEARCH_TARGET_AFTER_TURNS,
            self.APPROACH_AND_ALIGN_TARGET,
            self.HIT_TARGET,
            self.HIT_BACKOFF_AFTER_HIT,
            self.POST_HIT_LEFT_JUMP,
            self.APPROACH_SELECTED_OBSTACLE_AFTER_HIT,
            self.POST_HIT_OBS_TURN_1,
            self.POST_HIT_OBS_FORWARD,
            self.POST_HIT_OBS_TURN_2,
            self.POST_HIT_PRE_FINAL_FORWARD,
            self.POST_HIT_FINAL_FORWARD,
            self.FINAL_LEFT_JUMP,
            self.OBSTACLE_RESTORE_NORMAL_AFTER_FINAL_TURN,
            self.GLOBAL_FINAL_RIGHT_JUMP,
            self.GLOBAL_FINAL_YELLOW_FORWARD,
            self.GLOBAL_FINAL_LEFT_JUMP,
            self.GLOBAL_FINAL_P3_ALIGN,
            self.GLOBAL_FINAL_RIGHT_SHIFT_AFTER_LEFT_JUMP,
            self.DONE,
        ]

    def enter_initial_state(self):
        """
        根据 initial_state 参数进入指定初始状态，方便单独调试某一段流程。

        说明：
        1. 正常跑完整流程时，initial_state 保持默认 APPROACH_OBSTACLES。
        2. 如果直接从依赖 dashed_side 的状态开始，建议同时传入：
           -p debug_dashed_side:=left
           或
           -p debug_dashed_side:=right
        """
        valid_states = self.get_all_state_names()

        if self.p4_initial_state not in valid_states:
            self.get_logger().warn(
                f"[INIT_STATE] invalid initial_state='{self.p4_initial_state}', "
                f"fallback to {self.GLOBAL_INITIAL_LATERAL_SHIFT}. "
                f"valid_states={valid_states}"
            )
            self.p4_initial_state = self.GLOBAL_INITIAL_LATERAL_SHIFT

        if self.debug_dashed_side in ('left', 'right'):
            self.dashed_side = self.debug_dashed_side
            self.get_logger().info(f"[INIT_STATE] debug_dashed_side={self.dashed_side}")
        elif self.debug_dashed_side != 'auto':
            self.get_logger().warn(
                f"[INIT_STATE] invalid debug_dashed_side='{self.debug_dashed_side}', use auto. "
                "valid values: auto / left / right"
            )
            self.debug_dashed_side = 'auto'

        states_need_dashed_side = [
            self.DASH_PRE_SIDE_SHIFT,
            self.POST_DASH_TURN_1,
            self.POST_DASH_TURN_2,
            self.APPROACH_SELECTED_OBSTACLE_AFTER_HIT,
        ]
        if self.p4_initial_state in states_need_dashed_side and self.dashed_side not in ('left', 'right'):
            self.get_logger().warn(
                f"[INIT_STATE] {self.p4_initial_state} needs dashed_side, "
                "but dashed_side is None. You can run with: "
                "-p debug_dashed_side:=left or -p debug_dashed_side:=right"
            )

        self.get_logger().info(f"[INIT_STATE] start from: {self.p4_initial_state}")
        self.enter_state(self.p4_initial_state)

    def enter_state(self, new_state: str):
        self.state = new_state
        # 不在这里直接记录 now_s()。
        # 使用 Gazebo /clock 时，enter_state 可能发生在 /clock 首帧到达前，
        # 这会导致 state_enter_time=0，后续 elapsed 异常变大。
        # 让 state_elapsed_s() 在 control_loop 第一次真正执行时再开始计时。
        self.state_enter_time = None

        self.get_logger().info(f'ENTER STATE -> {new_state}')

        if new_state == self.GLOBAL_INITIAL_LATERAL_SHIFT:
            pass

        if new_state == self.GLOBAL_LATERAL_SEARCH:
            self.current_global_target = None
            self.global_center_stable_count = 0
            self.bar_center_stable_count = 0

        if new_state == self.GLOBAL_CENTER_BAR:
            self.global_center_stable_count = 0
            self.bar_center_stable_count = 0

        if new_state == self.GLOBAL_CENTER_OBSTACLE:
            self.global_center_stable_count = 0

        if new_state == self.GLOBAL_SHIFT_AFTER_SUBTASK:
            pass

        if new_state == self.BAR_FORWARD_UNDER:
            self.bar_center_stable_count = 0
            self.target_stable_count = 0
            self.stable_target_type = None
            self.locked_target = None
            self.latest_target = None
            # 注意：bar_return_target_depth_m 在 GLOBAL_CENTER_BAR 居中稳定时记录，不能清空

        if new_state == self.BAR_SEARCH_TARGET:
            self.target_stable_count = 0
            self.stable_target_type = None
            self.locked_target = None
            self.latest_target = None

        if new_state == self.BAR_APPROACH_TARGET:
            self.bar_hit_start_pose = None

        if new_state == self.BAR_HIT_TARGET:
            pass

        if new_state == self.BAR_BACKOFF_TO_BAR:
            self.bar_backoff_start_time = self.now_s()

        if new_state == self.OBSTACLE_FLOW_DONE:
            self.send_motion_cmd(0.0, 0.0, 0.0)

        if new_state == self.DASH_PRE_SIDE_SHIFT:
            self.dashed_pre_shift_start_pose = None
            self.dashed_pre_shift_dir_sign = self.get_pre_shift_dir_sign()
            self.dashed_center_count = 0
            self.dashed_lost_count = 0
            self.get_logger().info(
                f'[DASH_PRE_SHIFT] enter: side={self.dashed_side}, '
                f'dir_sign={self.dashed_pre_shift_dir_sign:.1f}, '
                f'duration={self.dashed_pre_shift_duration_s:.3f}s'
            )

        if new_state == self.ALIGN_DASHED_LINE:
            self.dashed_center_count = 0
            self.dashed_lost_count = 0

        if new_state == self.FOLLOW_DASHED_UNTIL_LOST:
            self.dashed_lost_count = 0

        if new_state == self.POST_DASH_FORWARD:
            pass

        if new_state == self.POST_DASH_TURN_1:
            self.turn_start_yaw = None
            self.current_turn_dir = self.get_first_turn_dir()
            self.current_turn_duration_s = self.post_dash_turn_duration_s
            self.current_turn_wz = self.post_dash_turn_wz

        if new_state == self.POST_TURN_FORWARD:
            pass

        if new_state == self.POST_DASH_TURN_2:
            self.turn_start_yaw = None
            self.current_turn_dir = -self.get_first_turn_dir()
            self.current_turn_duration_s = self.post_second_turn_duration_s
            self.current_turn_wz = self.post_second_turn_wz

        if new_state == self.SEARCH_TARGET_AFTER_TURNS:
            self.latest_target = None
            self.locked_target = None
            self.target_stable_count = 0
            self.stable_target_type = None
            self.hit_start_pose = None

        if new_state == self.APPROACH_AND_ALIGN_TARGET:
            self.hit_start_pose = None

        if new_state == self.HIT_TARGET:
            self.hit_start_pose = None
            self.get_logger().info(
                f'[HIT] start by sim time, target={self.locked_target.det_type if self.locked_target else None}')

        if new_state == self.HIT_BACKOFF_AFTER_HIT:
            self.after_hit_backoff_start_pose = None
            self.selected_obstacle_after_hit = None

        if new_state == self.POST_HIT_LEFT_JUMP:
            self.selected_obstacle_after_hit = None

        if new_state == self.APPROACH_SELECTED_OBSTACLE_AFTER_HIT:
            self.selected_obstacle_after_hit = None
            self.selected_obstacle_after_hit_side = None

        if new_state == self.POST_HIT_OBS_TURN_1:
            self.turn_start_yaw = None
            self.current_turn_dir = self.get_post_hit_obs_first_turn_dir()
            self.current_turn_duration_s = self.post_hit_obs_turn_duration_s
            self.current_turn_wz = self.post_hit_obs_turn_wz
            self.get_logger().warn(
                f'[POST_HIT_OBS_TURN_1] use dashed_side to turn by sim time: '
                f'dashed_side={self.dashed_side}, turn_dir={self.current_turn_dir}, '
                f'duration={self.current_turn_duration_s:.3f}s, '
                f'wz_cmd={self.current_turn_dir * abs(self.current_turn_wz):.3f}'
            )

        if new_state == self.POST_HIT_OBS_FORWARD:
            pass

        if new_state == self.POST_HIT_OBS_TURN_2:
            self.turn_start_yaw = None
            self.current_turn_dir = -self.get_post_hit_obs_first_turn_dir()
            self.current_turn_duration_s = self.post_hit_obs_turn_duration_s
            self.current_turn_wz = self.post_hit_obs_turn_wz
            self.get_logger().warn(
                f'[POST_HIT_OBS_TURN_2] reverse first turn by sim time: '
                f'dashed_side={self.dashed_side}, turn_dir={self.current_turn_dir}, '
                f'duration={self.current_turn_duration_s:.3f}s, '
                f'wz_cmd={self.current_turn_dir * abs(self.current_turn_wz):.3f}'
            )

        if new_state == self.POST_HIT_PRE_FINAL_FORWARD:
            # 第二次转回后，先按仿真时间向前走一段。
            self.post_hit_pre_final_forward_start_pose = None
            self.post_hit_final_forward_start_pose = None

        if new_state == self.POST_HIT_FINAL_FORWARD:
            # 固定距离前进完成后，再进入 RGB 横向黄线识别和朝向修正。
            # 新逻辑：先等黄线底部到达下方阈值，再继续前进等黄线从画面中消失。
            self.final_yellow_done_counter = 0
            self.final_yellow_reached_lower_area = False
            self.final_yellow_disappear_counter = 0
            self.latest_final_yellow_line = None

        if new_state == self.FINAL_LEFT_JUMP:
            # 障碍物流程内部的最终掉头动作，这里只负责进入状态；动作在 control_loop 中执行。
            self.send_motion_cmd(0.0, 0.0, 0.0)

        if new_state == self.OBSTACLE_RESTORE_NORMAL_AFTER_FINAL_TURN:
            # 最终掉头完成后才恢复 normal。恢复姿态函数内部会 STOP。
            self.send_motion_cmd(0.0, 0.0, 0.0)

        if new_state == self.GLOBAL_FINAL_RIGHT_JUMP:
            # 全部流程完成后的最终右跳，动作在 control_loop 中执行。
            self.send_motion_cmd(0.0, 0.0, 0.0)

        if new_state == self.GLOBAL_FINAL_YELLOW_FORWARD:
            # 右跳后，开始识别前方横向黄线并修正朝向。
            # 新逻辑：黄线到达下方阈值后继续前进，直到黄线消失再停。
            self.global_final_yellow_done_counter = 0
            self.global_final_yellow_reached_lower_area = False
            self.global_final_yellow_disappear_counter = 0
            self.latest_final_yellow_line = None

        if new_state == self.GLOBAL_FINAL_LEFT_JUMP:
            # 最终左跳一次，动作在 control_loop 中执行。
            self.send_motion_cmd(0.0, 0.0, 0.0)

        if new_state == self.GLOBAL_FINAL_P3_ALIGN:
            # 第四赛段最后左跳后，复用第三赛段结束的黄线近/远中心矫正逻辑。
            # 清空 P3 视觉缓存，避免刚切入时使用旧帧误差。
            self.p3_s4_lat = 0.0
            self.p3_s4_yaw = 0.0
            self.p3_s4_valid = 0.0
            self.p3_align_near_center = -1.0
            self.p3_align_far_center = -1.0
            self.send_motion_cmd(0.0, 0.0, 0.0)

        if new_state == self.DONE:
            self.task_done_stop_sent = False

    # ---------- TF 工具 ----------
    def normalize_angle(self, angle: float) -> float:
        """把角度归一化到 [-pi, pi]。"""
        return math.atan2(math.sin(angle), math.cos(angle))

    def quaternion_to_yaw(self, q) -> float:
        """四元数转 yaw，不依赖 tf_transformations。"""
        x = q.x
        y = q.y
        z = q.z
        w = q.w
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_current_pose_2d(self):
        """
        从 TF 获取当前平面位姿。
        返回 (x, y, yaw)，失败返回 None。
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                self.tf_parent_frame,
                self.tf_child_frame,
                rclpy.time.Time()
            )
            x = float(tf.transform.translation.x)
            y = float(tf.transform.translation.y)
            yaw = self.quaternion_to_yaw(tf.transform.rotation)
            return (x, y, yaw)
        except Exception as e:
            self.get_logger().warn(
                f'[TF] lookup {self.tf_parent_frame}->{self.tf_child_frame} failed: {e}',
                throttle_duration_sec=1.0
            )
            return None

    def get_current_yaw(self):
        pose = self.get_current_pose_2d()
        if pose is None:
            return None
        return pose[2]

    def distance_from_pose(self, start_pose) -> Optional[float]:
        """计算当前 TF 位置相对 start_pose 的平面位移距离。"""
        if start_pose is None:
            return None

        cur = self.get_current_pose_2d()
        if cur is None:
            return None

        dx = cur[0] - start_pose[0]
        dy = cur[1] - start_pose[1]
        return math.sqrt(dx * dx + dy * dy)

    def get_first_turn_dir(self) -> int:
        """
        虚线在左边：第一次右转，返回 -1。
        虚线在右边：第一次左转，返回 +1。

        当前假设 wz > 0 是左转，wz < 0 是右转。
        如果实测方向反了，把这里的返回值正负号对调。
        """
        if self.dashed_side == 'left':
            return -1
        if self.dashed_side == 'right':
            return 1
        return 1

    def turn_finished_by_tf(self) -> bool:
        """用 TF yaw 判断当前转向是否达到目标角度。"""
        if self.turn_start_yaw is None:
            return False

        cur_yaw = self.get_current_yaw()
        if cur_yaw is None:
            return False

        signed_delta = self.normalize_angle(cur_yaw - self.turn_start_yaw)
        target = self.current_turn_angle_rad
        tol = self.current_turn_tolerance_rad

        if self.current_turn_dir > 0:
            done = signed_delta >= (target - tol)
        else:
            done = signed_delta <= -(target - tol)

        self.get_logger().info(
            f'[TF_TURN] dir={self.current_turn_dir}, '
            f'delta_deg={math.degrees(signed_delta):.1f}, '
            f'target_deg={math.degrees(target):.1f}, done={done}',
            throttle_duration_sec=0.2
        )
        return done

    # ---------- 对齐计算 ----------
    def choose_obstacle_pair(self, candidates: List[Detection]) -> Optional[Tuple[Detection, Detection]]:
        if len(candidates) < 2:
            return None

        # 先取面积较大的两个，避免小噪声影响
        top = sorted(
            candidates,
            key=lambda d: d.extra.get('area', 0.0),
            reverse=True
        )[:2]

        left, right = sorted(top, key=lambda d: d.center_img[0])
        return left, right

    def compute_obstacle_mid_align_vy(self, left: Detection, right: Detection) -> float:
        img_center_x = self.latest_bgr.shape[1] / 2.0
        pair_center_x = (left.center_img[0] + right.center_img[0]) / 2.0

        err_px = pair_center_x - img_center_x

        if abs(err_px) <= self.obstacle_center_px_deadband:
            return 0.0

        err_norm = err_px / max(img_center_x, 1.0)

        # 如果方向反了，把这里的负号改成正号
        vy = -self.obstacle_align_vy_k * err_norm

        vy = float(np.clip(vy, -self.obstacle_align_vy_max, self.obstacle_align_vy_max))

        if 0.0 < abs(vy) < self.obstacle_align_vy_min:
            vy = math.copysign(self.obstacle_align_vy_min, vy)

        return vy

    def choose_selected_obstacle_after_hit(self, candidates: List[Detection]) -> Optional[Detection]:
        """
        左跳后再次前进识别障碍物时，只选择一个障碍物做居中对齐。

        规则：
        - 之前记录虚线在左边：选择右边障碍物居中
        - 之前记录虚线在右边：选择左边障碍物居中
        """
        pair = self.choose_obstacle_pair(candidates)
        if pair is None:
            return None

        left, right = pair
        if self.dashed_side == 'left':
            return right
        if self.dashed_side == 'right':
            return left

        # 兜底：如果没有记录虚线侧，就选更靠近图像中心的那个
        img_center_x = self.latest_bgr.shape[1] / 2.0
        return min(pair, key=lambda d: abs(d.center_img[0] - img_center_x))

    def get_obstacle_side_in_pair(self, obstacle: Detection, candidates: List[Detection]) -> str:
        """判断当前选中的障碍物是两障碍物中的左边还是右边。"""
        pair = self.choose_obstacle_pair(candidates)
        if pair is not None:
            left, right = pair
            if obstacle.center_img[0] <= left.center_img[0]:
                return 'left'
            if obstacle.center_img[0] >= right.center_img[0]:
                return 'right'

        img_center_x = self.latest_bgr.shape[1] / 2.0
        return 'left' if obstacle.center_img[0] < img_center_x else 'right'

    def get_post_hit_obs_first_turn_dir(self) -> int:
        """
        撞击后到达蓝色障碍物距离阈值后的第一次转向方向。

        现在不再根据 selected_obstacle_after_hit_side 判断，
        而是直接根据前面识别到的黄色竖直虚线方向 dashed_side 判断：

        - dashed_side == 'left'
            说明之前虚线在左边，流程里会偏向右侧障碍物路线，
            第一次转向按右转处理，返回 -1。

        - dashed_side == 'right'
            说明之前虚线在右边，流程里会偏向左侧障碍物路线，
            第一次转向按左转处理，返回 +1。

        当前假设 wz > 0 是左转，wz < 0 是右转。
        如果实测方向反了，只需要把这里的 +1 / -1 对调。
        """
        if self.dashed_side == 'right':
            return 1
        if self.dashed_side == 'left':
            return -1

        self.get_logger().warn(
            '[POST_HIT_OBS] dashed_side is None, fallback first turn LEFT',
            throttle_duration_sec=1.0
        )
        return 1

    def compute_selected_obstacle_align_vy_after_hit(self, obstacle: Detection) -> float:
        """左跳后，对单个选中的蓝色障碍物做居中对齐。"""
        img_center_x = self.latest_bgr.shape[1] / 2.0
        err_px = obstacle.center_img[0] - img_center_x

        if abs(err_px) <= self.post_hit_obstacle_center_px_deadband:
            return 0.0

        err_norm = err_px / max(img_center_x, 1.0)

        # 如果方向反了，把这里的负号改成正号
        vy = -self.post_hit_obstacle_align_vy_k * err_norm
        vy = float(np.clip(vy, -self.post_hit_obstacle_align_vy_max, self.post_hit_obstacle_align_vy_max))

        if 0.0 < abs(vy) < self.post_hit_obstacle_align_vy_min:
            vy = math.copysign(self.post_hit_obstacle_align_vy_min, vy)

        return vy

    def get_dashed_side(self, dashed: Detection) -> str:
        img_center_x = self.latest_bgr.shape[1] / 2.0
        return 'left' if dashed.center_img[0] < img_center_x else 'right'

    def get_forced_dashed_side(self) -> Optional[str]:
        """
        debug_dashed_side 控制规则：
        - left/right：强制使用这个方向
        - auto：不强制，交给视觉判断
        """
        side = str(getattr(self, 'debug_dashed_side', 'auto')).strip().lower()
        if side in ('left', 'right'):
            return side
        return None

    def get_dashed_target_x(self) -> float:
        img_center_x = self.latest_bgr.shape[1] / 2.0
        if self.dashed_side == 'left':
            return img_center_x + self.dashed_target_offset_px
        if self.dashed_side == 'right':
            return img_center_x - self.dashed_target_offset_px
        return img_center_x

    def is_dashed_valid_for_follow(self, dashed: Optional[Detection]) -> bool:
        """
        FOLLOW_DASHED_UNTIL_LOST 阶段专用判断。

        对齐虚线完成后，机器狗沿虚线向前走。
        此时不能只要检测器检测到黄色虚线就认为还在跟随。

        只有当前检测到的虚线中心 x 落在“对齐目标线 target_x”附近一定范围内，
        才认为这条虚线还是当前正在跟随的那条线。

        如果虚线中心偏离 target_x 太远：
            - 不参与跟随修正；
            - 按 dashed lost 处理；
            - 连续 lost 若干帧后进入 POST_DASH_FORWARD。
        """
        if dashed is None:
            return False

        if self.latest_bgr is None:
            return False

        target_x = self.get_dashed_target_x()
        cx = float(dashed.center_img[0])
        valid_range = float(self.follow_dashed_valid_x_range_px)

        err_px = cx - target_x
        return abs(err_px) <= valid_range

    def get_pre_shift_dir_sign(self) -> float:
        """
        DASH_PRE_SIDE_SHIFT 的横移方向符号。

        设计意图：
          dashed_side == 'left'  -> 朝左侧虚线方向横移
          dashed_side == 'right' -> 朝右侧虚线方向横移

        注意：如果实测方向反了，只需要把这里的 1.0 和 -1.0 对调。
        """
        if self.dashed_side == 'left':
            return 1.0
        if self.dashed_side == 'right':
            return -1.0
        return 0.0

    def get_pre_shift_vy(self) -> float:
        return self.get_pre_shift_dir_sign() * abs(self.dashed_pre_shift_speed)

    def get_local_lateral_displacement_from_start(self, start_pose, current_pose) -> float:
        """
        计算 current_pose 相对 start_pose 的横向位移。

        start_pose/current_pose 格式为 (x, y, yaw)。
        返回的是以 start_pose 的 yaw 为基准的侧向位移，避免把前后漂移算进预横移距离。
        """
        if start_pose is None or current_pose is None:
            return 0.0

        sx, sy, syaw = start_pose
        cx, cy, _ = current_pose

        dx = cx - sx
        dy = cy - sy

        # 起始朝向左侧法向量：(-sin(yaw), cos(yaw))
        lateral = -math.sin(syaw) * dx + math.cos(syaw) * dy
        return float(lateral)

    def compute_dashed_align_vy(
            self,
            dashed: Detection,
            k: Optional[float] = None,
            vy_max: Optional[float] = None,
            vy_min: Optional[float] = None,
    ) -> float:
        if k is None:
            k = self.dashed_align_vy_k
        if vy_max is None:
            vy_max = self.dashed_align_vy_max
        if vy_min is None:
            vy_min = self.dashed_align_vy_min

        # 不再对齐图像中心，而是对齐偏置目标点
        target_x = self.get_dashed_target_x()
        err_px = dashed.center_img[0] - target_x

        if abs(err_px) <= self.dashed_center_px_deadband:
            return 0.0

        img_center_x = self.latest_bgr.shape[1] / 2.0
        err_norm = err_px / max(img_center_x, 1.0)

        # 如果方向反了，把这里的负号改成正号
        vy = -float(k) * err_norm

        vy = float(np.clip(vy, -float(vy_max), float(vy_max)))

        if 0.0 < abs(vy) < float(vy_min):
            vy = math.copysign(float(vy_min), vy)

        return vy

    def is_dashed_centered(self, dashed: Detection) -> bool:
        target_x = self.get_dashed_target_x()
        err_px = dashed.center_img[0] - target_x
        return abs(err_px) <= self.dashed_center_px_deadband

    def compute_final_yellow_wz(self, yellow_line: Optional[Detection]) -> float:
        """
        根据前方横向黄线倾斜角修正 yaw。
        angle_deg 来自图像坐标系中的黄线斜率，0 度表示水平。

        当前符号：wz = -k * angle。
        如果实测发现越修越歪，把下面的负号改成正号即可。
        """
        if yellow_line is None:
            return 0.0

        angle_deg = float(yellow_line.extra.get('angle_deg', 0.0))
        if abs(angle_deg) <= self.final_yellow_tilt_deadband_deg:
            return 0.0

        angle_rad = math.radians(angle_deg)
        wz = -self.final_yellow_align_wz_k * angle_rad
        wz = float(np.clip(wz, -self.final_yellow_align_wz_max, self.final_yellow_align_wz_max))

        if 0.0 < abs(wz) < self.final_yellow_align_wz_min:
            wz = math.copysign(self.final_yellow_align_wz_min, wz)

        return wz

    def get_global_final_yellow_forward_speed(self, final_yellow_line: Optional[Detection]) -> float:
        """
        全局最终横向黄线阶段的前进速度选择。

        逻辑：
        1. 没看到黄线时：使用快速速度 global_final_yellow_forward_speed
        2. 看到黄线但还没到减速阈值：继续快速
        3. 黄线 bottom_ratio >= global_final_yellow_slow_start_ratio：切换慢速

        bottom_ratio 含义：
            bottom_ratio = 黄线检测框底部 y 坐标 / 图像高度

        例如图像高度 480，slow_start_ratio=0.85：
            480 * 0.85 = 408
        也就是横向黄线底部到达 y=408 附近后开始减速。
        """
        if final_yellow_line is None:
            return self.global_final_yellow_forward_speed

        bottom_ratio = float(final_yellow_line.extra.get('bottom_ratio', 0.0))

        if bottom_ratio >= self.global_final_yellow_slow_start_ratio:
            return self.global_final_yellow_slow_forward_speed

        return self.global_final_yellow_forward_speed

    # ---------- 第二次转向后的目标检测 / 对齐 / 撞击 ----------
    def detect_all_targets(self, frame_bgr) -> List[Detection]:
        """
        复用限高杆任务代码的目标检测逻辑：
        同时检测蓝球、白球、可乐，返回所有检测到的目标。
        """
        detections = []

        for detector in [
            self.blue_ball_detector,
            self.white_ball_detector,
            self.cola_detector,
        ]:
            det = detector.detect(frame_bgr)
            if det is not None:
                detections.append(det)

        return detections

    def choose_best_target(self, candidates: List[Detection]) -> Optional[Detection]:
        """
        和限高杆代码一样，优先选择最靠近图像中心的目标。
        """
        if not candidates or self.latest_bgr is None:
            return None

        img_center_x = self.latest_bgr.shape[1] / 2.0
        return min(candidates, key=lambda c: abs(c.center_img[0] - img_center_x))

    def compute_target_align_cmd(self, target: Detection):
        """
        复用限高杆代码中的目标接近对齐逻辑：
        - 根据目标中心 x 偏差算横移 vy
        - 根据目标深度选择远近前进速度
        """
        if self.latest_bgr is None:
            return 0.0, 0.0

        depth_m = self.estimate_depth_at_center(target.center_img)

        img_center_x = self.latest_bgr.shape[1] / 2.0
        err_px = target.center_img[0] - img_center_x

        if depth_m is not None and depth_m < 0.35:
            vx = self.align_forward_speed_near
        else:
            vx = self.align_forward_speed_far

        if abs(err_px) <= self.center_px_deadband:
            vy = 0.0
        else:
            err_norm = err_px / max(img_center_x, 1.0)

            # 和你前面所有视觉对齐一致：如果方向反了，把负号改成正号
            vy = -self.align_vy_k * err_norm
            vy = float(np.clip(vy, -self.align_vy_max, self.align_vy_max))

            if 0.0 < abs(vy) < self.align_vy_min:
                vy = math.copysign(self.align_vy_min, vy)

        return vx, vy

    # ---------- 主循环 ----------
    def fourth_control_loop(self):
        if self.latest_bgr is None or self.latest_depth is None:
            return

        if self.latest_bgr.shape[:2] != self.latest_depth.shape[:2]:
            self.get_logger().warn(
                f'RGB size {self.latest_bgr.shape[:2]} != DEPTH size {self.latest_depth.shape[:2]}',
                throttle_duration_sec=1.0
            )
            return

        frame = self.latest_bgr
        now = self.now_s()

        # 障碍物只需要完成一次；完成后在全局搜索阶段不再检测/触发障碍物。
        # 但如果已经进入障碍物子流程内部，则仍需要继续检测障碍物和虚线。
        if (self.completed_obstacle_count < self.required_obstacle_count) or self.is_obstacle_flow_state():
            obstacle_result = self.obstacle_detector.detect(frame, self.latest_depth)
            obstacle_candidates = obstacle_result['candidates']
        else:
            obstacle_candidates = []

        if self.is_obstacle_flow_state():
            dashed_lines = self.dashed_detector.detect_top_dashed_lines(frame)
            dashed = dashed_lines[0] if dashed_lines else None
        else:
            dashed = None

        chosen_pair = None
        target_candidates_for_vis: List[Detection] = []
        chosen_target_for_vis: Optional[Detection] = None
        final_yellow_line: Optional[Detection] = None
        bar_for_vis: Optional[Detection] = None

        if self.state == self.GLOBAL_INITIAL_LATERAL_SHIFT:
            # 启动预左移：只按固定仿真时间横移，不做任何目标识别。
            elapsed = self.state_elapsed_s()
            if self.global_initial_lateral_shift_duration_s <= 0.0:
                self.get_logger().info('[GLOBAL_INIT_SHIFT] duration <= 0, skip initial shift')
                self.enter_state(self.GLOBAL_LATERAL_SEARCH)
                return

            if elapsed >= self.global_initial_lateral_shift_duration_s:
                self.get_logger().info(
                    f'[GLOBAL_INIT_SHIFT] finished: elapsed={elapsed:.3f}/{self.global_initial_lateral_shift_duration_s:.3f}s, '
                    'enter GLOBAL_LATERAL_SEARCH'
                )
                self.enter_state(self.GLOBAL_LATERAL_SEARCH)
                return

            self.send_motion_cmd(0.0, self.global_initial_lateral_shift_vy, 0.0)
            self.get_logger().info(
                f'[GLOBAL_INIT_SHIFT] shifting left by sim time: elapsed={elapsed:.3f}/{self.global_initial_lateral_shift_duration_s:.3f}s, '
                f'vy={self.global_initial_lateral_shift_vy:.3f}',
                throttle_duration_sec=0.3
            )
            return


        elif self.state == self.GLOBAL_LATERAL_SEARCH:
            if self.all_global_tasks_done():
                self.enter_global_final_sequence()
                return

            # 只检测还没有完成的任务类型：
            # - 限高杆已经完成 required_bar_count 次后，后续不再识别/触发限高杆；
            # - 障碍物已经完成 required_obstacle_count 次后，后续不再识别/触发障碍物。
            need_bar = self.completed_bar_count < self.required_bar_count
            need_obstacle = self.completed_obstacle_count < self.required_obstacle_count

            bar = self.bar_detector.detect(frame) if need_bar else None
            bar_for_vis = bar

            obs_for_choice = obstacle_candidates if need_obstacle else []
            obj_type, det = self.choose_global_object(bar, obs_for_choice)

            if obj_type == 'bar':
                self.current_global_target = det
                self.get_logger().info(
                    f'[GLOBAL_SEARCH] detect BAR {self.completed_bar_count + 1}/{self.required_bar_count}, center={det.center_img}'
                )
                self.enter_state(self.GLOBAL_CENTER_BAR)
                return

            if obj_type == 'obstacle':
                self.current_global_target = det
                self.get_logger().info(
                    f'[GLOBAL_SEARCH] detect OBSTACLE {self.completed_obstacle_count + 1}/{self.required_obstacle_count}, center={det.center_img}'
                )
                self.enter_state(self.GLOBAL_CENTER_OBSTACLE)
                return

            self.send_motion_cmd(0.0, self.global_lateral_search_vy, 0.0)

        elif self.state == self.GLOBAL_CENTER_BAR:
            bar = self.bar_detector.detect(frame)
            bar_for_vis = bar

            if bar is None:
                self.global_center_stable_count = 0
                self.send_motion_cmd(0.0, self.global_lateral_search_vy, 0.0)

                self.get_logger().info(
                    '[GLOBAL_CENTER_BAR] bar lost, continue lateral search motion',
                    throttle_duration_sec=0.5
                )

            else:
                vy = self.compute_global_bar_center_fixed_vy(bar)
                wz = self.compute_bar_depth_yaw_align_wz(bar)
                self.send_motion_cmd(0.0, vy, wz)

                centered = self.is_bar_centered(
                    bar,
                    deadband_px=self.global_bar_center_px_deadband
                )

                if centered:
                    self.global_center_stable_count += 1
                else:
                    self.global_center_stable_count = 0

                img_center_x = self.latest_bgr.shape[1] / 2.0
                err_px = bar.center_img[0] - img_center_x

                self.get_logger().info(
                    f'[GLOBAL_CENTER_BAR] fixed-vy center align: '
                    f'center={bar.center_img}, err_px={err_px:.1f}, '
                    f'deadband={self.global_bar_center_px_deadband}, '
                    f'vy={vy:.3f}, wz={wz:.3f}, '
                    f'depth_yaw={self.latest_bar_depth_yaw_info}, '
                    f'stable={self.global_center_stable_count}/{self.global_center_stable_frames}',
                    throttle_duration_sec=0.2
                )

                if self.global_center_stable_count >= self.global_center_stable_frames:
                    d = self.estimate_bar_depth(bar)
                    self.bar_return_target_depth_m = d
                    self.current_bar_det = bar

                    self.get_logger().info(
                        f'[GLOBAL_CENTER_BAR] centered by fixed-vy, '
                        f'record bar depth={d}, enter BAR_FORWARD_UNDER'
                    )

                    self.enter_state(self.BAR_FORWARD_UNDER)
                    return

        elif self.state == self.GLOBAL_CENTER_OBSTACLE:
            if not obstacle_candidates:
                self.global_center_stable_count = 0
                self.send_motion_cmd(0.0, self.global_lateral_search_vy, 0.0)
                self.get_logger().info('[GLOBAL_CENTER_OBS] obstacle lost, continue lateral search motion',
                                       throttle_duration_sec=0.5)
            else:
                img_center_x = frame.shape[1] / 2.0
                obs = min(obstacle_candidates, key=lambda d: abs(d.center_img[0] - img_center_x))
                err_px = obs.center_img[0] - img_center_x
                if abs(err_px) <= self.obstacle_center_px_deadband:
                    vy = 0.0
                    self.global_center_stable_count += 1
                else:
                    err_norm = err_px / max(img_center_x, 1.0)
                    vy = -self.obstacle_align_vy_k * err_norm
                    vy = float(np.clip(vy, -self.obstacle_align_vy_max, self.obstacle_align_vy_max))
                    if 0.0 < abs(vy) < self.obstacle_align_vy_min:
                        vy = math.copysign(self.obstacle_align_vy_min, vy)
                    self.global_center_stable_count = 0
                self.send_motion_cmd(0.0, vy, 0.0)
                self.get_logger().info(
                    f'[GLOBAL_CENTER_OBS] center={obs.center_img}, vy={vy:.3f}, '
                    f'stable={self.global_center_stable_count}/{self.global_center_stable_frames}',
                    throttle_duration_sec=0.2
                )
                if self.global_center_stable_count >= self.global_center_stable_frames:
                    # 记录当前障碍物流程是不是全局第 3 个子流程。
                    # 例如 required_bar_count=2、required_obstacle_count=1 时：
                    # 两个限高杆都完成后才处理障碍物，则这个障碍物就是第 3 个物体。
                    flow_index = self.completed_bar_count + self.completed_obstacle_count + 1
                    total_required_flows = self.required_bar_count + self.required_obstacle_count
                    self.obstacle_flow_is_third_object = (flow_index >= total_required_flows)

                    self.get_logger().info(
                        f'[GLOBAL_CENTER_OBS] centered, enter obstacle flow, '
                        f'flow_index={flow_index}/{total_required_flows}, '
                        f'obstacle_flow_is_third_object={self.obstacle_flow_is_third_object}'
                    )
                    # 障碍物流程参数是按 low 姿态调好的：
                    # 先完成全局障碍物横向居中，再进入 low，随后进入正式障碍物流程。
                    self.set_body_low_for_obstacle_flow()
                    self.enter_state(self.APPROACH_OBSTACLES)
                    return

        elif self.state == self.GLOBAL_SHIFT_AFTER_SUBTASK:
            if self.all_global_tasks_done():
                self.enter_global_final_sequence()
                return

            elapsed = self.state_elapsed_s()
            target_duration = self.current_after_task_shift_duration_s if self.current_after_task_shift_duration_s > 0.0 else self.global_after_task_shift_duration_s

            if elapsed >= target_duration:
                self.get_logger().info(
                    f'[GLOBAL_SHIFT] finished: elapsed={elapsed:.3f}/{target_duration:.3f}s, '
                    f'reason={self.global_after_task_shift_reason}'
                )
                self.enter_state(self.GLOBAL_LATERAL_SEARCH)
                return

            self.send_motion_cmd(0.0, self.global_after_task_shift_vy, 0.0)
            self.get_logger().info(
                f'[GLOBAL_SHIFT] shifting left by sim time: elapsed={elapsed:.3f}/{target_duration:.3f}s, '
                f'reason={self.global_after_task_shift_reason}',
                throttle_duration_sec=0.3
            )


        elif self.state == self.BAR_FORWARD_UNDER:
            bar = self.bar_detector.detect(frame)
            bar_for_vis = bar
            vy = self.compute_bar_align_vy(bar) if bar is not None else 0.0
            wz = self.compute_bar_depth_yaw_align_wz(bar) if bar is not None else 0.0
            self.send_motion_cmd(self.bar_search_forward_speed, vy, wz)
            if bar is not None:
                d = self.estimate_bar_depth(bar)
                if d is not None and self.bar_return_target_depth_m is None:
                    self.bar_return_target_depth_m = d
                self.get_logger().info(
                    f'[BAR_FORWARD] depth={d}, target_depth={self.bar_return_target_depth_m}, '
                    f'vy={vy:.3f}, wz={wz:.3f}, depth_yaw={self.latest_bar_depth_yaw_info}',
                    throttle_duration_sec=0.2
                )
                if d is not None and d < self.bar_trigger_distance_m:
                    # 限高杆：靠近到规定距离、准备开始搜索目标物体时播报；
                    # 切 low 与播报共用同一个触发距离 bar_trigger_distance_m。
                    self.speak_bar_at_trigger()
                    self.set_body_low_for_bar_trigger()
                    self.enter_state(self.BAR_SEARCH_TARGET)
                    return

        elif self.state == self.BAR_SEARCH_TARGET:
            self.send_motion_cmd(self.target_search_forward_speed, 0.0, 0.0)
            target_candidates_for_vis = self.detect_all_targets(frame)
            target = self.choose_best_target(target_candidates_for_vis)
            chosen_target_for_vis = target
            if target is None:
                self.stable_target_type = None
                self.target_stable_count = 0
            else:
                if self.stable_target_type == target.det_type:
                    self.target_stable_count += 1
                else:
                    self.stable_target_type = target.det_type
                    self.target_stable_count = 1
                self.latest_target = target
                if self.target_stable_count >= self.target_stable_frames:
                    self.locked_target = target
                    self.enter_state(self.BAR_APPROACH_TARGET)
                    return

        elif self.state == self.BAR_APPROACH_TARGET:
            target_candidates_for_vis = self.detect_all_targets(frame)
            target = self.choose_best_target(target_candidates_for_vis)
            chosen_target_for_vis = target
            if target is None:
                self.locked_target = None
                self.enter_state(self.BAR_SEARCH_TARGET)
                return
            self.locked_target = target
            vx, vy = self.compute_target_align_cmd(target)
            self.send_motion_cmd(vx, vy, 0.0)
            d = self.estimate_depth_at_center(target.center_img)
            self.get_logger().info(
                f'[BAR_TARGET_ALIGN] target={target.det_type}, depth={d}, cmd=({vx:.3f},{vy:.3f},0)',
                throttle_duration_sec=0.2
            )
            if d is not None and d < self.hit_trigger_distance_m:
                # 目标物体：靠近到撞击距离、刚进入撞击状态前播报
                self.speak_target_at_hit_trigger(target.det_type)
                self.enter_state(self.BAR_HIT_TARGET)
                return

        elif self.state == self.BAR_HIT_TARGET:
            if self.locked_target is None:
                self.enter_state(self.BAR_SEARCH_TARGET)
                return

            chosen_target_for_vis = self.locked_target
            target_candidates_for_vis = [self.locked_target]
            params = self.hit_params.get(self.locked_target.det_type, {'speed': 0.20, 'duration_s': 0.85})
            elapsed = self.state_elapsed_s()
            duration = float(params.get('duration_s', 0.85))

            if elapsed >= duration:
                self.get_logger().info(
                    f'[BAR_HIT] finished by sim time: target={self.locked_target.det_type}, '
                    f'elapsed={elapsed:.3f}/{duration:.3f}s'
                )
                self.enter_state(self.BAR_BACKOFF_TO_BAR)
                return

            self.send_motion_cmd(params['speed'], 0.0, 0.0)
            self.get_logger().info(
                f'[BAR_HIT] target={self.locked_target.det_type}, elapsed={elapsed:.3f}/{duration:.3f}s',
                throttle_duration_sec=0.2
            )


        elif self.state == self.BAR_BACKOFF_TO_BAR:
            bar = self.bar_detector.detect(frame)
            bar_for_vis = bar

            if bar is None:
                # 看不到限高杆时，保持原来的固定后退，避免因为视觉丢失卡住。
                self.send_motion_cmd(-self.backoff_after_hit_speed, 0.0, 0.0)
                self.get_logger().info('[BAR_BACKOFF] bar=None, keep fixed backing', throttle_duration_sec=0.5)
            else:
                # 能看到限高杆时：
                # vx：用当前深度和返回目标深度做闭环；
                # vy：用限高杆中心误差做横向居中；
                # wz：继续复用左右深度差做朝向修正。
                d = self.estimate_bar_depth(bar)
                target_d = self.bar_return_target_depth_m
                vy = self.compute_bar_align_vy(bar)
                wz = self.compute_bar_depth_yaw_align_wz(bar)

                if d is not None and target_d is not None:
                    depth_err = d - target_d

                    if (self.state_elapsed_s() >= self.backoff_min_time_s and
                            abs(depth_err) <= self.backoff_bar_depth_tolerance_m):
                        self.send_motion_cmd(0.0, 0.0, 0.0)
                        self.get_logger().info(
                            f'[BAR_BACKOFF] reached target: depth={d:.3f}, target={target_d:.3f}, '
                            f'err={depth_err:.3f}, finish bar flow'
                        )
                        self.finish_bar_flow()
                        return

                    vx = self.compute_bar_backoff_vx_by_depth(d, target_d)
                    self.send_motion_cmd(vx, vy, wz)
                    self.get_logger().info(
                        f'[BAR_BACKOFF] depth={d:.3f}, target={target_d:.3f}, err={depth_err:.3f}, '
                        f'vx={vx:.3f}, vy={vy:.3f}, wz={wz:.3f}, '
                        f'depth_yaw={self.latest_bar_depth_yaw_info}',
                        throttle_duration_sec=0.2
                    )
                elif d is None:
                    # 看得到杆但深度不可用：仍然用中心做 vy、用默认速度后退。
                    self.send_motion_cmd(-self.backoff_after_hit_speed, vy, wz)
                    self.get_logger().info(
                        f'[BAR_BACKOFF] bar detected but depth=None, fixed backing with vy/wz: '
                        f'vy={vy:.3f}, wz={wz:.3f}',
                        throttle_duration_sec=0.5
                    )
                else:
                    # 没有返回目标深度，无法做 vx 深度闭环；保底固定后退，但保留 vy/wz。
                    self.send_motion_cmd(-self.backoff_after_hit_speed, vy, wz)
                    self.get_logger().warn(
                        f'[BAR_BACKOFF] return target depth is None, fixed backing with vy/wz: '
                        f'vy={vy:.3f}, wz={wz:.3f}',
                        throttle_duration_sec=1.0
                    )

        elif self.state == self.APPROACH_OBSTACLES:
            pair = self.choose_obstacle_pair(obstacle_candidates)
            chosen_pair = pair

            if pair is None:
                # 没有稳定看到两个障碍物，先慢速前进搜索
                self.send_motion_cmd(self.obstacle_search_forward_speed, 0.0, 0.0)
                self.get_logger().info(
                    '[OBS_ALIGN] no valid obstacle pair, searching forward',
                    throttle_duration_sec=0.5
                )
                return

            left, right = pair

            vy = self.compute_obstacle_mid_align_vy(left, right)

            d_left = left.extra.get('median_depth')
            d_right = right.extra.get('median_depth')

            depths = [d for d in [d_left, d_right] if d is not None]
            obstacle_dist = min(depths) if depths else None

            self.send_motion_cmd(self.obstacle_forward_speed, vy, 0.0)

            if obstacle_dist is None:
                self.get_logger().info(
                    f'[OBS_ALIGN] pair_center=({left.center_img[0]},{right.center_img[0]}), '
                    f'dist=None, trigger={self.obstacle_trigger_distance_m:.3f}, vy={vy:.3f}, keep approaching',
                    throttle_duration_sec=0.2
                )
                return

            self.get_logger().info(
                f'[OBS_ALIGN] pair_center=({left.center_img[0]},{right.center_img[0]}), '
                f'dist={obstacle_dist:.3f}, trigger={self.obstacle_trigger_distance_m:.3f}, '
                f'vy={vy:.3f}, keep approaching',
                throttle_duration_sec=0.2
            )

            if obstacle_dist > self.obstacle_trigger_distance_m:
                # 还没靠近到触发距离，继续靠近障碍物
                return

            # 只有距离真正小于等于 obstacle_trigger_distance_m，才进入虚线流程
            self.speak_obstacle_at_trigger()
            self.get_logger().info(
                f'[OBS_ALIGN] obstacle_dist={obstacle_dist:.3f} <= '
                f'{self.obstacle_trigger_distance_m:.3f}, switch to dashed align'
            )

            forced_side = self.get_forced_dashed_side()

            if forced_side is not None:
                self.dashed_side = forced_side
                self.get_logger().info(
                    f'[OBS_ALIGN] debug_dashed_side={forced_side}, force dashed_side={self.dashed_side}'
                )
            else:
                self.dashed_side = None
                self.get_logger().info(
                    '[OBS_ALIGN] debug_dashed_side=auto, dashed_side will be decided by vision'
                )

            self.dashed_pre_shift_start_time = None
            self.enter_state(self.ALIGN_DASHED_LINE)
            return

        elif self.state == self.ALIGN_DASHED_LINE:
            if dashed is None:
                self.dashed_center_count = 0
                self.send_motion_cmd(0.0, 0.0, 0.0)
                self.get_logger().info(
                    '[DASH_ALIGN] dashed=None, waiting',
                    throttle_duration_sec=0.5
                )
            else:
                # 第一次看到虚线时，先记录它在左边还是右边，随后进入预横移状态
                if self.dashed_side is None:
                    forced_side = self.get_forced_dashed_side()

                    if forced_side is not None:
                        self.dashed_side = forced_side
                        side_source = 'debug'
                    else:
                        self.dashed_side = self.get_dashed_side(dashed)
                        side_source = 'vision'

                    self.get_logger().info(
                        f'[DASH_ALIGN] first dashed side={self.dashed_side}, source={side_source}, '
                        f'center={dashed.center_img}, target_x={self.get_dashed_target_x():.1f}, '
                        f'enter pre side shift'
                    )

                    self.enter_state(self.DASH_PRE_SIDE_SHIFT)
                    return

                vy = self.compute_dashed_align_vy(dashed)
                self.send_motion_cmd(0.0, vy, 0.0)

                if self.is_dashed_centered(dashed):
                    self.dashed_center_count += 1
                else:
                    self.dashed_center_count = 0

                self.get_logger().info(
                    f'[DASH_ALIGN] side={self.dashed_side}, center={dashed.center_img}, '
                    f'target_x={self.get_dashed_target_x():.1f}, '
                    f'vy={vy:.3f}, stable={self.dashed_center_count}/{self.dashed_center_stable_frames}',
                    throttle_duration_sec=0.2
                )

                if self.dashed_center_count >= self.dashed_center_stable_frames:
                    self.enter_state(self.FOLLOW_DASHED_UNTIL_LOST)

        elif self.state == self.DASH_PRE_SIDE_SHIFT:
            if self.dashed_side not in ('left', 'right'):
                self.get_logger().warn('[DASH_PRE_SHIFT] dashed_side is None, skip pre-shift')
                self.enter_state(self.ALIGN_DASHED_LINE)
                return

            if self.dashed_pre_shift_dir_sign == 0.0:
                self.dashed_pre_shift_dir_sign = self.get_pre_shift_dir_sign()

            elapsed = self.state_elapsed_s()
            if elapsed >= self.dashed_pre_shift_duration_s:
                self.get_logger().info(
                    f'[DASH_PRE_SHIFT] done by sim time: side={self.dashed_side}, '
                    f'elapsed={elapsed:.3f}/{self.dashed_pre_shift_duration_s:.3f}s, '
                    f'go ALIGN_DASHED_LINE'
                )
                self.enter_state(self.ALIGN_DASHED_LINE)
                return

            vy = self.get_pre_shift_vy()
            self.send_motion_cmd(0.0, vy, 0.0)
            self.get_logger().info(
                f'[DASH_PRE_SHIFT] moving by sim time: side={self.dashed_side}, '
                f'elapsed={elapsed:.3f}/{self.dashed_pre_shift_duration_s:.3f}s, vy={vy:.3f}',
                throttle_duration_sec=0.2
            )


        elif self.state == self.FOLLOW_DASHED_UNTIL_LOST:
            # 新逻辑：
            # 对齐虚线之后，沿虚线向前走时，不是所有检测到的虚线都算有效。
            # 只有虚线中心 x 落在 get_dashed_target_x() 附近 follow_dashed_valid_x_range_px 范围内，
            # 才认为当前虚线仍然存在。
            # 如果检测到的虚线偏得太远，也按“虚线消失”处理。
            dashed_valid = self.is_dashed_valid_for_follow(dashed)

            if not dashed_valid:
                self.dashed_lost_count += 1

                if dashed is None:
                    self.get_logger().info(
                        f'[FOLLOW_DASH] dashed=None, '
                        f'lost_count={self.dashed_lost_count}/{self.dashed_lost_stop_frames}',
                        throttle_duration_sec=0.2
                    )
                else:
                    target_x = self.get_dashed_target_x()
                    cx = float(dashed.center_img[0])
                    err_px = cx - target_x

                    self.get_logger().info(
                        f'[FOLLOW_DASH] dashed detected but outside valid range, treat as lost: '
                        f'center_x={cx:.1f}, target_x={target_x:.1f}, '
                        f'err={err_px:.1f}px, valid_range=±{self.follow_dashed_valid_x_range_px}px, '
                        f'lost_count={self.dashed_lost_count}/{self.dashed_lost_stop_frames}',
                        throttle_duration_sec=0.2
                    )

                if self.dashed_lost_count >= self.dashed_lost_stop_frames:
                    self.get_logger().info(
                        f'[FOLLOW_DASH] dashed lost/out-of-range '
                        f'{self.dashed_lost_count} frames, go post dash forward'
                    )
                    self.enter_state(self.POST_DASH_FORWARD)
                else:
                    # 防止单帧漏检或单帧跳变，短暂继续向前。
                    # 注意这里不给 vy 修正，避免被错误虚线带偏。
                    self.send_motion_cmd(self.follow_forward_speed, 0.0, 0.0)

            else:
                self.dashed_lost_count = 0

                vy = self.compute_dashed_align_vy(
                    dashed,
                    k=self.follow_align_vy_k,
                    vy_max=self.follow_align_vy_max,
                    vy_min=self.follow_align_vy_min,
                )

                self.send_motion_cmd(self.follow_forward_speed, vy, 0.0)

                target_x = self.get_dashed_target_x()
                cx = float(dashed.center_img[0])
                err_px = cx - target_x

                self.get_logger().info(
                    f'[FOLLOW_DASH] valid dashed: center={dashed.center_img}, '
                    f'target_x={target_x:.1f}, err={err_px:.1f}px, '
                    f'valid_range=±{self.follow_dashed_valid_x_range_px}px, '
                    f'cmd=({self.follow_forward_speed:.3f},{vy:.3f},0.000), '
                    f'vy_limit=[{self.follow_align_vy_min:.3f},{self.follow_align_vy_max:.3f}]',
                    throttle_duration_sec=0.2
                )

        elif self.state == self.POST_DASH_FORWARD:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.post_dash_forward_duration_s:
                self.get_logger().info(
                    f'[POST_DASH_FORWARD] finished by sim time: elapsed={elapsed:.3f}/{self.post_dash_forward_duration_s:.3f}s, go first turn'
                )
                self.enter_state(self.POST_DASH_TURN_1)
            else:
                self.send_motion_cmd(self.post_dash_forward_speed, 0.0, 0.0)
                self.get_logger().info(
                    f'[POST_DASH_FORWARD] elapsed={elapsed:.3f}/{self.post_dash_forward_duration_s:.3f}s',
                    throttle_duration_sec=0.2
                )


        elif self.state == self.POST_DASH_TURN_1:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.current_turn_duration_s:
                self.get_logger().info(
                    f'[POST_DASH_TURN_1] finished by sim time: elapsed={elapsed:.3f}/{self.current_turn_duration_s:.3f}s, go forward'
                )
                self.enter_state(self.POST_TURN_FORWARD)
            else:
                wz = self.current_turn_dir * abs(self.current_turn_wz)
                self.send_motion_cmd(0.0, 0.0, wz)


        elif self.state == self.POST_TURN_FORWARD:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.post_turn_forward_duration_s:
                self.get_logger().info(
                    f'[POST_TURN_FORWARD] finished by sim time: elapsed={elapsed:.3f}/{self.post_turn_forward_duration_s:.3f}s, go second turn'
                )
                self.enter_state(self.POST_DASH_TURN_2)
            else:
                if elapsed < self.post_turn_forward_fast_duration_s:
                    vx = self.post_turn_forward_fast_speed
                    phase = 'FAST'
                else:
                    vx = self.post_turn_forward_slow_speed
                    phase = 'SLOW'

                self.send_motion_cmd(vx, 0.0, 0.0)
                self.get_logger().info(
                    f'[POST_TURN_FORWARD] phase={phase}, vx={vx:.3f}, '
                    f'elapsed={elapsed:.3f}/{self.post_turn_forward_duration_s:.3f}s, '
                    f'fast_until={self.post_turn_forward_fast_duration_s:.3f}s',
                    throttle_duration_sec=0.2
                )


        elif self.state == self.POST_DASH_TURN_2:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.current_turn_duration_s:
                self.get_logger().info(
                    f'[POST_DASH_TURN_2] finished by sim time: elapsed={elapsed:.3f}/{self.current_turn_duration_s:.3f}s, start target search'
                )
                self.enter_state(self.SEARCH_TARGET_AFTER_TURNS)
            else:
                wz = self.current_turn_dir * abs(self.current_turn_wz)
                self.send_motion_cmd(0.0, 0.0, wz)


        elif self.state == self.SEARCH_TARGET_AFTER_TURNS:
            self.send_motion_cmd(self.target_search_forward_speed, 0.0, 0.0)

            target_candidates_for_vis = self.detect_all_targets(frame)
            target = self.choose_best_target(target_candidates_for_vis)
            chosen_target_for_vis = target

            if target is None:
                self.stable_target_type = None
                self.target_stable_count = 0
                self.get_logger().info(
                    '[TARGET_SEARCH] target=None, keep moving forward',
                    throttle_duration_sec=0.5
                )
            else:
                if self.stable_target_type == target.det_type:
                    self.target_stable_count += 1
                else:
                    self.stable_target_type = target.det_type
                    self.target_stable_count = 1

                self.latest_target = target

                self.get_logger().info(
                    f'[TARGET_SEARCH] target={target.det_type}, center={target.center_img}, '
                    f'stable={self.target_stable_count}/{self.target_stable_frames}',
                    throttle_duration_sec=0.2
                )

                if self.target_stable_count >= self.target_stable_frames:
                    self.locked_target = target
                    self.enter_state(self.APPROACH_AND_ALIGN_TARGET)

        elif self.state == self.APPROACH_AND_ALIGN_TARGET:
            target_candidates_for_vis = self.detect_all_targets(frame)
            target = self.choose_best_target(target_candidates_for_vis)
            chosen_target_for_vis = target

            if target is None:
                self.locked_target = None
                self.enter_state(self.SEARCH_TARGET_AFTER_TURNS)
            else:
                self.locked_target = target
                vx, vy = self.compute_target_align_cmd(target)
                self.send_motion_cmd(vx, vy, 0.0)

                d = self.estimate_depth_at_center(target.center_img)

                self.get_logger().info(
                    f'[TARGET_ALIGN] target={target.det_type}, center={target.center_img}, '
                    f'depth={d}, cmd=({vx:.3f},{vy:.3f},0.000)',
                    throttle_duration_sec=0.2
                )

                if d is not None and d < self.hit_trigger_distance_m:
                    # 目标物体：靠近到撞击距离、刚进入撞击状态前播报
                    self.speak_target_at_hit_trigger(target.det_type)
                    self.enter_state(self.HIT_TARGET)

        elif self.state == self.HIT_TARGET:
            if self.locked_target is None:
                self.enter_state(self.SEARCH_TARGET_AFTER_TURNS)
            else:
                chosen_target_for_vis = self.locked_target
                target_candidates_for_vis = [self.locked_target]

                params = self.hit_params.get(
                    self.locked_target.det_type,
                    {'speed': 0.20, 'duration_s': 0.85}
                )
                elapsed = self.state_elapsed_s()
                duration = float(params.get('duration_s', 0.85))

                if elapsed >= duration:
                    self.get_logger().info(
                        f'[HIT] finished by sim time: target={self.locked_target.det_type}, '
                        f'elapsed={elapsed:.3f}/{duration:.3f}s, go backoff after hit'
                    )
                    self.enter_state(self.HIT_BACKOFF_AFTER_HIT)
                    return

                self.send_motion_cmd(params['speed'], 0.0, 0.0)
                self.get_logger().info(
                    f'[HIT] target={self.locked_target.det_type}, elapsed={elapsed:.3f}/{duration:.3f}s',
                    throttle_duration_sec=0.2
                )

                if elapsed >= self.hit_timeout_s:
                    self.get_logger().warn('[HIT] timeout reached, go backoff after hit')
                    self.enter_state(self.HIT_BACKOFF_AFTER_HIT)


        elif self.state == self.HIT_BACKOFF_AFTER_HIT:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.after_hit_backoff_duration_s:
                self.send_motion_cmd(0.0, 0.0, 0.0)
                self.get_logger().info(
                    f'[AFTER_HIT_BACKOFF] finished by sim time: elapsed={elapsed:.3f}/{self.after_hit_backoff_duration_s:.3f}s, go two left jumps'
                )
                self.enter_state(self.POST_HIT_LEFT_JUMP)
            else:
                self.send_motion_cmd(-self.after_hit_backoff_speed, 0.0, 0.0)
                self.get_logger().info(
                    f'[AFTER_HIT_BACKOFF] elapsed={elapsed:.3f}/{self.after_hit_backoff_duration_s:.3f}s',
                    throttle_duration_sec=0.2
                )


        elif self.state == self.POST_HIT_LEFT_JUMP:
            self.execute_left_jump_turn(
                jump_count=self.after_hit_left_jump_count,
                next_state=self.APPROACH_SELECTED_OBSTACLE_AFTER_HIT
            )
            return

        elif self.state == self.APPROACH_SELECTED_OBSTACLE_AFTER_HIT:
            # 这个状态有三种情况：
            # 1) 0 个障碍物：继续向前搜索。
            # 2) 1 个障碍物：不居中对齐，只检查距离；距离到阈值后进入后续转向。
            # 3) 2 个及以上障碍物：保持原逻辑，根据 dashed_side 选择左/右障碍物并居中对齐。
            obs_count = len(obstacle_candidates)

            if obs_count == 0:
                self.selected_obstacle_after_hit = None
                self.selected_obstacle_after_hit_side = None
                self.send_motion_cmd(self.post_hit_obstacle_search_forward_speed, 0.0, 0.0)
                self.get_logger().info(
                    '[POST_HIT_OBS] no obstacle detected, keep searching forward',
                    throttle_duration_sec=0.5
                )
                return

            if obs_count == 1:
                selected = obstacle_candidates[0]
                self.selected_obstacle_after_hit = selected

                img_w = self.latest_bgr.shape[1] if self.latest_bgr is not None else 640
                img_center_x = img_w // 2
                cx = selected.center_img[0]
                self.selected_obstacle_after_hit_side = 'left' if cx < img_center_x else 'right'

                d = selected.extra.get('median_depth')

                self.get_logger().warn(
                    f'[POST_HIT_OBS] only one obstacle detected, distance-only mode: '
                    f'side={self.selected_obstacle_after_hit_side}, center={selected.center_img}, '
                    f'depth={d}, threshold={self.post_hit_obstacle_trigger_distance_m:.3f}',
                    throttle_duration_sec=0.3
                )

                if d is not None and d <= self.post_hit_obstacle_trigger_distance_m:
                    self.get_logger().info(
                        f'[POST_HIT_OBS] one obstacle close enough: '
                        f'depth={d:.3f}/{self.post_hit_obstacle_trigger_distance_m:.3f}, '
                        f'dashed_side={self.dashed_side}, go post-hit turn task'
                    )
                    self.enter_state(self.POST_HIT_OBS_TURN_1)
                    return

                # 只有一个障碍物但还没到距离：不居中，不给 vy，只继续直走。
                self.send_motion_cmd(self.post_hit_obstacle_forward_speed, 0.0, 0.0)
                return

            # 2 个及以上障碍物：保持原逻辑，按 dashed_side 选左/右障碍物并居中对齐。
            selected = self.choose_selected_obstacle_after_hit(obstacle_candidates)
            self.selected_obstacle_after_hit = selected

            if selected is None:
                self.selected_obstacle_after_hit_side = None
                self.send_motion_cmd(self.post_hit_obstacle_search_forward_speed, 0.0, 0.0)
                self.get_logger().info(
                    f'[POST_HIT_OBS] selected=None, dashed_side={self.dashed_side}, keep searching forward',
                    throttle_duration_sec=0.5
                )
                return

            vy = self.compute_selected_obstacle_align_vy_after_hit(selected)
            d = selected.extra.get('median_depth')
            self.send_motion_cmd(self.post_hit_obstacle_forward_speed, vy, 0.0)

            self.get_logger().info(
                f'[POST_HIT_OBS] dashed_side={self.dashed_side}, selected={selected.center_img}, '
                f'depth={d}, cmd=({self.post_hit_obstacle_forward_speed:.3f},{vy:.3f},0.000)',
                throttle_duration_sec=0.2
            )

            if d is not None and d <= self.post_hit_obstacle_trigger_distance_m:
                self.selected_obstacle_after_hit_side = self.get_obstacle_side_in_pair(selected, obstacle_candidates)
                self.get_logger().info(
                    f'[POST_HIT_OBS] selected obstacle dist={d:.3f} <= '
                    f'{self.post_hit_obstacle_trigger_distance_m:.3f}, '
                    f'dashed_side={self.dashed_side}, go post-hit turn task'
                )
                self.enter_state(self.POST_HIT_OBS_TURN_1)
                return

        elif self.state == self.POST_HIT_OBS_TURN_1:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.current_turn_duration_s:
                self.get_logger().info(
                    f'[POST_HIT_OBS_TURN_1] finished by sim time: elapsed={elapsed:.3f}/{self.current_turn_duration_s:.3f}s, go forward'
                )
                self.enter_state(self.POST_HIT_OBS_FORWARD)
            else:
                wz = self.current_turn_dir * abs(self.current_turn_wz)
                self.send_motion_cmd(0.0, 0.0, wz)


        elif self.state == self.POST_HIT_OBS_FORWARD:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.post_hit_obs_forward_duration_s:
                self.get_logger().info(
                    f'[POST_HIT_OBS_FORWARD] finished by sim time: elapsed={elapsed:.3f}/{self.post_hit_obs_forward_duration_s:.3f}s, go opposite turn'
                )
                self.enter_state(self.POST_HIT_OBS_TURN_2)
            else:
                self.send_motion_cmd(self.post_hit_obs_forward_speed, 0.0, 0.0)
                self.get_logger().info(
                    f'[POST_HIT_OBS_FORWARD] elapsed={elapsed:.3f}/{self.post_hit_obs_forward_duration_s:.3f}s',
                    throttle_duration_sec=0.2
                )


        elif self.state == self.POST_HIT_OBS_TURN_2:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.current_turn_duration_s:
                self.get_logger().info(
                    f'[POST_HIT_OBS_TURN_2] finished by sim time: elapsed={elapsed:.3f}/{self.current_turn_duration_s:.3f}s, go pre-final fixed forward'
                )
                # 注意：这里不恢复 normal，也不额外发送 0 速度。
                # 后面还要继续前进识别横向黄线，并完成最终 180 度掉头，
                # 这些障碍物收尾动作仍保持 low 姿态。
                self.enter_state(self.POST_HIT_PRE_FINAL_FORWARD)
            else:
                wz = self.current_turn_dir * abs(self.current_turn_wz)
                self.send_motion_cmd(0.0, 0.0, wz)


        elif self.state == self.POST_HIT_PRE_FINAL_FORWARD:
            # 第二次转回后，先按仿真时间向前走一小段。
            # 新增：这一段如果已经能看到前方横向黄线，就只用黄线角度修正 wz，
            # 不用黄线提前结束该状态，仍然按 post_hit_final_forward_duration_s 到时后进入正式黄线阶段。
            elapsed = self.state_elapsed_s()
            if elapsed >= self.post_hit_final_forward_duration_s:
                self.get_logger().info(
                    f'[POST_HIT_PRE_FINAL_FORWARD] finished by sim time: elapsed={elapsed:.3f}/{self.post_hit_final_forward_duration_s:.3f}s, start final yellow detection'
                )
                self.enter_state(self.POST_HIT_FINAL_FORWARD)
            else:
                wz = 0.0
                final_yellow_line = None

                if self.post_hit_pre_final_angle_align_enabled:
                    final_yellow_line = self.final_yellow_detector.detect(frame)
                    self.latest_final_yellow_line = final_yellow_line
                    wz = self.compute_final_yellow_wz(final_yellow_line)

                self.send_motion_cmd(self.post_hit_final_forward_speed, 0.0, wz)

                if final_yellow_line is None:
                    self.get_logger().info(
                        f'[POST_HIT_PRE_FINAL_FORWARD] elapsed={elapsed:.3f}/{self.post_hit_final_forward_duration_s:.3f}s, '
                        f'no yellow angle reference, wz={wz:.3f}',
                        throttle_duration_sec=0.2
                    )
                else:
                    angle_deg = float(final_yellow_line.extra.get('angle_deg', 0.0))
                    abs_tilt = float(final_yellow_line.extra.get('abs_tilt_deg', abs(angle_deg)))
                    bottom_ratio = float(final_yellow_line.extra.get('bottom_ratio', 0.0))
                    self.get_logger().info(
                        f'[POST_HIT_PRE_FINAL_FORWARD] elapsed={elapsed:.3f}/{self.post_hit_final_forward_duration_s:.3f}s, '
                        f'angle={angle_deg:.1f}deg, abs_tilt={abs_tilt:.1f}, bottom_ratio={bottom_ratio:.3f}, wz={wz:.3f}',
                        throttle_duration_sec=0.2
                    )


        elif self.state == self.POST_HIT_FINAL_FORWARD:
            # 障碍物流程内部的最终横向黄线收尾：
            # 1. 没看到黄线：继续向前找黄线；
            # 2. 看到黄线但还没到下方阈值：vx 正常前进，同时用 wz 修正角度；
            # 3. 黄线已经到达下方阈值但角度没对正：vx=0, vy=0，只原地 wz 调角度；
            # 4. 黄线到达下方阈值且角度满足：进入 FINAL_LEFT_JUMP。
            final_yellow_line = self.final_yellow_detector.detect(frame)
            self.latest_final_yellow_line = final_yellow_line

            if final_yellow_line is None:
                self.final_yellow_done_counter = 0
                self.send_motion_cmd(self.post_hit_final_forward_speed, 0.0, 0.0)
                self.get_logger().info(
                    '[POST_HIT_FINAL_FORWARD] no horizontal yellow line, keep moving forward',
                    throttle_duration_sec=0.3
                )
                return

            bottom_y = int(final_yellow_line.extra.get('bottom_y', 0))
            bottom_ratio = float(final_yellow_line.extra.get('bottom_ratio', 0.0))
            angle_deg = float(final_yellow_line.extra.get('angle_deg', 0.0))
            abs_tilt = float(final_yellow_line.extra.get('abs_tilt_deg', abs(angle_deg)))
            wz = self.compute_final_yellow_wz(final_yellow_line)

            reached_line = bottom_ratio >= self.final_yellow_stop_line_y_ratio
            angle_ok = abs_tilt <= self.final_yellow_done_tilt_deg

            if reached_line and angle_ok:
                self.final_yellow_done_counter += 1
            else:
                self.final_yellow_done_counter = 0

            if self.final_yellow_done_counter >= self.final_yellow_confirm_count:
                self.get_logger().info(
                    '[POST_HIT_FINAL_FORWARD] yellow reached lower area and aligned, go FINAL_LEFT_JUMP'
                )
                self.send_motion_cmd(0.0, 0.0, 0.0)
                self.enter_state(self.FINAL_LEFT_JUMP)
                return

            if reached_line:
                # 黄线已经到图像下方，但角度还没满足：
                # 停止前进，原地只用 wz 调整朝向，避免黄线继续往下跑出画面。
                vx_cmd = 0.0
                vy_cmd = 0.0
                phase = 'reached_lower_align_in_place'
            else:
                # 黄线还没到下方阈值：
                # 正常向前靠近，同时用 wz 修正角度。
                vx_cmd = self.post_hit_final_forward_speed
                vy_cmd = 0.0
                phase = 'approach_with_angle_align'

            self.send_motion_cmd(vx_cmd, vy_cmd, wz)

            self.get_logger().info(
                f'[POST_HIT_FINAL_FORWARD] {phase}: '
                f'bottom={bottom_y}, ratio={bottom_ratio:.3f}/{self.final_yellow_stop_line_y_ratio:.3f}, '
                f'angle={angle_deg:.1f}deg, abs_tilt={abs_tilt:.1f}/{self.final_yellow_done_tilt_deg:.1f}, '
                f'vx={vx_cmd:.3f}, vy={vy_cmd:.3f}, wz={wz:.3f}, '
                f'counter={self.final_yellow_done_counter}/{self.final_yellow_confirm_count}',
                throttle_duration_sec=0.2
            )

        elif self.state == self.FINAL_LEFT_JUMP:
            # 障碍物流程内部的最终掉头。
            # 新姿态策略：
            #   POST_HIT_FINAL_FORWARD 前进识别横向黄线并对正后，
            #   先在 low 姿态下完成这里的最终掉头；
            #   掉头完成后进入 OBSTACLE_RESTORE_NORMAL_AFTER_FINAL_TURN，
            #   再 STOP -> NORMAL。
            if self.obstacle_flow_is_third_object:
                if self.completed_obstacle_count < self.required_obstacle_count:
                    self.completed_obstacle_count += 1

                self.after_obstacle_restore_next_state = self.GLOBAL_FINAL_YELLOW_FORWARD
                self.get_logger().info(
                    f'[FINAL_LEFT_JUMP] obstacle is the 3rd flow, '
                    f'execute final turn first, then restore NORMAL and jump directly to GLOBAL_FINAL_YELLOW_FORWARD. '
                    f'bar={self.completed_bar_count}/{self.required_bar_count}, '
                    f'obstacle={self.completed_obstacle_count}/{self.required_obstacle_count}'
                )
                self.execute_left_jump_turn(1, self.OBSTACLE_RESTORE_NORMAL_AFTER_FINAL_TURN)
                return

            self.after_obstacle_restore_next_state = self.OBSTACLE_FLOW_DONE
            self.get_logger().info(
                '[FINAL_LEFT_JUMP] timed left turn equivalent to 180 deg, then restore NORMAL and obstacle flow done')
            self.execute_left_jump_turn(2, self.OBSTACLE_RESTORE_NORMAL_AFTER_FINAL_TURN)

        elif self.state == self.OBSTACLE_RESTORE_NORMAL_AFTER_FINAL_TURN:
            next_state = getattr(self, 'after_obstacle_restore_next_state', self.OBSTACLE_FLOW_DONE)
            self.get_logger().warn(
                f'[BODY] obstacle final yellow + final turn finished, restore NORMAL, next={next_state}'
            )
            self.restore_body_normal_after_obstacle_final_turn()
            self.enter_state(next_state)
            return

        elif self.state == self.OBSTACLE_FLOW_DONE:
            self.finish_obstacle_flow()
            return

        elif self.state == self.GLOBAL_FINAL_RIGHT_JUMP:
            # 全部任务完成后的第一步：右跳一次。
            self.get_logger().info(
                '[GLOBAL_FINAL_RIGHT_JUMP] timed right turn equivalent to one right jump, then start final yellow alignment')
            self.execute_right_jump_turn(1, self.GLOBAL_FINAL_YELLOW_FORWARD)
            return

        elif self.state == self.GLOBAL_FINAL_YELLOW_FORWARD:
            # 右跳后继续前进，同时识别前方横向黄线并用倾斜角修正朝向。
            #
            # 新逻辑：
            # 1. 没看到黄线：快速前进
            # 2. 看到黄线但 bottom_ratio 还没到 slow_start_ratio：快速前进
            # 3. bottom_ratio >= slow_start_ratio：切换慢速前进
            # 4. bottom_ratio >= stop_line_y_ratio：认为黄线已经到图像下方区域
            # 5. 黄线到下方区域后，继续慢速前进，直到横向黄线从画面中消失
            # 6. 连续消失 global_final_yellow_disappear_confirm_count 帧后，进入最终左跳

            final_yellow_line = self.final_yellow_detector.detect(frame)
            self.latest_final_yellow_line = final_yellow_line

            if final_yellow_line is None:
                self.global_final_yellow_done_counter = 0

                if self.global_final_yellow_reached_lower_area:
                    # 已经确认黄线到过图像下方区域，现在看不到黄线，
                    # 说明机器狗可能已经越过黄线。
                    self.global_final_yellow_disappear_counter += 1

                    # 这里建议停住等待确认，避免继续冲太远。
                    self.send_motion_cmd(0.0, 0.0, 0.0)

                    self.get_logger().info(
                        f'[GLOBAL_FINAL_YELLOW] yellow disappeared after reaching lower area: '
                        f'disappear_counter={self.global_final_yellow_disappear_counter}/'
                        f'{self.global_final_yellow_disappear_confirm_count}',
                        throttle_duration_sec=0.2
                    )

                    if self.global_final_yellow_disappear_counter >= self.global_final_yellow_disappear_confirm_count:
                        self.get_logger().info(
                            '[GLOBAL_FINAL_YELLOW_FORWARD] yellow disappeared after reached lower area, '
                            'go forward before final left jump'
                        )
                        self.send_motion_cmd(0.0, 0.0, 0.0)
                        self.enter_state(self.GLOBAL_FINAL_LEFT_JUMP)
                        return

                else:
                    # 还没有确认黄线到达过下方区域。
                    # 没看到黄线时继续快速前进寻找。
                    self.global_final_yellow_disappear_counter = 0

                    vx = self.global_final_yellow_forward_speed
                    self.send_motion_cmd(vx, 0.0, 0.0)

                    self.get_logger().info(
                        f'[GLOBAL_FINAL_YELLOW] no horizontal yellow line before lower-area reached, '
                        f'keep moving forward, vx={vx:.3f}',
                        throttle_duration_sec=0.3
                    )

            else:
                # 看到横向黄线
                self.global_final_yellow_disappear_counter = 0

                bottom_y = int(final_yellow_line.extra.get('bottom_y', 0))
                bottom_ratio = float(final_yellow_line.extra.get('bottom_ratio', 0.0))
                angle_deg = float(final_yellow_line.extra.get('angle_deg', 0.0))

                wz = self.compute_final_yellow_wz(final_yellow_line)
                vx = self.get_global_final_yellow_forward_speed(final_yellow_line)

                reached_slow_area = bottom_ratio >= self.global_final_yellow_slow_start_ratio
                reached_line = bottom_ratio >= self.global_final_yellow_stop_line_y_ratio

                if reached_line:
                    # 黄线已经到达图像下方区域
                    self.global_final_yellow_done_counter += 1

                    if self.global_final_yellow_done_counter >= self.global_final_yellow_confirm_count:
                        self.global_final_yellow_reached_lower_area = True

                    # 到达下方区域后继续前进，等待黄线消失。
                    # 这里 vx 会因为 bottom_ratio 已经很大而自动变成慢速。
                    self.send_motion_cmd(vx, 0.0, wz)

                    self.get_logger().info(
                        f'[GLOBAL_FINAL_YELLOW] yellow reached lower area, keep moving until it disappears: '
                        f'bottom={bottom_y}, '
                        f'ratio={bottom_ratio:.3f}/{self.global_final_yellow_stop_line_y_ratio:.3f}, '
                        f'slow_start={self.global_final_yellow_slow_start_ratio:.3f}, '
                        f'slow={reached_slow_area}, '
                        f'angle={angle_deg:.1f}deg, '
                        f'vx={vx:.3f}, wz={wz:.3f}, '
                        f'reach_counter={self.global_final_yellow_done_counter}/'
                        f'{self.global_final_yellow_confirm_count}, '
                        f'armed={self.global_final_yellow_reached_lower_area}',
                        throttle_duration_sec=0.2
                    )

                else:
                    # 黄线还没到最终下方阈值
                    self.global_final_yellow_done_counter = 0

                    # 如果 bottom_ratio 已经超过 slow_start_ratio，这里会自动用慢速；
                    # 否则继续快速靠近。
                    self.send_motion_cmd(vx, 0.0, wz)

                    self.get_logger().info(
                        f'[GLOBAL_FINAL_YELLOW] approach and align: '
                        f'bottom={bottom_y}, '
                        f'ratio={bottom_ratio:.3f}/{self.global_final_yellow_stop_line_y_ratio:.3f}, '
                        f'slow_start={self.global_final_yellow_slow_start_ratio:.3f}, '
                        f'slow={reached_slow_area}, '
                        f'angle={angle_deg:.1f}deg, '
                        f'vx={vx:.3f}, wz={wz:.3f}',
                        throttle_duration_sec=0.2
                    )

            return

        elif self.state == self.GLOBAL_FINAL_LEFT_JUMP:
            self.get_logger().info(
                '[GLOBAL_FINAL_LEFT_JUMP] timed left turn equivalent to one left jump, then right shift'
            )
            self.execute_left_jump_turn(1, self.GLOBAL_FINAL_RIGHT_SHIFT_AFTER_LEFT_JUMP)
            return

        elif self.state == self.GLOBAL_FINAL_RIGHT_SHIFT_AFTER_LEFT_JUMP:
            elapsed = self.state_elapsed_s()

            if elapsed < self.global_final_after_left_jump_right_shift_duration_s:
                self.send_motion_cmd(
                    0.0,
                    self.global_final_after_left_jump_right_shift_vy,
                    0.0
                )
                self.get_logger().info(
                    f'[GLOBAL_FINAL_RIGHT_SHIFT_AFTER_LEFT_JUMP] right shift after left jump: '
                    f'elapsed={elapsed:.3f}/{self.global_final_after_left_jump_right_shift_duration_s:.3f}s, '
                    f'vy={self.global_final_after_left_jump_right_shift_vy:.3f}',
                    throttle_duration_sec=0.2
                )
                return

            self.get_logger().info(
                '[GLOBAL_FINAL_RIGHT_SHIFT_AFTER_LEFT_JUMP] done, go GLOBAL_FINAL_P3_ALIGN'
            )
            self.send_motion_cmd(0.0, 0.0, 0.0)
            self.enter_state(self.GLOBAL_FINAL_P3_ALIGN)
            return

        elif self.state == self.GLOBAL_FINAL_P3_ALIGN:
            # 第四赛段结束位置和第三赛段结束位置相同，直接复用第三赛段 P3_ALIGN_TRACK 的矫正逻辑。
            # 注意：第四赛段 RGB 回调不会跑 P3 视觉，所以这里主动调用 p3_process_yellow_track(frame)。
            self.p3_process_yellow_track(frame)
            elapsed = self.state_elapsed_s()

            if elapsed >= self.p3_align_max_duration_sec:
                self.get_logger().info(
                    f'[GLOBAL_FINAL_P3_ALIGN] timeout, finish all stages: '
                    f'elapsed={elapsed:.2f}/{self.p3_align_max_duration_sec:.2f}s'
                )
                if self.show_debug_vis:
                    self.p3_show_debug_window(frame)
                self.enter_state(self.DONE)
                return

            if self.p3_s4_valid > 0.5:
                err_lat = self.p3_s4_lat
                err_yaw = self.p3_s4_yaw

                if abs(err_lat) < self.p3_align_lat_tol and abs(err_yaw) < self.p3_align_yaw_tol:
                    self.get_logger().info(
                        f'[GLOBAL_FINAL_P3_ALIGN] complete: '
                        f'lat={err_lat:.4f}/{self.p3_align_lat_tol:.4f}, '
                        f'yaw={err_yaw:.4f}/{self.p3_align_yaw_tol:.4f}'
                    )
                    self.send_motion_cmd(0.0, 0.0, 0.0)
                    if self.show_debug_vis:
                        self.p3_show_debug_window(frame)
                    self.enter_state(self.DONE)
                    return

                lateral_speed = clamp(
                    err_lat * self.p3_align_lat_gain,
                    -self.p3_align_lat_max,
                    self.p3_align_lat_max
                )
                turn_speed = clamp(
                    err_yaw * self.p3_align_yaw_gain,
                    -self.p3_align_yaw_max,
                    self.p3_align_yaw_max
                )
                self.send_motion_cmd(0.0, lateral_speed, turn_speed)
                self.get_logger().info(
                    f'[GLOBAL_FINAL_P3_ALIGN] align: '
                    f'lat={err_lat:.4f}, yaw={err_yaw:.4f}, '
                    f'cmd=(0.000,{lateral_speed:.3f},{turn_speed:.3f})',
                    throttle_duration_sec=0.3
                )
                if self.show_debug_vis:
                    self.p3_show_debug_window(frame)
            else:
                self.send_motion_cmd(0.05, -0.04, 0.0)
                self.get_logger().info(
                    f'[GLOBAL_FINAL_P3_ALIGN] no valid track, searching: '
                    f'cmd=({self.p3_align_search_vx:.3f},0.000,{self.p3_align_search_wz:.3f})',
                    throttle_duration_sec=0.5
                )
                if self.show_debug_vis:
                    self.p3_show_debug_window(frame)
            return

        elif self.state == self.DONE:
            if not self.task_done_stop_sent:
                self.stop()
                self.task_done_stop_sent = True

        if self.show_debug_vis:
            self.update_debug_visualization(
                frame,
                obstacle_candidates,
                chosen_pair,
                dashed,
                target_candidates_for_vis,
                chosen_target_for_vis,
                final_yellow_line,
                bar_for_vis,
            )

        if now - self.last_log_time > 0.5:
            self.last_log_time = now
            vx, vy, wz = self.motion_cmd
            dashed_text = 'None' if dashed is None else f'{dashed.center_img}'
            self.get_logger().info(
                f'state={self.state} cmd=({vx:.3f},{vy:.3f},{wz:.3f}) bar={self.completed_bar_count}/{self.required_bar_count} obs={self.completed_obstacle_count}/{self.required_obstacle_count} '
                f'obs_candidates={len(obstacle_candidates)} dashed={dashed_text}'
            )

    # ---------- 可视化 ----------
    def update_debug_visualization(
            self,
            frame,
            obstacle_candidates: List[Detection],
            obstacle_pair: Optional[Tuple[Detection, Detection]],
            dashed: Optional[Detection],
            target_candidates: Optional[List[Detection]] = None,
            chosen_target: Optional[Detection] = None,
            final_yellow_line: Optional[Detection] = None,
            bar_det: Optional[Detection] = None,
    ):
        vis = frame.copy()
        h, w = vis.shape[:2]

        cv2.line(vis, (w // 2, 0), (w // 2, h - 1), (0, 255, 0), 1)

        # 黄线偏置对齐目标线：绿色是图像中心线，紫色是当前虚线对齐目标线
        if self.dashed_side is not None:
            target_x = int(round(self.get_dashed_target_x()))
            target_x = max(0, min(w - 1, target_x))
            cv2.line(vis, (target_x, 0), (target_x, h - 1), (255, 0, 255), 2)
            cv2.putText(
                vis,
                f'dash target side={self.dashed_side}',
                (target_x + 5, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                2
            )

        cv2.putText(
            vis,
            f'state: {self.state}',
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        vx, vy, wz = self.motion_cmd
        cv2.putText(
            vis,
            f'cmd: ({vx:.2f},{vy:.2f},{wz:.2f})',
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        # 画限高杆检测结果：红色框表示当前识别到的限高杆，中心点用于全局居中对齐。
        # 注意：限高杆完成 required_bar_count 次后，全局搜索阶段不再检测限高杆，窗口中也不会再更新 BAR 框。
        if bar_det is not None:
            x1, y1, x2, y2 = bar_det.bbox_img
            cx, cy = bar_det.center_img
            aspect = float(bar_det.extra.get('aspect_ratio', 0.0))
            depth_yaw_info = getattr(self, 'latest_bar_depth_yaw_info', {})
            depth_err = depth_yaw_info.get('depth_error', None)
            left_depth = depth_yaw_info.get('left_depth', None)
            right_depth = depth_yaw_info.get('right_depth', None)

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
            cv2.line(vis, (cx, 0), (cx, h - 1), (0, 0, 255), 1)
            cv2.putText(
                vis,
                f'BAR {self.completed_bar_count}/{self.required_bar_count} aspect={aspect:.1f} L={left_depth} R={right_depth} err={depth_err}',
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )
            left_pt = depth_yaw_info.get('left_point', None)
            right_pt = depth_yaw_info.get('right_point', None)
            if left_pt is not None:
                cv2.circle(vis, tuple(left_pt), 5, (255, 0, 255), -1)
            if right_pt is not None:
                cv2.circle(vis, tuple(right_pt), 5, (255, 0, 255), -1)

        # 左上角显示总流程完成进度，方便判断当前还会不会继续检测限高杆/障碍物。
        cv2.putText(
            vis,
            f'progress: BAR {self.completed_bar_count}/{self.required_bar_count}  OBS {self.completed_obstacle_count}/{self.required_obstacle_count}',
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2
        )

        if self.state in [self.POST_DASH_TURN_1, self.POST_DASH_TURN_2, self.POST_HIT_OBS_TURN_1,
                          self.POST_HIT_OBS_TURN_2]:
            turn_name = 'LEFT' if self.current_turn_dir > 0 else 'RIGHT'
            cv2.putText(
                vis,
                f'tf turn: {turn_name} target={math.degrees(self.current_turn_angle_rad):.0f}deg',
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 0, 255),
                2
            )

        # 画所有蓝色障碍物候选
        for i, det in enumerate(obstacle_candidates):
            x1, y1, x2, y2 = det.bbox_img
            cx, cy = det.center_img
            d = det.extra.get('median_depth')

            color = (255, 0, 0)
            thickness = 2

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(vis, (cx, cy), 4, color, -1)

            d_text = 'None' if d is None else f'{d:.2f}'
            cv2.putText(
                vis,
                f'OBS{i} d={d_text}',
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # 画用于居中的两个障碍物
        if obstacle_pair is not None:
            left, right = obstacle_pair
            lx, ly = left.center_img
            rx, ry = right.center_img
            mid_x = int((lx + rx) / 2)
            mid_y = int((ly + ry) / 2)

            cv2.circle(vis, (lx, ly), 6, (0, 255, 255), -1)
            cv2.circle(vis, (rx, ry), 6, (0, 255, 255), -1)
            cv2.circle(vis, (mid_x, mid_y), 7, (0, 0, 255), -1)
            cv2.line(vis, (lx, ly), (rx, ry), (0, 255, 255), 2)
            cv2.line(vis, (mid_x, 0), (mid_x, h - 1), (0, 0, 255), 1)

            cv2.putText(
                vis,
                'OBSTACLE MID',
                (mid_x + 5, max(20, mid_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )

        # 画虚线
        if dashed is not None:
            x1, y1, x2, y2 = dashed.bbox_img
            cx, cy = dashed.center_img

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 3)
            cv2.circle(vis, (cx, cy), 6, (0, 165, 255), -1)

            centers = dashed.extra.get('group_centers', [])
            for k, p in enumerate(centers):
                px = int(round(p[0]))
                py = int(round(p[1]))
                cv2.circle(vis, (px, py), 4, (0, 165, 255), -1)

                if k > 0:
                    qx = int(round(centers[k - 1][0]))
                    qy = int(round(centers[k - 1][1]))
                    cv2.line(vis, (qx, qy), (px, py), (0, 165, 255), 2)

            cv2.putText(
                vis,
                f'DASH seg={dashed.extra.get("segments", 0)} span={dashed.extra.get("total_span_y", 0):.0f}',
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2
            )

        # 画第二次转向后的目标检测结果
        if target_candidates is None:
            target_candidates = []

        for det in target_candidates:
            x1, y1, x2, y2 = det.bbox_img
            cx, cy = det.center_img

            if det.det_type == 'blue_ball':
                color = (255, 0, 0)
            elif det.det_type == 'white_ball':
                color = (255, 255, 255)
            else:
                color = (0, 0, 255)

            thickness = 2
            if chosen_target is not None and det.det_type == chosen_target.det_type and det.center_img == chosen_target.center_img:
                thickness = 4

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(vis, (cx, cy), 5, color, -1)
            cv2.putText(
                vis,
                f'TARGET {det.det_type}',
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2
            )

        # 画最后阶段前方横向黄线
        if final_yellow_line is not None:
            x1, y1, x2, y2 = final_yellow_line.bbox_img
            cx, cy = final_yellow_line.center_img
            angle_deg = float(final_yellow_line.extra.get('angle_deg', 0.0))
            bottom_y = int(final_yellow_line.extra.get('bottom_y', 0))
            bottom_ratio = float(final_yellow_line.extra.get('bottom_ratio', 0.0))

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.circle(vis, (cx, cy), 6, (0, 255, 255), -1)
            cv2.putText(
                vis,
                f'FINAL YELLOW bottom={bottom_y} ratio={bottom_ratio:.2f} angle={angle_deg:.1f}',
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

        cv2.imshow('obstacle_dashed_task_debug', vis)
        cv2.waitKey(1)


class CombinedStage1Stage2Node(Node):
    def __init__(self):
        super().__init__('combined_stage1_stage2_node')

        # 使用仿真时间；如果 launch/yaml 已经设置过，这里失败也不影响运行。
        try:
            self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        except Exception as e:
            self.get_logger().warn(f'failed to set use_sim_time: {e}')

        # =========================
        # 话题与 TF
        # =========================
        self.declare_parameter('rgb_topic', '/rgb_camera/rgb_camera/image_raw')
        self.declare_parameter('depth_topic', '/d435/depth/d435_depth/depth/image_raw')
        self.declare_parameter('global_frame', 'vodom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('control_hz', 30.0)
        # self.declare_parameter('initial_state', 'P1_STAND_WAIT')
        self.declare_parameter('initial_state', 'GLOBAL_INITIAL_LATERAL_SHIFT')
        # self.declare_parameter('initial_state', 'GLOBAL_FINAL_YELLOW_FORWARD')
        
        self.declare_parameter('second_stage_initial_state', 'STAGE1_CRUISE_BALL_AND_YELLOW')

        # OpenCV 可视化窗口：只用于调试，不参与控制逻辑
        self.declare_parameter('show_debug_vis', True)
        self.declare_parameter('show_yellow_mask', False)

        # =========================
        # 第一赛段参数（全部加 p1_ 前缀，避免和第二赛段变量冲突）
        # =========================
        self.declare_parameter('p1_stand_wait_sec', 0)
        self.declare_parameter('p1_stand_body_height', 0.28)

        self.declare_parameter('p1_stage1_max_duration_sec', 9.0)
        self.declare_parameter('p1_base_forward_speed', 0.40)
        self.declare_parameter('p1_min_forward_speed', 0.20)
        self.declare_parameter('p1_kp_turn', 0.25)
        self.declare_parameter('p1_kp_lat', 0.15)
        self.declare_parameter('p1_kd_slowdown', 0.05)
        self.declare_parameter('p1_max_turn_speed', 0.15)
        self.declare_parameter('p1_max_lateral_speed', 0.15)
        self.declare_parameter('p1_vision_timeout_sec', 1.0)

        self.declare_parameter('p1_brake_duration_sec', 0.3)
        self.declare_parameter('p1_align_max_duration_sec', 3.0)
        self.declare_parameter('p1_align_angle_deadband_rad', 0.05)
        self.declare_parameter('p1_align_turn_kp', 0.4)
        self.declare_parameter('p1_align_turn_max_wz', 0.10)

        self.declare_parameter('p1_turn_duration_sec', 3.5)
        self.declare_parameter('p1_turn_forward_vel', 0.13)
        self.declare_parameter('p1_turn_yaw_vel', 0.51)

        self.declare_parameter('p1_blue_target_distance_m', 0.25)
        self.declare_parameter('p1_approach_blue_max_duration_sec', 3.0)
        self.declare_parameter('p1_approach_blue_forward_speed', 0.20)

        self.declare_parameter('p1_blind_left_duration_sec', 3.0)
        self.declare_parameter('p1_blind_left_vy', 0.13)
        self.declare_parameter('p1_blind_left_vx', 0.14)

        self.declare_parameter('p1_yellow_h_min', 20)
        self.declare_parameter('p1_yellow_h_max', 40)
        self.declare_parameter('p1_yellow_s_min', 50)
        self.declare_parameter('p1_yellow_s_max', 255)
        self.declare_parameter('p1_yellow_v_min', 150)
        self.declare_parameter('p1_yellow_v_max', 255)

        self.declare_parameter('p1_stop_top_ratio', 0.80)
        self.declare_parameter('p1_stop_bottom_ratio', 0.95)
        self.declare_parameter('p1_stop_left_ratio', 0.35)
        self.declare_parameter('p1_stop_right_ratio', 0.65)
        self.declare_parameter('p1_stop_yellow_pixel_threshold', 5000)

        self.declare_parameter('p1_nav_top_ratio', 0.90)
        self.declare_parameter('p1_nav_bottom_ratio', 1.00)
        self.declare_parameter('p1_nav_crop_left_ratio', 0.15)
        self.declare_parameter('p1_nav_crop_right_ratio', 0.85)

        self.declare_parameter('p1_blue_h_min', 100)
        self.declare_parameter('p1_blue_h_max', 130)
        self.declare_parameter('p1_blue_s_min', 100)
        self.declare_parameter('p1_blue_s_max', 255)
        self.declare_parameter('p1_blue_v_min', 50)
        self.declare_parameter('p1_blue_v_max', 255)
        self.declare_parameter('p1_blue_min_area', 6500.0)
        self.declare_parameter('p1_blue_depth_patch_half', 1)

        # =========================
        # 状态机状态说明
        # =========================
        # STAGE1_CRUISE_BALL_AND_YELLOW:
        #   第一阶段巡航；沿两排球中间前进，同时看橙球和黄线。
        #   发现满足条件的橙球时转入 BALL_LATERAL_ALIGN；
        #   黄线达到第一阶段阈值时转入 STAGE1_ROTATE_LEFT_90。
        #
        # STAGE1_ROTATE_LEFT_90:
        #   第一阶段结束后的左转状态。
        #   使用固定角速度 + 固定仿真时间代替原地左跳，完成后直接进入 STAGE2_CRUISE_YELLOW_ONLY。
        #
        # STAGE2_CRUISE_YELLOW_ONLY:
        #   第二阶段巡航；主要看黄线，不进入撞球子状态。
        #   额外检测左侧蓝球/橙球：如果左侧近球靠近图像中心且距离足够近，
        #   就临时给固定右移 vy，避免机器狗左侧蹭球或撞球。
        #   黄线达到第二阶段阈值时转入 STAGE2_ROTATE_LEFT_90。
        #
        # STAGE2_ROTATE_LEFT_90:
        #   第二阶段结束后的左转状态。
        #   使用固定角速度 + 固定仿真时间代替原地左跳，完成后先进入
        #   STAGE2_MOVE_FORWARD_AFTER_LEFT_JUMP_TIME，按仿真时间前进固定时长，
        #   再进入 STAGE3_CRUISE_BALL_ONLY。
        #
        # STAGE2_MOVE_FORWARD_AFTER_LEFT_JUMP_TIME:
        #   第二阶段左跳完成后，按仿真时间以固定速度向前走固定时长。
        #   到时后进入 STAGE3_CRUISE_BALL_ONLY。
        #
        # STAGE3_CRUISE_BALL_ONLY:
        #   第三阶段巡航；主要处理橙球，同时继续看黄线。
        #   发现满足条件的橙球时转入 BALL_LATERAL_ALIGN；
        #   黄线达到第三阶段阈值时转入 STAGE3_ROTATE_BACK_180。
        #
        # STAGE3_ROTATE_BACK_180:
        #   第三阶段末尾回转状态。
        #   使用固定角速度 + 固定仿真时间代替两次原地左跳，近似完成 180° 掉头，
        #   完成后进入 STAGE3_FINAL_DECISION。
        #
        # STAGE3_FINAL_DECISION:
        #   第三阶段末尾决策状态。
        #   根据 orange_hit_count 判断是否已完成 4 个橙球：
        #   已完成则进入 STAGE3_GO_FINAL，否则进入 STAGE3_GO_SCAN。
        #
        # STAGE3_GO_SCAN:
        #   前往补扫区域的巡航状态。
        #   走到 yellow_ratio_scan 对应黄线位置后进入 STAGE3_SCAN_AND_HIT_LAST。
        #
        # STAGE3_SCAN_AND_HIT_LAST:
        #   在补扫区域寻找最后一个橙球。
        #   找到目标球则进入 BALL_LATERAL_ALIGN，未找到则继续巡航。
        #
        # STAGE3_GO_FINAL:
        #   前往最终出口线的巡航状态。
        #   黄线达到 yellow_ratio_final 后进入 STAGE3_ROTATE_LEFT_30。
        #
        # STAGE3_ROTATE_LEFT_30:
        #   最终出口前的收尾状态。
        #   不再使用 TF 判断结束，而是按仿真时间发送移动速度；
        #   到时后直接进入 STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT。
        #
        # STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT:
        #   移动完成后，不再使用 TF yaw 闭环，而是按仿真时间发送转向角速度；
        #   到时后进入 DONE。
        #
        # BALL_LATERAL_ALIGN:
        #   撞球子状态 1：横向对齐球。
        #   采用“小 vx + 主 vy”边前进边横移，把目标橙球送到机器狗正前方。
        #
        # BALL_HIT_CONFIRM_FORWARD:
        #   撞球子状态 2：确认后直接前冲撞击。
        #   进入时记录目标球深度，用“球深 + 额外前冲距离”生成撞击距离，
        #   并用仿真时间判断撞击是否完成。
        #
        # BALL_POST_HIT_SIDE_SHIFT:
        #   撞球子状态 3：撞完后只做左右横移。
        #   撞左球则向右移，撞右球则向左移。
        #   使用固定速度，持续指定仿真时间后直接回到保存的巡航状态 ball_return_state。
        #
        # DONE:
        #   全流程结束状态。
        #   持续发送停止命令，任务完成。

        # =========================
        # 橙球检测
        # =========================
        self.declare_parameter('orange_h_min', 5)
        self.declare_parameter('orange_h_max', 25)
        self.declare_parameter('orange_s_min', 100)
        self.declare_parameter('orange_s_max', 255)
        self.declare_parameter('orange_v_min', 80)
        self.declare_parameter('orange_v_max', 255)
        self.declare_parameter('orange_min_contour_area', 400.0)

        # =========================
        # 蓝球检测（只用于辅助中线）
        # =========================
        self.declare_parameter('blue_h_min', 90)
        self.declare_parameter('blue_h_max', 130)
        self.declare_parameter('blue_s_min', 80)
        self.declare_parameter('blue_s_max', 255)
        self.declare_parameter('blue_v_min', 50)
        self.declare_parameter('blue_v_max', 255)
        self.declare_parameter('blue_min_contour_area', 400.0)

        self.declare_parameter('prefer_nearest_ball', True)
        self.declare_parameter('min_ball_radius_to_trigger', 40.0)

        # =========================
        # 深度搜索
        # =========================
        self.declare_parameter('depth_search_half', 12)
        self.declare_parameter('valid_min_depth_m', 0.05)
        self.declare_parameter('valid_max_depth_m', 10.0)

        # =========================
        # 黄线检测
        # =========================
        self.declare_parameter('yellow_roi_top_ratio', 0.45)
        self.declare_parameter('yellow_roi_left_ratio', 0.4)
        self.declare_parameter('yellow_roi_right_ratio', 0.6)

        self.declare_parameter('yellow_h_min', 15)
        self.declare_parameter('yellow_h_max', 40)
        self.declare_parameter('yellow_s_min', 80)
        self.declare_parameter('yellow_s_max', 255)
        self.declare_parameter('yellow_v_min', 80)
        self.declare_parameter('yellow_v_max', 255)
        self.declare_parameter('yellow_min_contour_area', 100.0)

        self.declare_parameter('yellow_min_width_height_ratio', 2.5)
        self.declare_parameter('yellow_max_tilt_deg', 30.0)
        self.declare_parameter('yellow_center_tolerance_ratio', 0.15)
        self.declare_parameter('yellow_min_width_ratio', 0.45)

        self.declare_parameter('yellow_stop_line_y_ratio_stage1', 1.0)
        self.declare_parameter('yellow_stop_line_y_ratio_stage2', 0.75)
        self.declare_parameter('yellow_stop_line_y_ratio_stage3', 0.80)
        self.declare_parameter('yellow_stop_confirm_count', 1)

        self.declare_parameter('yellow_ratio_scan', 0.6)
        self.declare_parameter('yellow_ratio_final', 0.9)

        # =========================
        # 巡航中黄线角度矫正
        # =========================
        self.declare_parameter('yellow_angle_align_enabled', True)
        # 黄线角度矫正使用固定角速度：只根据 angle_deg 正负决定转向方向。
        self.declare_parameter('yellow_angle_align_fixed_wz', 0.15)
        self.declare_parameter('yellow_angle_align_deadband_deg', 0.5)

        # =========================
        # 巡航 / 中线
        # =========================
        self.declare_parameter('stage1_cruise_forward_speed', 0.30)
        self.declare_parameter('stage2_cruise_forward_speed', 0.40)
        self.declare_parameter('stage3_cruise_ball_only_speed', 0.30)

        # =========================
        # 第二赛段左侧近球固定右避让
        # =========================
        # STAGE2_CRUISE_YELLOW_ONLY 本来只看黄线向前走，容易蹭到左侧靠近中线的蓝球/橙球。
        # 这里不进入撞球子状态，只在危险球连续出现时给固定右移 vy。
        self.declare_parameter('stage2_left_ball_avoid_enabled', True)
        self.declare_parameter('stage2_left_ball_avoid_center_px', 130)
        self.declare_parameter('stage2_left_ball_avoid_depth_m', 0.45)
        self.declare_parameter('stage2_left_ball_avoid_vy', 0.12)
        self.declare_parameter('stage2_left_ball_avoid_confirm_frames', 2)
        self.declare_parameter('stage2_left_ball_avoid_min_radius', 8.0)

        self.declare_parameter('stage3_go_scan_speed', 0.30)
        self.declare_parameter('stage3_go_final_speed', 0.40)

        # 黄线预触发减速区：先减速，再真正触发切状态
        self.declare_parameter('yellow_slowdown_ratio_stage1', 0.90)
        self.declare_parameter('yellow_slowdown_ratio_stage2', 0.67)
        self.declare_parameter('yellow_slowdown_ratio_stage3', 0.70)
        self.declare_parameter('yellow_slowdown_ratio_scan', 0.52)
        self.declare_parameter('yellow_slowdown_ratio_final', 0.80)

        self.declare_parameter('stage1_yellow_slow_speed', 0.15)
        self.declare_parameter('stage2_yellow_slow_speed', 0.15)
        self.declare_parameter('stage3_yellow_slow_speed', 0.15)
        self.declare_parameter('stage3_go_scan_yellow_slow_speed', 0.15)
        self.declare_parameter('stage3_go_final_yellow_slow_speed', 0.15)

        self.declare_parameter('turn_trigger_distance_m', 0.45)

        # 中线对齐：使用固定 vy 横向平移修正。
        self.declare_parameter('center_cruise_vy_gain', 0.25)  # 保留兼容，当前不再使用
        self.declare_parameter('center_cruise_vy_max', 0.3)  # 保留兼容，当前不再使用
        self.declare_parameter('center_ok_px', 10.0)
        self.declare_parameter('center_cruise_fixed_vy', 0.10)
        # 左右参考球深度差太大时，不再按两球图像中点做中线对齐，
        # 而是向距离更远的小球一侧给一个较小固定 vy。
        self.declare_parameter('center_depth_diff_disable_align_m', 0.50)
        self.declare_parameter('center_far_side_fixed_vy', 0.03)

        # =========================
        # 对齐球阶段：小 vx + 主 vy
        # =========================
        self.declare_parameter('lateral_align_forward_speed', 0.125)
        self.declare_parameter('lateral_align_vy_gain', 0.30)
        self.declare_parameter('lateral_align_vy_max', 0.30)
        self.declare_parameter('lateral_align_vy_min', 0.10)
        self.declare_parameter('lateral_align_px_tol', 20.0)
        self.declare_parameter('lateral_align_confirm_count', 1)

        # =========================
        # 对齐球阶段：目标丢失 / 深度突然变远保护
        # =========================
        # 对齐过程中如果目标球突然识别不到：
        # 认为机器狗已经离球很近，球进入相机盲区/穿模，直接开始撞击。
        self.declare_parameter('ball_align_lost_go_hit', True)

        # 对齐过程中如果 best_target_ball 深度突然变大：
        # 认为近处 A 球丢失，当前识别到的是远处 B 球，不继续对齐，直接撞击。
        self.declare_parameter('ball_align_depth_jump_enabled', True)

        # 深度增加超过这个值，认为是跳变。
        # 例如上一帧 0.30m，下一帧 0.70m，增加 0.40m，就触发。
        self.declare_parameter('ball_align_depth_jump_threshold_m', 0.25)

        # 只有曾经看到目标小于这个距离，才启用“突然变远 -> 直接撞击”。
        # 避免远距离正常识别波动时误触发。
        self.declare_parameter('ball_align_near_depth_for_jump_m', 0.45)

        # =========================
        # 撞击 / 撞后移动
        # =========================
        # 撞击前冲：按仿真时间结束，不再用 TF 距离和 hit_extra_distance_m。
        self.declare_parameter('hit_forward_speed', 0.20)
        self.declare_parameter('hit_forward_duration_sec', 0.7)

        # 撞完后左右移动：固定速度 + 固定仿真时间。
        # 不再使用 post_hit_side_shift_distance_m / fast / slow / slowdown_ratio。
        self.declare_parameter('post_hit_side_shift_speed', 0.30)
        self.declare_parameter('post_hit_side_shift_duration_sec', 1.5)

        # =========================
        # 防重复撞同一颗球
        # =========================
        # 防重复触发现在只按仿真时间冷却判断，不再用 TF 位移。
        self.declare_parameter('ball_retrigger_cooldown_sec', 0.5)

        # 先按仿真时间移动一段，再按仿真时间转向一段。
        self.declare_parameter('stage3_final_left_shift_speed', 0.30)
        self.declare_parameter('stage3_final_left_shift_duration_sec', 0.6)
        self.declare_parameter('stage3_final_rotate_wz', 0.30)
        self.declare_parameter('stage3_final_rotate_duration_sec', 0.8)

        # =========================
        # 第二段左跳后按仿真时间前进
        # =========================
        self.declare_parameter('stage2_forward_after_left_jump_speed', 0.3)
        self.declare_parameter('stage2_forward_after_left_jump_duration_sec', 0.3)

        # =========================
        # 用固定角速度 + 固定仿真时间代替原地左跳转向
        # =========================
        # 实际值需要按仿真里机器狗的真实转角微调。
        self.declare_parameter('timed_turn_wz_90', 0.60)
        self.declare_parameter('timed_turn_duration_90_sec', 3.20)
        self.declare_parameter('timed_turn_wz_180', 0.60)
        self.declare_parameter('timed_turn_duration_180_sec', 6.4)
        self.declare_parameter('timed_turn_step_height', 0.02)

        # =========================
        # 第三赛段参数：S 弯巡航 + 出弯赛道对齐
        # 来自 part3_2.0.py / part3_vision.py，合并后不再使用 /vision 中间话题。
        # =========================
        self.declare_parameter('p3_stand_wait_sec', 2.0)
        self.declare_parameter('p3_stand_body_height', 0.20)
        self.declare_parameter('p3_stand_pitch', 0.30)
        self.declare_parameter('p3_step_height', 0.05)
        self.declare_parameter('p3_align_step_height', 0.10)

        self.declare_parameter('p3_s_curve_duration_sec', 16.5)
        self.declare_parameter('p3_base_forward_speed', 0.35)
        self.declare_parameter('p3_min_forward_speed', 0.00)
        self.declare_parameter('p3_kp_turn', 1.2)
        self.declare_parameter('p3_kp_lat', 0.2)
        self.declare_parameter('p3_kd_slowdown', 0.10)
        self.declare_parameter('p3_vision_timeout_sec', 1.0)
        self.declare_parameter('p3_fallback_forward_speed', 0.10)

        self.declare_parameter('p3_align_max_duration_sec', 8.0)
        self.declare_parameter('p3_align_lat_tol', 0.006)
        self.declare_parameter('p3_align_yaw_tol', 0.006)
        self.declare_parameter('p3_align_lat_gain', 0.6)
        self.declare_parameter('p3_align_yaw_gain', 2.0)
        self.declare_parameter('p3_align_lat_max', 0.15)
        self.declare_parameter('p3_align_yaw_max', 0.30)
        self.declare_parameter('p3_align_search_vx', 0.10)
        self.declare_parameter('p3_align_search_wz', 0.14)

        self.declare_parameter('p3_yellow_h_min', 20)
        self.declare_parameter('p3_yellow_h_max', 40)
        self.declare_parameter('p3_yellow_s_min', 50)
        self.declare_parameter('p3_yellow_s_max', 255)
        self.declare_parameter('p3_yellow_v_min', 150)
        self.declare_parameter('p3_yellow_v_max', 255)
        self.declare_parameter('p3_crop_left_ratio', 0.10)
        self.declare_parameter('p3_crop_right_ratio', 0.90)
        self.declare_parameter('p3_mid_top_ratio', 0.85)
        self.declare_parameter('p3_mid_bottom_ratio', 0.95)
        self.declare_parameter('p3_near_top_ratio', 0.95)
        self.declare_parameter('p3_near_bottom_ratio', 1.00)

        self.declare_parameter('p3_align_near_y_ratio', 0.90)
        self.declare_parameter('p3_align_far_y_ratio', 0.70)
        self.declare_parameter('p3_align_roi_left_ratio', 0.10)
        self.declare_parameter('p3_align_roi_right_ratio', 0.90)
        self.declare_parameter('p3_align_min_gap_px', 30)

        # =========================
        # 读取参数
        # =========================
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.global_frame = self.get_parameter('global_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.control_hz = float(self.get_parameter('control_hz').value)
        self.initial_state = self.get_parameter('initial_state').value
        self.second_stage_initial_state = self.get_parameter('second_stage_initial_state').value
        self.show_debug_vis = bool(self.get_parameter('show_debug_vis').value)
        self.show_yellow_mask = bool(self.get_parameter('show_yellow_mask').value)

        # =========================
        # 读取第一赛段参数（p1_ 前缀）
        # =========================
        self.p1_stand_wait_sec = float(self.get_parameter('p1_stand_wait_sec').value)
        self.p1_stand_body_height = float(self.get_parameter('p1_stand_body_height').value)

        self.p1_stage1_max_duration_sec = float(self.get_parameter('p1_stage1_max_duration_sec').value)
        self.p1_base_forward_speed = float(self.get_parameter('p1_base_forward_speed').value)
        self.p1_min_forward_speed = float(self.get_parameter('p1_min_forward_speed').value)
        self.p1_kp_turn = float(self.get_parameter('p1_kp_turn').value)
        self.p1_kp_lat = float(self.get_parameter('p1_kp_lat').value)
        self.p1_kd_slowdown = float(self.get_parameter('p1_kd_slowdown').value)
        self.p1_max_turn_speed = float(self.get_parameter('p1_max_turn_speed').value)
        self.p1_max_lateral_speed = float(self.get_parameter('p1_max_lateral_speed').value)
        self.p1_vision_timeout_sec = float(self.get_parameter('p1_vision_timeout_sec').value)

        self.p1_brake_duration_sec = float(self.get_parameter('p1_brake_duration_sec').value)
        self.p1_align_max_duration_sec = float(self.get_parameter('p1_align_max_duration_sec').value)
        self.p1_align_angle_deadband_rad = float(self.get_parameter('p1_align_angle_deadband_rad').value)
        self.p1_align_turn_kp = float(self.get_parameter('p1_align_turn_kp').value)
        self.p1_align_turn_max_wz = float(self.get_parameter('p1_align_turn_max_wz').value)

        self.p1_turn_duration_sec = float(self.get_parameter('p1_turn_duration_sec').value)
        self.p1_turn_forward_vel = float(self.get_parameter('p1_turn_forward_vel').value)
        self.p1_turn_yaw_vel = float(self.get_parameter('p1_turn_yaw_vel').value)

        self.p1_blue_target_distance_m = float(self.get_parameter('p1_blue_target_distance_m').value)
        self.p1_approach_blue_max_duration_sec = float(self.get_parameter('p1_approach_blue_max_duration_sec').value)
        self.p1_approach_blue_forward_speed = float(self.get_parameter('p1_approach_blue_forward_speed').value)

        self.p1_blind_left_duration_sec = float(self.get_parameter('p1_blind_left_duration_sec').value)
        self.p1_blind_left_vy = float(self.get_parameter('p1_blind_left_vy').value)
        self.p1_blind_left_vx = float(self.get_parameter('p1_blind_left_vx').value)

        self.p1_yellow_h_min = int(self.get_parameter('p1_yellow_h_min').value)
        self.p1_yellow_h_max = int(self.get_parameter('p1_yellow_h_max').value)
        self.p1_yellow_s_min = int(self.get_parameter('p1_yellow_s_min').value)
        self.p1_yellow_s_max = int(self.get_parameter('p1_yellow_s_max').value)
        self.p1_yellow_v_min = int(self.get_parameter('p1_yellow_v_min').value)
        self.p1_yellow_v_max = int(self.get_parameter('p1_yellow_v_max').value)

        self.p1_stop_top_ratio = float(self.get_parameter('p1_stop_top_ratio').value)
        self.p1_stop_bottom_ratio = float(self.get_parameter('p1_stop_bottom_ratio').value)
        self.p1_stop_left_ratio = float(self.get_parameter('p1_stop_left_ratio').value)
        self.p1_stop_right_ratio = float(self.get_parameter('p1_stop_right_ratio').value)
        self.p1_stop_yellow_pixel_threshold = int(self.get_parameter('p1_stop_yellow_pixel_threshold').value)

        self.p1_nav_top_ratio = float(self.get_parameter('p1_nav_top_ratio').value)
        self.p1_nav_bottom_ratio = float(self.get_parameter('p1_nav_bottom_ratio').value)
        self.p1_nav_crop_left_ratio = float(self.get_parameter('p1_nav_crop_left_ratio').value)
        self.p1_nav_crop_right_ratio = float(self.get_parameter('p1_nav_crop_right_ratio').value)

        self.p1_blue_h_min = int(self.get_parameter('p1_blue_h_min').value)
        self.p1_blue_h_max = int(self.get_parameter('p1_blue_h_max').value)
        self.p1_blue_s_min = int(self.get_parameter('p1_blue_s_min').value)
        self.p1_blue_s_max = int(self.get_parameter('p1_blue_s_max').value)
        self.p1_blue_v_min = int(self.get_parameter('p1_blue_v_min').value)
        self.p1_blue_v_max = int(self.get_parameter('p1_blue_v_max').value)
        self.p1_blue_min_area = float(self.get_parameter('p1_blue_min_area').value)
        self.p1_blue_depth_patch_half = int(self.get_parameter('p1_blue_depth_patch_half').value)

        self.orange_h_min = int(self.get_parameter('orange_h_min').value)
        self.orange_h_max = int(self.get_parameter('orange_h_max').value)
        self.orange_s_min = int(self.get_parameter('orange_s_min').value)
        self.orange_s_max = int(self.get_parameter('orange_s_max').value)
        self.orange_v_min = int(self.get_parameter('orange_v_min').value)
        self.orange_v_max = int(self.get_parameter('orange_v_max').value)
        self.orange_min_contour_area = float(self.get_parameter('orange_min_contour_area').value)

        self.blue_h_min = int(self.get_parameter('blue_h_min').value)
        self.blue_h_max = int(self.get_parameter('blue_h_max').value)
        self.blue_s_min = int(self.get_parameter('blue_s_min').value)
        self.blue_s_max = int(self.get_parameter('blue_s_max').value)
        self.blue_v_min = int(self.get_parameter('blue_v_min').value)
        self.blue_v_max = int(self.get_parameter('blue_v_max').value)
        self.blue_min_contour_area = float(self.get_parameter('blue_min_contour_area').value)

        self.prefer_nearest_ball = bool(self.get_parameter('prefer_nearest_ball').value)
        self.min_ball_radius_to_trigger = float(self.get_parameter('min_ball_radius_to_trigger').value)

        self.depth_search_half = int(self.get_parameter('depth_search_half').value)
        self.valid_min_depth_m = float(self.get_parameter('valid_min_depth_m').value)
        self.valid_max_depth_m = float(self.get_parameter('valid_max_depth_m').value)

        self.yellow_roi_top_ratio = float(self.get_parameter('yellow_roi_top_ratio').value)
        self.yellow_roi_left_ratio = float(self.get_parameter('yellow_roi_left_ratio').value)
        self.yellow_roi_right_ratio = float(self.get_parameter('yellow_roi_right_ratio').value)

        self.yellow_h_min = int(self.get_parameter('yellow_h_min').value)
        self.yellow_h_max = int(self.get_parameter('yellow_h_max').value)
        self.yellow_s_min = int(self.get_parameter('yellow_s_min').value)
        self.yellow_s_max = int(self.get_parameter('yellow_s_max').value)
        self.yellow_v_min = int(self.get_parameter('yellow_v_min').value)
        self.yellow_v_max = int(self.get_parameter('yellow_v_max').value)
        self.yellow_min_contour_area = float(self.get_parameter('yellow_min_contour_area').value)

        self.yellow_min_width_height_ratio = float(self.get_parameter('yellow_min_width_height_ratio').value)
        self.yellow_max_tilt_deg = float(self.get_parameter('yellow_max_tilt_deg').value)
        self.yellow_center_tolerance_ratio = float(self.get_parameter('yellow_center_tolerance_ratio').value)
        self.yellow_min_width_ratio = float(self.get_parameter('yellow_min_width_ratio').value)

        self.yellow_stop_line_y_ratio_stage1 = float(self.get_parameter('yellow_stop_line_y_ratio_stage1').value)
        self.yellow_stop_line_y_ratio_stage2 = float(self.get_parameter('yellow_stop_line_y_ratio_stage2').value)
        self.yellow_stop_line_y_ratio_stage3 = float(self.get_parameter('yellow_stop_line_y_ratio_stage3').value)
        self.yellow_stop_confirm_count = int(self.get_parameter('yellow_stop_confirm_count').value)

        self.yellow_ratio_scan = float(self.get_parameter('yellow_ratio_scan').value)
        self.yellow_ratio_final = float(self.get_parameter('yellow_ratio_final').value)

        self.yellow_angle_align_enabled = bool(self.get_parameter('yellow_angle_align_enabled').value)
        self.yellow_angle_align_fixed_wz = abs(float(self.get_parameter('yellow_angle_align_fixed_wz').value))
        self.yellow_angle_align_deadband_deg = float(self.get_parameter('yellow_angle_align_deadband_deg').value)

        self.stage1_cruise_forward_speed = float(self.get_parameter('stage1_cruise_forward_speed').value)
        self.stage2_cruise_forward_speed = float(self.get_parameter('stage2_cruise_forward_speed').value)
        self.stage3_cruise_ball_only_speed = float(self.get_parameter('stage3_cruise_ball_only_speed').value)

        self.stage2_left_ball_avoid_enabled = bool(self.get_parameter('stage2_left_ball_avoid_enabled').value)
        self.stage2_left_ball_avoid_center_px = float(self.get_parameter('stage2_left_ball_avoid_center_px').value)
        self.stage2_left_ball_avoid_depth_m = float(self.get_parameter('stage2_left_ball_avoid_depth_m').value)
        self.stage2_left_ball_avoid_vy = abs(float(self.get_parameter('stage2_left_ball_avoid_vy').value))
        self.stage2_left_ball_avoid_confirm_frames = int(self.get_parameter('stage2_left_ball_avoid_confirm_frames').value)
        self.stage2_left_ball_avoid_min_radius = float(self.get_parameter('stage2_left_ball_avoid_min_radius').value)
        self.stage3_go_scan_speed = float(self.get_parameter('stage3_go_scan_speed').value)
        self.stage3_go_final_speed = float(self.get_parameter('stage3_go_final_speed').value)

        self.yellow_slowdown_ratio_stage1 = float(self.get_parameter('yellow_slowdown_ratio_stage1').value)
        self.yellow_slowdown_ratio_stage2 = float(self.get_parameter('yellow_slowdown_ratio_stage2').value)
        self.yellow_slowdown_ratio_stage3 = float(self.get_parameter('yellow_slowdown_ratio_stage3').value)
        self.yellow_slowdown_ratio_scan = float(self.get_parameter('yellow_slowdown_ratio_scan').value)
        self.yellow_slowdown_ratio_final = float(self.get_parameter('yellow_slowdown_ratio_final').value)

        self.stage1_yellow_slow_speed = float(self.get_parameter('stage1_yellow_slow_speed').value)
        self.stage2_yellow_slow_speed = float(self.get_parameter('stage2_yellow_slow_speed').value)
        self.stage3_yellow_slow_speed = float(self.get_parameter('stage3_yellow_slow_speed').value)
        self.stage3_go_scan_yellow_slow_speed = float(self.get_parameter('stage3_go_scan_yellow_slow_speed').value)
        self.stage3_go_final_yellow_slow_speed = float(self.get_parameter('stage3_go_final_yellow_slow_speed').value)

        self.turn_trigger_distance_m = float(self.get_parameter('turn_trigger_distance_m').value)

        self.center_cruise_vy_gain = float(self.get_parameter('center_cruise_vy_gain').value)
        self.center_cruise_vy_max = float(self.get_parameter('center_cruise_vy_max').value)
        self.center_ok_px = float(self.get_parameter('center_ok_px').value)
        self.center_cruise_fixed_vy = abs(float(self.get_parameter('center_cruise_fixed_vy').value))
        self.center_depth_diff_disable_align_m = float(
            self.get_parameter('center_depth_diff_disable_align_m').value
        )
        self.center_far_side_fixed_vy = abs(float(self.get_parameter('center_far_side_fixed_vy').value))

        self.lateral_align_forward_speed = float(self.get_parameter('lateral_align_forward_speed').value)
        self.lateral_align_vy_gain = float(self.get_parameter('lateral_align_vy_gain').value)
        self.lateral_align_vy_max = float(self.get_parameter('lateral_align_vy_max').value)
        self.lateral_align_vy_min = float(self.get_parameter('lateral_align_vy_min').value)
        self.lateral_align_px_tol = float(self.get_parameter('lateral_align_px_tol').value)
        self.lateral_align_confirm_count = int(self.get_parameter('lateral_align_confirm_count').value)

        self.ball_align_lost_go_hit = bool(self.get_parameter('ball_align_lost_go_hit').value)
        self.ball_align_depth_jump_enabled = bool(self.get_parameter('ball_align_depth_jump_enabled').value)
        self.ball_align_depth_jump_threshold_m = float(
            self.get_parameter('ball_align_depth_jump_threshold_m').value
        )
        self.ball_align_near_depth_for_jump_m = float(
            self.get_parameter('ball_align_near_depth_for_jump_m').value
        )

        self.hit_forward_speed = float(self.get_parameter('hit_forward_speed').value)
        self.hit_forward_duration_sec = float(self.get_parameter('hit_forward_duration_sec').value)

        self.post_hit_side_shift_speed = float(self.get_parameter('post_hit_side_shift_speed').value)
        self.post_hit_side_shift_duration_sec = float(self.get_parameter('post_hit_side_shift_duration_sec').value)

        self.ball_retrigger_cooldown_sec = float(self.get_parameter('ball_retrigger_cooldown_sec').value)

        self.stage3_final_left_shift_speed = float(self.get_parameter('stage3_final_left_shift_speed').value)
        self.stage3_final_left_shift_duration_sec = float(
            self.get_parameter('stage3_final_left_shift_duration_sec').value
        )
        self.stage3_final_rotate_wz = float(self.get_parameter('stage3_final_rotate_wz').value)
        self.stage3_final_rotate_duration_sec = float(
            self.get_parameter('stage3_final_rotate_duration_sec').value
        )

        self.stage2_forward_after_left_jump_speed = float(
            self.get_parameter('stage2_forward_after_left_jump_speed').value)
        self.stage2_forward_after_left_jump_duration_sec = float(
            self.get_parameter('stage2_forward_after_left_jump_duration_sec').value)

        self.timed_turn_wz_90 = float(self.get_parameter('timed_turn_wz_90').value)
        self.timed_turn_duration_90_sec = float(self.get_parameter('timed_turn_duration_90_sec').value)
        self.timed_turn_wz_180 = float(self.get_parameter('timed_turn_wz_180').value)
        self.timed_turn_duration_180_sec = float(self.get_parameter('timed_turn_duration_180_sec').value)
        self.timed_turn_step_height = float(self.get_parameter('timed_turn_step_height').value)

        self.p3_stand_wait_sec = float(self.get_parameter('p3_stand_wait_sec').value)
        self.p3_stand_body_height = float(self.get_parameter('p3_stand_body_height').value)
        self.p3_stand_pitch = float(self.get_parameter('p3_stand_pitch').value)
        self.p3_step_height = float(self.get_parameter('p3_step_height').value)
        self.p3_align_step_height = float(self.get_parameter('p3_align_step_height').value)

        self.p3_s_curve_duration_sec = float(self.get_parameter('p3_s_curve_duration_sec').value)
        self.p3_base_forward_speed = float(self.get_parameter('p3_base_forward_speed').value)
        self.p3_min_forward_speed = float(self.get_parameter('p3_min_forward_speed').value)
        self.p3_kp_turn = float(self.get_parameter('p3_kp_turn').value)
        self.p3_kp_lat = float(self.get_parameter('p3_kp_lat').value)
        self.p3_kd_slowdown = float(self.get_parameter('p3_kd_slowdown').value)
        self.p3_vision_timeout_sec = float(self.get_parameter('p3_vision_timeout_sec').value)
        self.p3_fallback_forward_speed = float(self.get_parameter('p3_fallback_forward_speed').value)

        self.p3_align_max_duration_sec = float(self.get_parameter('p3_align_max_duration_sec').value)
        self.p3_align_lat_tol = float(self.get_parameter('p3_align_lat_tol').value)
        self.p3_align_yaw_tol = float(self.get_parameter('p3_align_yaw_tol').value)
        self.p3_align_lat_gain = float(self.get_parameter('p3_align_lat_gain').value)
        self.p3_align_yaw_gain = float(self.get_parameter('p3_align_yaw_gain').value)
        self.p3_align_lat_max = float(self.get_parameter('p3_align_lat_max').value)
        self.p3_align_yaw_max = float(self.get_parameter('p3_align_yaw_max').value)
        self.p3_align_search_vx = float(self.get_parameter('p3_align_search_vx').value)
        self.p3_align_search_wz = float(self.get_parameter('p3_align_search_wz').value)

        self.p3_yellow_h_min = int(self.get_parameter('p3_yellow_h_min').value)
        self.p3_yellow_h_max = int(self.get_parameter('p3_yellow_h_max').value)
        self.p3_yellow_s_min = int(self.get_parameter('p3_yellow_s_min').value)
        self.p3_yellow_s_max = int(self.get_parameter('p3_yellow_s_max').value)
        self.p3_yellow_v_min = int(self.get_parameter('p3_yellow_v_min').value)
        self.p3_yellow_v_max = int(self.get_parameter('p3_yellow_v_max').value)
        self.p3_crop_left_ratio = float(self.get_parameter('p3_crop_left_ratio').value)
        self.p3_crop_right_ratio = float(self.get_parameter('p3_crop_right_ratio').value)
        self.p3_mid_top_ratio = float(self.get_parameter('p3_mid_top_ratio').value)
        self.p3_mid_bottom_ratio = float(self.get_parameter('p3_mid_bottom_ratio').value)
        self.p3_near_top_ratio = float(self.get_parameter('p3_near_top_ratio').value)
        self.p3_near_bottom_ratio = float(self.get_parameter('p3_near_bottom_ratio').value)

        self.p3_align_near_y_ratio = float(self.get_parameter('p3_align_near_y_ratio').value)
        self.p3_align_far_y_ratio = float(self.get_parameter('p3_align_far_y_ratio').value)
        self.p3_align_roi_left_ratio = float(self.get_parameter('p3_align_roi_left_ratio').value)
        self.p3_align_roi_right_ratio = float(self.get_parameter('p3_align_roi_right_ratio').value)
        self.p3_align_min_gap_px = int(self.get_parameter('p3_align_min_gap_px').value)

        # =========================
        # 控制接口
        # =========================
        self.Ctrl = Robot_Ctrl()
        self.Ctrl.run()
        self.msg = robot_control_cmd_lcmt()
        if not hasattr(self.msg, 'life_count'):
            self.msg.life_count = 0

        self.bridge = CvBridge()

        # =========================
        # 第一赛段运行缓存（全部 p1_ 前缀，避免覆盖第二赛段 latest_* / yellow_* / ball_*）
        # =========================
        self.p1_state_start_time: Optional[float] = None
        self.p1_stand_sent = False
        self.p1_lateral_force = 0.0
        self.p1_stop_angle = 0.0
        self.p1_stop_flag = 0.0
        self.p1_last_update_time = 0.0
        self.p1_blue_distance_m = 0.0
        self.p1_blue_count = 0.0
        self.p1_blue_detections = []
        self.p1_latest_mask_yellow = None

        # =========================
        # 第三赛段运行缓存（p3_ 前缀）
        # =========================
        self.p3_state_start_time: Optional[float] = None
        self.p3_stand_sent = False
        self.p3_error_mid = 0.0
        self.p3_error_near = 0.0
        self.p3_last_update_time = 0.0
        self.p3_s4_lat = 0.0
        self.p3_s4_yaw = 0.0
        self.p3_s4_valid = 0.0
        self.p3_latest_mask = None
        self.p3_latest_mask_mid = None
        self.p3_latest_mask_near = None
        self.p3_align_near_center = -1.0
        self.p3_align_far_center = -1.0

        self.latest_depth = None
        self.latest_depth_encoding = None
        self.latest_bgr = None

        # TF 只作为可选调试/兼容信息使用。主状态机不再因为 TF 不可用而停止。
        self.last_known_pose: Optional[Tuple[float, float, float]] = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.rgb_w = 640
        self.rgb_h = 480

        self.latest_ball_result = {
            'has_ball': False,
            'ball_center': None,
            'ball_radius': None,
            'ball_depth_m': None,
            'img_shape': None,
            'error_x': None,
            'aligned': False,
            'depth_center': None,
            'depth_box': None,

            'orange_balls': [],
            'blue_balls': [],
            'left_balls': [],
            'right_balls': [],
            'has_center_reference': False,
            'center_error_px': None,
            'left_ref': None,
            'right_ref': None,
            'best_target_ball': None,
        }

        # 第二赛段左侧近球避让缓存：用于连续帧确认和可视化。
        self.stage2_left_ball_avoid_counter = 0
        self.stage2_left_ball_avoid_active = False
        self.stage2_left_ball_avoid_debug = {
            'enabled': self.stage2_left_ball_avoid_enabled,
            'active': False,
            'counter': 0,
            'danger_ball': None,
            'candidate_count': 0,
            'vy': 0.0,
            'reason': 'init',
        }

        self.latest_yellow_result = {
            'has_line': False,
            'line_bottom_y': None,
            'line_center': None,
            'img_shape': None,
            'angle_deg': None,
            'abs_tilt_deg': None,
            'bbox': None,
            'width_ratio': None,
            'wh_ratio': None,
            'require_front_horizontal': None,
        }

        self.state = self.initial_state
        # 整合后 initial_state 是 P1_STAND_WAIT；撞球子链回退状态必须默认指向第二赛段入口，
        # 避免异常路径下撞球结束后跳回第一赛段。
        self.ball_return_state = self.second_stage_initial_state

        self.yellow_stop_counter = 0

        # 第一阶段黄线“到底后出图”逻辑
        self.stage1_yellow_touched_bottom = False
        self.stage1_yellow_disappear_counter = 0

        self.pre_turn_pose: Optional[Tuple[float, float, float]] = None
        self.last_ball_done_time_sec: Optional[float] = None
        self.last_ball_done_pose: Optional[Tuple[float, float, float]] = None

        self.stage2_forward_after_left_jump_start_time_sec: Optional[float] = None
        self.stage3_final_left_shift_start_time_sec: Optional[float] = None
        self.stage3_final_rotate_start_time_sec: Optional[float] = None
        self.timed_turn_start_time_sec: Optional[float] = None

        self.lateral_align_counter = 0

        # BALL_LATERAL_ALIGN 阶段记录目标球深度变化。
        # 用于判断：近处球是否丢失、是否误切到远处其他球。
        self.ball_align_last_depth_m: Optional[float] = None
        self.ball_align_min_seen_depth_m: Optional[float] = None

        self.hit_start_pose: Optional[Tuple[float, float, float]] = None
        self.hit_start_depth_m: Optional[float] = None
        self.hit_start_time_sec: Optional[float] = None

        self.post_hit_side_shift_start_pose: Optional[Tuple[float, float, float]] = None
        self.post_hit_side_shift_start_time_sec: Optional[float] = None
        self.last_hit_side: Optional[str] = None
        self.side_shift_done: bool = False

        self.orange_hit_count = 0

        self.center_cruise_debug_info = {
            'mode': 'INIT',
            'left_depth': None,
            'right_depth': None,
            'depth_diff': None,
            'center_error_px': None,
            'vy': 0.0,
        }

        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.rgb_callback, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_profile_sensor_data)

        self.control_timer = self.create_timer(1.0 / self.control_hz, self.control_loop)

        self.send_stop_command()
        self.Ctrl.Wait_finish(12, 0)

        self.get_logger().info('CombinedStage1Stage2Node started.')
        self.get_logger().info(f'initial_state={self.state}')
        self.get_logger().info(f'rgb_topic={self.rgb_topic}')
        self.get_logger().info(f'depth_topic={self.depth_topic}')
        self.get_logger().info(f'tf: {self.global_frame} -> {self.base_frame}')

    # ============================================================
    # 第一赛段工具 / 视觉 / 控制状态机
    # ============================================================
    def p1_elapsed_in_state(self) -> float:
        now = self.now_sec()
        if self.p1_state_start_time is None:
            self.p1_state_start_time = now
        return now - self.p1_state_start_time

    def p1_send_stand_command(self):
        self.msg.mode = 12
        self.msg.gait_id = 0
        self._inc_life_count()
        self.msg.rpy_des = [0.0, 0.0, 0.0]
        self.msg.pos_des = [0.0, 0.0, self.p1_stand_body_height]
        self.Ctrl.Send_cmd(self.msg)
        self.get_logger().info('[P1 CMD] STAND', throttle_duration_sec=1.0)

    def p1_depth_to_meters_patch(self, patch: np.ndarray):
        if patch is None or patch.size == 0:
            return None

        if self.latest_depth_encoding == '16UC1':
            patch_m = patch.astype(np.float32) / 1000.0
        elif self.latest_depth_encoding == '32FC1':
            patch_m = patch.astype(np.float32)
        else:
            patch_m = patch.astype(np.float32)

        valid = patch_m[np.isfinite(patch_m)]
        valid = valid[(valid > self.valid_min_depth_m) & (valid < self.valid_max_depth_m)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def p1_process_stage1_yellow(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([self.p1_yellow_h_min, self.p1_yellow_s_min, self.p1_yellow_v_min], dtype=np.uint8)
        upper_yellow = np.array([self.p1_yellow_h_max, self.p1_yellow_s_max, self.p1_yellow_v_max], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        self.p1_latest_mask_yellow = mask_yellow

        lateral_force = 0.0
        stop_angle = 0.0
        stop_flag = 0.0

        stop_top = int(h * self.p1_stop_top_ratio)
        stop_bottom = int(h * self.p1_stop_bottom_ratio)
        stop_left = int(w * self.p1_stop_left_ratio)
        stop_right = int(w * self.p1_stop_right_ratio)
        mask_stop = mask_yellow[stop_top:stop_bottom, stop_left:stop_right]

        if cv2.countNonZero(mask_stop) > self.p1_stop_yellow_pixel_threshold:
            stop_flag = 1.0
            contours, _ = cv2.findContours(mask_stop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect).astype(np.int32)
                box_sorted = sorted(box, key=lambda p: p[0])
                left_pt, right_pt = box_sorted[0], box_sorted[-1]
                dx = right_pt[0] - left_pt[0]
                dy = right_pt[1] - left_pt[1]
                stop_angle = float(np.arctan2(dy, dx)) if dx != 0 else 0.0

        nav_top = int(h * self.p1_nav_top_ratio)
        nav_bottom = int(h * self.p1_nav_bottom_ratio)
        crop_left = int(w * self.p1_nav_crop_left_ratio)
        crop_right = int(w * self.p1_nav_crop_right_ratio)
        mask_nav = np.zeros_like(mask_yellow)
        mask_nav[nav_top:nav_bottom, crop_left:crop_right] = mask_yellow[nav_top:nav_bottom, crop_left:crop_right]

        M_nav = cv2.moments(mask_nav)
        if M_nav['m00'] > 0:
            cx_nav = int(M_nav['m10'] / M_nav['m00'])
            dist_nav = abs(cx_nav - w / 2)
            force_nav = ((w / 2 - dist_nav) / (w / 2)) ** 3
            lateral_force = float(force_nav) if cx_nav > w / 2 else -float(force_nav)

        self.p1_lateral_force = lateral_force
        self.p1_stop_angle = stop_angle
        self.p1_stop_flag = stop_flag
        self.p1_last_update_time = self.now_sec()

    def p1_process_blue_ball(self, frame: np.ndarray):
        self.p1_blue_detections = []
        self.p1_blue_distance_m = 0.0
        self.p1_blue_count = 0.0

        if self.latest_depth is None:
            return

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([self.p1_blue_h_min, self.p1_blue_s_min, self.p1_blue_v_min], dtype=np.uint8)
        upper_blue = np.array([self.p1_blue_h_max, self.p1_blue_s_max, self.p1_blue_v_max], dtype=np.uint8)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_depths = []
        dh, dw = self.latest_depth.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= self.p1_blue_min_area:
                continue
            M = cv2.moments(cnt)
            if M['m00'] <= 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            dx = int(cx * dw / max(w, 1))
            dy = int(cy * dh / max(h, 1))
            half = self.p1_blue_depth_patch_half
            x1 = max(0, dx - half)
            x2 = min(dw, dx + half + 1)
            y1 = max(0, dy - half)
            y2 = min(dh, dy + half + 1)
            depth_m = self.p1_depth_to_meters_patch(self.latest_depth[y1:y2, x1:x2])
            if depth_m is None:
                continue
            valid_depths.append(depth_m)
            self.p1_blue_detections.append({'center': (cx, cy), 'depth_m': depth_m, 'area': float(area)})

        if valid_depths:
            self.p1_blue_count = float(len(valid_depths))
            self.p1_blue_distance_m = float(min(valid_depths))

    def p1_control_loop(self):
        now = self.now_sec()
        if now <= 0.0:
            return

        elapsed = self.p1_elapsed_in_state()

        if self.state == 'P1_STAND_WAIT':
            if not self.p1_stand_sent:
                self.get_logger().info('[P1] 起立')
                self.p1_send_stand_command()
                self.p1_stand_sent = True
            if elapsed >= self.p1_stand_wait_sec:
                self.get_logger().info('[P1] 开始第一赛段黄线纠偏巡航')
                self.set_state('P1_STAGE1_CRUISE')
            return

        if self.state == 'P1_STAGE1_CRUISE':
            if self.p1_stop_flag > 0.5:
                self.get_logger().info('[P1] 看到横向黄线，进入刹车缓冲')
                self.set_state('P1_BRAKE_BUFFER')
                return

            if elapsed >= self.p1_stage1_max_duration_sec:
                self.get_logger().info('[P1] 第一赛段行走超时，进入刹车缓冲')
                self.set_state('P1_BRAKE_BUFFER')
                return

            if now - self.p1_last_update_time < self.p1_vision_timeout_sec:
                err = self.p1_lateral_force
                turn_speed = clamp(err * self.p1_kp_turn, -self.p1_max_turn_speed, self.p1_max_turn_speed)
                lateral_speed = clamp(err * self.p1_kp_lat, -self.p1_max_lateral_speed, self.p1_max_lateral_speed)
                speed_drop = abs(err) * self.p1_kd_slowdown
                forward_speed = max(self.p1_min_forward_speed, self.p1_base_forward_speed - speed_drop)
            else:
                forward_speed = self.p1_base_forward_speed
                lateral_speed = 0.0
                turn_speed = 0.0

            self.send_velocity_command(forward_speed, lateral_speed, turn_speed, step_height=0.13)
            return

        if self.state == 'P1_BRAKE_BUFFER':
            if elapsed >= self.p1_brake_duration_sec:
                self.get_logger().info('[P1] 开始根据横线角度调平')
                self.set_state('P1_ALIGN_STOP_LINE')
                return
            self.send_velocity_command(0.0, 0.0, 0.0, step_height=0.10)
            return

        if self.state == 'P1_ALIGN_STOP_LINE':
            angle_err = self.p1_stop_angle
            if abs(angle_err) < self.p1_align_angle_deadband_rad or self.p1_stop_flag < 0.5:
                self.get_logger().info(f'[P1] 调平完成或横线离开视野，angle_err={angle_err:.3f}')
                self.set_state('P1_TURN_LEFT_TO_STAGE2')
                return

            if elapsed >= self.p1_align_max_duration_sec:
                self.get_logger().info('[P1] 调平超时，进入左转')
                self.set_state('P1_TURN_LEFT_TO_STAGE2')
                return

            turn_speed = clamp(angle_err * self.p1_align_turn_kp, -self.p1_align_turn_max_wz, self.p1_align_turn_max_wz)
            self.send_velocity_command(0.0, 0.0, turn_speed, step_height=0.10)
            return

        if self.state == 'P1_TURN_LEFT_TO_STAGE2':
            if elapsed >= self.p1_turn_duration_sec:
                self.get_logger().info('[P1] 左转结束，开始寻找蓝球并前进')
                self.set_state('P1_APPROACH_BLUE_BALL')
                return
            self.send_velocity_command(self.p1_turn_forward_vel, 0.0, self.p1_turn_yaw_vel, step_height=0.10)
            return

        if self.state == 'P1_APPROACH_BLUE_BALL':
            if self.p1_blue_count >= 1.0:
                self.get_logger().info(
                    f'[P1] 锁定蓝球距离: {self.p1_blue_distance_m:.2f}m',
                    throttle_duration_sec=0.5
                )
                if self.p1_blue_distance_m <= self.p1_blue_target_distance_m:
                    self.get_logger().info('[P1] 到达蓝球目标距离，进入盲走左移')
                    self.set_state('P1_BLIND_LEFT_SHIFT')
                    return

            if elapsed >= self.p1_approach_blue_max_duration_sec:
                self.get_logger().info('[P1] 找蓝球前进超时，进入盲走左移')
                self.set_state('P1_BLIND_LEFT_SHIFT')
                return

            self.send_velocity_command(self.p1_approach_blue_forward_speed, 0.0, 0.0, step_height=0.10)
            return

        if self.state == 'P1_BLIND_LEFT_SHIFT':
            if elapsed >= self.p1_blind_left_duration_sec:
                # 关键：这里仍然不发 STOP，但进入第二赛段前重置第二赛段缓存，
                # 让第二赛段表现更接近“单独启动第二赛段”。
                self.get_logger().info(f'[P1] 第一赛段结束，不停顿切入第二赛段: {self.second_stage_initial_state}')
                self.enter_second_stage()
                return

            self.send_velocity_command(self.p1_blind_left_vx, self.p1_blind_left_vy, 0.0, step_height=0.10)
            return

        # 兜底：如果 P1 状态写错，直接切入第二赛段，避免卡死。
        self.get_logger().warn(f'[P1] unknown state={self.state}, jump to {self.second_stage_initial_state}')
        self.enter_second_stage()

    def enter_second_stage(self):
        """
        第一赛段结束后进入第二赛段。
        注意：这里不发送 STOP，保持连续衔接；只清理第二赛段内部状态缓存，
        尽量让第二赛段像单独启动时一样，从干净的状态机变量开始。
        """
        # 第二赛段黄线/球处理相关计数器
        self.yellow_stop_counter = 0
        self.stage1_yellow_touched_bottom = False
        self.stage1_yellow_disappear_counter = 0

        # 撞球子链相关缓存
        self.lateral_align_counter = 0
        self.ball_align_last_depth_m = None
        self.ball_align_min_seen_depth_m = None

        self.hit_start_pose = None
        self.hit_start_depth_m = None
        self.hit_start_time_sec = None
        self.post_hit_side_shift_start_pose = None
        self.post_hit_side_shift_start_time_sec = None
        self.last_hit_side = None
        self.side_shift_done = False
        self.ball_return_state = self.second_stage_initial_state

        # 第二赛段按时间运动状态缓存
        self.stage2_forward_after_left_jump_start_time_sec = None
        self.stage3_final_left_shift_start_time_sec = None
        self.stage3_final_rotate_start_time_sec = None
        self.timed_turn_start_time_sec = None

        # 防重复撞球缓存：进入第二赛段时清空，避免第一赛段运动时间影响第二赛段第一次触发。
        self.last_ball_done_time_sec = None
        self.last_ball_done_pose = None

        # 如果当前已经有最新图像，进入第二赛段前立刻用第二赛段算法刷新一次视觉结果，
        # 避免刚切状态的第一个 control tick 使用 P1 阶段的旧缓存。
        if self.latest_bgr is not None:
            self.latest_ball_result = self.detect_ball_scene(self.latest_bgr)
            self.latest_yellow_result = self.detect_yellow_stop_line(self.latest_bgr)

        self.set_state(self.second_stage_initial_state)

    # ============================================================
    # 基础工具
    # ============================================================
    def planar_distance(self, pose0: Tuple[float, float, float], pose1: Tuple[float, float, float]) -> float:
        x0, y0, _ = pose0
        x1, y1, _ = pose1
        return math.hypot(x1 - x0, y1 - y0)

    def local_lateral_displacement(self, start_pose: Tuple[float, float, float],
                                   current_pose: Tuple[float, float, float]) -> float:
        """
        计算 current_pose 相对 start_pose 的横向位移。
        返回值 > 0 表示相对 start_pose 的朝向向左移动；< 0 表示向右移动。
        这样可以避免把前后方向的漂移算进横移距离。
        """
        sx, sy, syaw = start_pose
        cx, cy, _ = current_pose
        dx = cx - sx
        dy = cy - sy
        return -math.sin(syaw) * dx + math.cos(syaw) * dy

    def apply_min_abs_velocity(self, v: float, v_min: float, deadband: float = 0.0) -> float:
        if abs(v) <= deadband:
            return 0.0
        if 0.0 < abs(v) < v_min:
            return math.copysign(v_min, v)
        return v

    def _inc_life_count(self):
        self.msg.life_count += 1
        if self.msg.life_count > 127:
            self.msg.life_count = 0

    # ============================================================
    # 控制
    # ============================================================
    def send_stop_command(self):
        self.msg.mode = 12
        self.msg.gait_id = 0
        self._inc_life_count()
        self.Ctrl.Send_cmd(self.msg)
        self.Ctrl.Wait_finish(12, 0)
        self.get_logger().info('[CMD] STOP', throttle_duration_sec=1.0)

    def send_velocity_command(self, vx: float, vy: float, wz: float, step_height: float = 0.02):
        self.msg.mode = 11
        self.msg.gait_id = 3
        self._inc_life_count()
        self.msg.vel_des = [vx, vy, wz]
        self.msg.step_height = [step_height, step_height]
        self.msg.rpy_des = [0.0, 0.0, 0.0]
        self.Ctrl.Send_cmd(self.msg)
        self.get_logger().info(
            f'[CMD] vel_des=[{vx:.3f}, {vy:.3f}, {wz:.3f}]',
            throttle_duration_sec=0.3
        )

    def send_left_jump_action_once(self):
        self.msg.mode = 16
        self.msg.gait_id = 0
        self._inc_life_count()
        self.Ctrl.Send_cmd(self.msg)
        self.Ctrl.Wait_finish(16, 0)

        self.msg.mode = 12
        self.msg.gait_id = 0
        self._inc_life_count()
        self.Ctrl.Send_cmd(self.msg)
        self.Ctrl.Wait_finish(12, 0)

    def execute_left_jump_turn(self, jump_count: int, next_state: str):
        for _ in range(jump_count):
            self.send_left_jump_action_once()
        self.set_state(next_state)

    def execute_timed_turn(self, wz: float, duration_sec: float, next_state: str) -> bool:
        """
        用固定角速度 + 固定仿真时间转向，代替原来的原地左跳。
        不发送 STOP，转完后直接切换到 next_state。
        """
        now = self.now_sec()

        if self.timed_turn_start_time_sec is None:
            self.timed_turn_start_time_sec = now
            self.get_logger().info(
                f'[TIMED_TURN] start: wz={wz:.3f}, duration={duration_sec:.2f}s, next={next_state}'
            )

        elapsed = now - self.timed_turn_start_time_sec

        if elapsed >= duration_sec:
            self.get_logger().info(
                f'[TIMED_TURN] done: elapsed={elapsed:.2f}s, next={next_state}'
            )
            self.timed_turn_start_time_sec = None
            self.set_state(next_state)
            return True

        self.send_velocity_command(
            0.0,
            0.0,
            wz,
            step_height=self.timed_turn_step_height
        )
        return True

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    # ============================================================
    # TF
    # ============================================================
    def get_current_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(self.global_frame, self.base_frame, Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        return (t.x, t.y, yaw)

    def can_trigger_ball_again(self, current_pose: Tuple[float, float, float]) -> bool:
        """
        防重复撞球：现在只按仿真时间冷却判断，不再依赖 TF 位移。
        current_pose 参数保留是为了兼容原调用位置。
        """
        if self.last_ball_done_time_sec is None:
            return True

        now_sec = self.now_sec()
        dt = now_sec - self.last_ball_done_time_sec
        cooldown_ok = dt >= self.ball_retrigger_cooldown_sec

        self.get_logger().info(
            f'ball retrigger check by sim time only: '
            f'dt={dt:.2f}s/{self.ball_retrigger_cooldown_sec:.2f}s, '
            f'cooldown_ok={cooldown_ok}',
            throttle_duration_sec=1.0
        )
        return cooldown_ok

    # ============================================================
    # 图像回调
    # ============================================================
    def depth_callback(self, msg: Image):
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'depth convert failed: {e}')
            return

        self.latest_depth = depth_img
        self.latest_depth_encoding = msg.encoding

    def rgb_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge convert failed: {e}')
            return

        self.latest_bgr = frame

        if isinstance(self.state, str) and self.state.startswith('P1_'):
            # 第一赛段运行时，只更新第一赛段视觉缓存。
            self.p1_process_stage1_yellow(frame)
            self.p1_process_blue_ball(frame)
            if self.show_debug_vis:
                self.show_debug_window(frame)
        elif isinstance(self.state, str) and self.state.startswith('P3_'):
            # 第三赛段运行时，只跑第三赛段黄线/S弯视觉，避免第二赛段球检测增加负载。
            self.p3_process_yellow_track(frame)
            if self.show_debug_vis:
                self.p3_show_debug_window(frame)
        else:
            # 第二赛段运行时，只跑第二赛段原本的视觉逻辑。
            self.latest_ball_result = self.detect_ball_scene(frame)
            self.latest_yellow_result = self.detect_yellow_stop_line(frame)
            if self.show_debug_vis:
                self.show_debug_window(frame)

    # ============================================================
    # 深度查值
    # ============================================================
    def get_depth_for_rgb_point(self, rgb_cx: int, rgb_cy: int):
        if self.latest_depth is None or self.latest_depth_encoding is None:
            return None, None, None

        depth = self.latest_depth
        encoding = self.latest_depth_encoding
        dh, dw = depth.shape[:2]

        depth_cx = int(rgb_cx * dw / max(self.rgb_w, 1))
        depth_cy = int(rgb_cy * dh / max(self.rgb_h, 1))

        x1 = max(0, depth_cx - self.depth_search_half)
        x2 = min(dw, depth_cx + self.depth_search_half + 1)
        y1 = max(0, depth_cy - self.depth_search_half)
        y2 = min(dh, depth_cy + self.depth_search_half + 1)

        patch = depth[y1:y2, x1:x2]

        if encoding == '16UC1':
            patch_m = patch.astype(np.float32) / 1000.0
        elif encoding == '32FC1':
            patch_m = patch.astype(np.float32)
        else:
            return None, (depth_cx, depth_cy), (x1, y1, x2, y2)

        valid = patch_m[np.isfinite(patch_m)]
        valid = valid[(valid > self.valid_min_depth_m) & (valid < self.valid_max_depth_m)]

        if valid.size == 0:
            return None, (depth_cx, depth_cy), (x1, y1, x2, y2)

        depth_m = float(np.percentile(valid, 20))
        return depth_m, (depth_cx, depth_cy), (x1, y1, x2, y2)

    # ============================================================
    # 球检测
    # ============================================================
    def detect_color_ball_candidates(
            self,
            frame: np.ndarray,
            h_min: int, h_max: int,
            s_min: int, s_max: int,
            v_min: int, v_max: int,
            min_contour_area: float,
            color_name: str
    ) -> List[Dict]:
        h, w = frame.shape[:2]
        self.rgb_w = w
        self.rgb_h = h

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_contour_area:
                continue

            # 圆心仍然用最小外接圆
            (cx_f, cy_f), r_circle = cv2.minEnclosingCircle(cnt)
            cx = int(cx_f)
            cy = int(cy_f)
            r_circle = float(r_circle)

            # 面积等效半径
            r_eq = math.sqrt(area / math.pi)

            # 最终半径：只信较小的那个
            radius = min(r_circle, r_eq)

            depth_m, depth_center, depth_box = self.get_depth_for_rgb_point(cx, cy)
            if depth_m is None:
                continue

            image_center_x = w // 2
            error_x = cx - image_center_x
            side = 'left' if cx < image_center_x else 'right'

            candidates.append({
                'color': color_name,
                'center': (cx, cy),
                'radius': radius,
                'radius_circle': r_circle,
                'radius_eq': r_eq,
                'area': float(area),
                'depth_m': depth_m,
                'error_x': int(error_x),
                'depth_center': depth_center,
                'depth_box': depth_box,
                'side': side,
            })

        return candidates

    def choose_side_reference_ball(self, balls: List[Dict]) -> Optional[Dict]:
        if len(balls) == 0:
            return None
        return min(balls, key=lambda b: b['depth_m'])

    def choose_best_target_orange_ball(self, orange_balls: List[Dict]) -> Optional[Dict]:
        if len(orange_balls) == 0:
            return None
        if self.prefer_nearest_ball:
            return min(orange_balls, key=lambda b: b['depth_m'])
        return min(orange_balls, key=lambda b: b['depth_m'] + 0.002 * abs(b['error_x']))

    def detect_ball_scene(self, frame: np.ndarray) -> Dict:
        h, w = frame.shape[:2]
        self.rgb_w = w
        self.rgb_h = h

        orange_balls = self.detect_color_ball_candidates(
            frame,
            self.orange_h_min, self.orange_h_max,
            self.orange_s_min, self.orange_s_max,
            self.orange_v_min, self.orange_v_max,
            self.orange_min_contour_area,
            'orange'
        )

        blue_balls = self.detect_color_ball_candidates(
            frame,
            self.blue_h_min, self.blue_h_max,
            self.blue_s_min, self.blue_s_max,
            self.blue_v_min, self.blue_v_max,
            self.blue_min_contour_area,
            'blue'
        )

        all_balls = orange_balls + blue_balls
        image_center_x = w // 2

        left_balls = [b for b in all_balls if b['center'][0] < image_center_x]
        right_balls = [b for b in all_balls if b['center'][0] >= image_center_x]

        left_ref = self.choose_side_reference_ball(left_balls)
        right_ref = self.choose_side_reference_ball(right_balls)

        has_center_reference = (left_ref is not None and right_ref is not None)
        center_error_px = None
        if has_center_reference:
            left_cx = left_ref['center'][0]
            right_cx = right_ref['center'][0]
            lane_mid_x = 0.5 * (left_cx + right_cx)
            center_error_px = lane_mid_x - image_center_x

        best_target_ball = self.choose_best_target_orange_ball(orange_balls)

        if best_target_ball is None:
            return {
                'has_ball': False,
                'ball_center': None,
                'ball_radius': None,
                'ball_depth_m': None,
                'img_shape': (h, w),
                'error_x': None,
                'aligned': False,
                'depth_center': None,
                'depth_box': None,

                'orange_balls': orange_balls,
                'blue_balls': blue_balls,
                'left_balls': left_balls,
                'right_balls': right_balls,
                'has_center_reference': has_center_reference,
                'center_error_px': center_error_px,
                'left_ref': left_ref,
                'right_ref': right_ref,
                'best_target_ball': None,
            }

        return {
            'has_ball': True,
            'ball_center': best_target_ball['center'],
            'ball_radius': best_target_ball['radius'],
            'ball_depth_m': best_target_ball['depth_m'],
            'img_shape': (h, w),
            'error_x': best_target_ball['error_x'],
            'aligned': abs(best_target_ball['error_x']) <= self.lateral_align_px_tol,
            'depth_center': best_target_ball['depth_center'],
            'depth_box': best_target_ball['depth_box'],

            'orange_balls': orange_balls,
            'blue_balls': blue_balls,
            'left_balls': left_balls,
            'right_balls': right_balls,
            'has_center_reference': has_center_reference,
            'center_error_px': center_error_px,
            'left_ref': left_ref,
            'right_ref': right_ref,
            'best_target_ball': best_target_ball,
        }

    # ============================================================
    # 黄线检测
    # ============================================================
    def is_front_horizontal_yellow_line(self, cnt, roi_shape) -> bool:
        """
        判断黄色轮廓是否为前方横向停止线。

        改进版：不再使用 minAreaRect / fitLine 的角度作为过滤条件，
        避免同一条横线在 0° 和 90° 之间跳变导致误拒绝。

        只使用更严格的 bbox 条件：
        1. wh_ratio = bbox_width / bbox_height 足够大，必须像横向长条；
        2. width_ratio = bbox_width / roi_width 足够大，必须横跨较大前方区域；
        3. center_offset_ratio 足够小，必须靠近 ROI 中心，避免旁边黄线误判。
        """
        _, roi_w = roi_shape[:2]

        area = cv2.contourArea(cnt)
        if area < self.yellow_min_contour_area:
            return False

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh <= 0:
            return False

        wh_ratio = bw / float(bh)
        if wh_ratio < self.yellow_min_width_height_ratio:
            return False

        width_ratio = bw / float(max(roi_w, 1))
        if width_ratio < self.yellow_min_width_ratio:
            return False

        cx = x + bw / 2.0
        roi_cx = roi_w / 2.0
        center_offset_ratio = abs(cx - roi_cx) / float(max(roi_w, 1))
        if center_offset_ratio > self.yellow_center_tolerance_ratio:
            return False

        return True

    def get_signed_yellow_line_angle_deg(self, cnt) -> float:
        """
        复用原黄线轮廓，估计其相对图像水平线的有符号角度。
        0 度表示基本水平；正负号只用于后面的 wz 矫正。
        """
        if cnt is None or len(cnt) < 2:
            return 0.0
        vx, vy, _, _ = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        vx = float(vx)
        vy = float(vy)
        angle = math.degrees(math.atan2(vy, vx))
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        return float(angle)

    def detect_yellow_stop_line(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]

        roi_top = int(h * self.yellow_roi_top_ratio)
        roi_left = int(w * self.yellow_roi_left_ratio)
        roi_right = int(w * self.yellow_roi_right_ratio)

        roi = frame[roi_top:h, roi_left:roi_right]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([self.yellow_h_min, self.yellow_s_min, self.yellow_v_min], dtype=np.uint8)
        upper_yellow = np.array([self.yellow_h_max, self.yellow_s_max, self.yellow_v_max], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_score = -1.0

        # STAGE1_CRUISE_BALL_AND_YELLOW 和 STAGE3_CRUISE_BALL_ONLY
        # 这两个状态不要求黄线候选必须是“前方横线”。
        require_front_horizontal = self.state not in (
            'STAGE1_CRUISE_BALL_AND_YELLOW',
            'STAGE3_CRUISE_BALL_ONLY',
            'STAGE3_GO_SCAN',
            'STAGE3_GO_FINAL',
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.yellow_min_contour_area:
                continue

            if require_front_horizontal and not self.is_front_horizontal_yellow_line(cnt, roi.shape):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            score = y + bh

            if score > best_score:
                best_score = score
                best_contour = cnt

        if best_contour is None:
            return {
                'has_line': False,
                'line_bottom_y': None,
                'line_center': None,
                'img_shape': (h, w),
                'angle_deg': None,
                'abs_tilt_deg': None,
                'bbox': None,
                'width_ratio': None,
                'wh_ratio': None,
                'require_front_horizontal': bool(require_front_horizontal),
            }

        x, y, bw, bh = cv2.boundingRect(best_contour)
        line_bottom_y = roi_top + y + bh
        cx = roi_left + x + bw // 2
        cy = roi_top + y + bh // 2
        angle_deg = self.get_signed_yellow_line_angle_deg(best_contour)
        abs_tilt_deg = abs(angle_deg)
        width_ratio = bw / float(max(roi_right - roi_left, 1))
        wh_ratio = bw / float(max(bh, 1))

        return {
            'has_line': True,
            'line_bottom_y': int(line_bottom_y),
            'line_center': (int(cx), int(cy)),
            'img_shape': (h, w),
            'angle_deg': float(angle_deg),
            'abs_tilt_deg': float(abs_tilt_deg),
            'bbox': (int(roi_left + x), int(roi_top + y), int(roi_left + x + bw), int(roi_top + y + bh)),
            'width_ratio': float(width_ratio),
            'wh_ratio': float(wh_ratio),
            'require_front_horizontal': bool(require_front_horizontal),
        }

    # ============================================================
    # 状态切换
    # ============================================================
    def set_state(self, new_state: str):
        if new_state != self.state:
            self.get_logger().info(f'STATE: {self.state} -> {new_state}')
            self.state = new_state

            if new_state.startswith('P1_'):
                self.p1_state_start_time = None

            if new_state.startswith('P3_'):
                self.p3_state_start_time = None
                if new_state == 'P3_STAND_WAIT':
                    self.p3_stand_sent = False

            if new_state in (
                    'STAGE1_CRUISE_BALL_AND_YELLOW',
                    'STAGE2_CRUISE_YELLOW_ONLY',
                    'STAGE3_CRUISE_BALL_ONLY',
                    'STAGE3_GO_SCAN',
                    'STAGE3_GO_FINAL'
            ):
                self.yellow_stop_counter = 0

            if new_state == 'STAGE1_CRUISE_BALL_AND_YELLOW':
                self.stage1_yellow_touched_bottom = False
                self.stage1_yellow_disappear_counter = 0

            if new_state in (
                    'STAGE1_ROTATE_LEFT_90',
                    'STAGE2_ROTATE_LEFT_90',
                    'STAGE3_ROTATE_BACK_180',
            ):
                self.timed_turn_start_time_sec = None

            if new_state == 'STAGE3_ROTATE_LEFT_30':
                self.stage3_final_left_shift_start_time_sec = None

            if new_state == 'STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT':
                self.stage3_final_rotate_start_time_sec = None

            if new_state == 'STAGE2_MOVE_FORWARD_AFTER_LEFT_JUMP_TIME':
                self.stage2_forward_after_left_jump_start_time_sec = None

            if new_state == 'BALL_LATERAL_ALIGN':
                self.lateral_align_counter = 0

                # 每次进入对齐球阶段，都重新记录深度变化。
                self.ball_align_last_depth_m = None
                self.ball_align_min_seen_depth_m = None

                # 锁定“开始对齐时”的目标球所在侧。
                # 后面对齐过程中目标球可能因为机器人横移跑到画面另一边，
                # 撞后横移方向仍然使用这里锁定的初始 side，不再在撞击前冲时覆盖。
                target = self.latest_ball_result.get('best_target_ball') if isinstance(self.latest_ball_result, dict) else None

                if target is not None:
                    self.last_hit_side = target.get('side')

                    depth = target.get('depth_m', None)
                    if depth is not None:
                        self.ball_align_last_depth_m = float(depth)
                        self.ball_align_min_seen_depth_m = float(depth)

                    self.get_logger().info(
                        f'BALL_LATERAL_ALIGN lock hit side at align start: '
                        f'last_hit_side={self.last_hit_side}, '
                        f'target_center={target.get("center")}, '
                        f'error_x={target.get("error_x")}, '
                        f'depth={target.get("depth_m")}, '
                        f'radius={target.get("radius")}'
                    )
                else:
                    self.last_hit_side = None
                    self.get_logger().warn(
                        'BALL_LATERAL_ALIGN start but target is None; '
                        'last_hit_side=None, depth cache cleared'
                    )
            if new_state == 'BALL_HIT_CONFIRM_FORWARD':
                self.hit_start_pose = None
                self.hit_start_depth_m = None
                self.hit_start_time_sec = None

            if new_state == 'BALL_POST_HIT_SIDE_SHIFT':
                self.post_hit_side_shift_start_pose = None
                self.post_hit_side_shift_start_time_sec = None
                self.side_shift_done = False

    # ============================================================
    # 判定
    # ============================================================
    def yellow_reached(self, yellow_result: dict, ratio: float) -> bool:
        if yellow_result['img_shape'] is None or not yellow_result['has_line']:
            self.yellow_stop_counter = 0
            return False

        h, _ = yellow_result['img_shape']
        stop_y_threshold = int(h * ratio)

        if yellow_result['line_bottom_y'] is not None and yellow_result['line_bottom_y'] >= stop_y_threshold:
            self.yellow_stop_counter += 1
        else:
            self.yellow_stop_counter = 0

        self.get_logger().info(
            f'yellow line check: bottom={yellow_result["line_bottom_y"]}, '
            f'threshold={stop_y_threshold}, counter={self.yellow_stop_counter}/{self.yellow_stop_confirm_count}',
            throttle_duration_sec=0.5
        )
        return self.yellow_stop_counter >= self.yellow_stop_confirm_count

    def stage1_yellow_passed(self, yellow_result: dict) -> bool:
        if yellow_result['img_shape'] is None:
            self.stage1_yellow_disappear_counter = 0
            return False

        h, _ = yellow_result['img_shape']
        bottom_threshold = int(h * self.yellow_stop_line_y_ratio_stage1)

        # 还看得到黄线
        if yellow_result['has_line'] and yellow_result['line_bottom_y'] is not None:
            current_bottom = yellow_result['line_bottom_y']

            # 先记录：黄线已经到过图像最底下
            if current_bottom >= bottom_threshold:
                if not self.stage1_yellow_touched_bottom:
                    self.get_logger().info(
                        f'STAGE1 yellow touched bottom: bottom={current_bottom}, '
                        f'threshold={bottom_threshold}'
                    )
                self.stage1_yellow_touched_bottom = True
                self.stage1_yellow_disappear_counter = 0
                return False

            # 如果之前已经到底过，现在又重新低于阈值
            # 就认为近处这条黄线已经过去了
            if self.stage1_yellow_touched_bottom and current_bottom < bottom_threshold:
                self.stage1_yellow_disappear_counter += 1
                self.get_logger().info(
                    f'STAGE1 yellow dropped below threshold after touching bottom: '
                    f'bottom={current_bottom}, threshold={bottom_threshold}, '
                    f'counter={self.stage1_yellow_disappear_counter}/{self.yellow_stop_confirm_count}',
                    throttle_duration_sec=0.2
                )
                return self.stage1_yellow_disappear_counter >= self.yellow_stop_confirm_count

            self.stage1_yellow_disappear_counter = 0
            return False

        # 如果已经到底过，并且现在彻底看不到黄线，也算通过
        if self.stage1_yellow_touched_bottom:
            self.stage1_yellow_disappear_counter += 1
            self.get_logger().info(
                f'STAGE1 yellow disappeared after touching bottom: '
                f'counter={self.stage1_yellow_disappear_counter}/{self.yellow_stop_confirm_count}',
                throttle_duration_sec=0.2
            )
            return self.stage1_yellow_disappear_counter >= self.yellow_stop_confirm_count

        self.stage1_yellow_disappear_counter = 0
        return False

    # ============================================================
    # 巡航中线
    # ============================================================
    def send_center_cruise_command(self, ball: Dict, vx: float):
        self.send_center_cruise_command_with_wz(ball, vx, 0.0)

    def compute_center_cruise_vy(self, ball: Dict) -> float:
        """
        中线对齐固定 vy 横向平移版本，加入“左右深度差保护”。

        1. 如果左右参考球都有，并且两个球深度差太大：
           不再使用两球图像中点 center_error_px 对齐，
           而是向距离更远的小球那一侧给一个较小固定 vy。

           left_depth > right_depth：左球更远 -> 向左小幅横移 -> vy > 0
           right_depth > left_depth：右球更远 -> 向右小幅横移 -> vy < 0

        2. 如果左右参考球深度差不大：
           使用原来的图像中点对齐逻辑，超过 center_ok_px 后给固定 center_cruise_fixed_vy。

        如果实测方向反了，只需要对调对应分支里的正负号。
        """
        self.center_cruise_debug_info = {
            'mode': 'NO_CENTER_REF',
            'left_depth': None,
            'right_depth': None,
            'depth_diff': None,
            'center_error_px': ball.get('center_error_px', None),
            'vy': 0.0,
        }

        if not ball.get('has_center_reference', False):
            return 0.0

        left_ref = ball.get('left_ref')
        right_ref = ball.get('right_ref')
        err_px = ball.get('center_error_px', None)

        if left_ref is None or right_ref is None:
            return 0.0

        left_depth = left_ref.get('depth_m', None)
        right_depth = right_ref.get('depth_m', None)

        self.center_cruise_debug_info.update({
            'left_depth': left_depth,
            'right_depth': right_depth,
            'center_error_px': err_px,
        })

        # 先判断左右参考球深度差。
        # 如果深度差太大，说明这两个球不太适合直接拿来做“中点对齐”。
        if left_depth is not None and right_depth is not None:
            depth_diff = abs(float(left_depth) - float(right_depth))
            self.center_cruise_debug_info['depth_diff'] = depth_diff

            if depth_diff >= self.center_depth_diff_disable_align_m:
                if float(left_depth) > float(right_depth):
                    # 左边球更远：向左边给一个小 vy
                    vy = abs(self.center_far_side_fixed_vy)
                    far_side = 'left'
                else:
                    # 右边球更远：向右边给一个小 vy
                    vy = -abs(self.center_far_side_fixed_vy)
                    far_side = 'right'

                self.center_cruise_debug_info.update({
                    'mode': 'FAR_SIDE_BIAS',
                    'far_side': far_side,
                    'vy': vy,
                })

                self.get_logger().info(
                    f'center far-side bias: left_depth={float(left_depth):.3f}, '
                    f'right_depth={float(right_depth):.3f}, '
                    f'diff={depth_diff:.3f}/{self.center_depth_diff_disable_align_m:.3f}, '
                    f'far_side={far_side}, vy={vy:.3f}',
                    throttle_duration_sec=0.3
                )
                return vy

        # 深度差不大，或者深度无效时，退回原来的图像中线对齐。
        if err_px is None:
            self.center_cruise_debug_info['mode'] = 'NO_CENTER_ERR'
            return 0.0

        err_px = float(err_px)
        self.center_cruise_debug_info['center_error_px'] = err_px

        if abs(err_px) <= self.center_ok_px:
            self.center_cruise_debug_info.update({
                'mode': 'CENTER_OK',
                'vy': 0.0,
            })
            return 0.0

        # 沿用原来 vy 横向平移对齐的方向约定：
        # center_error_px > 0 -> vy < 0；center_error_px < 0 -> vy > 0。
        if err_px > 0.0:
            vy = -abs(self.center_cruise_fixed_vy)
        else:
            vy = abs(self.center_cruise_fixed_vy)

        self.center_cruise_debug_info.update({
            'mode': 'NORMAL_ALIGN',
            'vy': vy,
        })

        self.get_logger().info(
            f'center lateral align fixed: center_error_px={err_px:.1f}, '
            f'deadband={self.center_ok_px:.1f}, vy={vy:.3f}',
            throttle_duration_sec=0.3
        )
        return vy

    def send_center_cruise_command_with_wz(self, ball: Dict, vx: float, wz: float):
        center_vy = self.compute_center_cruise_vy(ball)

        self.get_logger().info(
            f'cruise correction: center_vy={center_vy:.3f}, yellow_wz={wz:.3f}',
            throttle_duration_sec=0.3
        )

        # 中线对齐使用固定 vy 横向平移；黄线角度矫正仍然使用 wz。
        # 两者不冲突，可以同时发送。
        self.send_velocity_command(vx, center_vy, wz)

    def choose_stage2_left_danger_ball(self, ball: Dict) -> Optional[Dict]:
        """
        第二赛段左侧近球避让目标选择。

        只使用已经由 detect_ball_scene() 检出的蓝球和橙球，不单独新增视觉检测。
        触发条件：
        1. 球在图像中心左侧；
        2. 球距离图像中心不能太远：image_center_x - cx <= stage2_left_ball_avoid_center_px；
        3. 球深度足够近：depth_m <= stage2_left_ball_avoid_depth_m；
        4. 球半径达到最小值，避免小噪声触发。
        """
        if not self.stage2_left_ball_avoid_enabled:
            return None
        if ball is None or ball.get('img_shape') is None:
            return None

        h, w = ball['img_shape']
        image_center_x = w / 2.0

        candidates = []
        for b in ball.get('orange_balls', []) + ball.get('blue_balls', []):
            center = b.get('center')
            depth_m = b.get('depth_m')
            radius = float(b.get('radius', 0.0))
            if center is None or depth_m is None:
                continue

            cx = float(center[0])
            if cx >= image_center_x:
                continue

            dist_to_center_px = image_center_x - cx
            if dist_to_center_px > self.stage2_left_ball_avoid_center_px:
                continue
            if float(depth_m) > self.stage2_left_ball_avoid_depth_m:
                continue
            if radius < self.stage2_left_ball_avoid_min_radius:
                continue

            item = dict(b)
            item['stage2_avoid_dist_to_center_px'] = float(dist_to_center_px)
            candidates.append(item)

        self.stage2_left_ball_avoid_debug['candidate_count'] = len(candidates)

        if not candidates:
            return None

        # 优先避让最近的；如果深度接近，再优先避让更靠近图像中心的。
        return min(candidates, key=lambda b: (float(b.get('depth_m', 999.0)), float(b.get('stage2_avoid_dist_to_center_px', 9999.0))))

    def compute_stage2_left_ball_avoid_vy(self, ball: Dict) -> float:
        """
        STAGE2_CRUISE_YELLOW_ONLY 专用：左侧蓝球/橙球靠近路线时，固定向右偏移。

        当前代码约定：vy < 0 通常表示向右移动；如果实测方向反了，
        只需要把下面 return 的 -abs(...) 改成 +abs(...)，或者把参数值改负后自行扩展。
        """
        debug = {
            'enabled': self.stage2_left_ball_avoid_enabled,
            'active': False,
            'counter': self.stage2_left_ball_avoid_counter,
            'danger_ball': None,
            'candidate_count': 0,
            'vy': 0.0,
            'reason': 'disabled' if not self.stage2_left_ball_avoid_enabled else 'no_danger_ball',
        }
        self.stage2_left_ball_avoid_debug = debug

        if not self.stage2_left_ball_avoid_enabled:
            self.stage2_left_ball_avoid_counter = 0
            self.stage2_left_ball_avoid_active = False
            return 0.0

        danger = self.choose_stage2_left_danger_ball(ball)
        debug['candidate_count'] = self.stage2_left_ball_avoid_debug.get('candidate_count', 0)

        if danger is None:
            self.stage2_left_ball_avoid_counter = 0
            self.stage2_left_ball_avoid_active = False
            debug.update({
                'active': False,
                'counter': 0,
                'danger_ball': None,
                'vy': 0.0,
                'reason': 'no_danger_ball',
            })
            self.stage2_left_ball_avoid_debug = debug
            return 0.0

        self.stage2_left_ball_avoid_counter += 1
        confirm_frames = max(1, int(self.stage2_left_ball_avoid_confirm_frames))
        active = self.stage2_left_ball_avoid_counter >= confirm_frames
        self.stage2_left_ball_avoid_active = active

        vy = -abs(self.stage2_left_ball_avoid_vy) if active else 0.0
        debug.update({
            'active': active,
            'counter': self.stage2_left_ball_avoid_counter,
            'danger_ball': danger,
            'vy': vy,
            'reason': 'avoid_right' if active else 'confirming',
        })
        self.stage2_left_ball_avoid_debug = debug

        self.get_logger().info(
            f'[STAGE2_LEFT_BALL_AVOID] danger={danger.get("color")} '
            f'center={danger.get("center")}, depth={danger.get("depth_m")}, '
            f'radius={danger.get("radius", 0.0):.1f}, '
            f'dist_to_center={danger.get("stage2_avoid_dist_to_center_px", 0.0):.1f}/'
            f'{self.stage2_left_ball_avoid_center_px:.1f}px, '
            f'counter={self.stage2_left_ball_avoid_counter}/{confirm_frames}, '
            f'active={active}, vy={vy:.3f}',
            throttle_duration_sec=0.2
        )
        return vy

    def compute_yellow_angle_align_wz(self, yellow_result: dict) -> float:
        """
        使用原 detect_yellow_stop_line() 的检测结果做角度矫正。
        不改变原来的黄线筛选逻辑，只把 angle_deg 的正负转换成固定 wz。
        """
        if not self.yellow_angle_align_enabled:
            return 0.0
        if yellow_result is None or not yellow_result.get('has_line', False):
            return 0.0

        angle_deg = yellow_result.get('angle_deg', None)
        if angle_deg is None:
            return 0.0

        angle_deg = float(angle_deg)
        if abs(angle_deg) <= self.yellow_angle_align_deadband_deg:
            return 0.0

        # 固定角速度版本：只看 angle_deg 正负，不按角度大小改变速度。
        # 当前符号：黄线角度为正时给负 wz。
        # 如果实测发现越修越歪，把下面 if/else 的正负号对调。
        if angle_deg > 0.0:
            wz = -abs(self.yellow_angle_align_fixed_wz)
        else:
            wz = abs(self.yellow_angle_align_fixed_wz)

        self.get_logger().info(
            f'yellow angle align fixed: angle={angle_deg:.2f}deg, '
            f'deadband={self.yellow_angle_align_deadband_deg:.2f}deg, '
            f'wz={wz:.3f}, '
            f'require_front_horizontal={yellow_result.get("require_front_horizontal")}',
            throttle_duration_sec=0.3
        )
        return wz

    def get_yellow_slowdown_speed(
            self,
            yellow_result: dict,
            normal_speed: float,
            slow_speed: float,
            slowdown_ratio: float
    ) -> float:
        if yellow_result['img_shape'] is None or not yellow_result['has_line']:
            return normal_speed

        h, _ = yellow_result['img_shape']
        slow_threshold = int(h * slowdown_ratio)
        bottom = yellow_result['line_bottom_y']

        if bottom is not None and bottom >= slow_threshold:
            return min(normal_speed, slow_speed)
        return normal_speed

    # ============================================================
    # 可视化调试窗口
    # ============================================================
    def _make_yellow_mask_for_debug(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        roi_top = int(h * self.yellow_roi_top_ratio)
        roi_left = int(w * self.yellow_roi_left_ratio)
        roi_right = int(w * self.yellow_roi_right_ratio)

        roi_top = max(0, min(h - 1, roi_top))
        roi_left = max(0, min(w - 1, roi_left))
        roi_right = max(roi_left + 1, min(w, roi_right))

        roi = frame[roi_top:h, roi_left:roi_right]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([self.yellow_h_min, self.yellow_s_min, self.yellow_v_min], dtype=np.uint8)
        upper_yellow = np.array([self.yellow_h_max, self.yellow_s_max, self.yellow_v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask, (roi_left, roi_top, roi_right, h)

    def get_current_yellow_ratio_for_debug(self):
        if self.state == 'STAGE1_CRUISE_BALL_AND_YELLOW':
            return self.yellow_stop_line_y_ratio_stage1
        if self.state == 'STAGE2_CRUISE_YELLOW_ONLY':
            return self.yellow_stop_line_y_ratio_stage2
        if self.state == 'STAGE3_CRUISE_BALL_ONLY':
            return self.yellow_stop_line_y_ratio_stage3
        if self.state == 'STAGE3_GO_SCAN':
            return self.yellow_ratio_scan
        if self.state == 'STAGE3_GO_FINAL':
            return self.yellow_ratio_final
        return None

    def show_debug_window(self, frame: np.ndarray):
        """
        第二赛段调试窗口。
        只显示当前识别结果，不改变状态机逻辑。
        """
        try:
            vis = frame.copy()
            h, w = vis.shape[:2]
            image_center_x = w // 2
            image_center_y = h // 2

            ball = self.latest_ball_result
            yellow = self.latest_yellow_result

            # 画图像中心线
            cv2.line(vis, (image_center_x, 0), (image_center_x, h - 1), (255, 255, 255), 1)
            cv2.line(vis, (0, image_center_y), (w - 1, image_center_y), (80, 80, 80), 1)

            cv2.putText(vis, f'state={self.state}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            cv2.putText(vis, f'orange_hit_count={self.orange_hit_count}', (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (255, 255, 255), 2)

            # 画黄色 ROI
            roi_top = int(h * self.yellow_roi_top_ratio)
            roi_left = int(w * self.yellow_roi_left_ratio)
            roi_right = int(w * self.yellow_roi_right_ratio)
            roi_top = max(0, min(h - 1, roi_top))
            roi_left = max(0, min(w - 1, roi_left))
            roi_right = max(roi_left + 1, min(w, roi_right))
            cv2.rectangle(vis, (roi_left, roi_top), (roi_right, h - 1), (0, 255, 255), 1)
            cv2.putText(vis, 'yellow ROI', (roi_left + 3, max(18, roi_top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1)

            # 当前状态的黄线触发阈值线
            ratio = self.get_current_yellow_ratio_for_debug()
            if ratio is not None:
                threshold_y = int(h * ratio)
                cv2.line(vis, (0, threshold_y), (w - 1, threshold_y), (0, 180, 255), 2)
                cv2.putText(
                    vis,
                    f'th={threshold_y} ratio={ratio:.2f}',
                    (10, max(78, threshold_y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 180, 255),
                    2
                )

            # 画所有蓝球/橙球候选
            for color_name, balls, draw_color in (
                    ('B', ball.get('blue_balls', []), (255, 0, 0)),
                    ('O', ball.get('orange_balls', []), (0, 140, 255)),
            ):
                for idx, b in enumerate(balls):
                    cx, cy = b['center']
                    radius = int(max(2, round(b.get('radius', 2))))
                    depth_m = b.get('depth_m')
                    error_x = b.get('error_x')
                    cv2.circle(vis, (cx, cy), radius, draw_color, 2)
                    cv2.circle(vis, (cx, cy), 4, draw_color, -1)
                    depth_text = 'None' if depth_m is None else f'{depth_m:.2f}m'
                    cv2.putText(
                        vis,
                        f'{color_name}{idx} r={b.get("radius", 0):.1f} d={depth_text} ex={error_x}',
                        (max(5, cx - 45), max(18, cy - radius - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.43,
                        draw_color,
                        2
                    )

                    # 深度取样窗口映射到 RGB 上的大致位置
                    box = b.get('depth_box')
                    if box is not None and self.latest_depth is not None:
                        dh, dw = self.latest_depth.shape[:2]
                        x1, y1, x2, y2 = box
                        rx1 = int(x1 * w / max(dw, 1))
                        rx2 = int(x2 * w / max(dw, 1))
                        ry1 = int(y1 * h / max(dh, 1))
                        ry2 = int(y2 * h / max(dh, 1))
                        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), draw_color, 1)

            # 画左右参考球和中线
            left_ref = ball.get('left_ref')
            right_ref = ball.get('right_ref')
            if left_ref is not None:
                cx, cy = left_ref['center']
                cv2.circle(vis, (cx, cy), 8, (255, 255, 0), 3)
                cv2.putText(vis, 'LEFT_REF', (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 2)
            if right_ref is not None:
                cx, cy = right_ref['center']
                cv2.circle(vis, (cx, cy), 8, (255, 255, 0), 3)
                cv2.putText(vis, 'RIGHT_REF', (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 2)

            if left_ref is not None and right_ref is not None:
                lx = left_ref['center'][0]
                rx = right_ref['center'][0]
                lane_mid_x = int(0.5 * (lx + rx))
                cv2.line(vis, (lane_mid_x, 0), (lane_mid_x, h - 1), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f'lane_mid={lane_mid_x}, center_err={ball.get("center_error_px")}',
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 0),
                    2
                )

            # 画当前最佳目标球
            target = ball.get('best_target_ball')
            if target is not None:
                cx, cy = target['center']
                radius = int(max(8, round(target.get('radius', 8))))
                cv2.circle(vis, (cx, cy), radius + 4, (0, 0, 255), 3)
                cv2.putText(
                    vis,
                    f'TARGET {target.get("color")} side={target.get("side")} d={target.get("depth_m", -1):.2f}',
                    (max(5, cx - 70), min(h - 10, cy + radius + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 0, 255),
                    2
                )

            # 第二赛段左侧近球避让可视化
            avoid_debug = getattr(self, 'stage2_left_ball_avoid_debug', {})
            danger = avoid_debug.get('danger_ball')
            if danger is not None and danger.get('center') is not None:
                cx, cy = danger['center']
                radius = int(max(8, round(float(danger.get('radius', 8)))))
                color = (0, 0, 255) if avoid_debug.get('active') else (0, 180, 255)
                cv2.circle(vis, (int(cx), int(cy)), radius + 10, color, 3)
                cv2.putText(
                    vis,
                    f'S2_AVOID {danger.get("color")} active={avoid_debug.get("active")} vy={avoid_debug.get("vy", 0.0):.2f}',
                    (max(5, int(cx) - 90), min(h - 10, int(cy) + radius + 42)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    color,
                    2
                )

            if self.state == 'STAGE2_CRUISE_YELLOW_ONLY':
                cv2.putText(
                    vis,
                    f'S2 left-ball avoid: {avoid_debug.get("reason", "none")} '
                    f'cnt={avoid_debug.get("counter", 0)} vy={avoid_debug.get("vy", 0.0):.2f}',
                    (10, 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 0, 255) if avoid_debug.get('active') else (0, 180, 255),
                    2
                )

            # 画黄线检测结果：底部线、中心点、bbox、角度
            if yellow.get('has_line') and yellow.get('line_bottom_y') is not None:
                bottom_y = int(yellow['line_bottom_y'])
                line_center = yellow.get('line_center')
                cv2.line(vis, (0, bottom_y), (w - 1, bottom_y), (0, 255, 255), 2)

                bbox = yellow.get('bbox')
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

                if line_center is not None:
                    cx, cy = line_center
                    cv2.circle(vis, (cx, cy), 6, (0, 255, 255), -1)
                    angle = yellow.get('angle_deg')
                    angle_text = 'None' if angle is None else f'{float(angle):.1f}deg'
                    cv2.putText(
                        vis,
                        f'YELLOW bottom={bottom_y} angle={angle_text}',
                        (max(5, cx - 100), max(18, cy - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 255, 255),
                        2
                    )

                # 画拟合角度方向线，方便看矫正方向
                angle = yellow.get('angle_deg')
                if line_center is not None and angle is not None:
                    cx, cy = line_center
                    length = 80
                    rad = math.radians(float(angle))
                    dx = int(math.cos(rad) * length)
                    dy = int(math.sin(rad) * length)
                    cv2.line(vis, (cx - dx, cy - dy), (cx + dx, cy + dy), (0, 0, 255), 2)

            else:
                cv2.putText(vis, 'YELLOW not detected', (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # 画当前修正量信息
            yellow_wz = self.compute_yellow_angle_align_wz(yellow)
            center_vy = self.compute_center_cruise_vy(ball)
            center_dbg = getattr(self, 'center_cruise_debug_info', {})
            mode = center_dbg.get('mode', 'NA')
            ld = center_dbg.get('left_depth')
            rd = center_dbg.get('right_depth')
            dd = center_dbg.get('depth_diff')
            ld_txt = 'None' if ld is None else f'{float(ld):.2f}'
            rd_txt = 'None' if rd is None else f'{float(rd):.2f}'
            dd_txt = 'None' if dd is None else f'{float(dd):.2f}'
            cv2.putText(
                vis,
                f'center_vy={center_vy:.2f} mode={mode} Ld={ld_txt} Rd={rd_txt} diff={dd_txt} yellow_wz={yellow_wz:.2f}',
                (10, h - 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2
            )
            cv2.putText(
                vis,
                f'orange_cnt={len(ball.get("orange_balls", []))} blue_cnt={len(ball.get("blue_balls", []))} '
                f'center_ref={ball.get("has_center_reference")}',
                (10, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2
            )
            cv2.putText(
                vis,
                f'yellow_has={yellow.get("has_line")} bottom={yellow.get("line_bottom_y")} '
                f'angle={yellow.get("angle_deg")} require_front={yellow.get("require_front_horizontal")}',
                (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                2
            )

            cv2.imshow('second_stage_orange_yellow_debug', vis)

            if self.show_yellow_mask:
                mask, _ = self._make_yellow_mask_for_debug(frame)
                cv2.imshow('second_stage_yellow_mask', mask)

            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f'show_debug_window failed: {e}', throttle_duration_sec=1.0)

    # ============================================================
    # 球子链
    # ============================================================
    def ball_align_should_go_hit(self, target: Optional[Dict]) -> bool:
        """
        BALL_LATERAL_ALIGN 阶段保护逻辑。

        目的：
        对齐 A 球时，如果 A 球太近导致识别不到，
        或者 A 球丢失后 best_target_ball 突然变成远处 B 球，
        不继续对齐 B 球，而是直接进入 BALL_HIT_CONFIRM_FORWARD。

        触发条件：
        1. target is None：
        认为球已经太近 / 进入盲区 / 穿模，直接撞击。

        2. 当前 target 深度比上一帧或历史最近深度突然变大：
        认为当前识别到的不是原来的近处球，而是远处其他球，直接撞击。
        """
        if target is None:
            if self.ball_align_lost_go_hit:
                self.get_logger().warn(
                    '[BALL_ALIGN_PROTECT] target=None during BALL_LATERAL_ALIGN, '
                    'assume ball is too close/lost, go BALL_HIT_CONFIRM_FORWARD'
                )
                return True

            return False

        if not self.ball_align_depth_jump_enabled:
            return False

        cur_depth = target.get('depth_m', None)

        if cur_depth is None:
            self.get_logger().warn(
                '[BALL_ALIGN_PROTECT] target depth=None during BALL_LATERAL_ALIGN, '
                'assume ball is too close/lost, go BALL_HIT_CONFIRM_FORWARD'
            )
            return True

        cur_depth = float(cur_depth)

        # 初始化历史深度
        if self.ball_align_last_depth_m is None:
            self.ball_align_last_depth_m = cur_depth

        if self.ball_align_min_seen_depth_m is None:
            self.ball_align_min_seen_depth_m = cur_depth
        else:
            self.ball_align_min_seen_depth_m = min(self.ball_align_min_seen_depth_m, cur_depth)

        last_depth = float(self.ball_align_last_depth_m)
        min_seen = float(self.ball_align_min_seen_depth_m)

        jump_from_last = cur_depth - last_depth
        jump_from_min = cur_depth - min_seen

        near_enough_before = min_seen <= self.ball_align_near_depth_for_jump_m

        depth_jump = (
            jump_from_last >= self.ball_align_depth_jump_threshold_m
            or jump_from_min >= self.ball_align_depth_jump_threshold_m
        )

        if near_enough_before and depth_jump:
            self.get_logger().warn(
                f'[BALL_ALIGN_PROTECT] target depth suddenly increased, '
                f'treat current target as another far ball and go hit: '
                f'cur={cur_depth:.3f}, last={last_depth:.3f}, min_seen={min_seen:.3f}, '
                f'jump_last={jump_from_last:.3f}, jump_min={jump_from_min:.3f}, '
                f'jump_th={self.ball_align_depth_jump_threshold_m:.3f}, '
                f'near_th={self.ball_align_near_depth_for_jump_m:.3f}, '
                f'center={target.get("center")}, side={target.get("side")}, '
                f'error_x={target.get("error_x")}, radius={target.get("radius")}'
            )
            return True

        self.ball_align_last_depth_m = cur_depth

        self.get_logger().info(
            f'[BALL_ALIGN_PROTECT] normal target depth: '
            f'cur={cur_depth:.3f}, last={last_depth:.3f}, min_seen={min_seen:.3f}, '
            f'center={target.get("center")}, side={target.get("side")}',
            throttle_duration_sec=0.3
        )

        return False

    def finish_ball_task_and_return(self, x: float, y: float, yaw: float):
        self.last_ball_done_time_sec = self.now_sec()
        self.last_ball_done_pose = (x, y, yaw)
        self.orange_hit_count += 1
        self.get_logger().info(
            f'Ball task finished. orange_hit_count={self.orange_hit_count} | '
            f'last_ball_done_time_sec={self.last_ball_done_time_sec:.2f} | '
            f'last_ball_done_pose=({x:.3f}, {y:.3f}, {yaw:.3f})'
        )
        self.set_state(self.ball_return_state)

    def handle_ball_subchain(self, x: float, y: float, yaw: float) -> bool:
        ball = self.latest_ball_result
        target = ball['best_target_ball']

        # 1) 中线 -> 对齐球：只这里加最小 vy
        if self.state == 'BALL_LATERAL_ALIGN':
            # 新逻辑：
            # 1. 对齐时目标突然丢失：认为已经很近，直接撞击。
            # 2. 对齐时目标深度突然变远：认为 A 球丢失、误识别到远处 B 球，直接撞击。
            # 注意：这里不发 STOP，也不进入 BALL_POST_HIT_SIDE_SHIFT。
            if self.ball_align_should_go_hit(target):
                self.get_logger().warn(
                    '[BALL_LATERAL_ALIGN] target lost or depth jumped far, '
                    'switch directly to BALL_HIT_CONFIRM_FORWARD'
                )
                self.set_state('BALL_HIT_CONFIRM_FORWARD')
                return True
            error_px = target['error_x']
            err_norm = error_px / max(self.rgb_w * 0.5, 1.0)

            vx = self.lateral_align_forward_speed
            vy = clamp(
                -self.lateral_align_vy_gain * err_norm,
                -self.lateral_align_vy_max,
                self.lateral_align_vy_max
            )
            vy = self.apply_min_abs_velocity(vy, self.lateral_align_vy_min, deadband=0.01)

            if abs(error_px) <= self.lateral_align_px_tol:
                self.lateral_align_counter += 1
                self.get_logger().info(
                    f'BALL_LATERAL_ALIGN ok: error_x={error_px}, '
                    f'counter={self.lateral_align_counter}/{self.lateral_align_confirm_count}',
                    throttle_duration_sec=0.2
                )
                if self.lateral_align_counter >= self.lateral_align_confirm_count:
                    self.set_state('BALL_HIT_CONFIRM_FORWARD')
                    return True
            else:
                self.lateral_align_counter = 0

            self.get_logger().info(
                f'BALL_LATERAL_ALIGN target=({target["color"]}, side={target["side"]}) '
                f'depth={target["depth_m"]:.3f} error_x={error_px} '
                f'radius={target["radius"]:.1f} '
                f'(circle={target.get("radius_circle", -1):.1f}, eq={target.get("radius_eq", -1):.1f}) '
                f'-> cmd vx={vx:.3f}, vy={vy:.3f}',
                throttle_duration_sec=0.3
            )
            self.send_velocity_command(vx, vy, 0.0)
            return True

        # 2) 直接撞击：按仿真时间结束，不再用 TF 位移
        if self.state == 'BALL_HIT_CONFIRM_FORWARD':
            now_sec = self.now_sec()

            if self.hit_start_time_sec is None:
                self.hit_start_time_sec = now_sec

                # last_hit_side 不再在这里记录/覆盖。
                # 它已经在进入 BALL_LATERAL_ALIGN 的第一刻锁定，避免对齐过程中
                # 球从画面左侧跑到右侧后，撞后横移方向被错误改掉。
                if target is not None:
                    self.hit_start_depth_m = target.get('depth_m')
                else:
                    self.hit_start_depth_m = None

                self.get_logger().info(
                    f'BALL_HIT_CONFIRM_FORWARD start by sim time: '
                    f'duration={self.hit_forward_duration_sec:.3f}s, '
                    f'speed={self.hit_forward_speed:.3f}, '
                    f'last_hit_side={self.last_hit_side}, '
                    f'depth_at_start={self.hit_start_depth_m}'
                )

            elapsed = now_sec - self.hit_start_time_sec

            self.get_logger().info(
                f'BALL_HIT_CONFIRM_FORWARD elapsed={elapsed:.3f}/'
                f'{self.hit_forward_duration_sec:.3f}s',
                throttle_duration_sec=0.2
            )

            if elapsed >= self.hit_forward_duration_sec:
                self.set_state('BALL_POST_HIT_SIDE_SHIFT')
                return True

            self.send_velocity_command(self.hit_forward_speed, 0.0, 0.0)
            return True

        # 3) 撞后左右移动：固定速度 + 固定仿真时间，不再分前半段/后半段速度
        if self.state == 'BALL_POST_HIT_SIDE_SHIFT':
            now_sec = self.now_sec()

            if self.post_hit_side_shift_start_time_sec is None:
                self.post_hit_side_shift_start_time_sec = now_sec
                self.get_logger().info(
                    f'BALL_POST_HIT_SIDE_SHIFT start by sim time: '
                    f'last_hit_side={self.last_hit_side}, '
                    f'duration={self.post_hit_side_shift_duration_sec:.3f}s, '
                    f'fixed_speed={self.post_hit_side_shift_speed:.3f}'
                )

            elapsed = now_sec - self.post_hit_side_shift_start_time_sec

            self.get_logger().info(
                f'BALL_POST_HIT_SIDE_SHIFT elapsed={elapsed:.3f}/'
                f'{self.post_hit_side_shift_duration_sec:.3f}s, '
                f'fixed_speed={self.post_hit_side_shift_speed:.3f}',
                throttle_duration_sec=0.2
            )

            if elapsed >= self.post_hit_side_shift_duration_sec:
                self.finish_ball_task_and_return(x, y, yaw)
                return True

            if self.last_hit_side == 'left':
                self.send_velocity_command(0.0, -abs(self.post_hit_side_shift_speed), 0.0)
                return True

            if self.last_hit_side == 'right':
                self.send_velocity_command(0.0, abs(self.post_hit_side_shift_speed), 0.0)
                return True

            self.finish_ball_task_and_return(x, y, yaw)
            return True

        return False

    # ============================================================
    # 第三赛段：视觉处理 + 控制状态机
    # ============================================================
    def p3_elapsed_in_state(self) -> float:
        now = self.now_sec()
        if self.p3_state_start_time is None:
            self.p3_state_start_time = now
        return now - self.p3_state_start_time

    def p3_send_stand_command(self):
        self.msg.mode = 12
        self.msg.gait_id = 0
        self._inc_life_count()
        self.msg.rpy_des = [0.0, self.p3_stand_pitch, 0.0]
        self.msg.pos_des = [0.0, 0.0, self.p3_stand_body_height]
        self.Ctrl.Send_cmd(self.msg)
        self.get_logger().info('[P3 CMD] STAND / LOW BODY', throttle_duration_sec=1.0)

    def p3_send_velocity_command(self, vx: float, vy: float, wz: float, step_height: Optional[float] = None):
        self.msg.mode = 11
        self.msg.gait_id = 3
        self._inc_life_count()
        h = self.p3_step_height if step_height is None else float(step_height)
        self.msg.step_height = [h, h]
        self.msg.rpy_des = [0.0, self.p3_stand_pitch, 0.0]
        self.msg.pos_des = [0.0, 0.0, self.p3_stand_body_height]
        self.msg.vel_des = [float(vx), float(vy), float(wz)]
        self.Ctrl.Send_cmd(self.msg)
        self.get_logger().info(
            f'[P3 CMD] vel_des=[{vx:.3f}, {vy:.3f}, {wz:.3f}], step_height={h:.3f}',
            throttle_duration_sec=0.4
        )

    def p3_process_yellow_track(self, frame: np.ndarray):
        """
        合并 part3_vision.py：
        1. S 弯阶段：计算中距离 error_mid 和近距离 error_near。
        2. 出弯对齐阶段：双行前瞻，计算 s4_lat / s4_yaw / s4_valid。
        """
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([self.p3_yellow_h_min, self.p3_yellow_s_min, self.p3_yellow_v_min], dtype=np.uint8)
        upper_yellow = np.array([self.p3_yellow_h_max, self.p3_yellow_s_max, self.p3_yellow_v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        crop_left = int(width * self.p3_crop_left_ratio)
        crop_right = int(width * self.p3_crop_right_ratio)
        mask[:, 0:crop_left] = 0
        mask[:, crop_right:width] = 0

        mid_top = int(height * self.p3_mid_top_ratio)
        mid_bottom = int(height * self.p3_mid_bottom_ratio)
        near_top = int(height * self.p3_near_top_ratio)
        near_bottom = int(height * self.p3_near_bottom_ratio)

        mask_mid = np.zeros_like(mask)
        mask_near = np.zeros_like(mask)
        mask_mid[mid_top:mid_bottom, 0:width] = mask[mid_top:mid_bottom, 0:width]
        mask_near[near_top:near_bottom, 0:width] = mask[near_top:near_bottom, 0:width]

        err_mid = 0.0
        err_near = 0.0

        M_mid = cv2.moments(mask_mid)
        if M_mid['m00'] > 0:
            cx_mid = int(M_mid['m10'] / M_mid['m00'])
            dist_mid = abs(cx_mid - width / 2)
            force_mid = ((width / 2 - dist_mid) / (width / 2)) ** 3
            err_mid = float(force_mid) if cx_mid > width / 2 else -float(force_mid)

        M_near = cv2.moments(mask_near)
        if M_near['m00'] > 0:
            cx_near = int(M_near['m10'] / M_near['m00'])
            dist_near = abs(cx_near - width / 2)
            force_near = ((width / 2 - dist_near) / (width / 2)) ** 3
            err_near = float(force_near) if cx_near > width / 2 else -float(force_near)

        self.p3_error_mid = err_mid
        self.p3_error_near = err_near
        self.p3_last_update_time = self.now_sec()

        near_y = int(height * self.p3_align_near_y_ratio)
        far_y = int(height * self.p3_align_far_y_ratio)
        roi_left = int(width * self.p3_align_roi_left_ratio)
        roi_right = int(width * self.p3_align_roi_right_ratio)

        def get_road_center(y_idx: int) -> float:
            y_idx = max(0, min(height - 1, int(y_idx)))
            row = mask[y_idx, :]
            yellow_idx = np.where(row > 128)[0]
            valid_idx = [idx for idx in yellow_idx if roi_left < idx < roi_right]
            if len(valid_idx) < 2:
                return -1.0
            diffs = np.diff(valid_idx)
            if len(diffs) == 0:
                return -1.0
            max_gap_idx = int(np.argmax(diffs))
            if diffs[max_gap_idx] > self.p3_align_min_gap_px:
                l_edge = valid_idx[max_gap_idx]
                r_edge = valid_idx[max_gap_idx + 1]
                return 0.5 * (l_edge + r_edge)
            return -1.0

        cx_n = get_road_center(near_y)
        cx_f = get_road_center(far_y)
        self.p3_align_near_center = cx_n
        self.p3_align_far_center = cx_f

        if cx_n != -1 and cx_f != -1:
            self.p3_s4_lat = (width / 2.0 - cx_n) / (width / 2.0)
            self.p3_s4_yaw = (cx_n - cx_f) / (width / 2.0)
            self.p3_s4_valid = 1.0
        else:
            self.p3_s4_lat = 0.0
            self.p3_s4_yaw = 0.0
            self.p3_s4_valid = 0.0

        self.p3_latest_mask = mask
        self.p3_latest_mask_mid = mask_mid
        self.p3_latest_mask_near = mask_near

    def p3_show_debug_window(self, frame: np.ndarray):
        try:
            vis = frame.copy()
            height, width = vis.shape[:2]
            crop_left = int(width * self.p3_crop_left_ratio)
            crop_right = int(width * self.p3_crop_right_ratio)
            mid_top = int(height * self.p3_mid_top_ratio)
            mid_bottom = int(height * self.p3_mid_bottom_ratio)
            near_top = int(height * self.p3_near_top_ratio)
            near_y = int(height * self.p3_align_near_y_ratio)
            far_y = int(height * self.p3_align_far_y_ratio)
            roi_left = int(width * self.p3_align_roi_left_ratio)
            roi_right = int(width * self.p3_align_roi_right_ratio)

            cv2.putText(vis, f'P3 state={self.state}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(vis, f'err_mid={self.p3_error_mid:.3f} err_near={self.p3_error_near:.3f}', (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(vis, f's4_valid={self.p3_s4_valid:.1f} lat={self.p3_s4_lat:.3f} yaw={self.p3_s4_yaw:.3f}',
                        (10, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            vx, vy, wz = getattr(self, 'motion_cmd', (0.0, 0.0, 0.0))
            cv2.putText(
                vis,
                f'cmd vx={vx:.3f} vy={vy:.3f} wz={wz:.3f}',
                (10, 106),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )
            if self.state == getattr(self, 'GLOBAL_FINAL_P3_ALIGN', None):
                cv2.putText(
                    vis,
                    'P4 FINAL uses P3 align logic',
                    (10, 133),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

            cv2.line(vis, (crop_left, 0), (crop_left, height), (0, 255, 255), 2)
            cv2.line(vis, (crop_right, 0), (crop_right, height), (0, 255, 255), 2)
            cv2.line(vis, (0, mid_top), (width, mid_top), (255, 0, 0), 2)
            cv2.line(vis, (0, mid_bottom), (width, mid_bottom), (0, 255, 0), 2)
            cv2.line(vis, (0, near_top), (width, near_top), (0, 180, 255), 1)

            cv2.line(vis, (0, near_y), (width, near_y), (255, 255, 0), 1)
            cv2.line(vis, (0, far_y), (width, far_y), (255, 255, 0), 1)
            cv2.line(vis, (roi_left, 0), (roi_left, height), (255, 0, 255), 2)
            cv2.line(vis, (roi_right, 0), (roi_right, height), (255, 0, 255), 2)

            if self.p3_align_near_center != -1 and self.p3_align_far_center != -1:
                cv2.line(
                    vis,
                    (int(self.p3_align_near_center), near_y),
                    (int(self.p3_align_far_center), far_y),
                    (0, 0, 255),
                    3
                )

            cv2.imshow('part3_origin_debug', vis)
            if self.p3_latest_mask_mid is not None:
                cv2.imshow('part3_mask_mid', self.p3_latest_mask_mid)
            if self.p3_latest_mask_near is not None:
                cv2.imshow('part3_mask_near', self.p3_latest_mask_near)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f'p3_show_debug_window failed: {e}', throttle_duration_sec=1.0)

    def p3_control_loop(self):
        elapsed = self.p3_elapsed_in_state()
        now = self.now_sec()

        if self.state == 'P3_STAND_WAIT':
            if not self.p3_stand_sent:
                self.p3_send_stand_command()
                self.p3_stand_sent = True
                self.get_logger().info(
                    f'P3_STAND_WAIT start: duration={self.p3_stand_wait_sec:.2f}s, '
                    f'body_height={self.p3_stand_body_height:.2f}, pitch={self.p3_stand_pitch:.2f}'
                )

            if elapsed >= self.p3_stand_wait_sec:
                self.set_state('P3_S_CURVE_CRUISE')
                return
            return

        if self.state == 'P3_S_CURVE_CRUISE':
            if elapsed >= self.p3_s_curve_duration_sec:
                self.set_state('P3_ALIGN_TRACK')
                return

            if now - self.p3_last_update_time < self.p3_vision_timeout_sec:
                err_mid = self.p3_error_mid
                err_near = self.p3_error_near
                raw_turn = (err_mid / 500.0 + err_near) * self.p3_kp_turn
                turn_speed = clamp(raw_turn, -0.5, 0.5)
                raw_lateral = err_near * self.p3_kp_lat
                lateral_speed = clamp(raw_lateral, -0.10, 0.10)
                speed_drop = abs(err_near) * self.p3_kd_slowdown
                forward_speed = max(self.p3_min_forward_speed, self.p3_base_forward_speed - speed_drop)
            else:
                forward_speed = self.p3_fallback_forward_speed
                lateral_speed = 0.0
                turn_speed = 0.0

            self.get_logger().info(
                f'P3_S_CURVE_CRUISE elapsed={elapsed:.2f}/{self.p3_s_curve_duration_sec:.2f}s | '
                f'err_mid={self.p3_error_mid:.3f}, err_near={self.p3_error_near:.3f}, '
                f'cmd=[{forward_speed:.3f},{lateral_speed:.3f},{turn_speed:.3f}]',
                throttle_duration_sec=0.5
            )
            self.p3_send_velocity_command(forward_speed, lateral_speed, turn_speed, step_height=self.p3_step_height)
            return

        if self.state == 'P3_ALIGN_TRACK':
            if elapsed >= self.p3_align_max_duration_sec:
                self.get_logger().info('P3_ALIGN_TRACK timeout, finish all stages.')
                self.set_state('DONE')
                return

            if self.p3_s4_valid > 0.5:
                err_lat = self.p3_s4_lat
                err_yaw = self.p3_s4_yaw

                if abs(err_lat) < self.p3_align_lat_tol and abs(err_yaw) < self.p3_align_yaw_tol:
                    self.get_logger().info('P3_ALIGN_TRACK complete: centered and aligned.')
                    self.set_state('DONE')
                    return

                lateral_speed = clamp(err_lat * self.p3_align_lat_gain, -self.p3_align_lat_max, self.p3_align_lat_max)
                turn_speed = clamp(err_yaw * self.p3_align_yaw_gain, -self.p3_align_yaw_max, self.p3_align_yaw_max)
                self.p3_send_velocity_command(0.0, lateral_speed, turn_speed, step_height=self.p3_align_step_height)
            else:
                self.p3_send_velocity_command(self.p3_align_search_vx, 0.0, self.p3_align_search_wz,
                                              step_height=self.p3_align_step_height)
            return

    # ============================================================
    # 主循环
    # ============================================================
    def control_loop(self):
        # 第一赛段 P1_* 状态优先执行；结束时会直接 set_state 到第二赛段状态，不额外发 STOP。
        if isinstance(self.state, str) and self.state.startswith('P1_'):
            self.p1_control_loop()
            return

        if isinstance(self.state, str) and self.state.startswith('P3_'):
            self.p3_control_loop()
            return

        pose = self.get_current_pose()

        # TF 现在不是状态机运行的必要条件。
        # 可用时记录用于日志/兼容；不可用时使用上一次位姿或 0 值占位，继续按图像和仿真时间运行。
        if pose is not None:
            self.last_known_pose = pose
            x, y, yaw = pose
        elif self.last_known_pose is not None:
            x, y, yaw = self.last_known_pose
            self.get_logger().warn(
                'TF pose unavailable, use last_known_pose and continue sim-time/image control.',
                throttle_duration_sec=1.0
            )
        else:
            x, y, yaw = 0.0, 0.0, 0.0
            self.get_logger().warn(
                'TF pose unavailable and no last_known_pose, use zero pose and continue sim-time/image control.',
                throttle_duration_sec=1.0
            )

        ball = self.latest_ball_result
        yellow = self.latest_yellow_result

        if ball['img_shape'] is not None:
            self.get_logger().info(
                f"state={self.state} | orange_hit_count={self.orange_hit_count} | "
                f"has_ball={ball['has_ball']} | ball_center={ball['ball_center']} | "
                f"ball_depth={ball['ball_depth_m']} | ball_radius={ball['ball_radius']} | "
                f"ball_error_x={ball['error_x']} | has_center_ref={ball['has_center_reference']} | "
                f"center_error_px={ball['center_error_px']} | "
                f"left_ref={'Y' if ball['left_ref'] is not None else 'N'} | "
                f"right_ref={'Y' if ball['right_ref'] is not None else 'N'} | "
                f"orange_cnt={len(ball['orange_balls'])} | blue_cnt={len(ball['blue_balls'])} | "
                f"yellow_has_line={yellow['has_line']} | yellow_bottom={yellow['line_bottom_y']} | "
                f"yellow_angle={yellow.get('angle_deg')}",
                throttle_duration_sec=0.6
            )

        if self.handle_ball_subchain(x, y, yaw):
            return

        if self.state == 'STAGE1_ROTATE_LEFT_90':
            self.execute_timed_turn(
                wz=self.timed_turn_wz_90,
                duration_sec=self.timed_turn_duration_90_sec,
                next_state='STAGE2_CRUISE_YELLOW_ONLY'
            )
            return

        if self.state == 'STAGE2_ROTATE_LEFT_90':
            self.execute_timed_turn(
                wz=self.timed_turn_wz_90,
                duration_sec=self.timed_turn_duration_90_sec,
                next_state='STAGE2_MOVE_FORWARD_AFTER_LEFT_JUMP_TIME'
            )
            return

        if self.state == 'STAGE3_ROTATE_BACK_180':
            self.execute_timed_turn(
                wz=self.timed_turn_wz_180,
                duration_sec=self.timed_turn_duration_180_sec,
                next_state='STAGE3_FINAL_DECISION'
            )
            return

        if self.state == 'STAGE3_ROTATE_LEFT_30':
            # 最终出口前的移动阶段：不再使用 TF 距离判断。
            # 使用 ROS2 节点时钟 now_sec() 计时；启用 use_sim_time 后就是仿真时间。
            now_sec = self.now_sec()
            if self.stage3_final_left_shift_start_time_sec is None:
                self.stage3_final_left_shift_start_time_sec = now_sec
                self.get_logger().info(
                    f'STAGE3_ROTATE_LEFT_30 start time-based shift: '
                    f'sim_time_start={now_sec:.3f}s, '
                    f'duration={self.stage3_final_left_shift_duration_sec:.3f}s, '
                    f'vy={self.stage3_final_left_shift_speed:.3f}'
                )

            elapsed = now_sec - self.stage3_final_left_shift_start_time_sec
            self.get_logger().info(
                f'STAGE3_ROTATE_LEFT_30 time shift: '
                f'elapsed={elapsed:.3f}/{self.stage3_final_left_shift_duration_sec:.3f}s, '
                f'vy={self.stage3_final_left_shift_speed:.3f}',
                throttle_duration_sec=0.2
            )

            if elapsed >= self.stage3_final_left_shift_duration_sec:
                # 不在移动和转向之间调用 STOP，避免 mode=12 + Wait_finish 带来的停顿。
                self.set_state('STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT')
                return

            # 默认发送 vy > 0 作为移动命令。
            # 如果实测方向反了，把 abs(...) 改成 -abs(...)。
            self.send_velocity_command(0.0, abs(self.stage3_final_left_shift_speed), 0.0)
            return

        if self.state == 'STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT':
            # 最终出口前的转向阶段：不再使用 TF yaw 闭环。
            # 按仿真时间发送 wz，到时后进入 DONE，由 DONE 统一 STOP。
            now_sec = self.now_sec()
            if self.stage3_final_rotate_start_time_sec is None:
                self.stage3_final_rotate_start_time_sec = now_sec
                self.get_logger().info(
                    f'STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT start time-based rotate: '
                    f'sim_time_start={now_sec:.3f}s, '
                    f'duration={self.stage3_final_rotate_duration_sec:.3f}s, '
                    f'wz={self.stage3_final_rotate_wz:.3f}'
                )

            elapsed = now_sec - self.stage3_final_rotate_start_time_sec
            self.get_logger().info(
                f'STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT time rotate: '
                f'elapsed={elapsed:.3f}/{self.stage3_final_rotate_duration_sec:.3f}s, '
                f'wz={self.stage3_final_rotate_wz:.3f}',
                throttle_duration_sec=0.2
            )

            if elapsed >= self.stage3_final_rotate_duration_sec:
                # 第二赛段结束后直接进入第三赛段入口；不先进入 DONE，避免提前全流程停止。
                self.set_state('P3_S_CURVE_CRUISE')
                return

            # 默认 wz > 0 为左转。
            # 如果实测转向反了，把 abs(...) 改成 -abs(...)。
            self.send_velocity_command(0.0, 0.0, abs(self.stage3_final_rotate_wz))
            return

        if self.state == 'STAGE1_CRUISE_BALL_AND_YELLOW':
            if self.stage1_yellow_passed(yellow):
                self.set_state('STAGE1_ROTATE_LEFT_90')
                return

            target = ball['best_target_ball']
            if target is not None:
                if (target['depth_m'] <= self.turn_trigger_distance_m and
                        target['radius'] >= self.min_ball_radius_to_trigger):
                    if self.can_trigger_ball_again((x, y, yaw)):
                        self.pre_turn_pose = (x, y, yaw)
                        self.ball_return_state = 'STAGE1_CRUISE_BALL_AND_YELLOW'
                        self.get_logger().info(
                            f"Saved pre_turn_pose = ({x:.3f}, {y:.3f}, {yaw:.3f}) | "
                            f"target=({target['color']}, side={target['side']}, depth={target['depth_m']:.3f}, "
                            f"error_x={target['error_x']})"
                        )
                        self.set_state('BALL_LATERAL_ALIGN')
                        return

            vx = self.get_yellow_slowdown_speed(
                yellow, self.stage1_cruise_forward_speed,
                self.stage1_yellow_slow_speed,
                self.yellow_slowdown_ratio_stage1
            )
            wz = self.compute_yellow_angle_align_wz(yellow)
            self.send_center_cruise_command_with_wz(ball, vx, wz)
            return

        if self.state == 'STAGE1_MOVE_RIGHT_FIXED_DISTANCE':
            if self.stage1_right_shift_start_pose is None:
                time.sleep(2)
                self.stage1_right_shift_start_pose = (x, y, yaw)
                self.get_logger().info(
                    f"Saved stage1_right_shift_start_pose = ({x:.3f}, {y:.3f}, {yaw:.3f})"
                )

            x0, y0, _ = self.stage1_right_shift_start_pose
            dist = math.hypot(x - x0, y - y0)

            self.get_logger().info(
                f"Stage1 fixed right shift: dist={dist:.3f} / {self.stage1_right_shift_distance_m:.3f}",
                throttle_duration_sec=0.5
            )

            if dist >= self.stage1_right_shift_distance_m - self.stage1_right_shift_tolerance_m:
                self.set_state('STAGE2_CRUISE_YELLOW_ONLY')
                return

            self.send_move_right_command(self.right_speed)
            return

        if self.state == 'STAGE2_MOVE_FORWARD_AFTER_LEFT_JUMP':
            if self.stage2_forward_after_left_jump_start_pose is None:
                time.sleep(2)
                self.stage2_forward_after_left_jump_start_pose = (x, y, yaw)
                self.get_logger().info(
                    f"Saved stage2_forward_after_left_jump_start_pose = ({x:.3f}, {y:.3f}, {yaw:.3f})"
                )

            x0, y0, _ = self.stage2_forward_after_left_jump_start_pose
            dist = math.hypot(x - x0, y - y0)

            self.get_logger().info(
                f"Stage2 forward after left jump: dist={dist:.3f} / "
                f"{self.stage2_forward_after_left_jump_distance_m:.3f}",
                throttle_duration_sec=0.5
            )

            if dist >= self.stage2_forward_after_left_jump_distance_m - self.stage2_forward_after_left_jump_tolerance_m:
                self.set_state('STAGE3_CRUISE_BALL_ONLY')
                return

            self.send_velocity_command(self.stage2_forward_after_left_jump_speed, 0.0, 0.0)
            return

        if self.state == 'STAGE2_MOVE_FORWARD_AFTER_LEFT_JUMP_TIME':
            now_sec = self.now_sec()
            if self.stage2_forward_after_left_jump_start_time_sec is None:
                self.stage2_forward_after_left_jump_start_time_sec = now_sec
                self.get_logger().info(
                    f'STAGE2_MOVE_FORWARD_AFTER_LEFT_JUMP_TIME start: '
                    f'sim_time_start={now_sec:.3f}s, '
                    f'duration={self.stage2_forward_after_left_jump_duration_sec:.3f}s, '
                    f'speed={self.stage2_forward_after_left_jump_speed:.3f}'
                )

            elapsed = now_sec - self.stage2_forward_after_left_jump_start_time_sec
            self.get_logger().info(
                f'STAGE2_MOVE_FORWARD_AFTER_LEFT_JUMP_TIME elapsed='
                f'{elapsed:.3f}/{self.stage2_forward_after_left_jump_duration_sec:.3f}s',
                throttle_duration_sec=0.5
            )

            if elapsed >= self.stage2_forward_after_left_jump_duration_sec:
                self.set_state('STAGE3_CRUISE_BALL_ONLY')
                return

            self.send_velocity_command(self.stage2_forward_after_left_jump_speed, 0.0, 0.0)
            return

        if self.state == 'STAGE2_CRUISE_YELLOW_ONLY':
            if self.yellow_reached(yellow, self.yellow_stop_line_y_ratio_stage2):
                self.set_state('STAGE2_ROTATE_LEFT_90')
                return

            vx = self.get_yellow_slowdown_speed(
                yellow, self.stage2_cruise_forward_speed,
                self.stage2_yellow_slow_speed,
                self.yellow_slowdown_ratio_stage2
            )
            wz = self.compute_yellow_angle_align_wz(yellow)

            # 第二赛段不进入撞球逻辑，但会检查左侧蓝球/橙球是否过近。
            # 如果左侧近球靠近图像中心，就给固定右移 vy，避免左侧擦碰。
            avoid_vy = self.compute_stage2_left_ball_avoid_vy(ball)
            self.send_velocity_command(vx, avoid_vy, wz)
            return

        if self.state == 'STAGE3_CRUISE_BALL_ONLY':
            if self.yellow_reached(yellow, self.yellow_stop_line_y_ratio_stage3):
                self.set_state('STAGE3_ROTATE_BACK_180')
                return

            target = ball['best_target_ball']
            if target is not None:
                if (target['depth_m'] <= self.turn_trigger_distance_m and
                        target['radius'] >= self.min_ball_radius_to_trigger):
                    if self.can_trigger_ball_again((x, y, yaw)):
                        self.pre_turn_pose = (x, y, yaw)
                        self.ball_return_state = 'STAGE3_CRUISE_BALL_ONLY'
                        self.get_logger().info(
                            f"Saved pre_turn_pose = ({x:.3f}, {y:.3f}, {yaw:.3f}) | "
                            f"target=({target['color']}, side={target['side']}, depth={target['depth_m']:.3f}, "
                            f"error_x={target['error_x']})"
                        )
                        self.set_state('BALL_LATERAL_ALIGN')
                        return

            vx = self.get_yellow_slowdown_speed(
                yellow, self.stage3_cruise_ball_only_speed,
                self.stage3_yellow_slow_speed,
                self.yellow_slowdown_ratio_stage3
            )
            wz = self.compute_yellow_angle_align_wz(yellow)
            self.send_center_cruise_command_with_wz(ball, vx, wz)
            return

        if self.state == 'STAGE3_FINAL_DECISION':
            if self.orange_hit_count >= 4:
                self.get_logger().info('已完成4个橙球，直接前往最终出口线')
                self.set_state('STAGE3_GO_FINAL')
            else:
                self.get_logger().info('当前只完成3个橙球，先前往0.6位置补扫最后一个球')
                self.set_state('STAGE3_GO_SCAN')
            return

        if self.state == 'STAGE3_GO_SCAN':
            if self.yellow_reached(yellow, self.yellow_ratio_scan):
                self.set_state('STAGE3_SCAN_AND_HIT_LAST')
                return

            vx = self.get_yellow_slowdown_speed(
                yellow, self.stage3_go_scan_speed,
                self.stage3_go_scan_yellow_slow_speed,
                self.yellow_slowdown_ratio_scan
            )
            wz = self.compute_yellow_angle_align_wz(yellow)
            self.send_center_cruise_command_with_wz(ball, vx, wz)
            return

        if self.state == 'STAGE3_SCAN_AND_HIT_LAST':
            target = ball['best_target_ball']
            if target is not None:
                self.pre_turn_pose = (x, y, yaw)
                self.ball_return_state = 'STAGE3_GO_FINAL'
                self.get_logger().info(
                    f"Scan area found final ball. pre_turn_pose=({x:.3f}, {y:.3f}, {yaw:.3f}) | "
                    f"target=({target['color']}, side={target['side']}, depth={target['depth_m']:.3f}, "
                    f"error_x={target['error_x']})"
                )
                self.set_state('BALL_LATERAL_ALIGN')
                return

            self.send_center_cruise_command(ball, self.stage3_go_scan_speed)
            return

        if self.state == 'STAGE3_GO_FINAL':
            if self.yellow_reached(yellow, self.yellow_ratio_final):
                self.set_state('STAGE3_ROTATE_LEFT_30')
                return

            vx = self.get_yellow_slowdown_speed(
                yellow, self.stage3_go_final_speed,
                self.stage3_go_final_yellow_slow_speed,
                self.yellow_slowdown_ratio_final
            )

            # STAGE3_GO_FINAL 不做黄线角度矫正：
            # 这里只保留原来的黄线到底判断和预减速逻辑。
            # 中线对齐仍然使用固定 vy 横向平移。
            # 如果你连中线横移也不想要，可以把下一行改成：
            # self.send_velocity_command(vx, 0.0, 0.0)
            self.send_center_cruise_command_with_wz(ball, vx, 0.0)
            return

        if self.state == 'DONE':
            self.send_stop_command()
            return


class FullCompetitionNode(FourthStageMixin, CombinedStage1Stage2Node):
    """
    One-node integration of stages 1-3 and stage 4.

    Vision processing is explicitly state-gated:
      - P1_* only runs P1 yellow/blue processing;
      - P3_* only runs P3 S-curve/yellow processing;
      - fourth-stage states only store RGB frames and run fourth-stage detectors inside fourth_control_loop;
      - second-stage states only run ball/yellow-stop detection.
    """

    def __init__(self):
        CombinedStage1Stage2Node.__init__(self)
        self.fourth_stage_init()
        self.get_logger().info('FullCompetitionNode ready: P1 -> P2/P3 -> P4 in one ROS2 node.')

    # yaml_pub compatibility: the fourth-stage code calls self.yaml_node.publish_*.
    def publish_yaml_kDOUBLE(self, name: str, value: float, is_user: int = 0):
        msg = YamlParam()
        msg.name = name
        msg.kind = ControlParameterValueKind.kDOUBLE
        msg.s64_value = float(value)
        msg.is_user = int(is_user)
        if not hasattr(self, '_integrated_para_pub'):
            self._integrated_para_pub = self.create_publisher(YamlParam, 'yaml_parameter', 10)
        self._integrated_para_pub.publish(msg)

    def publish_yaml_s64(self, name: str, value: int, is_user: int = 0):
        msg = YamlParam()
        msg.name = name
        msg.kind = ControlParameterValueKind.kS64
        msg.s64_value = int(value)
        msg.is_user = int(is_user)
        if not hasattr(self, '_integrated_para_pub'):
            self._integrated_para_pub = self.create_publisher(YamlParam, 'yaml_parameter', 10)
        self._integrated_para_pub.publish(msg)

    def publish_yaml_vecxd(self, name: str, values, is_user: int = 1):
        msg = YamlParam()
        msg.name = name
        msg.kind = ControlParameterValueKind.kVEC_X_DOUBLE
        vec = [0.0] * 12
        for i, v in enumerate(values):
            if i < 12:
                vec[i] = float(v)
        msg.vecxd_value = vec
        msg.is_user = int(is_user)
        if not hasattr(self, '_integrated_para_pub'):
            self._integrated_para_pub = self.create_publisher(YamlParam, 'yaml_parameter', 10)
        self._integrated_para_pub.publish(msg)

    def publish_apply_force(self, link_name: str, rel_pos, force, duration: float):
        msg = ApplyForce()
        msg.link_name = link_name
        msg.rel_pos = [float(x) for x in rel_pos]
        msg.force = [float(x) for x in force]
        msg.time = float(duration)
        if not hasattr(self, '_integrated_force_pub'):
            self._integrated_force_pub = self.create_publisher(ApplyForce, 'apply_force', 10)
        self._integrated_force_pub.publish(msg)

    def is_fourth_stage_state(self, state: str) -> bool:
        return isinstance(state, str) and state in self.get_all_state_names()

    def clear_pre_fourth_vision_caches(self):
        # Stop using stale previous-stage visual caches after handoff.
        self.p1_latest_mask_yellow = None
        self.p1_blue_detections = []
        self.p3_latest_mask = None
        self.p3_latest_mask_mid = None
        self.p3_latest_mask_near = None
        self.latest_ball_result = {
            'has_ball': False, 'ball_center': None, 'ball_radius': None,
            'ball_depth_m': None, 'img_shape': None, 'error_x': None,
            'aligned': False, 'depth_center': None, 'depth_box': None,
            'orange_balls': [], 'blue_balls': [], 'left_balls': [], 'right_balls': [],
            'has_center_reference': False, 'center_error_px': None,
            'left_ref': None, 'right_ref': None, 'best_target_ball': None,
        }
        self.latest_yellow_result = {
            'has_line': False, 'line_bottom_y': None, 'line_center': None,
            'img_shape': None, 'angle_deg': None, 'abs_tilt_deg': None,
            'bbox': None, 'width_ratio': None, 'wh_ratio': None,
            'require_front_horizontal': None,
        }

    def handoff_to_fourth_stage(self, reason: str):
        self.get_logger().info(f'[HANDOFF] P3 -> P4: {reason}')
        self.clear_pre_fourth_vision_caches()

        if self.show_debug_vis:
            try:
                cv2.destroyWindow('second_stage_orange_yellow_debug')
                cv2.destroyWindow('second_stage_yellow_mask')
                cv2.destroyWindow('part3_origin_debug')
                cv2.destroyWindow('part3_mask_mid')
                cv2.destroyWindow('part3_mask_near')
            except Exception as e:
                self.get_logger().warn(f'[HANDOFF] destroy old debug windows failed: {e}')

        # 第四赛段开始保持 normal，不再全程 low。
        self.set_body_normal(do_stop=True, reason='handoff_to_fourth_stage', force=True)
        self.enter_initial_state()

    def p3_control_loop(self):
        elapsed = self.p3_elapsed_in_state()
        now = self.now_sec()

        if self.state == 'P3_STAND_WAIT':
            if not self.p3_stand_sent:
                self.p3_send_stand_command()
                self.p3_stand_sent = True
                self.get_logger().info(
                    f'P3_STAND_WAIT start: duration={self.p3_stand_wait_sec:.2f}s, '
                    f'body_height={self.p3_stand_body_height:.2f}, pitch={self.p3_stand_pitch:.2f}'
                )
            if elapsed >= self.p3_stand_wait_sec:
                self.set_state('P3_S_CURVE_CRUISE')
                return
            return

        if self.state == 'P3_S_CURVE_CRUISE':
            if elapsed >= self.p3_s_curve_duration_sec:
                self.set_state('P3_ALIGN_TRACK')
                return

            if now - self.p3_last_update_time < self.p3_vision_timeout_sec:
                err_mid = self.p3_error_mid
                err_near = self.p3_error_near
                raw_turn = (err_mid / 500.0 + err_near) * self.p3_kp_turn
                turn_speed = clamp(raw_turn, -0.5, 0.5)
                raw_lateral = err_near * self.p3_kp_lat
                lateral_speed = clamp(raw_lateral, -0.10, 0.10)
                speed_drop = abs(err_near) * self.p3_kd_slowdown
                forward_speed = max(self.p3_min_forward_speed, self.p3_base_forward_speed - speed_drop)
            else:
                forward_speed = self.p3_fallback_forward_speed
                lateral_speed = 0.0
                turn_speed = 0.0

            self.get_logger().info(
                f'P3_S_CURVE_CRUISE elapsed={elapsed:.2f}/{self.p3_s_curve_duration_sec:.2f}s | '
                f'err_mid={self.p3_error_mid:.3f}, err_near={self.p3_error_near:.3f}, '
                f'cmd=[{forward_speed:.3f},{lateral_speed:.3f},{turn_speed:.3f}]',
                throttle_duration_sec=0.5
            )
            self.p3_send_velocity_command(forward_speed, lateral_speed, turn_speed, step_height=self.p3_step_height)
            return

        if self.state == 'P3_ALIGN_TRACK':
            if elapsed >= self.p3_align_max_duration_sec:
                self.handoff_to_fourth_stage('P3_ALIGN_TRACK timeout')
                return

            if self.p3_s4_valid > 0.5:
                err_lat = self.p3_s4_lat
                err_yaw = self.p3_s4_yaw
                if abs(err_lat) < self.p3_align_lat_tol and abs(err_yaw) < self.p3_align_yaw_tol:
                    self.handoff_to_fourth_stage('P3 centered and aligned')
                    return
                lateral_speed = clamp(err_lat * self.p3_align_lat_gain, -self.p3_align_lat_max, self.p3_align_lat_max)
                turn_speed = clamp(err_yaw * self.p3_align_yaw_gain, -self.p3_align_yaw_max, self.p3_align_yaw_max)
                self.p3_send_velocity_command(0.0, lateral_speed, turn_speed, step_height=self.p3_align_step_height)
            else:
                self.p3_send_velocity_command(self.p3_align_search_vx, 0.0, self.p3_align_search_wz,
                                              step_height=self.p3_align_step_height)
            return

    def rgb_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge convert failed: {e}')
            return

        self.latest_bgr = frame

        if isinstance(self.state, str) and self.state.startswith('P1_'):
            self.p1_process_stage1_yellow(frame)
            self.p1_process_blue_ball(frame)
            if self.show_debug_vis:
                self.show_debug_window(frame)
        elif isinstance(self.state, str) and self.state.startswith('P3_'):
            self.p3_process_yellow_track(frame)
            if self.show_debug_vis:
                self.p3_show_debug_window(frame)
        elif self.is_fourth_stage_state(self.state):
            # Fourth-stage detectors and visualization are called inside fourth_control_loop
            # so the RGB callback does not run extra perception work.
            pass
        else:
            self.latest_ball_result = self.detect_ball_scene(frame)
            self.latest_yellow_result = self.detect_yellow_stop_line(frame)
            if self.show_debug_vis:
                self.show_debug_window(frame)

    def control_loop(self):
        if self.is_fourth_stage_state(self.state):
            self.fourth_control_loop()
            return
        CombinedStage1Stage2Node.control_loop(self)


def main(args=None):
    rclpy.init(args=args)
    node = FullCompetitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down, sending stop command...')
        try:
            node.send_stop_command()
        except Exception:
            try:
                node.stop()
            except Exception:
                pass
        try:
            node.Ctrl.quit()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
