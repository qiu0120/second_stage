#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
from threading import Thread, Lock

import cv2
import lcm
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import rclpy.time
from tf2_ros import Buffer, TransformListener

from fourth_stage.robot_control_cmd_lcmt import robot_control_cmd_lcmt
from fourth_stage.robot_control_response_lcmt import robot_control_response_lcmt
from cyberdog_msg.msg import YamlParam, ApplyForce


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
            'bar': 'bar.wav',                  # 识别到限高杆
            'obstacle': 'obstacle.wav',        # 识别到无法跨越障碍
            'cola': 'cola.wav',                # 识别到可乐瓶
            'orange_ball': 'orange_ball.wav',  # 识别到橙色小球
            'football': 'football.wav',        # 识别到足球
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


class yaml_pub(Node):
    def __init__(self):
        super().__init__("cyberdogmsg_node")
        self.para_pub = self.create_publisher(YamlParam, "yaml_parameter", 10)
        self.force_pub = self.create_publisher(ApplyForce, "apply_force", 10)

    def publish_yaml_kDOUBLE(self, name: str, value: float, is_user: int = 0):
        msg = YamlParam()
        msg.name = name
        msg.kind = ControlParameterValueKind.kDOUBLE
        msg.s64_value = float(value)
        msg.is_user = int(is_user)
        self.para_pub.publish(msg)

    def publish_yaml_s64(self, name: str, value: int, is_user: int = 0):
        msg = YamlParam()
        msg.name = name
        msg.kind = ControlParameterValueKind.kS64
        msg.s64_value = int(value)
        msg.is_user = int(is_user)
        self.para_pub.publish(msg)

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
        self.para_pub.publish(msg)

    def publish_apply_force(self, link_name: str, rel_pos, force, duration: float):
        msg = ApplyForce()
        msg.link_name = link_name
        msg.rel_pos = [float(x) for x in rel_pos]
        msg.force = [float(x) for x in force]
        msg.time = float(duration)
        self.force_pub.publish(msg)


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

            candidates.append((score, rx, ry, rw, rh, aspect_ratio))
        if not candidates:
            return None
        score, rx, ry, rw, rh, aspect_ratio = max(candidates, key=lambda x: x[0])
        bx1, by1 = x1 + rx, y1 + ry
        bx2, by2 = bx1 + rw, by1 + rh
        cx = bx1 + rw // 2
        cy = by1 + rh // 2
        return Detection('bar', (cx, cy), (bx1, by1, bx2, by2), float(score), {'aspect_ratio': float(aspect_ratio)})


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
            score = (radius * self.radius_score_gain) * max(circularity, 0.0) * (self.center_weight_base + self.center_weight_gain * center_bonus)
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




# ============================================================
# 蓝色障碍物检测：两个蓝色方块 + 深度
# ============================================================

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


# ============================================================
# 黄色竖直虚线检测
# ============================================================

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


# ============================================================
# 前方横向黄线检测：用于最后结束前边走边对正黄线
# ============================================================

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
# Robot_Ctrl
# ============================================================

class Robot_Ctrl(object):
    def __init__(self):
        self.rec_thread = Thread(target=self.rec_responce)
        self.send_thread = Thread(target=self.send_publish)
        self.lc_r = lcm.LCM("udpm://239.255.76.67:7670?ttl=255")
        self.lc_s = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
        self.cmd_msg = robot_control_cmd_lcmt()
        self.rec_msg = robot_control_response_lcmt()
        self.send_lock = Lock()
        self.delay_cnt = 0
        self.mode_ok = 0
        self.gait_ok = 0
        self.runing = 1

    def run(self):
        self.lc_r.subscribe("robot_control_response", self.msg_handler)
        self.send_thread.start()
        self.rec_thread.start()

    def msg_handler(self, channel, data):
        self.rec_msg = robot_control_response_lcmt().decode(data)
        if self.rec_msg.order_process_bar >= 95:
            self.mode_ok = self.rec_msg.mode
            self.gait_ok = self.rec_msg.gait_id
        else:
            self.mode_ok = 0
            self.gait_ok = 0

    def rec_responce(self):
        while self.runing:
            self.lc_r.handle()
            time.sleep(0.002)

    def Wait_finish(self, mode, gait_id):
        count = 0
        while self.runing and count < 2000:
            if self.mode_ok == mode and self.gait_ok == gait_id:
                return True
            time.sleep(0.005)
            count += 1
        return False

    def send_publish(self):
        while self.runing:
            self.send_lock.acquire()
            if self.delay_cnt > 20:
                self.lc_s.publish("robot_control_cmd", self.cmd_msg.encode())
                self.delay_cnt = 0
            self.delay_cnt += 1
            self.send_lock.release()
            time.sleep(0.005)

    def Send_cmd(self, msg):
        self.send_lock.acquire()
        self.delay_cnt = 50
        self.cmd_msg = msg
        self.send_lock.release()

    def quit(self):
        self.runing = 0
        self.rec_thread.join()
        self.send_thread.join()


# ============================================================
# 主任务节点
# ============================================================

class ObstacleDashedTaskNode(Node):
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

    # 新增：最终对准横向黄线后，再执行两次左跳动作，然后进入障碍物流程完成计数
    FINAL_LEFT_JUMP = 'FINAL_LEFT_JUMP'

    # 全部 2 次限高杆 + 1 次障碍物完成后的最终收尾动作：
    # 右跳一次 -> 前进识别前方横向黄线并矫正朝向 -> 黄线到达图像下方阈值 -> 左跳一次 -> DONE
    GLOBAL_FINAL_RIGHT_JUMP = 'GLOBAL_FINAL_RIGHT_JUMP'
    GLOBAL_FINAL_YELLOW_FORWARD = 'GLOBAL_FINAL_YELLOW_FORWARD'
    GLOBAL_FINAL_LEFT_JUMP = 'GLOBAL_FINAL_LEFT_JUMP'

    DONE = 'DONE'

    def __init__(self):
        super().__init__('obstacle_dashed_task_node')

        self.bridge = CvBridge()
        self.yaml_node = yaml_pub()

        # TF：用于后续转向角度和前进距离判断
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.declare_parameter('rgb_topic', '/rgb_camera/rgb_camera/image_raw')
        self.declare_parameter('depth_topic', '/d435/depth/d435_depth/depth/image_raw')
        self.declare_parameter('control_hz', 30.0)
        self.declare_parameter('show_debug_vis', True)
        self.declare_parameter('voice_enabled', True)
        self.declare_parameter('voice_dir', '/home/cyberdog_sim/voice')

        # ============================================================
        # 调试用：指定程序启动后的初始状态
        # 默认从 GLOBAL_INITIAL_LATERAL_SHIFT 开始完整整合流程。
        #
        # 可选状态及含义：
        # GLOBAL_INITIAL_LATERAL_SHIFT
        #   程序启动后的预左移状态。先固定向左移动一段 TF 距离，不识别目标；
        #   完成后进入 GLOBAL_LATERAL_SEARCH。
        #
        # GLOBAL_LATERAL_SEARCH
        #   全局向左搜索状态。只检测还没完成的任务类型：
        #   限高杆未满 2 次才检测限高杆，障碍物未满 1 次才检测障碍物。
        #
        # GLOBAL_CENTER_BAR / GLOBAL_CENTER_OBSTACLE
        #   识别到目标后先横向居中，稳定若干帧后再进入对应子流程。
        #
        # GLOBAL_SHIFT_AFTER_SUBTASK
        #   完成一个子流程后，如果总任务还没完成，就向左移动一段距离再继续搜索；
        #   如果总任务已完成，则不进入该状态，直接 DONE。
        #
        # APPROACH_OBSTACLES
        #   启动后向前走，识别两个蓝色障碍物，并对齐两个障碍物中点。
        #
        # DASH_PRE_SIDE_SHIFT
        #   第一次识别到黄色竖直虚线后，先朝虚线所在方向横移一小段。
        #   注意：单独从这个状态启动时，需要配合 debug_dashed_side 使用。
        #
        # ALIGN_DASHED_LINE
        #   对黄色竖直虚线做偏置对齐。
        #   左边虚线让它位于图像中间偏右，右边虚线让它位于图像中间偏左。
        #
        # FOLLOW_DASHED_UNTIL_LOST
        #   虚线对齐稳定后继续向前走，直到虚线连续丢失若干帧。
        #
        # POST_DASH_FORWARD
        #   虚线消失后继续向前走一小段 TF 距离。
        #
        # POST_DASH_TURN_1
        #   虚线消失后的第一次 TF 转向。
        #   dashed_side == 'left' 时右转，dashed_side == 'right' 时左转。
        #
        # POST_TURN_FORWARD
        #   第一次转向结束后继续向前走一段 TF 距离。
        #
        # POST_DASH_TURN_2
        #   第二次 TF 转向，方向和第一次相反。
        #
        # SEARCH_TARGET_AFTER_TURNS
        #   第二次转向后开始搜索目标：blue_ball、white_ball、cola。
        #
        # APPROACH_AND_ALIGN_TARGET
        #   锁定目标后，边向前走边根据目标中心做横向对齐。
        #
        # HIT_TARGET
        #   根据 locked_target 类型执行对应速度和距离的撞击。
        #   注意：单独从这个状态启动时，一般需要已有 locked_target，不建议直接从这里启动。
        #
        # HIT_BACKOFF_AFTER_HIT
        #   撞击完成后，按 TF 距离后退一小段。
        #
        # POST_HIT_LEFT_JUMP
        #   后退完成后执行两次左跳。
        #
        # APPROACH_SELECTED_OBSTACLE_AFTER_HIT
        #   两次左跳后继续前进识别蓝色障碍物。
        #   dashed_side == 'left' 时对齐右边障碍物；
        #   dashed_side == 'right' 时对齐左边障碍物。
        #   距离小于 post_hit_obstacle_trigger_distance_m 后进入后续转向任务。
        #
        # POST_HIT_OBS_TURN_1
        #   到达选中障碍物距离后第一次 TF 转向。
        #   如果当前对齐的是左边障碍物就左转；如果当前对齐的是右边障碍物就右转。
        #
        # POST_HIT_OBS_FORWARD
        #   第一次转向后，按 TF 距离向前走一段。
        #
        # POST_HIT_OBS_TURN_2
        #   第二次 TF 转向，方向和第一次相反。
        #
        # POST_HIT_PRE_FINAL_FORWARD
        #   第二次转回后，先不识别黄线，按 TF 向前走一段固定距离。
        #   距离由 post_hit_final_forward_distance_m 控制，速度由 post_hit_final_forward_speed 控制。
        #
        # POST_HIT_FINAL_FORWARD
        #   固定距离前进完成后，开始边前进边识别前方横向黄线。
        #   根据黄线倾斜角修正 wz，让机器狗正对黄线；
        #   黄线底部到达 final_yellow_stop_line_y_ratio 对应图像位置，且角度基本对正后进入 FINAL_LEFT_JUMP。
        #
        # FINAL_LEFT_JUMP
        #   最终对准横向黄线后，执行两次原地左跳动作。
        #   两次左跳完成后进入 DONE。
        #
        # DONE
        #   任务结束，发送 STOP。
        # ============================================================
        self.declare_parameter('initial_state', 'GLOBAL_INITIAL_LATERAL_SHIFT')

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

        # 程序开始先固定向左移动一段距离，不检测限高杆/障碍物，避免一启动就在原地误触发。
        self.declare_parameter('global_initial_lateral_shift_distance_m', 0.20)
        self.declare_parameter('global_initial_lateral_shift_vy', 0.30)

        # 启动预左移完成后，进入全局搜索；搜索阶段会按完成计数决定是否检测限高杆/障碍物。
        self.declare_parameter('global_lateral_search_vy', 0.30)
        self.declare_parameter('global_center_stable_frames', 1)
        self.declare_parameter('global_after_task_shift_vy', 0.30)
        self.declare_parameter('global_after_task_shift_distance_m', 0.75)
        # 障碍物流程完成后，如果虚线在右边，左移距离更大
        self.declare_parameter('global_after_obstacle_shift_distance_left_dash_m', 0.10)
        self.declare_parameter('global_after_obstacle_shift_distance_right_dash_m', 1.0)

        # 完成一次限高杆流程后，下一轮全局搜索只在图像左半边寻找目标。
        # 这样可以避免刚完成的限高杆仍出现在右半边时被重复选中。
        self.declare_parameter('search_left_half_after_bar_done', True)
        self.declare_parameter('after_bar_search_x_ratio_max', 0.50)

        # 限高杆流程参数
        self.declare_parameter('bar_search_forward_speed', 0.60)
        self.declare_parameter('bar_trigger_distance_m', 0.40)
        self.declare_parameter('bar_align_vy_k', 0.35)
        self.declare_parameter('bar_align_vy_max', 0.30)
        self.declare_parameter('bar_align_vy_min', 0.20)
        self.declare_parameter('bar_center_px_deadband', 7)
        self.declare_parameter('bar_center_stable_frames', 3)
        self.declare_parameter('backoff_after_hit_speed', 0.60)
        self.declare_parameter('backoff_bar_depth_tolerance_m', 0.05)
        self.declare_parameter('backoff_min_time_s', 0.30)

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
        self.declare_parameter('dashed_center_stable_frames', 3)

        # 黄线开始对齐前，先朝虚线所在方向横移一小段
        # 现在不再按时间结束，而是用 TF 判断横向位移达到 dashed_pre_shift_distance_m 后结束。
        self.declare_parameter('dashed_pre_shift_speed', 0.25)
        self.declare_parameter('dashed_pre_shift_distance_m', 0.35)

        # 偏置对齐目标，单位：像素
        # 左边有虚线：目标点 = 图像中心 + offset，也就是中间偏右
        # 右边有虚线：目标点 = 图像中心 - offset，也就是中间偏左
        self.declare_parameter('dashed_target_offset_px', 50)

        self.declare_parameter('follow_forward_speed', 0.40)
        self.declare_parameter('follow_align_vy_k', 0.18)
        # FOLLOW_DASHED_UNTIL_LOST 阶段单独使用的横向速度上下限，
        # 不再和 ALIGN_DASHED_LINE 共用 dashed_align_vy_min / dashed_align_vy_max。
        self.declare_parameter('follow_align_vy_max', 0.15)
        self.declare_parameter('follow_align_vy_min', 0.05)
        self.declare_parameter('dashed_lost_stop_frames', 2)

        # =========================
        # 虚线消失后的后续任务参数
        # =========================
        self.declare_parameter('tf_parent_frame', 'vodom')
        self.declare_parameter('tf_child_frame', 'base_link')

        # 虚线识别不到后，继续向前走一小段距离，单位：米
        self.declare_parameter('post_dash_forward_distance_m', 0.23)
        self.declare_parameter('post_dash_forward_speed', 0.30)

        # 第一次转向角度，单位：度。左边虚线 -> 右转；右边虚线 -> 左转
        self.declare_parameter('post_dash_turn_angle_deg', 30.0)
        self.declare_parameter('post_dash_turn_wz', 0.30)
        self.declare_parameter('post_dash_turn_tolerance_deg', 1.5)

        # 第一次转向完成后，继续前进一段距离，单位：米
        self.declare_parameter('post_turn_forward_distance_m', 0.50)
        self.declare_parameter('post_turn_forward_speed', 0.30)

        # 第二次转向角度，方向与第一次相反，单位：度
        self.declare_parameter('post_second_turn_angle_deg', 30.0)
        self.declare_parameter('post_second_turn_wz', 0.30)
        self.declare_parameter('post_second_turn_tolerance_deg', 1.5)

        # =========================
        # 第二次转向后的目标检测 / 对齐 / 撞击参数
        # 复用限高杆任务代码里的目标检测逻辑：蓝球、白球、可乐
        # =========================
        self.declare_parameter('target_search_forward_speed', 0.20)
        self.declare_parameter('align_forward_speed_far', 0.40)
        self.declare_parameter('align_forward_speed_near', 0.40)
        self.declare_parameter('align_vy_k', 0.35)
        self.declare_parameter('align_vy_max', 0.30)
        self.declare_parameter('align_vy_min', 0.20)
        self.declare_parameter('target_stable_frames', 3)
        self.declare_parameter('hit_trigger_distance_m', 0.20)
        self.declare_parameter('center_px_deadband', 7)

        self.declare_parameter('hit_blue_ball_speed', 0.30)
        self.declare_parameter('hit_blue_ball_distance', 0.25)
        self.declare_parameter('hit_white_ball_speed', 0.30)
        self.declare_parameter('hit_white_ball_distance', 0.45)
        self.declare_parameter('hit_cola_speed', 0.30)
        self.declare_parameter('hit_cola_distance', 0.25)
        self.declare_parameter('hit_timeout_s', 10.0)

        # 撞击完成后：先后退一小段，再连续左跳两次
        self.declare_parameter('after_hit_backoff_distance_m', 0.35)
        self.declare_parameter('after_hit_backoff_speed', 0.40)
        self.declare_parameter('after_hit_left_jump_count', 2)

        # 左跳完成后：前进并识别两个蓝色障碍物，按虚线侧选择其中一个居中
        # 之前虚线在左边 -> 对齐右边障碍物；之前虚线在右边 -> 对齐左边障碍物
        self.declare_parameter('post_hit_obstacle_forward_speed', 0.30)
        self.declare_parameter('post_hit_obstacle_search_forward_speed', 0.30)
        self.declare_parameter('post_hit_obstacle_trigger_distance_m', 0.20)
        self.declare_parameter('post_hit_obstacle_align_vy_k', 0.35)
        self.declare_parameter('post_hit_obstacle_align_vy_max', 0.30)
        self.declare_parameter('post_hit_obstacle_align_vy_min', 0.20)
        self.declare_parameter('post_hit_obstacle_center_px_deadband', 7)

        # 对齐选中障碍物并到达距离后：转向 -> 前进 -> 反向转向 -> 最后前进
        # 如果对齐左边障碍物：第一次左转；如果对齐右边障碍物：第一次右转
        self.declare_parameter('post_hit_obs_turn_angle_deg', 30.0)
        self.declare_parameter('post_hit_obs_turn_wz', 0.30)
        self.declare_parameter('post_hit_obs_turn_tolerance_deg', 1.5)
        self.declare_parameter('post_hit_obs_forward_distance_m', 0.40)
        self.declare_parameter('post_hit_obs_forward_speed', 0.30)
        # 第二次转回后，先按 TF 向前走一段固定距离，再进入最终横向黄线识别对正
        self.declare_parameter('post_hit_final_forward_distance_m', 0.20)
        self.declare_parameter('post_hit_final_forward_speed', 0.40)

        # 最后前进阶段：前方横向黄线检测 + 朝向修正
        self.declare_parameter('final_yellow_stop_line_y_ratio', 0.95)
        self.declare_parameter('final_yellow_align_wz_k', 1.20)
        self.declare_parameter('final_yellow_align_wz_max', 0.30)
        self.declare_parameter('final_yellow_align_wz_min', 0.15)
        self.declare_parameter('final_yellow_tilt_deadband_deg', 1.5)
        self.declare_parameter('final_yellow_done_tilt_deg', 3.0)
        self.declare_parameter('final_yellow_confirm_count', 1)

        # 全局最终收尾阶段：所有流程完成后，右跳 -> 前进识别横向黄线并矫正 -> 左跳一次。
        # 黄线检测仍复用 final_yellow.* 的 HSV/ROI/形状过滤参数。
        self.declare_parameter('global_final_yellow_forward_speed', 0.30)
        self.declare_parameter('global_final_yellow_stop_line_y_ratio', 1.0)
        self.declare_parameter('global_final_yellow_confirm_count', 2)

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
        self.initial_state = str(self.get_parameter('initial_state').value)
        self.debug_dashed_side = str(self.get_parameter('debug_dashed_side').value).lower().strip()

        self.required_bar_count = int(self.get_parameter('required_bar_count').value)
        self.required_obstacle_count = int(self.get_parameter('required_obstacle_count').value)
        self.global_initial_lateral_shift_distance_m = float(self.get_parameter('global_initial_lateral_shift_distance_m').value)
        self.global_initial_lateral_shift_vy = float(self.get_parameter('global_initial_lateral_shift_vy').value)
        self.global_lateral_search_vy = float(self.get_parameter('global_lateral_search_vy').value)
        self.global_center_stable_frames = int(self.get_parameter('global_center_stable_frames').value)
        self.global_after_task_shift_vy = float(self.get_parameter('global_after_task_shift_vy').value)
        self.global_after_task_shift_distance_m = float(self.get_parameter('global_after_task_shift_distance_m').value)
        self.global_after_obstacle_shift_distance_left_dash_m = float(self.get_parameter('global_after_obstacle_shift_distance_left_dash_m').value)
        self.global_after_obstacle_shift_distance_right_dash_m = float(self.get_parameter('global_after_obstacle_shift_distance_right_dash_m').value)
        self.search_left_half_after_bar_done = bool(self.get_parameter('search_left_half_after_bar_done').value)
        self.after_bar_search_x_ratio_max = float(self.get_parameter('after_bar_search_x_ratio_max').value)

        self.bar_search_forward_speed = float(self.get_parameter('bar_search_forward_speed').value)
        self.bar_trigger_distance_m = float(self.get_parameter('bar_trigger_distance_m').value)
        self.bar_align_vy_k = float(self.get_parameter('bar_align_vy_k').value)
        self.bar_align_vy_max = float(self.get_parameter('bar_align_vy_max').value)
        self.bar_align_vy_min = float(self.get_parameter('bar_align_vy_min').value)
        self.bar_center_px_deadband = int(self.get_parameter('bar_center_px_deadband').value)
        self.bar_center_stable_frames = int(self.get_parameter('bar_center_stable_frames').value)
        self.backoff_after_hit_speed = abs(float(self.get_parameter('backoff_after_hit_speed').value))
        self.backoff_bar_depth_tolerance_m = float(self.get_parameter('backoff_bar_depth_tolerance_m').value)
        self.backoff_min_time_s = float(self.get_parameter('backoff_min_time_s').value)

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
        self.dashed_pre_shift_distance_m = float(self.get_parameter('dashed_pre_shift_distance_m').value)
        self.dashed_target_offset_px = int(self.get_parameter('dashed_target_offset_px').value)

        self.follow_forward_speed = float(self.get_parameter('follow_forward_speed').value)
        self.follow_align_vy_k = float(self.get_parameter('follow_align_vy_k').value)
        self.follow_align_vy_max = float(self.get_parameter('follow_align_vy_max').value)
        self.follow_align_vy_min = float(self.get_parameter('follow_align_vy_min').value)
        self.dashed_lost_stop_frames = int(self.get_parameter('dashed_lost_stop_frames').value)

        self.tf_parent_frame = str(self.get_parameter('tf_parent_frame').value)
        self.tf_child_frame = str(self.get_parameter('tf_child_frame').value)

        self.post_dash_forward_distance_m = float(self.get_parameter('post_dash_forward_distance_m').value)
        self.post_dash_forward_speed = float(self.get_parameter('post_dash_forward_speed').value)

        self.post_dash_turn_angle_rad = math.radians(float(self.get_parameter('post_dash_turn_angle_deg').value))
        self.post_dash_turn_wz = float(self.get_parameter('post_dash_turn_wz').value)
        self.post_dash_turn_tolerance_rad = math.radians(float(self.get_parameter('post_dash_turn_tolerance_deg').value))

        self.post_turn_forward_distance_m = float(self.get_parameter('post_turn_forward_distance_m').value)
        self.post_turn_forward_speed = float(self.get_parameter('post_turn_forward_speed').value)

        self.post_second_turn_angle_rad = math.radians(float(self.get_parameter('post_second_turn_angle_deg').value))
        self.post_second_turn_wz = float(self.get_parameter('post_second_turn_wz').value)
        self.post_second_turn_tolerance_rad = math.radians(float(self.get_parameter('post_second_turn_tolerance_deg').value))

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

        self.after_hit_backoff_distance_m = float(self.get_parameter('after_hit_backoff_distance_m').value)
        self.after_hit_backoff_speed = abs(float(self.get_parameter('after_hit_backoff_speed').value))
        self.after_hit_left_jump_count = int(self.get_parameter('after_hit_left_jump_count').value)

        self.post_hit_obstacle_forward_speed = float(self.get_parameter('post_hit_obstacle_forward_speed').value)
        self.post_hit_obstacle_search_forward_speed = float(self.get_parameter('post_hit_obstacle_search_forward_speed').value)
        self.post_hit_obstacle_trigger_distance_m = float(self.get_parameter('post_hit_obstacle_trigger_distance_m').value)
        self.post_hit_obstacle_align_vy_k = float(self.get_parameter('post_hit_obstacle_align_vy_k').value)
        self.post_hit_obstacle_align_vy_max = float(self.get_parameter('post_hit_obstacle_align_vy_max').value)
        self.post_hit_obstacle_align_vy_min = float(self.get_parameter('post_hit_obstacle_align_vy_min').value)
        self.post_hit_obstacle_center_px_deadband = int(self.get_parameter('post_hit_obstacle_center_px_deadband').value)

        self.post_hit_obs_turn_angle_rad = math.radians(float(self.get_parameter('post_hit_obs_turn_angle_deg').value))
        self.post_hit_obs_turn_wz = float(self.get_parameter('post_hit_obs_turn_wz').value)
        self.post_hit_obs_turn_tolerance_rad = math.radians(float(self.get_parameter('post_hit_obs_turn_tolerance_deg').value))
        self.post_hit_obs_forward_distance_m = float(self.get_parameter('post_hit_obs_forward_distance_m').value)
        self.post_hit_obs_forward_speed = float(self.get_parameter('post_hit_obs_forward_speed').value)
        self.post_hit_final_forward_distance_m = float(self.get_parameter('post_hit_final_forward_distance_m').value)
        self.post_hit_final_forward_speed = float(self.get_parameter('post_hit_final_forward_speed').value)

        self.final_yellow_stop_line_y_ratio = float(self.get_parameter('final_yellow_stop_line_y_ratio').value)
        self.final_yellow_align_wz_k = float(self.get_parameter('final_yellow_align_wz_k').value)
        self.final_yellow_align_wz_max = float(self.get_parameter('final_yellow_align_wz_max').value)
        self.final_yellow_align_wz_min = float(self.get_parameter('final_yellow_align_wz_min').value)
        self.final_yellow_tilt_deadband_deg = float(self.get_parameter('final_yellow_tilt_deadband_deg').value)
        self.final_yellow_done_tilt_deg = float(self.get_parameter('final_yellow_done_tilt_deg').value)
        self.final_yellow_confirm_count = int(self.get_parameter('final_yellow_confirm_count').value)

        self.global_final_yellow_forward_speed = float(self.get_parameter('global_final_yellow_forward_speed').value)
        self.global_final_yellow_stop_line_y_ratio = float(self.get_parameter('global_final_yellow_stop_line_y_ratio').value)
        self.global_final_yellow_confirm_count = int(self.get_parameter('global_final_yellow_confirm_count').value)

        self.hit_params = {
            'blue_ball': {
                'speed': float(self.get_parameter('hit_blue_ball_speed').value),
                'distance': float(self.get_parameter('hit_blue_ball_distance').value),
            },
            'white_ball': {
                'speed': float(self.get_parameter('hit_white_ball_speed').value),
                'distance': float(self.get_parameter('hit_white_ball_distance').value),
            },
            'cola': {
                'speed': float(self.get_parameter('hit_cola_speed').value),
                'distance': float(self.get_parameter('hit_cola_distance').value),
            },
        }

        self.bar_detector = BarColorDetector(self._read_bar_cfg())
        self.obstacle_detector = ObstacleBlueDepthDetector(self._read_obstacle_cfg())
        self.dashed_detector = YellowDashedLineDetector(self._read_yellow_cfg())
        self.final_yellow_detector = YellowHorizontalLineDetector(self._read_final_yellow_cfg())

        self.blue_ball_detector = BallDetector(self._read_ball_cfg('blue_ball'), 'blue_ball')
        self.white_ball_detector = BallDetector(self._read_ball_cfg('white_ball'), 'white_ball')
        self.cola_detector = ColaDetector(self._read_cola_cfg())

        self.latest_bgr = None
        self.latest_depth = None

        # 语音播报：用事件 ID 防止同一个触发点重复播报。
        # bar_1 / bar_2 可以分别播报；同一个目标类型只在准备撞击时播一次。
        self.voice = VoicePlayer(self.voice_dir, enabled=self.voice_enabled)
        self.voice_events_spoken = set()

        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            qos_profile_sensor_data
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile_sensor_data
        )

        self.Ctrl = Robot_Ctrl()
        self.Ctrl.run()

        self.msg = robot_control_cmd_lcmt()
        if not hasattr(self.msg, 'life_count'):
            self.msg.life_count = 0

        self.motion_cmd = (0.0, 0.0, 0.0)
        self.body_height_cmd = 0.25

        self.state = self.APPROACH_OBSTACLES
        self.state_enter_time = self.now_s()

        self.dashed_center_count = 0
        self.dashed_lost_count = 0

        # 第一次看到虚线时，记录它在图像左边还是右边。
        # 后续预横移和偏置对齐都会使用这个方向，不在 ALIGN 状态里清空。
        self.dashed_side = None          # None / 'left' / 'right'
        # DASH_PRE_SIDE_SHIFT 不再按时间结束，而是记录 TF 起点，按横向位移结束。
        self.dashed_pre_shift_start_pose = None
        self.dashed_pre_shift_dir_sign = 0.0

        # 后续任务使用的 TF 起点
        self.post_forward_start_pose = None
        self.post_turn_forward_start_pose = None
        self.turn_start_yaw = None
        self.current_turn_dir = 0       # +1 左转，-1 右转
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
        self.selected_obstacle_after_hit_side = None   # 'left' / 'right'
        self.post_hit_obs_forward_start_pose = None
        self.post_hit_pre_final_forward_start_pose = None
        self.post_hit_final_forward_start_pose = None  # 保留旧变量名，避免外部引用出错

        # 最后阶段前方横向黄线对正 / 到达判定
        self.final_yellow_done_counter = 0
        self.latest_final_yellow_line: Optional[Detection] = None

        # 全局最终收尾黄线确认计数器
        self.global_final_yellow_done_counter = 0

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

        self.global_center_stable_count = 0
        self.current_global_target: Optional[Detection] = None
        self.global_initial_lateral_shift_start_pose = None
        self.global_after_task_shift_start_pose = None
        self.current_after_task_shift_distance_m = self.global_after_task_shift_distance_m
        self.global_after_task_shift_reason = 'init'

        # 限高杆子流程变量
        self.current_bar_det: Optional[Detection] = None
        self.bar_return_target_depth_m: Optional[float] = None
        self.bar_center_stable_count = 0
        self.bar_hit_start_pose = None
        self.bar_backoff_start_time = None

        self.task_done_stop_sent = False
        self.last_log_time = self.now_s()

        self.timer = self.create_timer(1.0 / self.control_hz, self.control_loop)

        self.get_logger().info('obstacle_dashed_task_node started')
        self.get_logger().info(f'RGB: {self.rgb_topic}')
        self.get_logger().info(f'DEPTH: {self.depth_topic}')

        # 程序开始就切换到 low 机身高度，后续全程保持 low
        self.set_body_low()
        # self.set_body_normal()
        # self.send_left_jump_action_once()
        # self.send_left_jump_action_once()

        # 根据 initial_state 参数进入指定调试状态
        self.enter_initial_state()

    # ---------- 全局整合 / 限高杆辅助 ----------
    def _declare_bar_params(self):
        p = self.declare_parameter
        p('bar.h_min', 85); p('bar.h_max', 100)
        p('bar.s_min', 15); p('bar.s_max', 45)
        p('bar.v_min', 35); p('bar.v_max', 80)
        p('bar.roi_x_ratio_min', 0.20); p('bar.roi_x_ratio_max', 0.80)
        p('bar.roi_y_ratio_min', 0.10); p('bar.roi_y_ratio_max', 0.90)
        p('bar.open_kernel', 3)
        p('bar.close_kernel_h', 7); p('bar.close_kernel_w', 11)
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
        x_limit = img_w * self.after_bar_search_x_ratio_max
        return det.center_img[0] < x_limit

    def choose_global_object(self, bar: Optional[Detection], obs_candidates: List[Detection]):
        if self.latest_bgr is None:
            return None, None

        img_w = self.latest_bgr.shape[1]
        img_center_x = img_w / 2.0
        choices = []

        if self.only_search_left_half_after_bar:
            x_limit = img_w * self.after_bar_search_x_ratio_max
            self.get_logger().info(
                f'[GLOBAL_FILTER] after bar done: only search left region, '
                f'center_x < {x_limit:.1f} ({self.after_bar_search_x_ratio_max:.2f} * image_width)',
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

        return obj_type, det

    def finish_bar_flow(self):
        self.completed_bar_count += 1
        self.get_logger().info(
            f'[GLOBAL] bar flow finished: bar={self.completed_bar_count}/{self.required_bar_count}, '
            f'obstacle={self.completed_obstacle_count}/{self.required_obstacle_count}'
        )
        if self.all_global_tasks_done():
            self.get_logger().info('[GLOBAL] all flows done after bar flow, start final sequence')
            self.enter_global_final_sequence()
            return

        # 新增：完成一次限高杆后，下一轮 GLOBAL_LATERAL_SEARCH 只在左半边找目标。
        # 目的：避免刚刚通过的限高杆还在右半边画面里，被重复当成下一个目标。
        if self.search_left_half_after_bar_done:
            self.only_search_left_half_after_bar = True
            self.get_logger().info(
                f'[GLOBAL_FILTER] bar done: next global search only uses left region, '
                f'x_ratio_max={self.after_bar_search_x_ratio_max:.2f}'
            )

        self.current_after_task_shift_distance_m = self.global_after_task_shift_distance_m
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
        if self.dashed_side == 'right':
            self.current_after_task_shift_distance_m = self.global_after_obstacle_shift_distance_right_dash_m
        elif self.dashed_side == 'left':
            self.current_after_task_shift_distance_m = self.global_after_obstacle_shift_distance_left_dash_m
        else:
            self.current_after_task_shift_distance_m = self.global_after_task_shift_distance_m
            self.get_logger().warn('[GLOBAL] obstacle finished but dashed_side is None, use default shift distance')
        self.global_after_task_shift_reason = f'obstacle_done_dash_{self.dashed_side}'
        self.get_logger().info(
            f'[GLOBAL] after obstacle shift distance={self.current_after_task_shift_distance_m:.3f}m, '
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
        sx1 = max(0, min(iw - 1, sx1)); sx2 = max(sx1 + 1, min(iw, sx2))
        sy1 = max(0, min(ih - 1, sy1)); sy2 = max(sy1 + 1, min(ih, sy2))
        dx1 = int(sx1 * dw / max(iw, 1)); dx2 = int(sx2 * dw / max(iw, 1))
        dy1 = int(sy1 * dh / max(ih, 1)); dy2 = int(sy2 * dh / max(ih, 1))
        dx1 = max(0, min(dw - 1, dx1)); dx2 = max(dx1 + 1, min(dw, dx2))
        dy1 = max(0, min(dh - 1, dy1)); dy2 = max(dy1 + 1, min(dh, dy2))
        patch = depth_m[dy1:dy2, dx1:dx2]
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > 0.05) & (valid < 10.0)]
        if valid.size == 0:
            return None
        return float(np.percentile(valid, 20))

    def is_bar_centered(self, bar: Detection) -> bool:
        if self.latest_bgr is None or bar is None:
            return False
        img_center_x = self.latest_bgr.shape[1] / 2.0
        err_px = bar.center_img[0] - img_center_x
        return abs(err_px) <= self.bar_center_px_deadband

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
        p('cola.h_min', 0); p('cola.h_max', 20)
        p('cola.s_min', 0); p('cola.s_max', 20)
        p('cola.v_min', 0); p('cola.v_max', 20)
        p('cola.roi_x_ratio_min', 0.20); p('cola.roi_x_ratio_max', 0.80)
        p('cola.roi_y_ratio_min', 0.00); p('cola.roi_y_ratio_max', 1.00)
        p('cola.open_kernel', 3); p('cola.close_kernel', 5)
        p('cola.min_area', 250); p('cola.max_area', 80000)
        p('cola.min_width', 8); p('cola.max_width', 5000)
        p('cola.min_height', 20); p('cola.max_height', 10000)
        p('cola.min_hw_ratio', 1.5); p('cola.max_hw_ratio', 20.0)
        p('cola.max_center_y_ratio_in_roi', 1.0)
        p('cola.center_weight_base', 0.3); p('cola.center_weight_gain', 0.7)

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

    def rgb_callback(self, msg: Image):
        try:
            self.latest_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB convert failed: {e}')

    def depth_callback(self, msg: Image):
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
    def set_body_low(self):
        """
        启动时调用一次，把机身高度设置为 low。
        后续状态不再重复调用，避免每次切状态都 stop。
        """
        self.body_height_cmd = 0.17

        values = [0.0] * 12
        values[0] = 0.0   # roll
        values[2] = 0.17  # height

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

        self.stop()
        self.get_logger().info('[BODY] set low height: 0.17')

    def set_body_normal(self):
        """
        备用：恢复 normal 高度。当前任务默认不调用。
        """
        self.body_height_cmd = 0.25

        values = [0.0] * 12
        values[0] = 0.0   # roll
        values[2] = 0.25  # height

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

        self.stop()
        self.get_logger().info('[BODY] set normal height: 0.25')

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

    def execute_left_jump_turn(self, jump_count: int, next_state: str):
        """连续执行 jump_count 次左跳，然后进入 next_state。"""
        for i in range(max(0, int(jump_count))):
            self.get_logger().info(f'[LEFT_JUMP] {i + 1}/{jump_count}')
            self.send_left_jump_action_once()
        self.enter_state(next_state)

    def execute_right_jump_turn(self, jump_count: int, next_state: str):
        """连续执行 jump_count 次右跳，然后进入 next_state。"""
        for i in range(max(0, int(jump_count))):
            self.get_logger().info(f'[RIGHT_JUMP] {i + 1}/{jump_count}')
            self.send_right_jump_action_once()
        self.enter_state(next_state)

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
            self.GLOBAL_FINAL_RIGHT_JUMP,
            self.GLOBAL_FINAL_YELLOW_FORWARD,
            self.GLOBAL_FINAL_LEFT_JUMP,
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

        if self.initial_state not in valid_states:
            self.get_logger().warn(
                f"[INIT_STATE] invalid initial_state='{self.initial_state}', "
                f"fallback to {self.GLOBAL_INITIAL_LATERAL_SHIFT}. "
                f"valid_states={valid_states}"
            )
            self.initial_state = self.GLOBAL_INITIAL_LATERAL_SHIFT

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
        if self.initial_state in states_need_dashed_side and self.dashed_side not in ('left', 'right'):
            self.get_logger().warn(
                f"[INIT_STATE] {self.initial_state} needs dashed_side, "
                "but dashed_side is None. You can run with: "
                "-p debug_dashed_side:=left or -p debug_dashed_side:=right"
            )

        self.get_logger().info(f"[INIT_STATE] start from: {self.initial_state}")
        self.enter_state(self.initial_state)

    def enter_state(self, new_state: str):
        self.state = new_state
        self.state_enter_time = self.now_s()

        self.get_logger().info(f'ENTER STATE -> {new_state}')

        if new_state == self.GLOBAL_INITIAL_LATERAL_SHIFT:
            self.global_initial_lateral_shift_start_pose = None

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
            self.global_after_task_shift_start_pose = None

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
            self.bar_hit_start_pose = self.get_current_pose_2d()

        if new_state == self.BAR_BACKOFF_TO_BAR:
            self.bar_backoff_start_time = self.now_s()

        if new_state == self.OBSTACLE_FLOW_DONE:
            self.send_motion_cmd(0.0, 0.0, 0.0)

        if new_state == self.DASH_PRE_SIDE_SHIFT:
            self.dashed_pre_shift_start_pose = self.get_current_pose_2d()
            self.dashed_pre_shift_dir_sign = self.get_pre_shift_dir_sign()
            self.dashed_center_count = 0
            self.dashed_lost_count = 0

            if self.dashed_pre_shift_start_pose is None:
                self.get_logger().warn('[DASH_PRE_SHIFT] TF unavailable when entering state')

            self.get_logger().info(
                f'[DASH_PRE_SHIFT] enter: side={self.dashed_side}, '
                f'dir_sign={self.dashed_pre_shift_dir_sign:.1f}, '
                f'target_dist={self.dashed_pre_shift_distance_m:.3f}m'
            )

        if new_state == self.ALIGN_DASHED_LINE:
            self.dashed_center_count = 0
            self.dashed_lost_count = 0

        if new_state == self.FOLLOW_DASHED_UNTIL_LOST:
            self.dashed_lost_count = 0

        if new_state == self.POST_DASH_FORWARD:
            self.post_forward_start_pose = self.get_current_pose_2d()
            if self.post_forward_start_pose is None:
                self.get_logger().warn('[POST_DASH_FORWARD] TF unavailable at state enter')

        if new_state == self.POST_DASH_TURN_1:
            self.turn_start_yaw = self.get_current_yaw()
            self.current_turn_dir = self.get_first_turn_dir()
            self.current_turn_angle_rad = self.post_dash_turn_angle_rad
            self.current_turn_tolerance_rad = self.post_dash_turn_tolerance_rad
            self.current_turn_wz = self.post_dash_turn_wz
            if self.turn_start_yaw is None:
                self.get_logger().warn('[POST_DASH_TURN_1] TF yaw unavailable at state enter')

        if new_state == self.POST_TURN_FORWARD:
            self.post_turn_forward_start_pose = self.get_current_pose_2d()
            if self.post_turn_forward_start_pose is None:
                self.get_logger().warn('[POST_TURN_FORWARD] TF unavailable at state enter')

        if new_state == self.POST_DASH_TURN_2:
            self.turn_start_yaw = self.get_current_yaw()
            self.current_turn_dir = -self.get_first_turn_dir()
            self.current_turn_angle_rad = self.post_second_turn_angle_rad
            self.current_turn_tolerance_rad = self.post_second_turn_tolerance_rad
            self.current_turn_wz = self.post_second_turn_wz
            if self.turn_start_yaw is None:
                self.get_logger().warn('[POST_DASH_TURN_2] TF yaw unavailable at state enter')

        if new_state == self.SEARCH_TARGET_AFTER_TURNS:
            self.latest_target = None
            self.locked_target = None
            self.target_stable_count = 0
            self.stable_target_type = None
            self.hit_start_pose = None

        if new_state == self.APPROACH_AND_ALIGN_TARGET:
            self.hit_start_pose = None

        if new_state == self.HIT_TARGET:
            self.hit_start_pose = self.get_current_pose_2d()
            self.get_logger().info(f'[HIT] start pose={self.hit_start_pose}, target={self.locked_target.det_type if self.locked_target else None}')

        if new_state == self.HIT_BACKOFF_AFTER_HIT:
            self.after_hit_backoff_start_pose = self.get_current_pose_2d()
            self.selected_obstacle_after_hit = None
            if self.after_hit_backoff_start_pose is None:
                self.get_logger().warn('[AFTER_HIT_BACKOFF] TF unavailable at state enter')

        if new_state == self.POST_HIT_LEFT_JUMP:
            self.selected_obstacle_after_hit = None

        if new_state == self.APPROACH_SELECTED_OBSTACLE_AFTER_HIT:
            self.selected_obstacle_after_hit = None
            self.selected_obstacle_after_hit_side = None

        if new_state == self.POST_HIT_OBS_TURN_1:
            self.turn_start_yaw = self.get_current_yaw()
            self.current_turn_dir = self.get_post_hit_obs_first_turn_dir()
            self.current_turn_angle_rad = self.post_hit_obs_turn_angle_rad
            self.current_turn_tolerance_rad = self.post_hit_obs_turn_tolerance_rad
            self.current_turn_wz = self.post_hit_obs_turn_wz
            self.get_logger().warn(
                f'[POST_HIT_OBS_TURN_1] use dashed_side to turn: '
                f'dashed_side={self.dashed_side}, turn_dir={self.current_turn_dir}, '
                f'wz_cmd={self.current_turn_dir * abs(self.current_turn_wz):.3f}'
            )
            if self.turn_start_yaw is None:
                self.get_logger().warn('[POST_HIT_OBS_TURN_1] TF yaw unavailable at state enter')

        if new_state == self.POST_HIT_OBS_FORWARD:
            self.post_hit_obs_forward_start_pose = self.get_current_pose_2d()
            if self.post_hit_obs_forward_start_pose is None:
                self.get_logger().warn('[POST_HIT_OBS_FORWARD] TF unavailable at state enter')

        if new_state == self.POST_HIT_OBS_TURN_2:
            self.turn_start_yaw = self.get_current_yaw()
            self.current_turn_dir = -self.get_post_hit_obs_first_turn_dir()
            self.current_turn_angle_rad = self.post_hit_obs_turn_angle_rad
            self.current_turn_tolerance_rad = self.post_hit_obs_turn_tolerance_rad
            self.current_turn_wz = self.post_hit_obs_turn_wz
            self.get_logger().warn(
                f'[POST_HIT_OBS_TURN_2] reverse first turn: '
                f'dashed_side={self.dashed_side}, turn_dir={self.current_turn_dir}, '
                f'wz_cmd={self.current_turn_dir * abs(self.current_turn_wz):.3f}'
            )
            if self.turn_start_yaw is None:
                self.get_logger().warn('[POST_HIT_OBS_TURN_2] TF yaw unavailable at state enter')

        if new_state == self.POST_HIT_PRE_FINAL_FORWARD:
            # 第二次转回后，先按 TF 向前走一段固定距离。
            self.post_hit_pre_final_forward_start_pose = self.get_current_pose_2d()
            self.post_hit_final_forward_start_pose = self.post_hit_pre_final_forward_start_pose
            if self.post_hit_pre_final_forward_start_pose is None:
                self.get_logger().warn('[POST_HIT_PRE_FINAL_FORWARD] TF unavailable at state enter')

        if new_state == self.POST_HIT_FINAL_FORWARD:
            # 固定距离前进完成后，再进入 RGB 横向黄线识别和朝向修正。
            self.final_yellow_done_counter = 0
            self.latest_final_yellow_line = None

        if new_state == self.FINAL_LEFT_JUMP:
            # 障碍物流程内部的最终左跳是阻塞动作，这里只负责进入状态；动作在 control_loop 中执行。
            self.send_motion_cmd(0.0, 0.0, 0.0)

        if new_state == self.GLOBAL_FINAL_RIGHT_JUMP:
            # 全部流程完成后的最终右跳，动作在 control_loop 中执行。
            self.send_motion_cmd(0.0, 0.0, 0.0)

        if new_state == self.GLOBAL_FINAL_YELLOW_FORWARD:
            # 右跳后，开始识别前方横向黄线并修正朝向。
            self.global_final_yellow_done_counter = 0
            self.latest_final_yellow_line = None

        if new_state == self.GLOBAL_FINAL_LEFT_JUMP:
            # 最终左跳一次，动作在 control_loop 中执行。
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

    def get_dashed_target_x(self) -> float:
        img_center_x = self.latest_bgr.shape[1] / 2.0
        if self.dashed_side == 'left':
            return img_center_x + self.dashed_target_offset_px
        if self.dashed_side == 'right':
            return img_center_x - self.dashed_target_offset_px
        return img_center_x

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
    def control_loop(self):
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
            # 启动预左移：只按 TF 横移固定距离，不做任何目标识别。
            # 完成后再进入 GLOBAL_LATERAL_SEARCH，开始按剩余任务数检测限高杆/障碍物。
            if self.global_initial_lateral_shift_distance_m <= 0.0:
                self.get_logger().info('[GLOBAL_INIT_SHIFT] distance <= 0, skip initial shift')
                self.enter_state(self.GLOBAL_LATERAL_SEARCH)
                return

            if self.global_initial_lateral_shift_start_pose is None:
                self.global_initial_lateral_shift_start_pose = self.get_current_pose_2d()
                if self.global_initial_lateral_shift_start_pose is None:
                    self.send_motion_cmd(0.0, 0.0, 0.0)
                    self.get_logger().warn('[GLOBAL_INIT_SHIFT] waiting TF start pose', throttle_duration_sec=0.5)
                    return

            dist = self.distance_from_pose(self.global_initial_lateral_shift_start_pose)

            if dist is None:
                self.send_motion_cmd(0.0, 0.0, 0.0)
                self.get_logger().warn('[GLOBAL_INIT_SHIFT] current TF unavailable', throttle_duration_sec=0.5)
                return

            if dist >= self.global_initial_lateral_shift_distance_m:
                self.get_logger().info(
                    f'[GLOBAL_INIT_SHIFT] finished: dist={dist:.3f}/{self.global_initial_lateral_shift_distance_m:.3f}, '
                    'enter GLOBAL_LATERAL_SEARCH'
                )
                self.enter_state(self.GLOBAL_LATERAL_SEARCH)
                return

            self.send_motion_cmd(0.0, self.global_initial_lateral_shift_vy, 0.0)
            self.get_logger().info(
                f'[GLOBAL_INIT_SHIFT] shifting left: dist={dist:.3f}/{self.global_initial_lateral_shift_distance_m:.3f}, '
                f'vy={self.global_initial_lateral_shift_vy:.3f}',
                throttle_duration_sec=0.3
            )
            return

        if self.state == self.GLOBAL_LATERAL_SEARCH:
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
                self.get_logger().info('[GLOBAL_CENTER_BAR] bar lost, continue lateral search motion', throttle_duration_sec=0.5)
            else:
                vy = self.compute_bar_align_vy(bar)
                self.send_motion_cmd(0.0, vy, 0.0)
                if self.is_bar_centered(bar):
                    self.global_center_stable_count += 1
                else:
                    self.global_center_stable_count = 0
                self.get_logger().info(
                    f'[GLOBAL_CENTER_BAR] center={bar.center_img}, vy={vy:.3f}, '
                    f'stable={self.global_center_stable_count}/{self.global_center_stable_frames}',
                    throttle_duration_sec=0.2
                )
                if self.global_center_stable_count >= self.global_center_stable_frames:
                    d = self.estimate_bar_depth(bar)
                    self.bar_return_target_depth_m = d
                    self.current_bar_det = bar
                    self.get_logger().info(f'[GLOBAL_CENTER_BAR] centered, record bar depth={d}, enter BAR_FORWARD_UNDER')
                    self.enter_state(self.BAR_FORWARD_UNDER)
                    return

        elif self.state == self.GLOBAL_CENTER_OBSTACLE:
            if not obstacle_candidates:
                self.global_center_stable_count = 0
                self.send_motion_cmd(0.0, self.global_lateral_search_vy, 0.0)
                self.get_logger().info('[GLOBAL_CENTER_OBS] obstacle lost, continue lateral search motion', throttle_duration_sec=0.5)
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
                    self.enter_state(self.APPROACH_OBSTACLES)
                    return

        elif self.state == self.GLOBAL_SHIFT_AFTER_SUBTASK:
            if self.all_global_tasks_done():
                self.enter_global_final_sequence()
                return
            if self.global_after_task_shift_start_pose is None:
                self.global_after_task_shift_start_pose = self.get_current_pose_2d()
                if self.global_after_task_shift_start_pose is None:
                    self.send_motion_cmd(0.0, 0.0, 0.0)
                    self.get_logger().warn('[GLOBAL_SHIFT] waiting TF start pose', throttle_duration_sec=0.5)
                    return
            dist = self.distance_from_pose(self.global_after_task_shift_start_pose)
            target_dist = self.current_after_task_shift_distance_m if self.current_after_task_shift_distance_m > 0.0 else self.global_after_task_shift_distance_m
            if dist is None:
                self.send_motion_cmd(0.0, 0.0, 0.0)
                self.get_logger().warn('[GLOBAL_SHIFT] current TF unavailable', throttle_duration_sec=0.5)
            elif dist >= target_dist:
                self.get_logger().info(
                    f'[GLOBAL_SHIFT] finished: dist={dist:.3f}/{target_dist:.3f}, '
                    f'reason={self.global_after_task_shift_reason}'
                )
                self.enter_state(self.GLOBAL_LATERAL_SEARCH)
                return
            else:
                self.send_motion_cmd(0.0, self.global_after_task_shift_vy, 0.0)
                self.get_logger().info(
                    f'[GLOBAL_SHIFT] shifting left: dist={dist:.3f}/{target_dist:.3f}, '
                    f'reason={self.global_after_task_shift_reason}',
                    throttle_duration_sec=0.3
                )

        elif self.state == self.BAR_FORWARD_UNDER:
            bar = self.bar_detector.detect(frame)
            bar_for_vis = bar
            vy = self.compute_bar_align_vy(bar) if bar is not None else 0.0
            self.send_motion_cmd(self.bar_search_forward_speed, vy, 0.0)
            if bar is not None:
                d = self.estimate_bar_depth(bar)
                if d is not None and self.bar_return_target_depth_m is None:
                    self.bar_return_target_depth_m = d
                self.get_logger().info(
                    f'[BAR_FORWARD] depth={d}, target_depth={self.bar_return_target_depth_m}, vy={vy:.3f}',
                    throttle_duration_sec=0.2
                )
                if d is not None and d < self.bar_trigger_distance_m:
                    # 限高杆：靠近到规定距离、准备开始搜索目标物体时播报
                    self.speak_bar_at_trigger()
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
            params = self.hit_params.get(self.locked_target.det_type, {'speed': 0.20, 'distance': 0.25})
            self.send_motion_cmd(params['speed'], 0.0, 0.0)
            if self.bar_hit_start_pose is None:
                self.bar_hit_start_pose = self.get_current_pose_2d()
            moved = self.distance_from_pose(self.bar_hit_start_pose)
            if moved is not None:
                self.get_logger().info(
                    f'[BAR_HIT] target={self.locked_target.det_type}, moved={moved:.3f}/{params["distance"]:.3f}',
                    throttle_duration_sec=0.2
                )
                if moved >= params['distance']:
                    self.enter_state(self.BAR_BACKOFF_TO_BAR)
                    return
            if now - self.state_enter_time >= self.hit_timeout_s:
                self.get_logger().warn('[BAR_HIT] timeout, enter BAR_BACKOFF_TO_BAR')
                self.enter_state(self.BAR_BACKOFF_TO_BAR)
                return

        elif self.state == self.BAR_BACKOFF_TO_BAR:
            bar = self.bar_detector.detect(frame)
            bar_for_vis = bar
            vy = self.compute_bar_align_vy(bar) if bar is not None else 0.0
            self.send_motion_cmd(-self.backoff_after_hit_speed, vy, 0.0)
            if bar is None:
                self.get_logger().info('[BAR_BACKOFF] bar=None, keep backing', throttle_duration_sec=0.5)
            else:
                d = self.estimate_bar_depth(bar)
                target_d = self.bar_return_target_depth_m
                if d is not None and target_d is not None:
                    depth_err = d - target_d
                    self.get_logger().info(
                        f'[BAR_BACKOFF] depth={d:.3f}, target={target_d:.3f}, err={depth_err:.3f}, vy={vy:.3f}',
                        throttle_duration_sec=0.2
                    )
                    if (now - self.state_enter_time >= self.backoff_min_time_s and
                            abs(depth_err) <= self.backoff_bar_depth_tolerance_m):
                        self.send_motion_cmd(0.0, 0.0, 0.0)
                        self.finish_bar_flow()
                        return
                elif d is None:
                    self.get_logger().info('[BAR_BACKOFF] bar detected but depth=None, keep backing', throttle_duration_sec=0.5)
                else:
                    self.get_logger().warn('[BAR_BACKOFF] return target depth is None, keep backing', throttle_duration_sec=1.0)

        elif self.state == self.APPROACH_OBSTACLES:
            pair = self.choose_obstacle_pair(obstacle_candidates)
            chosen_pair = pair

            if pair is None:
                # 没有稳定看到两个障碍物，先慢速前进搜索
                self.send_motion_cmd(self.obstacle_search_forward_speed, 0.0, 0.0)
            else:
                left, right = pair

                vy = self.compute_obstacle_mid_align_vy(left, right)

                d_left = left.extra.get('median_depth')
                d_right = right.extra.get('median_depth')

                depths = [d for d in [d_left, d_right] if d is not None]
                obstacle_dist = min(depths) if depths else None

                self.send_motion_cmd(self.obstacle_forward_speed, vy, 0.0)

                if obstacle_dist is not None:
                    self.get_logger().info(
                        f'[OBS_ALIGN] pair_center=({left.center_img[0]},{right.center_img[0]}), '
                        f'dist={obstacle_dist:.3f}, vy={vy:.3f}',
                        throttle_duration_sec=0.2
                    )

                    if obstacle_dist <= self.obstacle_trigger_distance_m:
                        # 无法跨越障碍：靠近到规定距离、准备绕障前播报
                        self.speak_obstacle_at_trigger()
                        self.get_logger().info(
                            f'[OBS_ALIGN] obstacle_dist={obstacle_dist:.3f} <= '
                            f'{self.obstacle_trigger_distance_m:.3f}, switch to dashed align'
                        )

                        # 新一轮黄线逻辑开始，先清空方向记录
                        self.dashed_side = None
                        self.dashed_pre_shift_start_time = None

                        self.enter_state(self.ALIGN_DASHED_LINE)

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
                    self.dashed_side = self.get_dashed_side(dashed)
                    self.get_logger().info(
                        f'[DASH_ALIGN] first dashed side={self.dashed_side}, '
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

            if self.dashed_pre_shift_start_pose is None:
                self.dashed_pre_shift_start_pose = self.get_current_pose_2d()
                if self.dashed_pre_shift_start_pose is None:
                    vy = self.get_pre_shift_vy()
                    self.send_motion_cmd(0.0, vy, 0.0)
                    self.get_logger().warn(
                        f'[DASH_PRE_SHIFT] start TF unavailable, keep shifting vy={vy:.3f}',
                        throttle_duration_sec=0.5
                    )
                    return

            current_pose = self.get_current_pose_2d()
            if current_pose is None:
                vy = self.get_pre_shift_vy()
                self.send_motion_cmd(0.0, vy, 0.0)
                self.get_logger().warn(
                    f'[DASH_PRE_SHIFT] current TF unavailable, keep shifting vy={vy:.3f}',
                    throttle_duration_sec=0.5
                )
                return

            lateral = self.get_local_lateral_displacement_from_start(
                self.dashed_pre_shift_start_pose,
                current_pose
            )
            target_abs = abs(self.dashed_pre_shift_distance_m)
            moved_along_target = lateral * self.dashed_pre_shift_dir_sign
            target_signed = self.dashed_pre_shift_dir_sign * target_abs

            if moved_along_target >= target_abs:
                self.send_motion_cmd(0.0, 0.0, 0.0)
                self.get_logger().info(
                    f'[DASH_PRE_SHIFT] done by TF: side={self.dashed_side}, '
                    f'lateral={lateral:.3f}, target={target_signed:.3f}, '
                    f'go ALIGN_DASHED_LINE'
                )
                self.enter_state(self.ALIGN_DASHED_LINE)
                return

            vy = self.get_pre_shift_vy()
            self.send_motion_cmd(0.0, vy, 0.0)

            self.get_logger().info(
                f'[DASH_PRE_SHIFT] moving by TF: side={self.dashed_side}, '
                f'lateral={lateral:.3f}, target={target_signed:.3f}, '
                f'moved={moved_along_target:.3f}/{target_abs:.3f}, vy={vy:.3f}',
                throttle_duration_sec=0.2
            )

        elif self.state == self.FOLLOW_DASHED_UNTIL_LOST:
            if dashed is None:
                self.dashed_lost_count += 1

                if self.dashed_lost_count >= self.dashed_lost_stop_frames:
                    self.get_logger().info(
                        f'[FOLLOW_DASH] dashed lost {self.dashed_lost_count} frames, go post dash forward'
                    )
                    self.enter_state(self.POST_DASH_FORWARD)
                else:
                    # 防止单帧漏检，短暂继续向前
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

                self.get_logger().info(
                    f'[FOLLOW_DASH] center={dashed.center_img}, '
                    f'cmd=({self.follow_forward_speed:.3f},{vy:.3f},0.000), '
                    f'vy_limit=[{self.follow_align_vy_min:.3f},{self.follow_align_vy_max:.3f}]',
                    throttle_duration_sec=0.2
                )

        elif self.state == self.POST_DASH_FORWARD:
            dist = self.distance_from_pose(self.post_forward_start_pose)

            if dist is None:
                self.send_motion_cmd(self.post_dash_forward_speed, 0.0, 0.0)
                self.get_logger().warn(
                    '[POST_DASH_FORWARD] TF distance unavailable, keep moving forward',
                    throttle_duration_sec=0.5
                )
            elif dist < self.post_dash_forward_distance_m:
                self.send_motion_cmd(self.post_dash_forward_speed, 0.0, 0.0)
                self.get_logger().info(
                    f'[POST_DASH_FORWARD] dist={dist:.3f}/{self.post_dash_forward_distance_m:.3f}',
                    throttle_duration_sec=0.2
                )
            else:
                self.get_logger().info(
                    f'[POST_DASH_FORWARD] finished dist={dist:.3f}, go first turn'
                )
                self.enter_state(self.POST_DASH_TURN_1)

        elif self.state == self.POST_DASH_TURN_1:
            if self.turn_finished_by_tf():
                self.send_motion_cmd(0.0, 0.0, 0.0)
                self.get_logger().info('[POST_DASH_TURN_1] finished, go forward')
                self.enter_state(self.POST_TURN_FORWARD)
            else:
                wz = self.current_turn_dir * abs(self.current_turn_wz)
                self.send_motion_cmd(0.0, 0.0, wz)

        elif self.state == self.POST_TURN_FORWARD:
            dist = self.distance_from_pose(self.post_turn_forward_start_pose)

            if dist is None:
                self.send_motion_cmd(self.post_turn_forward_speed, 0.0, 0.0)
                self.get_logger().warn(
                    '[POST_TURN_FORWARD] TF distance unavailable, keep moving forward',
                    throttle_duration_sec=0.5
                )
            elif dist < self.post_turn_forward_distance_m:
                self.send_motion_cmd(self.post_turn_forward_speed, 0.0, 0.0)
                self.get_logger().info(
                    f'[POST_TURN_FORWARD] dist={dist:.3f}/{self.post_turn_forward_distance_m:.3f}',
                    throttle_duration_sec=0.2
                )
            else:
                self.get_logger().info(
                    f'[POST_TURN_FORWARD] finished dist={dist:.3f}, go second turn'
                )
                self.enter_state(self.POST_DASH_TURN_2)

        elif self.state == self.POST_DASH_TURN_2:
            if self.turn_finished_by_tf():
                self.send_motion_cmd(0.0, 0.0, 0.0)
                self.get_logger().info('[POST_DASH_TURN_2] finished, start target search')
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
                    {'speed': 0.20, 'distance': 0.25}
                )

                self.send_motion_cmd(params['speed'], 0.0, 0.0)

                if self.hit_start_pose is None:
                    self.hit_start_pose = self.get_current_pose_2d()

                moved = self.distance_from_pose(self.hit_start_pose)

                if moved is not None:
                    self.get_logger().info(
                        f'[HIT] target={self.locked_target.det_type}, '
                        f'moved={moved:.3f}/{params["distance"]:.3f}',
                        throttle_duration_sec=0.2
                    )

                    if moved >= params['distance']:
                        self.get_logger().info('[HIT] finished, go backoff after hit')
                        self.enter_state(self.HIT_BACKOFF_AFTER_HIT)

                if now - self.state_enter_time >= self.hit_timeout_s:
                    self.get_logger().warn('[HIT] timeout reached, go backoff after hit')
                    self.enter_state(self.HIT_BACKOFF_AFTER_HIT)

        elif self.state == self.HIT_BACKOFF_AFTER_HIT:
            if self.after_hit_backoff_start_pose is None:
                self.after_hit_backoff_start_pose = self.get_current_pose_2d()
                if self.after_hit_backoff_start_pose is None:
                    self.send_motion_cmd(0.0, 0.0, 0.0)
                    self.get_logger().warn(
                        '[AFTER_HIT_BACKOFF] waiting TF start pose, stop temporarily',
                        throttle_duration_sec=0.5
                    )
                    return

            dist = self.distance_from_pose(self.after_hit_backoff_start_pose)

            if dist is None:
                self.send_motion_cmd(0.0, 0.0, 0.0)
                self.get_logger().warn(
                    '[AFTER_HIT_BACKOFF] current TF unavailable, stop temporarily',
                    throttle_duration_sec=0.5
                )
            elif dist < self.after_hit_backoff_distance_m:
                self.send_motion_cmd(-self.after_hit_backoff_speed, 0.0, 0.0)
                self.get_logger().info(
                    f'[AFTER_HIT_BACKOFF] dist={dist:.3f}/{self.after_hit_backoff_distance_m:.3f}',
                    throttle_duration_sec=0.2
                )
            else:
                self.get_logger().info(
                    f'[AFTER_HIT_BACKOFF] finished dist={dist:.3f}, go two left jumps'
                )
                self.enter_state(self.POST_HIT_LEFT_JUMP)

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
                    self.send_motion_cmd(0.0, 0.0, 0.0)
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
                self.send_motion_cmd(0.0, 0.0, 0.0)
                self.enter_state(self.POST_HIT_OBS_TURN_1)
                return

        elif self.state == self.POST_HIT_OBS_TURN_1:
            if self.turn_start_yaw is None:
                self.turn_start_yaw = self.get_current_yaw()
                if self.turn_start_yaw is None:
                    self.send_motion_cmd(0.0, 0.0, 0.0)
                    self.get_logger().warn('[POST_HIT_OBS_TURN_1] waiting TF yaw', throttle_duration_sec=0.5)
                    return

            if self.turn_finished_by_tf():
                self.send_motion_cmd(0.0, 0.0, 0.0)
                self.get_logger().info('[POST_HIT_OBS_TURN_1] finished, go forward')
                self.enter_state(self.POST_HIT_OBS_FORWARD)
            else:
                wz = self.current_turn_dir * abs(self.current_turn_wz)
                self.send_motion_cmd(0.0, 0.0, wz)

        elif self.state == self.POST_HIT_OBS_FORWARD:
            if self.post_hit_obs_forward_start_pose is None:
                self.post_hit_obs_forward_start_pose = self.get_current_pose_2d()
                if self.post_hit_obs_forward_start_pose is None:
                    self.send_motion_cmd(0.0, 0.0, 0.0)
                    self.get_logger().warn('[POST_HIT_OBS_FORWARD] waiting TF start pose', throttle_duration_sec=0.5)
                    return

            dist = self.distance_from_pose(self.post_hit_obs_forward_start_pose)

            if dist is None:
                self.send_motion_cmd(0.0, 0.0, 0.0)
                self.get_logger().warn('[POST_HIT_OBS_FORWARD] current TF unavailable, stop temporarily', throttle_duration_sec=0.5)
            elif dist < self.post_hit_obs_forward_distance_m:
                self.send_motion_cmd(self.post_hit_obs_forward_speed, 0.0, 0.0)
                self.get_logger().info(
                    f'[POST_HIT_OBS_FORWARD] dist={dist:.3f}/{self.post_hit_obs_forward_distance_m:.3f}',
                    throttle_duration_sec=0.2
                )
            else:
                self.get_logger().info('[POST_HIT_OBS_FORWARD] finished, go opposite turn')
                self.enter_state(self.POST_HIT_OBS_TURN_2)

        elif self.state == self.POST_HIT_OBS_TURN_2:
            if self.turn_start_yaw is None:
                self.turn_start_yaw = self.get_current_yaw()
                if self.turn_start_yaw is None:
                    self.send_motion_cmd(0.0, 0.0, 0.0)
                    self.get_logger().warn('[POST_HIT_OBS_TURN_2] waiting TF yaw', throttle_duration_sec=0.5)
                    return

            if self.turn_finished_by_tf():
                self.send_motion_cmd(0.0, 0.0, 0.0)
                self.get_logger().info('[POST_HIT_OBS_TURN_2] finished, go pre-final fixed forward')
                self.enter_state(self.POST_HIT_PRE_FINAL_FORWARD)
            else:
                wz = self.current_turn_dir * abs(self.current_turn_wz)
                self.send_motion_cmd(0.0, 0.0, wz)

        elif self.state == self.POST_HIT_PRE_FINAL_FORWARD:
            # After the second turn, move forward a fixed TF distance first,
            # then start RGB horizontal yellow-line detection/alignment.
            if self.post_hit_pre_final_forward_start_pose is None:
                self.post_hit_pre_final_forward_start_pose = self.get_current_pose_2d()
                self.post_hit_final_forward_start_pose = self.post_hit_pre_final_forward_start_pose
                if self.post_hit_pre_final_forward_start_pose is None:
                    self.send_motion_cmd(0.0, 0.0, 0.0)
                    self.get_logger().warn('[POST_HIT_PRE_FINAL_FORWARD] waiting TF start pose', throttle_duration_sec=0.5)
                    return

            dist = self.distance_from_pose(self.post_hit_pre_final_forward_start_pose)

            if dist is None:
                self.send_motion_cmd(0.0, 0.0, 0.0)
                self.get_logger().warn('[POST_HIT_PRE_FINAL_FORWARD] current TF unavailable, stop temporarily', throttle_duration_sec=0.5)
            elif dist < self.post_hit_final_forward_distance_m:
                self.send_motion_cmd(self.post_hit_final_forward_speed, 0.0, 0.0)
                self.get_logger().info(
                    f'[POST_HIT_PRE_FINAL_FORWARD] dist={dist:.3f}/{self.post_hit_final_forward_distance_m:.3f}',
                    throttle_duration_sec=0.2
                )
            else:
                self.send_motion_cmd(0.0, 0.0, 0.0)
                self.get_logger().info('[POST_HIT_PRE_FINAL_FORWARD] fixed forward finished, start final yellow detection')
                self.enter_state(self.POST_HIT_FINAL_FORWARD)

        elif self.state == self.POST_HIT_FINAL_FORWARD:
            # Final stage: RGB-only horizontal yellow-line approach.
            # It no longer uses TF fixed forward distance or depth distance.
            # Stop condition follows the previous second-stage idea:
            # line_bottom_y >= image_height * final_yellow_stop_line_y_ratio.
            # At the same time, use the line tilt angle to correct yaw.
            final_yellow_line = self.final_yellow_detector.detect(frame)
            self.latest_final_yellow_line = final_yellow_line

            if final_yellow_line is None:
                self.final_yellow_done_counter = 0
                self.send_motion_cmd(self.post_hit_final_forward_speed, 0.0, 0.0)
                self.get_logger().info(
                    '[POST_HIT_FINAL_FORWARD] no horizontal yellow line, keep moving forward',
                    throttle_duration_sec=0.3
                )
            else:
                bottom_y = int(final_yellow_line.extra.get('bottom_y', 0))
                bottom_ratio = float(final_yellow_line.extra.get('bottom_ratio', 0.0))
                angle_deg = float(final_yellow_line.extra.get('angle_deg', 0.0))
                abs_tilt_deg = abs(angle_deg)
                wz = self.compute_final_yellow_wz(final_yellow_line)

                reached_line = bottom_ratio >= self.final_yellow_stop_line_y_ratio
                angle_ok = abs_tilt_deg <= self.final_yellow_done_tilt_deg

                if reached_line and angle_ok:
                    self.final_yellow_done_counter += 1
                    self.send_motion_cmd(0.0, 0.0, 0.0)
                    self.get_logger().info(
                        f'[POST_HIT_FINAL_FORWARD] yellow reached and aligned: '
                        f'bottom={bottom_y}, ratio={bottom_ratio:.3f}/{self.final_yellow_stop_line_y_ratio:.3f}, '
                        f'angle={angle_deg:.1f}deg, '
                        f'counter={self.final_yellow_done_counter}/{self.final_yellow_confirm_count}',
                        throttle_duration_sec=0.2
                    )
                    if self.final_yellow_done_counter >= self.final_yellow_confirm_count:
                        self.get_logger().info('[POST_HIT_FINAL_FORWARD] final yellow reached and aligned, go final left jumps')
                        self.enter_state(self.FINAL_LEFT_JUMP)

                elif reached_line and not angle_ok:
                    # Already close according to RGB bottom position, but yaw is not aligned.
                    # Rotate in place to avoid pushing across the line.
                    self.final_yellow_done_counter = 0
                    self.send_motion_cmd(0.0, 0.0, wz)
                    self.get_logger().info(
                        f'[POST_HIT_FINAL_FORWARD] reached yellow but not aligned, align in place: '
                        f'bottom={bottom_y}, ratio={bottom_ratio:.3f}/{self.final_yellow_stop_line_y_ratio:.3f}, '
                        f'angle={angle_deg:.1f}deg, wz={wz:.3f}',
                        throttle_duration_sec=0.2
                    )

                else:
                    # Not close enough yet: keep moving forward and correct yaw by line tilt.
                    self.final_yellow_done_counter = 0
                    self.send_motion_cmd(self.post_hit_final_forward_speed, 0.0, wz)
                    self.get_logger().info(
                        f'[POST_HIT_FINAL_FORWARD] approach yellow by RGB: '
                        f'bottom={bottom_y}, ratio={bottom_ratio:.3f}/{self.final_yellow_stop_line_y_ratio:.3f}, '
                        f'angle={angle_deg:.1f}deg, wz={wz:.3f}',
                        throttle_duration_sec=0.2
                    )
        elif self.state == self.FINAL_LEFT_JUMP:
            # 障碍物流程内部的最终左跳。
            # 普通情况：左跳两次 -> OBSTACLE_FLOW_DONE -> 全局收尾右跳 -> 全局黄线前进。
            # 特殊情况：如果障碍物是全局第 3 个识别/处理的物体，
            # 说明 2 次限高杆已经完成，障碍物流程完成后不需要再执行全局右跳，
            # 这里改为：左跳一次 -> 直接进入 GLOBAL_FINAL_YELLOW_FORWARD。
            if self.obstacle_flow_is_third_object:
                if self.completed_obstacle_count < self.required_obstacle_count:
                    self.completed_obstacle_count += 1

                self.get_logger().info(
                    f'[FINAL_LEFT_JUMP] obstacle is the 3rd flow, '
                    f'execute one left jump, then jump directly to GLOBAL_FINAL_YELLOW_FORWARD. '
                    f'bar={self.completed_bar_count}/{self.required_bar_count}, '
                    f'obstacle={self.completed_obstacle_count}/{self.required_obstacle_count}'
                )
                self.execute_left_jump_turn(1, self.GLOBAL_FINAL_YELLOW_FORWARD)
                return

            # execute_left_jump_turn 内部会连续调用 send_left_jump_action_once()，
            # 每次左跳 mode=16/gait_id=0，之后接 Recovery stand mode=12/gait_id=0。
            self.get_logger().info('[FINAL_LEFT_JUMP] execute two left jumps, then obstacle flow done')
            self.execute_left_jump_turn(2, self.OBSTACLE_FLOW_DONE)

        elif self.state == self.OBSTACLE_FLOW_DONE:
            self.finish_obstacle_flow()
            return

        elif self.state == self.GLOBAL_FINAL_RIGHT_JUMP:
            # 全部任务完成后的第一步：右跳一次。
            self.get_logger().info('[GLOBAL_FINAL_RIGHT_JUMP] execute one right jump, then start final yellow alignment')
            self.execute_right_jump_turn(1, self.GLOBAL_FINAL_YELLOW_FORWARD)
            return

        elif self.state == self.GLOBAL_FINAL_YELLOW_FORWARD:
            # 右跳后继续前进，同时识别前方横向黄线并用倾斜角修正朝向。
            # 到达图像下方阈值后停止，再执行最后一次左跳。
            final_yellow_line = self.final_yellow_detector.detect(frame)
            self.latest_final_yellow_line = final_yellow_line

            if final_yellow_line is None:
                self.global_final_yellow_done_counter = 0
                self.send_motion_cmd(self.global_final_yellow_forward_speed, 0.0, 0.0)
                self.get_logger().info(
                    '[GLOBAL_FINAL_YELLOW] no horizontal yellow line, keep moving forward',
                    throttle_duration_sec=0.3
                )
            else:
                bottom_y = int(final_yellow_line.extra.get('bottom_y', 0))
                bottom_ratio = float(final_yellow_line.extra.get('bottom_ratio', 0.0))
                angle_deg = float(final_yellow_line.extra.get('angle_deg', 0.0))
                wz = self.compute_final_yellow_wz(final_yellow_line)
                reached_line = bottom_ratio >= self.global_final_yellow_stop_line_y_ratio

                if reached_line:
                    self.global_final_yellow_done_counter += 1
                    self.send_motion_cmd(0.0, 0.0, 0.0)
                    self.get_logger().info(
                        f'[GLOBAL_FINAL_YELLOW] yellow reached lower area: '
                        f'bottom={bottom_y}, ratio={bottom_ratio:.3f}/{self.global_final_yellow_stop_line_y_ratio:.3f}, '
                        f'angle={angle_deg:.1f}deg, '
                        f'counter={self.global_final_yellow_done_counter}/{self.global_final_yellow_confirm_count}',
                        throttle_duration_sec=0.2
                    )
                    if self.global_final_yellow_done_counter >= self.global_final_yellow_confirm_count:
                        self.get_logger().info('[GLOBAL_FINAL_YELLOW] stop and execute one final left jump')
                        self.enter_state(self.GLOBAL_FINAL_LEFT_JUMP)
                        return
                else:
                    self.global_final_yellow_done_counter = 0
                    self.send_motion_cmd(self.global_final_yellow_forward_speed, 0.0, wz)
                    self.get_logger().info(
                        f'[GLOBAL_FINAL_YELLOW] approach and align: '
                        f'bottom={bottom_y}, ratio={bottom_ratio:.3f}/{self.global_final_yellow_stop_line_y_ratio:.3f}, '
                        f'angle={angle_deg:.1f}deg, wz={wz:.3f}',
                        throttle_duration_sec=0.2
                    )
            return

        elif self.state == self.GLOBAL_FINAL_LEFT_JUMP:
            # 黄线到达图像下方阈值后，最后执行一次左跳，然后 DONE。
            self.get_logger().info('[GLOBAL_FINAL_LEFT_JUMP] execute one left jump, then DONE')
            self.execute_left_jump_turn(1, self.DONE)
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

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
            cv2.line(vis, (cx, 0), (cx, h - 1), (0, 0, 255), 1)
            cv2.putText(
                vis,
                f'BAR {self.completed_bar_count}/{self.required_bar_count} aspect={aspect:.1f}',
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )

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

        if self.state in [self.POST_DASH_TURN_1, self.POST_DASH_TURN_2, self.POST_HIT_OBS_TURN_1, self.POST_HIT_OBS_TURN_2]:
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


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDashedTaskNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            node.Ctrl.quit()
        except Exception:
            pass

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()