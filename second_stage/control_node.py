#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from second_stage.my_gait import Robot_Ctrl
from second_stage.robot_control_cmd_lcmt import robot_control_cmd_lcmt


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class MultiStageOrangeYellowTaskNode(Node):
    def __init__(self):
        super().__init__('multi_stage_orange_yellow_task_node')

        # =========================
        # 话题与 TF
        # =========================
        self.declare_parameter('rgb_topic', '/rgb_camera/rgb_camera/image_raw')
        self.declare_parameter('depth_topic', '/d435/depth/d435_depth/depth/image_raw')
        self.declare_parameter('global_frame', 'vodom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('control_hz', 30.0)

        # 测试用：初始状态
        self.declare_parameter('initial_state', 'STAGE1_CRUISE_BALL_AND_YELLOW')

        # =========================
        # 橙色球检测参数
        # =========================
        self.declare_parameter('orange_h_min', 5)
        self.declare_parameter('orange_h_max', 25)
        self.declare_parameter('orange_s_min', 100)
        self.declare_parameter('orange_s_max', 255)
        self.declare_parameter('orange_v_min', 80)
        self.declare_parameter('orange_v_max', 255)
        self.declare_parameter('orange_min_contour_area', 60.0)
        self.declare_parameter('prefer_nearest_ball', True)
        self.declare_parameter('min_ball_radius_to_trigger', 29.0)

        # =========================
        # depth 搜索（橙球距离用）
        # =========================
        self.declare_parameter('depth_search_half', 12)
        self.declare_parameter('valid_min_depth_m', 0.05)
        self.declare_parameter('valid_max_depth_m', 10.0)

        # =========================
        # 黄线检测参数
        # =========================
        self.declare_parameter('yellow_roi_top_ratio', 0.45)
        self.declare_parameter('yellow_roi_left_ratio', 0.35)
        self.declare_parameter('yellow_roi_right_ratio', 0.65)

        self.declare_parameter('yellow_h_min', 15)
        self.declare_parameter('yellow_h_max', 40)
        self.declare_parameter('yellow_s_min', 80)
        self.declare_parameter('yellow_s_max', 255)
        self.declare_parameter('yellow_v_min', 80)
        self.declare_parameter('yellow_v_max', 255)
        self.declare_parameter('yellow_min_contour_area', 600.0)

        # 只接受“前方横线”的几何约束
        self.declare_parameter('yellow_min_width_height_ratio', 2.0)
        self.declare_parameter('yellow_max_tilt_deg', 20.0)
        self.declare_parameter('yellow_center_tolerance_ratio', 0.28)
        self.declare_parameter('yellow_min_width_ratio', 0.18)

        self.declare_parameter('yellow_stop_line_y_ratio_stage1', 1.0)
        self.declare_parameter('yellow_stop_line_y_ratio_stage2', 0.77)
        self.declare_parameter('yellow_stop_line_y_ratio_stage3', 0.7)
        self.declare_parameter('yellow_stop_confirm_count', 3)

        # 第三段收尾两条线
        self.declare_parameter('yellow_ratio_scan', 0.6)
        self.declare_parameter('yellow_ratio_final', 0.9)

        # =========================
        # 巡航 / 搜索 / 对准
        # =========================
        self.declare_parameter('cruise_forward_speed', 0.15)
        self.declare_parameter('turn_trigger_distance_m', 0.5)

        self.declare_parameter('align_tolerance_px', 10)
        self.declare_parameter('align_confirm_count', 3)
        self.declare_parameter('align_turn_wz', 0.15)

        # =========================
        # 橙球碰撞模式
        # =========================
        self.declare_parameter('pre_collision_forward_speed', 0.15)
        self.declare_parameter('collision_trigger_distance_m', 0.20)
        self.declare_parameter('collision_forward_speed', 0.15)
        self.declare_parameter('collision_forward_distance_m', 0.25)
        self.declare_parameter('settle_after_collision_sec', 0.0)

        # =========================
        # 返回控制
        # =========================
        self.declare_parameter('backward_speed', 0.15)
        self.declare_parameter('collision_return_tolerance_m', 0.08)
        self.declare_parameter('preturn_return_tolerance_m', 0.08)
        self.declare_parameter('return_yaw_tolerance_rad', 0.03)

        self.declare_parameter('return_lateral_kp', 0.5)
        self.declare_parameter('return_max_vy', 0.06)

        self.declare_parameter('restore_turn_wz', 0.15)
        self.declare_parameter('restore_confirm_count', 3)

        # 返回到 0.2m 深度的确认
        self.declare_parameter('return_depth_tolerance_m', 0.015)
        self.declare_parameter('return_depth_confirm_count', 1)

        # =========================
        # 防重复撞同一颗球
        # =========================
        self.declare_parameter('ball_retrigger_cooldown_sec', 2.0)
        self.declare_parameter('ball_retrigger_min_travel_m', 0.5)

        # =========================
        # 第一段固定右移
        # =========================
        self.declare_parameter('right_speed', 0.05)
        self.declare_parameter('stage1_right_shift_distance_m', 0.01)
        self.declare_parameter('stage1_right_shift_tolerance_m', 0.005)

        # =========================
        # TF 旋转参数
        # =========================
        self.declare_parameter('rotate_left_90_tolerance_rad', 0.02)
        self.declare_parameter('rotate_left_90_confirm_count', 3)
        self.declare_parameter('rotate_left_90_wz', 0.15)

        self.declare_parameter('rotate_back_180_tolerance_rad', 0.02)
        self.declare_parameter('rotate_back_180_confirm_count', 3)
        self.declare_parameter('rotate_back_180_wz', 0.15)

        self.declare_parameter('rotate_left_30_tolerance_rad', 0.02)
        self.declare_parameter('rotate_left_30_confirm_count', 3)
        self.declare_parameter('rotate_left_30_wz', 0.15)

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.global_frame = self.get_parameter('global_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.control_hz = float(self.get_parameter('control_hz').value)
        self.initial_state = self.get_parameter('initial_state').value

        self.orange_h_min = int(self.get_parameter('orange_h_min').value)
        self.orange_h_max = int(self.get_parameter('orange_h_max').value)
        self.orange_s_min = int(self.get_parameter('orange_s_min').value)
        self.orange_s_max = int(self.get_parameter('orange_s_max').value)
        self.orange_v_min = int(self.get_parameter('orange_v_min').value)
        self.orange_v_max = int(self.get_parameter('orange_v_max').value)
        self.orange_min_contour_area = float(self.get_parameter('orange_min_contour_area').value)
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

        self.cruise_forward_speed = float(self.get_parameter('cruise_forward_speed').value)
        self.turn_trigger_distance_m = float(self.get_parameter('turn_trigger_distance_m').value)
        self.align_tolerance_px = int(self.get_parameter('align_tolerance_px').value)
        self.align_confirm_count = int(self.get_parameter('align_confirm_count').value)
        self.align_turn_wz = float(self.get_parameter('align_turn_wz').value)

        self.pre_collision_forward_speed = float(self.get_parameter('pre_collision_forward_speed').value)
        self.collision_trigger_distance_m = float(self.get_parameter('collision_trigger_distance_m').value)
        self.collision_forward_speed = float(self.get_parameter('collision_forward_speed').value)
        self.collision_forward_distance_m = float(self.get_parameter('collision_forward_distance_m').value)
        self.settle_after_collision_sec = float(self.get_parameter('settle_after_collision_sec').value)

        self.backward_speed = float(self.get_parameter('backward_speed').value)
        self.collision_return_tolerance_m = float(self.get_parameter('collision_return_tolerance_m').value)
        self.preturn_return_tolerance_m = float(self.get_parameter('preturn_return_tolerance_m').value)
        self.return_yaw_tolerance_rad = float(self.get_parameter('return_yaw_tolerance_rad').value)
        self.return_lateral_kp = float(self.get_parameter('return_lateral_kp').value)
        self.return_max_vy = float(self.get_parameter('return_max_vy').value)
        self.restore_turn_wz = float(self.get_parameter('restore_turn_wz').value)
        self.restore_confirm_count = int(self.get_parameter('restore_confirm_count').value)

        self.return_depth_tolerance_m = float(self.get_parameter('return_depth_tolerance_m').value)
        self.return_depth_confirm_count = int(self.get_parameter('return_depth_confirm_count').value)

        self.ball_retrigger_cooldown_sec = float(self.get_parameter('ball_retrigger_cooldown_sec').value)
        self.ball_retrigger_min_travel_m = float(self.get_parameter('ball_retrigger_min_travel_m').value)

        self.right_speed = float(self.get_parameter('right_speed').value)
        self.stage1_right_shift_distance_m = float(self.get_parameter('stage1_right_shift_distance_m').value)
        self.stage1_right_shift_tolerance_m = float(self.get_parameter('stage1_right_shift_tolerance_m').value)

        self.rotate_left_90_tolerance_rad = float(self.get_parameter('rotate_left_90_tolerance_rad').value)
        self.rotate_left_90_confirm_count = int(self.get_parameter('rotate_left_90_confirm_count').value)
        self.rotate_left_90_wz = float(self.get_parameter('rotate_left_90_wz').value)

        self.rotate_back_180_tolerance_rad = float(self.get_parameter('rotate_back_180_tolerance_rad').value)
        self.rotate_back_180_confirm_count = int(self.get_parameter('rotate_back_180_confirm_count').value)
        self.rotate_back_180_wz = float(self.get_parameter('rotate_back_180_wz').value)

        self.rotate_left_30_tolerance_rad = float(self.get_parameter('rotate_left_30_tolerance_rad').value)
        self.rotate_left_30_confirm_count = int(self.get_parameter('rotate_left_30_confirm_count').value)
        self.rotate_left_30_wz = float(self.get_parameter('rotate_left_30_wz').value)

        self.Ctrl = Robot_Ctrl()
        self.Ctrl.run()
        self.msg = robot_control_cmd_lcmt()
        if not hasattr(self.msg, 'life_count'):
            self.msg.life_count = 0

        self.bridge = CvBridge()

        self.latest_depth = None
        self.latest_depth_encoding = None

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
        }

        self.latest_yellow_result = {
            'has_line': False,
            'line_bottom_y': None,
            'line_center': None,
            'img_shape': None,
        }

        self.state = self.initial_state
        self.ball_return_state = self.initial_state

        self.align_counter = 0
        self.restore_counter = 0
        self.yellow_stop_counter = 0
        self.rotate_counter = 0

        self.pre_turn_pose: Optional[Tuple[float, float, float]] = None
        self.collision_start_pose: Optional[Tuple[float, float, float]] = None
        self.settle_deadline_sec: Optional[float] = None

        self.align_yaw_delta: Optional[float] = None
        self.restore_start_yaw: Optional[float] = None

        self.align_pose: Optional[Tuple[float, float, float]] = None
        self.pose_at_0p2: Optional[Tuple[float, float, float]] = None
        self.depth_return_counter = 0
        self.segment_start_pose: Optional[Tuple[float, float, float]] = None
        self.align_to_0p2_distance_m: float = 0.0

        self.last_ball_done_time_sec: Optional[float] = None
        self.last_ball_done_pose: Optional[Tuple[float, float, float]] = None

        self.rotation_target_yaw: Optional[float] = None
        self.stage1_right_shift_start_pose: Optional[Tuple[float, float, float]] = None

        self.orange_hit_count = 0

        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.rgb_callback, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_profile_sensor_data)

        self.control_timer = self.create_timer(1.0 / self.control_hz, self.control_loop)

        self.send_stop_command()
        self.Ctrl.Wait_finish(12, 0)

        self.get_logger().info('MultiStageOrangeYellowTaskNode started.')
        self.get_logger().info(f'initial_state={self.state}')
        self.get_logger().info(f'rgb_topic={self.rgb_topic}')
        self.get_logger().info(f'depth_topic={self.depth_topic}')
        self.get_logger().info(f'tf: {self.global_frame} -> {self.base_frame}')

    def _inc_life_count(self):
        self.msg.life_count += 1
        if self.msg.life_count > 127:
            self.msg.life_count = 0

    def send_stop_command(self):
        self.msg.mode = 12
        self.msg.gait_id = 0
        self._inc_life_count()
        self.Ctrl.Send_cmd(self.msg)
        self.get_logger().info('[CMD] STOP', throttle_duration_sec=1.0)

    def send_velocity_command(self, vx: float, vy: float, wz: float):
        self.msg.mode = 11
        self.msg.gait_id = 3
        self._inc_life_count()
        self.msg.vel_des = [vx, vy, wz]
        self.msg.step_height = [0.02, 0.02]
        self.msg.rpy_des = [0.0, 0.0, 0.0]
        self.Ctrl.Send_cmd(self.msg)

    def send_move_right_command(self, speed: float):
        self.msg.mode = 11
        self.msg.gait_id = 3
        self._inc_life_count()
        self.msg.vel_des = [0.0, -speed, 0.0]
        self.msg.step_height = [0.02, 0.02]
        self.msg.rpy_des = [0.0, 0.0, 0.0]
        self.Ctrl.Send_cmd(self.msg)

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
        self.send_stop_command()
        for _ in range(jump_count):
            self.send_left_jump_action_once()
        self.set_state(next_state)

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

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
        if self.last_ball_done_time_sec is None or self.last_ball_done_pose is None:
            return True

        now_sec = self.now_sec()
        dt = now_sec - self.last_ball_done_time_sec

        x, y, _ = current_pose
        x0, y0, _ = self.last_ball_done_pose
        dist = math.hypot(x - x0, y - y0)

        cooldown_ok = dt >= self.ball_retrigger_cooldown_sec
        travel_ok = dist >= self.ball_retrigger_min_travel_m

        self.get_logger().info(
            f'ball retrigger check: dt={dt:.2f}s/{self.ball_retrigger_cooldown_sec:.2f}s, '
            f'dist={dist:.3f}m/{self.ball_retrigger_min_travel_m:.3f}m, '
            f'cooldown_ok={cooldown_ok}, travel_ok={travel_ok}',
            throttle_duration_sec=1.0
        )
        return cooldown_ok and travel_ok

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

        self.latest_ball_result = self.detect_orange_ball_with_depth(frame)
        self.latest_yellow_result = self.detect_yellow_stop_line(frame)

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

    def detect_orange_ball_with_depth(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        self.rgb_w = w
        self.rgb_h = h

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_orange = np.array([self.orange_h_min, self.orange_s_min, self.orange_v_min], dtype=np.uint8)
        upper_orange = np.array([self.orange_h_max, self.orange_s_max, self.orange_v_max], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.orange_min_contour_area:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx = int(cx)
            cy = int(cy)
            radius = float(radius)

            depth_m, depth_center, depth_box = self.get_depth_for_rgb_point(cx, cy)
            if depth_m is None:
                continue

            image_center_x = w // 2
            error_x = cx - image_center_x

            candidates.append({
                'center': (cx, cy),
                'radius': radius,
                'area': float(area),
                'depth_m': depth_m,
                'error_x': int(error_x),
                'depth_center': depth_center,
                'depth_box': depth_box,
            })

        if len(candidates) == 0:
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
            }

        if self.prefer_nearest_ball:
            best = min(candidates, key=lambda c: c['depth_m'])
        else:
            best = min(candidates, key=lambda c: c['depth_m'] + 0.002 * abs(c['error_x']))

        aligned = abs(best['error_x']) <= self.align_tolerance_px

        return {
            'has_ball': True,
            'ball_center': best['center'],
            'ball_radius': best['radius'],
            'ball_depth_m': best['depth_m'],
            'img_shape': (h, w),
            'error_x': best['error_x'],
            'aligned': aligned,
            'depth_center': best['depth_center'],
            'depth_box': best['depth_box'],
        }

    def is_front_horizontal_yellow_line(self, cnt, roi_shape) -> bool:
        roi_h, roi_w = roi_shape[:2]

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

        rect = cv2.minAreaRect(cnt)
        (_, _), (rw, rh), angle = rect

        if rw < rh:
            tilt_deg = abs(angle - 90.0)
        else:
            tilt_deg = abs(angle)

        if tilt_deg > 45.0:
            tilt_deg = abs(90.0 - tilt_deg)

        if tilt_deg > self.yellow_max_tilt_deg:
            return False

        return True

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

        for cnt in contours:
            if not self.is_front_horizontal_yellow_line(cnt, roi.shape):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            score = bw * 1.0 + (y + bh) * 0.5

            if score > best_score:
                best_score = score
                best_contour = cnt

        if best_contour is None:
            return {
                'has_line': False,
                'line_bottom_y': None,
                'line_center': None,
                'img_shape': (h, w),
            }

        x, y, bw, bh = cv2.boundingRect(best_contour)
        line_bottom_y = roi_top + y + bh
        cx = roi_left + x + bw // 2
        cy = roi_top + y + bh // 2

        return {
            'has_line': True,
            'line_bottom_y': int(line_bottom_y),
            'line_center': (int(cx), int(cy)),
            'img_shape': (h, w),
        }

    def set_state(self, new_state: str):
        if new_state != self.state:
            self.get_logger().info(f'STATE: {self.state} -> {new_state}')
            self.state = new_state

            if new_state in (
                'STAGE1_CRUISE_BALL_AND_YELLOW',
                'STAGE2_CRUISE_YELLOW_ONLY',
                'STAGE3_CRUISE_BALL_ONLY',
                'STAGE3_GO_SCAN',
                'STAGE3_GO_FINAL'
            ):
                self.yellow_stop_counter = 0

            if new_state in (
                'STAGE1_ROTATE_LEFT_90',
                'STAGE2_ROTATE_LEFT_90',
                'STAGE3_ROTATE_BACK_180',
                'STAGE3_ROTATE_LEFT_30'
            ):
                self.rotate_counter = 0
                self.rotation_target_yaw = None

            if new_state == 'STAGE1_MOVE_RIGHT_FIXED_DISTANCE':
                self.stage1_right_shift_start_pose = None

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

    def handle_rotation_state(self, current_yaw: float, angle_rad: float, tolerance_rad: float,
                              confirm_count: int, wz_mag: float, next_state: str) -> bool:
        if self.rotation_target_yaw is None:
            self.rotation_target_yaw = wrap_to_pi(current_yaw + angle_rad)
            self.rotate_counter = 0
            self.get_logger().info(
                f'Rotation target set: current_yaw={current_yaw:.3f}, target_yaw={self.rotation_target_yaw:.3f}'
            )

        yaw_err = wrap_to_pi(self.rotation_target_yaw - current_yaw)

        if abs(yaw_err) <= tolerance_rad:
            self.rotate_counter += 1
            self.send_stop_command()

            self.get_logger().info(
                f'Rotation aligned. yaw_err={yaw_err:.3f}, rotate_counter={self.rotate_counter}/{confirm_count}',
                throttle_duration_sec=0.2
            )

            if self.rotate_counter >= confirm_count:
                self.send_stop_command()
                self.set_state(next_state)
                return True
            return True

        self.rotate_counter = 0
        wz = wz_mag if yaw_err > 0 else -wz_mag
        self.send_velocity_command(0.0, 0.0, wz)
        return True

    def handle_ball_subchain(self, x: float, y: float, yaw: float) -> bool:
        ball = self.latest_ball_result

        if self.state == 'BALL_ALIGN':
            if not ball['has_ball'] or ball['ball_depth_m'] is None:
                self.align_counter = 0
                self.send_stop_command()
                return True

            if ball['aligned']:
                self.align_counter += 1
                self.get_logger().info(
                    f"Ball horizontally aligned. align_counter={self.align_counter}/{self.align_confirm_count}",
                    throttle_duration_sec=0.2
                )

                if self.align_counter >= self.align_confirm_count:
                    if self.pre_turn_pose is not None:
                        _, _, yaw_detect = self.pre_turn_pose
                        yaw_aligned = yaw
                        self.align_yaw_delta = wrap_to_pi(yaw_aligned - yaw_detect)
                    else:
                        self.align_yaw_delta = None

                    self.align_pose = (x, y, yaw)
                    self.pose_at_0p2 = None
                    self.align_to_0p2_distance_m = 0.0
                    self.depth_return_counter = 0
                    self.segment_start_pose = None

                    self.get_logger().info(
                        f"Saved align_pose = ({x:.3f}, {y:.3f}, {yaw:.3f})"
                    )

                    self.set_state('BALL_WAIT_COLLISION_TRIGGER')
                    self.send_stop_command()
                    return True

                self.send_stop_command()
                return True

            self.align_counter = 0
            wz = -self.align_turn_wz if ball['error_x'] > 0 else self.align_turn_wz
            self.send_velocity_command(0.0, 0.0, wz)
            return True

        if self.state == 'BALL_WAIT_COLLISION_TRIGGER':
            if not ball['has_ball'] or ball['ball_depth_m'] is None:
                self.send_stop_command()
                return True

            if ball['ball_depth_m'] > self.collision_trigger_distance_m:
                self.send_velocity_command(self.pre_collision_forward_speed, 0.0, 0.0)
                return True

            self.collision_start_pose = (x, y, yaw)
            self.pose_at_0p2 = (x, y, yaw)

            if self.align_pose is not None:
                ax, ay, _ = self.align_pose
                self.align_to_0p2_distance_m = math.hypot(x - ax, y - ay)
            else:
                self.align_to_0p2_distance_m = 0.0

            self.get_logger().info(
                f"Saved pose_at_0p2 = ({x:.3f}, {y:.3f}, {yaw:.3f}) | "
                f"align_to_0p2_distance_m = {self.align_to_0p2_distance_m:.3f}"
            )

            self.set_state('BALL_COLLISION_FORWARD')
            return True

        if self.state == 'BALL_COLLISION_FORWARD':
            if self.collision_start_pose is None:
                self.send_stop_command()
                self.set_state(self.ball_return_state)
                return True

            x0, y0, _ = self.collision_start_pose
            dist = math.hypot(x - x0, y - y0)

            if dist >= self.collision_forward_distance_m:
                self.depth_return_counter = 0
                self.set_state('BALL_RETURN_TO_0P2_BY_DEPTH')
                return True

            self.send_velocity_command(self.collision_forward_speed, 0.0, 0.0)
            return True

        if self.state == 'BALL_RETURN_TO_0P2_BY_DEPTH':
            if not ball['has_ball'] or ball['ball_depth_m'] is None:
                self.send_stop_command()
                return True

            depth_err = ball['ball_depth_m'] - self.collision_trigger_distance_m

            self.get_logger().info(
                f"[BALL_RETURN_TO_0P2_BY_DEPTH] depth={ball['ball_depth_m']:.3f}, "
                f"target={self.collision_trigger_distance_m:.3f}, err={depth_err:.3f}",
                throttle_duration_sec=0.3
            )

            if abs(depth_err) <= self.return_depth_tolerance_m:
                self.depth_return_counter += 1
                self.send_stop_command()

                if self.depth_return_counter >= self.return_depth_confirm_count:
                    self.segment_start_pose = (x, y, yaw)
                    self.get_logger().info(
                        f"Depth returned to 0.2m. segment_start_pose=({x:.3f}, {y:.3f}, {yaw:.3f})"
                    )
                    self.set_state('BALL_BACKWARD_ALIGN_TO_0P2_DISTANCE')
                return True

            self.depth_return_counter = 0
            self.send_velocity_command(-self.backward_speed, 0.0, 0.0)
            return True

        if self.state == 'BALL_BACKWARD_ALIGN_TO_0P2_DISTANCE':
            if self.segment_start_pose is None:
                self.send_stop_command()
                self.set_state('BALL_RESTORE_PRETURN_YAW')
                return True

            x0, y0, _ = self.segment_start_pose
            dist = math.hypot(x - x0, y - y0)

            self.get_logger().info(
                f"[BALL_BACKWARD_ALIGN_TO_0P2_DISTANCE] dist={dist:.3f}/{self.align_to_0p2_distance_m:.3f}",
                throttle_duration_sec=0.3
            )

            if dist >= self.align_to_0p2_distance_m:
                self.send_stop_command()
                self.restore_start_yaw = yaw
                self.get_logger().info(
                    f"Saved restore_start_yaw = {self.restore_start_yaw:.3f} "
                    f"({math.degrees(self.restore_start_yaw):.2f} deg)"
                )
                self.set_state('BALL_RESTORE_PRETURN_YAW')
                return True

            self.send_velocity_command(-self.backward_speed, 0.0, 0.0)
            return True

        if self.state == 'BALL_RESTORE_PRETURN_YAW':
            if self.restore_start_yaw is None:
                self.send_stop_command()
                self.set_state(self.ball_return_state)
                return True

            if self.align_yaw_delta is not None:
                yaw_target = wrap_to_pi(self.restore_start_yaw - self.align_yaw_delta)
            else:
                yaw_target = self.restore_start_yaw

            yaw_err = wrap_to_pi(yaw_target - yaw)
            delta_show = self.align_yaw_delta if self.align_yaw_delta is not None else float('nan')

            self.get_logger().info(
                f"Restore using reversed yaw delta: "
                f"restore_start_yaw={self.restore_start_yaw:.3f}, "
                f"align_yaw_delta={delta_show:.3f}, "
                f"yaw_target={yaw_target:.3f}, "
                f"current_yaw={yaw:.3f}, yaw_err={yaw_err:.3f}",
                throttle_duration_sec=0.5
            )

            if abs(yaw_err) <= self.return_yaw_tolerance_rad:
                self.restore_counter += 1
                self.get_logger().info(
                    f"Yaw restored in this frame. restore_counter={self.restore_counter}/{self.restore_confirm_count}",
                    throttle_duration_sec=0.2
                )

                if self.restore_counter >= self.restore_confirm_count:
                    self.send_stop_command()
                    self.last_ball_done_time_sec = self.now_sec()
                    self.last_ball_done_pose = (x, y, yaw)
                    self.orange_hit_count += 1

                    self.get_logger().info(
                        f'Ball task finished. orange_hit_count={self.orange_hit_count} | '
                        f'last_ball_done_time_sec={self.last_ball_done_time_sec:.2f} | '
                        f'last_ball_done_pose=({x:.3f}, {y:.3f}, {yaw:.3f})'
                    )

                    self.set_state(self.ball_return_state)
                    return True

                self.send_stop_command()
                return True

            self.restore_counter = 0
            wz = self.restore_turn_wz if yaw_err > 0 else -self.restore_turn_wz
            self.send_velocity_command(0.0, 0.0, wz)
            return True

        return False

    def control_loop(self):
        pose = self.get_current_pose()

        if pose is None:
            self.get_logger().warn('TF pose unavailable. Stop for safety.', throttle_duration_sec=1.0)
            self.send_stop_command()
            return

        x, y, yaw = pose
        ball = self.latest_ball_result
        yellow = self.latest_yellow_result

        if ball['img_shape'] is not None:
            self.get_logger().info(
                f"state={self.state} | orange_hit_count={self.orange_hit_count} | "
                f"ball_center={ball['ball_center']} | ball_depth={ball['ball_depth_m']} | "
                f"ball_radius={ball['ball_radius']} | ball_error_x={ball['error_x']} | "
                f"ball_aligned={ball['aligned']} | yellow_has_line={yellow['has_line']} | "
                f"yellow_bottom={yellow['line_bottom_y']} | align_counter={self.align_counter} | "
                f"restore_counter={self.restore_counter} | yellow_stop_counter={self.yellow_stop_counter}",
                throttle_duration_sec=0.6
            )

        if self.handle_ball_subchain(x, y, yaw):
            return

        if self.state == 'STAGE1_ROTATE_LEFT_90':
            self.execute_left_jump_turn(
                jump_count=1,
                next_state='STAGE1_MOVE_RIGHT_FIXED_DISTANCE'
            )
            return

        if self.state == 'STAGE2_ROTATE_LEFT_90':
            self.execute_left_jump_turn(
                jump_count=1,
                next_state='STAGE3_CRUISE_BALL_ONLY'
            )
            return

        if self.state == 'STAGE3_ROTATE_BACK_180':
            self.execute_left_jump_turn(
                jump_count=2,
                next_state='STAGE3_FINAL_DECISION'
            )
            return

        if self.state == 'STAGE3_ROTATE_LEFT_30':
            if self.handle_rotation_state(yaw, math.pi / 6.0, self.rotate_left_30_tolerance_rad,
                                          self.rotate_left_30_confirm_count, self.rotate_left_30_wz,
                                          'DONE'):
                return

        if self.state == 'STAGE1_CRUISE_BALL_AND_YELLOW':
            self.align_counter = 0
            self.restore_counter = 0

            if self.yellow_reached(yellow, self.yellow_stop_line_y_ratio_stage1):
                self.send_stop_command()
                self.set_state('STAGE1_ROTATE_LEFT_90')
                return

            if ball['has_ball'] and ball['ball_depth_m'] is not None and ball['ball_radius'] is not None:
                if ball['ball_depth_m'] <= self.turn_trigger_distance_m and ball['ball_radius'] >= self.min_ball_radius_to_trigger:
                    if self.can_trigger_ball_again((x, y, yaw)):
                        self.pre_turn_pose = (x, y, yaw)
                        self.align_yaw_delta = None
                        self.restore_start_yaw = None
                        self.align_pose = None
                        self.pose_at_0p2 = None
                        self.align_to_0p2_distance_m = 0.0
                        self.depth_return_counter = 0
                        self.segment_start_pose = None
                        self.ball_return_state = 'STAGE1_CRUISE_BALL_AND_YELLOW'
                        self.get_logger().info(f"Saved pre_turn_pose = ({x:.3f}, {y:.3f}, {yaw:.3f})")
                        self.set_state('BALL_ALIGN')
                        self.send_stop_command()
                        return

            self.send_velocity_command(self.cruise_forward_speed, 0.0, 0.0)
            return

        if self.state == 'STAGE1_MOVE_RIGHT_FIXED_DISTANCE':
            time.sleep(1)
            if self.stage1_right_shift_start_pose is None:
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
                self.send_stop_command()
                self.set_state('STAGE2_CRUISE_YELLOW_ONLY')
                return

            self.send_move_right_command(self.right_speed)
            return

        if self.state == 'STAGE2_CRUISE_YELLOW_ONLY':
            self.align_counter = 0
            self.restore_counter = 0

            if self.yellow_reached(yellow, self.yellow_stop_line_y_ratio_stage2):
                self.send_stop_command()
                self.set_state('STAGE2_ROTATE_LEFT_90')
                return

            self.send_velocity_command(self.cruise_forward_speed, 0.0, 0.0)
            return

        if self.state == 'STAGE3_CRUISE_BALL_ONLY':
            self.align_counter = 0
            self.restore_counter = 0

            if self.yellow_reached(yellow, self.yellow_stop_line_y_ratio_stage3):
                self.send_stop_command()
                self.set_state('STAGE3_ROTATE_BACK_180')
                return

            if ball['has_ball'] and ball['ball_depth_m'] is not None and ball['ball_radius'] is not None:
                if ball['ball_depth_m'] <= self.turn_trigger_distance_m and ball['ball_radius'] >= self.min_ball_radius_to_trigger:
                    if self.can_trigger_ball_again((x, y, yaw)):
                        self.pre_turn_pose = (x, y, yaw)
                        self.align_yaw_delta = None
                        self.restore_start_yaw = None
                        self.align_pose = None
                        self.pose_at_0p2 = None
                        self.align_to_0p2_distance_m = 0.0
                        self.depth_return_counter = 0
                        self.segment_start_pose = None
                        self.ball_return_state = 'STAGE3_CRUISE_BALL_ONLY'
                        self.get_logger().info(f"Saved pre_turn_pose = ({x:.3f}, {y:.3f}, {yaw:.3f})")
                        self.set_state('BALL_ALIGN')
                        self.send_stop_command()
                        return

            self.send_velocity_command(self.cruise_forward_speed, 0.0, 0.0)
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
                self.send_stop_command()
                self.set_state('STAGE3_SCAN_AND_HIT_LAST')
                return

            self.send_velocity_command(self.cruise_forward_speed, 0.0, 0.0)
            return

        if self.state == 'STAGE3_SCAN_AND_HIT_LAST':
            if ball['has_ball'] and ball['ball_depth_m'] is not None and ball['ball_radius'] is not None:
                self.pre_turn_pose = (x, y, yaw)
                self.align_yaw_delta = None
                self.restore_start_yaw = None
                self.align_pose = None
                self.pose_at_0p2 = None
                self.align_to_0p2_distance_m = 0.0
                self.depth_return_counter = 0
                self.segment_start_pose = None
                self.ball_return_state = 'STAGE3_GO_FINAL'
                self.get_logger().info(
                    f"Scan area found final ball. pre_turn_pose=({x:.3f}, {y:.3f}, {yaw:.3f})"
                )
                self.set_state('BALL_ALIGN')
                self.send_stop_command()
                return

            self.send_velocity_command(0.05, 0.0, 0.0)
            return

        if self.state == 'STAGE3_GO_FINAL':
            if self.yellow_reached(yellow, self.yellow_ratio_final):
                self.send_stop_command()
                self.set_state('STAGE3_ROTATE_LEFT_30')
                return

            self.send_velocity_command(self.cruise_forward_speed, 0.0, 0.0)
            return

        if self.state == 'DONE':
            self.send_stop_command()
            return


def main(args=None):
    rclpy.init(args=args)
    node = MultiStageOrangeYellowTaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down, sending stop command...')
        node.send_stop_command()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()