#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
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


class TfReturnToRecordedPoseTestNode(Node):
    """
    流程：
    1. 先对齐小球，记录 align_pose
    2. 前进直到球距离 <= 0.2m，记录 pose_at_0p2
    3. 再前进 0.25m
    4. 后退直到球距离重新回到 0.2m
    5. 再继续后退 distance(align_pose, pose_at_0p2)
    """

    def __init__(self):
        super().__init__('tf_return_to_recorded_pose_test_node')

        # =========================
        # topics / tf
        # =========================
        self.declare_parameter('rgb_topic', '/rgb_camera/rgb_camera/image_raw')
        self.declare_parameter('depth_topic', '/d435/depth/d435_depth/depth/image_raw')
        self.declare_parameter('global_frame', 'vodom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('control_hz', 30.0)

        # =========================
        # orange detect
        # =========================
        self.declare_parameter('orange_h_min', 5)
        self.declare_parameter('orange_h_max', 25)
        self.declare_parameter('orange_s_min', 100)
        self.declare_parameter('orange_s_max', 255)
        self.declare_parameter('orange_v_min', 80)
        self.declare_parameter('orange_v_max', 255)
        self.declare_parameter('orange_min_contour_area', 60.0)
        self.declare_parameter('prefer_nearest_ball', True)

        # =========================
        # depth
        # =========================
        self.declare_parameter('depth_search_half', 12)
        self.declare_parameter('valid_min_depth_m', 0.05)
        self.declare_parameter('valid_max_depth_m', 10.0)

        # =========================
        # motion params
        # =========================
        self.declare_parameter('approach_speed', 0.10)
        self.declare_parameter('forward_speed', 0.10)
        self.declare_parameter('backward_speed', 0.08)
        self.declare_parameter('target_ball_distance_m', 0.20)
        self.declare_parameter('forward_after_record_m', 0.25)

        # 对齐相关
        self.declare_parameter('align_tolerance_px', 10)
        self.declare_parameter('align_confirm_count', 3)
        self.declare_parameter('align_turn_wz', 0.10)

        # 返回到 0.2m 的深度确认
        self.declare_parameter('return_depth_tolerance_m', 0.015)
        self.declare_parameter('return_depth_confirm_count', 3)

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.global_frame = self.get_parameter('global_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.control_hz = float(self.get_parameter('control_hz').value)

        self.orange_h_min = int(self.get_parameter('orange_h_min').value)
        self.orange_h_max = int(self.get_parameter('orange_h_max').value)
        self.orange_s_min = int(self.get_parameter('orange_s_min').value)
        self.orange_s_max = int(self.get_parameter('orange_s_max').value)
        self.orange_v_min = int(self.get_parameter('orange_v_min').value)
        self.orange_v_max = int(self.get_parameter('orange_v_max').value)
        self.orange_min_contour_area = float(self.get_parameter('orange_min_contour_area').value)
        self.prefer_nearest_ball = bool(self.get_parameter('prefer_nearest_ball').value)

        self.depth_search_half = int(self.get_parameter('depth_search_half').value)
        self.valid_min_depth_m = float(self.get_parameter('valid_min_depth_m').value)
        self.valid_max_depth_m = float(self.get_parameter('valid_max_depth_m').value)

        self.approach_speed = float(self.get_parameter('approach_speed').value)
        self.forward_speed = float(self.get_parameter('forward_speed').value)
        self.backward_speed = float(self.get_parameter('backward_speed').value)
        self.target_ball_distance_m = float(self.get_parameter('target_ball_distance_m').value)
        self.forward_after_record_m = float(self.get_parameter('forward_after_record_m').value)

        self.align_tolerance_px = int(self.get_parameter('align_tolerance_px').value)
        self.align_confirm_count = int(self.get_parameter('align_confirm_count').value)
        self.align_turn_wz = float(self.get_parameter('align_turn_wz').value)

        self.return_depth_tolerance_m = float(self.get_parameter('return_depth_tolerance_m').value)
        self.return_depth_confirm_count = int(self.get_parameter('return_depth_confirm_count').value)

        self.bridge = CvBridge()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.Ctrl = Robot_Ctrl()
        self.Ctrl.run()
        self.msg = robot_control_cmd_lcmt()
        if not hasattr(self.msg, 'life_count'):
            self.msg.life_count = 0

        self.latest_depth = None
        self.latest_depth_encoding = None
        self.rgb_w = 640
        self.rgb_h = 480

        self.latest_ball_result = {
            'has_ball': False,
            'ball_center': None,
            'ball_radius': None,
            'ball_depth_m': None,
            'img_shape': None,
            'error_x': None,
        }

        # =========================
        # state
        # =========================
        self.state = 'ALIGN_BALL'

        self.align_pose: Optional[Tuple[float, float, float]] = None
        self.pose_at_0p2: Optional[Tuple[float, float, float]] = None
        self.pose_after_forward_0p25: Optional[Tuple[float, float, float]] = None
        self.pose_returned_to_0p2: Optional[Tuple[float, float, float]] = None
        self.final_return_pose: Optional[Tuple[float, float, float]] = None

        self.align_to_0p2_distance_m: float = 0.0
        self.segment_start_pose: Optional[Tuple[float, float, float]] = None

        self.align_counter = 0
        self.return_depth_counter = 0

        self.rgb_sub = self.create_subscription(
            Image, self.rgb_topic, self.rgb_callback, qos_profile_sensor_data
        )
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, qos_profile_sensor_data
        )
        self.control_timer = self.create_timer(1.0 / self.control_hz, self.control_loop)

        self.send_stop_command()
        self.Ctrl.Wait_finish(12, 0)

        self.get_logger().info('TfReturnToRecordedPoseTestNode started')
        self.get_logger().info(f'rgb_topic={self.rgb_topic}')
        self.get_logger().info(f'depth_topic={self.depth_topic}')
        self.get_logger().info(f'tf: {self.global_frame} -> {self.base_frame}')

    # =========================
    # control
    # =========================
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
        self.get_logger().info(
            f'[CMD] vel_des=[{vx:.3f}, {vy:.3f}, {wz:.3f}]',
            throttle_duration_sec=0.5
        )

    # =========================
    # tf / pose
    # =========================
    def get_current_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.base_frame,
                Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        return (t.x, t.y, yaw)

    def xy_distance(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def log_pose(self, tag: str, pose: Tuple[float, float, float]):
        x, y, yaw = pose
        self.get_logger().info(
            f'[{tag}] tf_pose = (x={x:.3f}, y={y:.3f}, yaw={yaw:.6f} rad, {math.degrees(yaw):.3f} deg)'
        )

    # =========================
    # callbacks
    # =========================
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

    # =========================
    # orange detect
    # =========================
    def get_depth_for_rgb_point(self, rgb_cx: int, rgb_cy: int):
        if self.latest_depth is None or self.latest_depth_encoding is None:
            return None

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
            return None

        valid = patch_m[np.isfinite(patch_m)]
        valid = valid[
            (valid > self.valid_min_depth_m) &
            (valid < self.valid_max_depth_m)
        ]

        if valid.size == 0:
            return None

        return float(np.percentile(valid, 20))

    def detect_orange_ball_with_depth(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        self.rgb_w = w
        self.rgb_h = h

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_orange = np.array(
            [self.orange_h_min, self.orange_s_min, self.orange_v_min], dtype=np.uint8
        )
        upper_orange = np.array(
            [self.orange_h_max, self.orange_s_max, self.orange_v_max], dtype=np.uint8
        )

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

            depth_m = self.get_depth_for_rgb_point(cx, cy)
            if depth_m is None:
                continue

            error_x = cx - (w // 2)
            candidates.append({
                'center': (cx, cy),
                'radius': radius,
                'depth_m': depth_m,
                'error_x': int(error_x),
            })

        if len(candidates) == 0:
            return {
                'has_ball': False,
                'ball_center': None,
                'ball_radius': None,
                'ball_depth_m': None,
                'img_shape': (h, w),
                'error_x': None,
            }

        if self.prefer_nearest_ball:
            best = min(candidates, key=lambda c: c['depth_m'])
        else:
            best = candidates[0]

        return {
            'has_ball': True,
            'ball_center': best['center'],
            'ball_radius': best['radius'],
            'ball_depth_m': best['depth_m'],
            'img_shape': (h, w),
            'error_x': best['error_x'],
        }

    # =========================
    # main state machine
    # =========================
    def control_loop(self):
        pose = self.get_current_pose()
        if pose is None:
            self.get_logger().warn('TF unavailable, stop for safety.', throttle_duration_sec=1.0)
            self.send_stop_command()
            return

        x, y, yaw = pose
        ball = self.latest_ball_result

        self.get_logger().info(
            f'state={self.state} | '
            f'ball_center={ball["ball_center"]} | '
            f'ball_depth={ball["ball_depth_m"]} | '
            f'ball_radius={ball["ball_radius"]} | '
            f'ball_error_x={ball["error_x"]}',
            throttle_duration_sec=0.6
        )

        # 1. 先对齐小球
        if self.state == 'ALIGN_BALL':
            if not ball['has_ball'] or ball['error_x'] is None:
                self.get_logger().warn(
                    'Ball not detected in ALIGN_BALL, stop and wait.',
                    throttle_duration_sec=1.0
                )
                self.send_stop_command()
                return

            if abs(ball['error_x']) <= self.align_tolerance_px:
                self.align_counter += 1
                self.send_stop_command()

                if self.align_counter >= self.align_confirm_count:
                    self.align_pose = pose
                    self.log_pose('ALIGN_POSE', self.align_pose)
                    self.state = 'APPROACH_TO_0P2'
                return

            self.align_counter = 0
            wz = -self.align_turn_wz if ball['error_x'] > 0 else self.align_turn_wz
            self.send_velocity_command(0.0, 0.0, wz)
            return

        # 2. 前进直到球距离 <= 0.2m
        if self.state == 'APPROACH_TO_0P2':
            if not ball['has_ball'] or ball['ball_depth_m'] is None:
                self.get_logger().warn(
                    'Ball/depth unavailable in APPROACH_TO_0P2, stop and wait.',
                    throttle_duration_sec=1.0
                )
                self.send_stop_command()
                return

            if ball['ball_depth_m'] > self.target_ball_distance_m:
                self.send_velocity_command(self.approach_speed, 0.0, 0.0)
                return

            self.send_stop_command()
            self.pose_at_0p2 = pose
            self.log_pose('POSE_AT_BALL_0P2', self.pose_at_0p2)

            if self.align_pose is not None:
                self.align_to_0p2_distance_m = self.xy_distance(self.align_pose, self.pose_at_0p2)
                self.get_logger().info(
                    f'[ALIGN_TO_0P2_DISTANCE] {self.align_to_0p2_distance_m:.6f} m'
                )

            self.segment_start_pose = pose
            self.state = 'FORWARD_0P25'
            return

        # 3. 再前进 0.25m
        if self.state == 'FORWARD_0P25':
            if self.segment_start_pose is None:
                self.send_stop_command()
                return

            dist = self.xy_distance(pose, self.segment_start_pose)
            self.get_logger().info(
                f'[FORWARD_0P25] dist={dist:.3f}/{self.forward_after_record_m:.3f}',
                throttle_duration_sec=0.3
            )

            if dist >= self.forward_after_record_m:
                self.send_stop_command()
                self.pose_after_forward_0p25 = pose
                self.log_pose('POSE_AFTER_FORWARD_0P25', self.pose_after_forward_0p25)
                self.state = 'RETURN_TO_0P2_BY_DEPTH'
                return

            self.send_velocity_command(self.forward_speed, 0.0, 0.0)
            return

        # 4. 后退直到深度重新回到 0.2m
        if self.state == 'RETURN_TO_0P2_BY_DEPTH':
            if not ball['has_ball'] or ball['ball_depth_m'] is None:
                self.get_logger().warn(
                    'Ball/depth unavailable in RETURN_TO_0P2_BY_DEPTH, stop and wait.',
                    throttle_duration_sec=1.0
                )
                self.send_stop_command()
                return

            depth_err = ball['ball_depth_m'] - self.target_ball_distance_m

            self.get_logger().info(
                f'[RETURN_TO_0P2_BY_DEPTH] depth={ball["ball_depth_m"]:.6f}, '
                f'target={self.target_ball_distance_m:.3f}, err={depth_err:.6f}',
                throttle_duration_sec=0.3
            )

            if abs(depth_err) <= self.return_depth_tolerance_m:
                self.return_depth_counter += 1
                self.send_stop_command()

                if self.return_depth_counter >= self.return_depth_confirm_count:
                    self.pose_returned_to_0p2 = pose
                    self.log_pose('POSE_RETURNED_TO_0P2_BY_DEPTH', self.pose_returned_to_0p2)
                    self.segment_start_pose = pose
                    self.state = 'BACKWARD_ALIGN_TO_0P2_DISTANCE'
                return

            self.return_depth_counter = 0
            self.send_velocity_command(-self.backward_speed, 0.0, 0.0)
            return

        # 5. 再继续后退 distance(align_pose, pose_at_0p2)
        if self.state == 'BACKWARD_ALIGN_TO_0P2_DISTANCE':
            if self.segment_start_pose is None:
                self.send_stop_command()
                return

            dist = self.xy_distance(pose, self.segment_start_pose)

            self.get_logger().info(
                f'[BACKWARD_ALIGN_TO_0P2_DISTANCE] dist={dist:.6f}/{self.align_to_0p2_distance_m:.6f}',
                throttle_duration_sec=0.3
            )

            if dist >= self.align_to_0p2_distance_m:
                self.send_stop_command()
                self.final_return_pose = pose
                self.log_pose('FINAL_RETURN_POSE', self.final_return_pose)
                self.get_logger().info('=' * 70)
                self.get_logger().info('Test finished')
                self.get_logger().info('=' * 70)
                self.state = 'DONE'
                return

            self.send_velocity_command(-self.backward_speed, 0.0, 0.0)
            return

        if self.state == 'DONE':
            self.send_stop_command()
            return


def main(args=None):
    rclpy.init(args=args)
    node = TfReturnToRecordedPoseTestNode()
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