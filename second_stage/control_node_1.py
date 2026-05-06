#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
第一赛段合并版代码：视觉识别 + 机器狗控制放在同一个 ROS2 节点里。

合并内容：
1. 原 part1_vision.txt：
   - 订阅 RGB 图像
   - 订阅深度图像
   - 检测黄色边界 / 终点横向黄线
   - 检测蓝球距离
   - OpenCV 可视化窗口

2. 原 part1_2.0.txt：
   - LCM 控制 Robot_Ctrl
   - 起立
   - 第一赛段黄线纠偏行走
   - 看到横向黄线后刹车
   - 原地旋转调平
   - 按仿真时间左转
   - 找蓝球并前进到指定距离
   - 盲走左移
   - 停止

说明：
- 这版不再通过 /vision/line_error 和 /vision/stage2_error 两个中间话题通信。
- 视觉回调直接更新 self.lateral_force / self.stop_flag / self.stop_angle / self.s2_err_x 等变量。
- 控制逻辑使用 ROS 仿真时间，不使用 TF。
"""

import sys
import time
from threading import Thread, Lock
from typing import Optional

import cv2
import lcm
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from robot_control_cmd_lcmt import robot_control_cmd_lcmt
from robot_control_response_lcmt import robot_control_response_lcmt


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


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
            # 原代码这里没有更新 gait_ok，Wait_finish 会不稳定；合并版修正。
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
        while self.runing and count < 2000:  # 最大等待约 10 秒
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


class Part1CombinedNode(Node):
    STAND_WAIT = 'STAND_WAIT'
    STAGE1_CRUISE = 'STAGE1_CRUISE'
    BRAKE_BUFFER = 'BRAKE_BUFFER'
    ALIGN_STOP_LINE = 'ALIGN_STOP_LINE'
    TURN_LEFT_TO_STAGE2 = 'TURN_LEFT_TO_STAGE2'
    APPROACH_BLUE_BALL = 'APPROACH_BLUE_BALL'
    BLIND_LEFT_SHIFT = 'BLIND_LEFT_SHIFT'
    DONE = 'DONE'

    def __init__(self):
        super().__init__('part1_combined_node')

        # 使用仿真时间
        try:
            self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        except Exception as e:
            self.get_logger().warn(f'failed to set use_sim_time: {e}')

        # =========================
        # 话题参数
        # =========================
        self.declare_parameter('rgb_topic', '/rgb_camera/rgb_camera/image_raw')
        self.declare_parameter('depth_topic', '/d435/depth/d435_depth/depth/image_raw')
        self.declare_parameter('control_hz', 10.0)
        self.declare_parameter('show_debug_vis', True)

        # =========================
        # 起立参数
        # =========================
        self.declare_parameter('stand_wait_sec', 3.0)
        self.declare_parameter('stand_body_height', 0.28)

        # =========================
        # 第一赛段巡航参数
        # =========================
        self.declare_parameter('stage1_max_duration_sec', 8.0)
        self.declare_parameter('base_forward_speed', 0.40)
        self.declare_parameter('min_forward_speed', 0.20)
        self.declare_parameter('kp_turn', 0.25)
        self.declare_parameter('kp_lat', 0.15)
        self.declare_parameter('kd_slowdown', 0.05)
        self.declare_parameter('max_turn_speed', 0.15)
        self.declare_parameter('max_lateral_speed', 0.15)
        self.declare_parameter('vision_timeout_sec', 1.0)

        # =========================
        # 刹车 / 横线调平
        # =========================
        self.declare_parameter('brake_duration_sec', 0.3)
        self.declare_parameter('align_max_duration_sec', 3.0)
        self.declare_parameter('align_angle_deadband_rad', 0.05)
        self.declare_parameter('align_turn_kp', 0.4)
        self.declare_parameter('align_turn_max_wz', 0.10)

        # =========================
        # 左转进入下一段
        # =========================
        self.declare_parameter('turn_duration_sec', 3.5)
        self.declare_parameter('turn_forward_vel', 0.13)
        self.declare_parameter('turn_yaw_vel', 0.50)

        # =========================
        # 找蓝球前进
        # =========================
        self.declare_parameter('blue_target_distance_m', 0.25)
        self.declare_parameter('approach_blue_max_duration_sec', 6.0)
        self.declare_parameter('approach_blue_forward_speed', 0.20)

        # =========================
        # 最后盲走左移
        # =========================
        self.declare_parameter('blind_left_duration_sec', 3.0)
        self.declare_parameter('blind_left_vy', 0.13)
        self.declare_parameter('blind_left_vx', 0.14)

        # =========================
        # 视觉参数：黄色
        # =========================
        self.declare_parameter('yellow_h_min', 20)
        self.declare_parameter('yellow_h_max', 40)
        self.declare_parameter('yellow_s_min', 50)
        self.declare_parameter('yellow_s_max', 255)
        self.declare_parameter('yellow_v_min', 150)
        self.declare_parameter('yellow_v_max', 255)

        self.declare_parameter('stop_top_ratio', 0.80)
        self.declare_parameter('stop_bottom_ratio', 0.95)
        self.declare_parameter('stop_left_ratio', 0.35)
        self.declare_parameter('stop_right_ratio', 0.65)
        self.declare_parameter('stop_yellow_pixel_threshold', 1500)

        self.declare_parameter('nav_top_ratio', 0.90)
        self.declare_parameter('nav_bottom_ratio', 1.00)
        self.declare_parameter('nav_crop_left_ratio', 0.15)
        self.declare_parameter('nav_crop_right_ratio', 0.85)

        # =========================
        # 视觉参数：蓝球
        # =========================
        self.declare_parameter('blue_h_min', 100)
        self.declare_parameter('blue_h_max', 130)
        self.declare_parameter('blue_s_min', 100)
        self.declare_parameter('blue_s_max', 255)
        self.declare_parameter('blue_v_min', 50)
        self.declare_parameter('blue_v_max', 255)
        self.declare_parameter('blue_min_area', 6500.0)
        self.declare_parameter('blue_depth_patch_half', 1)
        self.declare_parameter('valid_min_depth_m', 0.05)
        self.declare_parameter('valid_max_depth_m', 10.0)

        # =========================
        # 读取参数
        # =========================
        self.rgb_topic = str(self.get_parameter('rgb_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.control_hz = float(self.get_parameter('control_hz').value)
        self.show_debug_vis = bool(self.get_parameter('show_debug_vis').value)

        self.stand_wait_sec = float(self.get_parameter('stand_wait_sec').value)
        self.stand_body_height = float(self.get_parameter('stand_body_height').value)

        self.stage1_max_duration_sec = float(self.get_parameter('stage1_max_duration_sec').value)
        self.base_forward_speed = float(self.get_parameter('base_forward_speed').value)
        self.min_forward_speed = float(self.get_parameter('min_forward_speed').value)
        self.kp_turn = float(self.get_parameter('kp_turn').value)
        self.kp_lat = float(self.get_parameter('kp_lat').value)
        self.kd_slowdown = float(self.get_parameter('kd_slowdown').value)
        self.max_turn_speed = float(self.get_parameter('max_turn_speed').value)
        self.max_lateral_speed = float(self.get_parameter('max_lateral_speed').value)
        self.vision_timeout_sec = float(self.get_parameter('vision_timeout_sec').value)

        self.brake_duration_sec = float(self.get_parameter('brake_duration_sec').value)
        self.align_max_duration_sec = float(self.get_parameter('align_max_duration_sec').value)
        self.align_angle_deadband_rad = float(self.get_parameter('align_angle_deadband_rad').value)
        self.align_turn_kp = float(self.get_parameter('align_turn_kp').value)
        self.align_turn_max_wz = float(self.get_parameter('align_turn_max_wz').value)

        self.turn_duration_sec = float(self.get_parameter('turn_duration_sec').value)
        self.turn_forward_vel = float(self.get_parameter('turn_forward_vel').value)
        self.turn_yaw_vel = float(self.get_parameter('turn_yaw_vel').value)

        self.blue_target_distance_m = float(self.get_parameter('blue_target_distance_m').value)
        self.approach_blue_max_duration_sec = float(self.get_parameter('approach_blue_max_duration_sec').value)
        self.approach_blue_forward_speed = float(self.get_parameter('approach_blue_forward_speed').value)

        self.blind_left_duration_sec = float(self.get_parameter('blind_left_duration_sec').value)
        self.blind_left_vy = float(self.get_parameter('blind_left_vy').value)
        self.blind_left_vx = float(self.get_parameter('blind_left_vx').value)

        self.yellow_h_min = int(self.get_parameter('yellow_h_min').value)
        self.yellow_h_max = int(self.get_parameter('yellow_h_max').value)
        self.yellow_s_min = int(self.get_parameter('yellow_s_min').value)
        self.yellow_s_max = int(self.get_parameter('yellow_s_max').value)
        self.yellow_v_min = int(self.get_parameter('yellow_v_min').value)
        self.yellow_v_max = int(self.get_parameter('yellow_v_max').value)

        self.stop_top_ratio = float(self.get_parameter('stop_top_ratio').value)
        self.stop_bottom_ratio = float(self.get_parameter('stop_bottom_ratio').value)
        self.stop_left_ratio = float(self.get_parameter('stop_left_ratio').value)
        self.stop_right_ratio = float(self.get_parameter('stop_right_ratio').value)
        self.stop_yellow_pixel_threshold = int(self.get_parameter('stop_yellow_pixel_threshold').value)

        self.nav_top_ratio = float(self.get_parameter('nav_top_ratio').value)
        self.nav_bottom_ratio = float(self.get_parameter('nav_bottom_ratio').value)
        self.nav_crop_left_ratio = float(self.get_parameter('nav_crop_left_ratio').value)
        self.nav_crop_right_ratio = float(self.get_parameter('nav_crop_right_ratio').value)

        self.blue_h_min = int(self.get_parameter('blue_h_min').value)
        self.blue_h_max = int(self.get_parameter('blue_h_max').value)
        self.blue_s_min = int(self.get_parameter('blue_s_min').value)
        self.blue_s_max = int(self.get_parameter('blue_s_max').value)
        self.blue_v_min = int(self.get_parameter('blue_v_min').value)
        self.blue_v_max = int(self.get_parameter('blue_v_max').value)
        self.blue_min_area = float(self.get_parameter('blue_min_area').value)
        self.blue_depth_patch_half = int(self.get_parameter('blue_depth_patch_half').value)
        self.valid_min_depth_m = float(self.get_parameter('valid_min_depth_m').value)
        self.valid_max_depth_m = float(self.get_parameter('valid_max_depth_m').value)

        # =========================
        # 控制接口
        # =========================
        self.Ctrl = Robot_Ctrl()
        self.Ctrl.run()

        self.msg = robot_control_cmd_lcmt()
        if not hasattr(self.msg, 'life_count'):
            self.msg.life_count = 0

        # =========================
        # 视觉缓存
        # =========================
        self.bridge = CvBridge()
        self.depth_image = None
        self.depth_encoding = None

        self.lateral_force = 0.0
        self.stop_angle = 0.0
        self.stop_flag = 0.0
        self.last_update_time = 0.0

        self.s2_err_x = 0.0       # 最近蓝球距离，单位 m
        self.s2_spheres = 0.0     # 蓝球数量
        self.blue_detections = []

        self.latest_frame = None
        self.latest_mask_yellow = None

        # =========================
        # 状态机
        # =========================
        self.state = self.STAND_WAIT
        self.state_start_time: Optional[float] = None
        self.stand_sent = False
        self.done_stop_sent = False

        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.image_callback,
            qos_profile_sensor_data
        )
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile_sensor_data
        )

        self.timer = self.create_timer(1.0 / self.control_hz, self.control_loop)

        self.get_logger().info('Part1CombinedNode started.')
        self.get_logger().info(f'rgb_topic={self.rgb_topic}')
        self.get_logger().info(f'depth_topic={self.depth_topic}')

    # ============================================================
    # 时间 / 状态
    # ============================================================
    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def set_state(self, new_state: str):
        if new_state != self.state:
            self.get_logger().info(f'STATE: {self.state} -> {new_state}')
            self.state = new_state
            self.state_start_time = None
            if new_state == self.DONE:
                self.done_stop_sent = False

    def elapsed_in_state(self) -> float:
        now = self.now_sec()
        if self.state_start_time is None:
            self.state_start_time = now
        return now - self.state_start_time

    # ============================================================
    # 控制命令
    # ============================================================
    def _inc_life_count(self):
        self.msg.life_count += 1
        if self.msg.life_count > 127:
            self.msg.life_count = 0

    def send_stand_command(self):
        self.msg.mode = 12
        self.msg.gait_id = 0
        self._inc_life_count()
        self.msg.rpy_des = [0.0, 0.0, 0.0]
        self.msg.pos_des = [0.0, 0.0, self.stand_body_height]
        self.Ctrl.Send_cmd(self.msg)
        self.get_logger().info('[CMD] STAND', throttle_duration_sec=1.0)

    def send_velocity_command(self, vx: float, vy: float, wz: float, step_height: float = 0.13):
        self.msg.mode = 11
        self.msg.gait_id = 3
        self._inc_life_count()
        self.msg.step_height = [step_height, step_height]
        self.msg.vel_des = [float(vx), float(vy), float(wz)]
        self.Ctrl.Send_cmd(self.msg)
        self.get_logger().info(
            f'[CMD] vel_des=[{vx:.3f}, {vy:.3f}, {wz:.3f}]',
            throttle_duration_sec=0.4
        )

    def send_zero_velocity(self):
        self.send_velocity_command(0.0, 0.0, 0.0)

    # ============================================================
    # 深度处理
    # ============================================================
    def depth_callback(self, msg: Image):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_encoding = msg.encoding
        except Exception as e:
            self.get_logger().error(f'depth convert failed: {e}')

    def depth_to_meters_patch(self, patch: np.ndarray):
        if patch is None or patch.size == 0:
            return None

        if self.depth_encoding == '16UC1':
            patch_m = patch.astype(np.float32) / 1000.0
        elif self.depth_encoding == '32FC1':
            patch_m = patch.astype(np.float32)
        else:
            patch_m = patch.astype(np.float32)

        valid = patch_m[np.isfinite(patch_m)]
        valid = valid[(valid > self.valid_min_depth_m) & (valid < self.valid_max_depth_m)]

        if valid.size == 0:
            return None
        return float(np.median(valid))

    # ============================================================
    # 视觉处理
    # ============================================================
    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image convert failed: {e}')
            return

        self.latest_frame = cv_image
        height, width = cv_image.shape[:2]
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # 黄色检测
        lower_yellow = np.array([self.yellow_h_min, self.yellow_s_min, self.yellow_v_min], dtype=np.uint8)
        upper_yellow = np.array([self.yellow_h_max, self.yellow_s_max, self.yellow_v_max], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        self.latest_mask_yellow = mask_yellow

        # 第一赛段：终点横向黄线 + 导航斥力
        self.process_stage1_yellow(cv_image, mask_yellow)

        # 后续衔接：蓝球绝对距离
        self.process_blue_ball(cv_image, hsv)

        if self.show_debug_vis:
            self.show_debug_window(cv_image, mask_yellow)

    def process_stage1_yellow(self, cv_image: np.ndarray, mask_yellow: np.ndarray):
        height, width = cv_image.shape[:2]

        lateral_force = 0.0
        stop_angle = 0.0
        stop_flag = 0.0

        # 终点横线探测：中央区域
        stop_top = int(height * self.stop_top_ratio)
        stop_bottom = int(height * self.stop_bottom_ratio)
        stop_left = int(width * self.stop_left_ratio)
        stop_right = int(width * self.stop_right_ratio)
        mask_stop = mask_yellow[stop_top:stop_bottom, stop_left:stop_right]

        if cv2.countNonZero(mask_stop) > self.stop_yellow_pixel_threshold:
            stop_flag = 1.0

            contours, _ = cv2.findContours(mask_stop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect).astype(np.int32)

                # 原代码：找最左和最右点算倾角。
                box_sorted = sorted(box, key=lambda p: p[0])
                left_pt, right_pt = box_sorted[0], box_sorted[-1]

                dx = right_pt[0] - left_pt[0]
                dy = right_pt[1] - left_pt[1]
                stop_angle = float(np.arctan2(dy, dx)) if dx != 0 else 0.0

        # 底部导航区域
        nav_top = int(height * self.nav_top_ratio)
        nav_bottom = int(height * self.nav_bottom_ratio)
        crop_left = int(width * self.nav_crop_left_ratio)
        crop_right = int(width * self.nav_crop_right_ratio)

        mask_nav = np.zeros_like(mask_yellow)
        mask_nav[nav_top:nav_bottom, crop_left:crop_right] = mask_yellow[nav_top:nav_bottom, crop_left:crop_right]

        M_nav = cv2.moments(mask_nav)
        if M_nav['m00'] > 0:
            cx_nav = int(M_nav['m10'] / M_nav['m00'])
            dist_nav = abs(cx_nav - width / 2)
            force_nav = ((width / 2 - dist_nav) / (width / 2)) ** 3
            lateral_force = float(force_nav) if cx_nav > width / 2 else -float(force_nav)

        self.lateral_force = lateral_force
        self.stop_angle = stop_angle
        self.stop_flag = stop_flag
        self.last_update_time = self.now_sec()

    def process_blue_ball(self, cv_image: np.ndarray, hsv: np.ndarray):
        height, width = cv_image.shape[:2]
        self.blue_detections = []
        self.s2_err_x = 0.0
        self.s2_spheres = 0.0

        if self.depth_image is None:
            return

        lower_blue = np.array([self.blue_h_min, self.blue_s_min, self.blue_v_min], dtype=np.uint8)
        upper_blue = np.array([self.blue_h_max, self.blue_s_max, self.blue_v_max], dtype=np.uint8)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_depths = []
        ball_count = 0

        dh, dw = self.depth_image.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= self.blue_min_area:
                continue

            M = cv2.moments(cnt)
            if M['m00'] <= 0:
                continue

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # RGB 点映射到深度图坐标，避免 RGB/Depth 分辨率不同
            dx = int(cx * dw / max(width, 1))
            dy = int(cy * dh / max(height, 1))

            half = self.blue_depth_patch_half
            x1 = max(0, dx - half)
            x2 = min(dw, dx + half + 1)
            y1 = max(0, dy - half)
            y2 = min(dh, dy + half + 1)

            region = self.depth_image[y1:y2, x1:x2]
            depth_m = self.depth_to_meters_patch(region)

            if depth_m is not None:
                valid_depths.append(depth_m)
                ball_count += 1
                self.blue_detections.append({
                    'center': (cx, cy),
                    'depth_m': depth_m,
                    'area': float(area),
                })

        if ball_count > 0:
            self.s2_spheres = float(ball_count)
            self.s2_err_x = float(min(valid_depths))

    # ============================================================
    # 可视化
    # ============================================================
    def show_debug_window(self, cv_image: np.ndarray, mask_yellow: np.ndarray):
        vis = cv_image.copy()
        height, width = vis.shape[:2]

        # 画终点横线检测区域
        stop_top = int(height * self.stop_top_ratio)
        stop_bottom = int(height * self.stop_bottom_ratio)
        stop_left = int(width * self.stop_left_ratio)
        stop_right = int(width * self.stop_right_ratio)
        cv2.rectangle(vis, (stop_left, stop_top), (stop_right, stop_bottom), (0, 0, 255), 2)

        # 画导航区域
        nav_top = int(height * self.nav_top_ratio)
        nav_bottom = int(height * self.nav_bottom_ratio)
        crop_left = int(width * self.nav_crop_left_ratio)
        crop_right = int(width * self.nav_crop_right_ratio)
        cv2.rectangle(vis, (crop_left, nav_top), (crop_right, nav_bottom), (0, 255, 0), 2)

        # 画导航黄色质心
        mask_nav = np.zeros_like(mask_yellow)
        mask_nav[nav_top:nav_bottom, crop_left:crop_right] = mask_yellow[nav_top:nav_bottom, crop_left:crop_right]
        M_nav = cv2.moments(mask_nav)
        if M_nav['m00'] > 0:
            cx_nav = int(M_nav['m10'] / M_nav['m00'])
            cy_nav = int((nav_top + nav_bottom) / 2)
            cv2.circle(vis, (cx_nav, cy_nav), 8, (0, 255, 255), -1)

        # 画蓝球
        for det in self.blue_detections:
            cx, cy = det['center']
            depth_m = det['depth_m']
            cv2.circle(vis, (cx, cy), 10, (255, 0, 255), -1)
            cv2.putText(
                vis,
                f'{depth_m:.2f}m',
                (cx - 20, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        cv2.putText(
            vis,
            f'state={self.state}',
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )
        cv2.putText(
            vis,
            f'force={self.lateral_force:.3f} stop={self.stop_flag:.1f} angle={np.degrees(self.stop_angle):.1f}deg',
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2
        )
        cv2.putText(
            vis,
            f'blue_count={self.s2_spheres:.0f} nearest_blue={self.s2_err_x:.2f}m enc={self.depth_encoding}',
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 0, 255),
            2
        )

        cv2.imshow('CyberDog_Part1_Combined', vis)
        cv2.waitKey(1)

    # ============================================================
    # 控制状态机
    # ============================================================
    def control_loop(self):
        now = self.now_sec()
        if now <= 0.0:
            return

        elapsed = self.elapsed_in_state()

        if self.state == self.STAND_WAIT:
            if not self.stand_sent:
                self.get_logger().info('1. 起立.')
                self.send_stand_command()
                self.stand_sent = True

            if elapsed >= self.stand_wait_sec:
                self.get_logger().info('2. 开始行走')
                self.set_state(self.STAGE1_CRUISE)
            return

        if self.state == self.STAGE1_CRUISE:
            # 看到终点横向黄线，退出直走状态
            if self.stop_flag > 0.5:
                self.get_logger().info('看到横向黄线，退出直走状态')
                self.set_state(self.BRAKE_BUFFER)
                return

            if elapsed >= self.stage1_max_duration_sec:
                self.get_logger().info('第一赛段行走超时，进入刹车缓冲')
                self.set_state(self.BRAKE_BUFFER)
                return

            if now - self.last_update_time < self.vision_timeout_sec:
                err = self.lateral_force
                turn_speed = clamp(err * self.kp_turn, -self.max_turn_speed, self.max_turn_speed)
                lateral_speed = clamp(err * self.kp_lat, -self.max_lateral_speed, self.max_lateral_speed)
                speed_drop = abs(err) * self.kd_slowdown
                forward_speed = max(self.min_forward_speed, self.base_forward_speed - speed_drop)
            else:
                forward_speed = self.base_forward_speed
                lateral_speed = 0.0
                turn_speed = 0.0

            self.send_velocity_command(forward_speed, lateral_speed, turn_speed, step_height=0.13)
            return

        if self.state == self.BRAKE_BUFFER:
            if elapsed >= self.brake_duration_sec:
                self.get_logger().info('4. 开始原地旋转调平姿态...')
                self.set_state(self.ALIGN_STOP_LINE)
                return
            self.send_zero_velocity()
            return

        if self.state == self.ALIGN_STOP_LINE:
            angle_err = self.stop_angle

            if abs(angle_err) < self.align_angle_deadband_rad or self.stop_flag < 0.5:
                self.get_logger().info(f'姿态调平完毕或横线超出视野，误差={angle_err:.3f}')
                self.set_state(self.TURN_LEFT_TO_STAGE2)
                return

            if elapsed >= self.align_max_duration_sec:
                self.get_logger().info('姿态调平超时，进入持续左转')
                self.set_state(self.TURN_LEFT_TO_STAGE2)
                return

            turn_speed = clamp(angle_err * self.align_turn_kp, -self.align_turn_max_wz, self.align_turn_max_wz)
            self.send_velocity_command(0.0, 0.0, turn_speed, step_height=0.13)
            return

        if self.state == self.TURN_LEFT_TO_STAGE2:
            if elapsed >= self.turn_duration_sec:
                self.get_logger().info('5. 阶段A：寻找右侧蓝球，根据距离笔直往前走...')
                self.set_state(self.APPROACH_BLUE_BALL)
                return

            self.send_velocity_command(self.turn_forward_vel, 0.0, self.turn_yaw_vel, step_height=0.13)
            return

        if self.state == self.APPROACH_BLUE_BALL:
            if self.s2_spheres >= 1.0:
                self.get_logger().info(
                    f'锁定蓝球当前距离: {self.s2_err_x:.2f} 米',
                    throttle_duration_sec=0.5
                )
                if self.s2_err_x <= self.blue_target_distance_m:
                    self.get_logger().info(f'到达目标距离 {self.blue_target_distance_m:.2f} 米，停止前进')
                    self.set_state(self.BLIND_LEFT_SHIFT)
                    return

            if elapsed >= self.approach_blue_max_duration_sec:
                self.get_logger().info('寻找蓝球前进超时，进入盲走横向平移')
                self.set_state(self.BLIND_LEFT_SHIFT)
                return

            self.send_velocity_command(self.approach_blue_forward_speed, 0.0, 0.0, step_height=0.10)
            return

        if self.state == self.BLIND_LEFT_SHIFT:
            if elapsed >= self.blind_left_duration_sec:
                self.get_logger().info('5. 停止')
                self.set_state(self.DONE)
                return

            self.send_velocity_command(self.blind_left_vx, self.blind_left_vy, 0.0, step_height=0.10)
            return

        if self.state == self.DONE:
            if not self.done_stop_sent:
                self.send_stand_command()
                self.done_stop_sent = True
            return

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            self.Ctrl.quit()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Part1CombinedNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        sys.exit()


if __name__ == '__main__':
    main()
