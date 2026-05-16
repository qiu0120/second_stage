#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单独测试：第四赛段结束后的最终视觉矫正

用途：
  不跑完整四赛段状态机，只测试你现在加在第四赛段最后的
  GLOBAL_FINAL_P3_ALIGN 逻辑。

逻辑：
  1. 订阅 RGB 图像；
  2. 复用第三赛段结束 P3_ALIGN_TRACK 的黄线轨迹检测方法；
  3. 用近处中心 cx_n 控制 vy 横向居中；
  4. 用近处中心 cx_n 和远处中心 cx_f 的差控制 wz 朝向；
  5. vx 固定为 0，不向前走；
  6. OpenCV 窗口显示 near/far 中心、ROI、误差、当前命令。

运行：
  python3 test_p4_final_p3_align_vis.py

退出：
  Ctrl+C，会自动发送 STOP。
"""

import math
from typing import Optional

import cv2
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from second_stage.my_gait import Robot_Ctrl
from second_stage.robot_control_cmd_lcmt import robot_control_cmd_lcmt


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class P4FinalP3AlignTestNode(Node):
    def __init__(self):
        super().__init__('p4_final_p3_align_test_node')

        # 使用 Gazebo 仿真时间。
        try:
            self.declare_parameter('use_sim_time', True)
        except Exception:
            pass
        try:
            self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        except Exception as e:
            self.get_logger().warn(f'failed to set use_sim_time=True: {e}')

        # =========================
        # 基本参数
        # =========================
        self.declare_parameter('rgb_topic', '/rgb_camera/rgb_camera/image_raw')
        self.declare_parameter('control_hz', 30.0)
        self.declare_parameter('show_debug_vis', True)

        # 测试阶段是否先发送一次低身站立命令。
        self.declare_parameter('send_low_stand_on_start', True)

        # 找不到有效黄线轨迹时，保持和源代码 P3_ALIGN_TRACK 一样：
        # vx=p3_align_search_vx, vy=0, wz=p3_align_search_wz，边向前边搜索轨迹。
        self.declare_parameter('use_source_search_when_lost', True)

        # 进入对齐阈值后是否自动停住。
        self.declare_parameter('stop_when_aligned', False)
        self.declare_parameter('aligned_stable_frames', 5)

        # =========================
        # 第三赛段结束矫正参数：复用 P3_ALIGN_TRACK 逻辑
        # =========================
        self.declare_parameter('p3_stand_body_height', 0.20)
        self.declare_parameter('p3_stand_pitch', 0.30)
        self.declare_parameter('p3_step_height', 0.05)
        self.declare_parameter('p3_align_step_height', 0.10)

        self.declare_parameter('p3_align_lat_tol', 0.01)
        self.declare_parameter('p3_align_yaw_tol', 0.010)
        self.declare_parameter('p3_align_lat_gain', 0.5)
        self.declare_parameter('p3_align_yaw_gain', 2.0)
        self.declare_parameter('p3_align_lat_max', 0.15)
        self.declare_parameter('p3_align_yaw_max', 0.30)
        self.declare_parameter('p3_align_search_vx', 0.11)
        self.declare_parameter('p3_align_search_wz', 0.13)

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
        self.declare_parameter('p3_align_roi_left_ratio', 0.15)
        self.declare_parameter('p3_align_roi_right_ratio', 0.85)
        self.declare_parameter('p3_align_min_gap_px', 30)

        # =========================
        # 读取参数
        # =========================
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.control_hz = float(self.get_parameter('control_hz').value)
        self.show_debug_vis = bool(self.get_parameter('show_debug_vis').value)
        self.send_low_stand_on_start = bool(self.get_parameter('send_low_stand_on_start').value)
        self.use_source_search_when_lost = bool(self.get_parameter('use_source_search_when_lost').value)
        self.stop_when_aligned = bool(self.get_parameter('stop_when_aligned').value)
        self.aligned_stable_frames = int(self.get_parameter('aligned_stable_frames').value)

        self.p3_stand_body_height = float(self.get_parameter('p3_stand_body_height').value)
        self.p3_stand_pitch = float(self.get_parameter('p3_stand_pitch').value)
        self.p3_step_height = float(self.get_parameter('p3_step_height').value)
        self.p3_align_step_height = float(self.get_parameter('p3_align_step_height').value)

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
        # 运行缓存
        # =========================
        self.bridge = CvBridge()
        self.latest_bgr: Optional[np.ndarray] = None
        self.motion_cmd = (0.0, 0.0, 0.0)
        self.aligned_count = 0
        self.is_aligned_and_stopped = False

        self.p3_error_mid = 0.0
        self.p3_error_near = 0.0
        self.p3_s4_lat = 0.0
        self.p3_s4_yaw = 0.0
        self.p3_s4_valid = 0.0
        self.p3_align_near_center = -1.0
        self.p3_align_far_center = -1.0
        self.p3_latest_mask = None
        self.p3_latest_mask_mid = None
        self.p3_latest_mask_near = None

        # =========================
        # 控制接口
        # =========================
        self.Ctrl = Robot_Ctrl()
        self.Ctrl.run()
        self.msg = robot_control_cmd_lcmt()
        if not hasattr(self.msg, 'life_count'):
            self.msg.life_count = 0

        if self.send_low_stand_on_start:
            self.send_stand_low_command()

        # =========================
        # ROS IO
        # =========================
        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            qos_profile_sensor_data,
        )
        self.timer = self.create_timer(1.0 / self.control_hz, self.control_loop)

        self.get_logger().info(
            'P4 final P3-align test started. '
            'Control: vx=0, vy=p3_s4_lat, wz=p3_s4_yaw. '
            'No forward movement in this test.'
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _inc_life_count(self):
        self.msg.life_count += 1
        if self.msg.life_count > 127:
            self.msg.life_count = 0

    def send_stand_low_command(self):
        self.msg.mode = 12
        self.msg.gait_id = 0
        self._inc_life_count()
        self.msg.rpy_des = [0.0, self.p3_stand_pitch, 0.0]
        self.msg.pos_des = [0.0, 0.0, self.p3_stand_body_height]
        self.Ctrl.Send_cmd(self.msg)
        self.get_logger().info(
            f'[CMD] LOW STAND body_height={self.p3_stand_body_height:.3f}, pitch={self.p3_stand_pitch:.3f}',
            throttle_duration_sec=1.0,
        )

    def send_stop_command(self):
        self.motion_cmd = (0.0, 0.0, 0.0)
        self.msg.mode = 12
        self.msg.gait_id = 0
        self._inc_life_count()
        self.msg.vel_des = [0.0, 0.0, 0.0]
        self.Ctrl.Send_cmd(self.msg)
        try:
            self.Ctrl.Wait_finish(12, 0)
        except Exception:
            pass
        self.get_logger().info('[CMD] STOP', throttle_duration_sec=1.0)

    def send_velocity_command(self, vx: float, vy: float, wz: float, step_height: Optional[float] = None):
        h = self.p3_align_step_height if step_height is None else float(step_height)
        vx = float(vx)
        vy = float(vy)
        wz = float(wz)
        self.motion_cmd = (vx, vy, wz)

        self.msg.mode = 11
        self.msg.gait_id = 3
        self._inc_life_count()
        self.msg.vel_des = [vx, vy, wz]
        self.msg.step_height = [h, h]
        self.msg.rpy_des = [0.0, self.p3_stand_pitch, 0.0]
        self.msg.pos_des = [0.0, 0.0, self.p3_stand_body_height]
        self.Ctrl.Send_cmd(self.msg)
        self.get_logger().info(
            f'[CMD] vel_des=[{vx:.3f}, {vy:.3f}, {wz:.3f}], step_height={h:.3f}',
            throttle_duration_sec=0.3,
        )

    def rgb_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge convert failed: {e}')
            return

        self.latest_bgr = frame
        self.process_yellow_track(frame)
        if self.show_debug_vis:
            self.show_debug_window(frame)

    def process_yellow_track(self, frame: np.ndarray):
        """
        复用第三赛段结束 P3_ALIGN_TRACK 的视觉计算：
          p3_s4_lat = (width / 2 - near_center) / (width / 2)
          p3_s4_yaw = (near_center - far_center) / (width / 2)
        """
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array(
            [self.p3_yellow_h_min, self.p3_yellow_s_min, self.p3_yellow_v_min],
            dtype=np.uint8,
        )
        upper_yellow = np.array(
            [self.p3_yellow_h_max, self.p3_yellow_s_max, self.p3_yellow_v_max],
            dtype=np.uint8,
        )
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

        # 保留第三赛段巡航误差的显示，但本测试不用它控制。
        self.p3_error_mid = self._moment_center_error(mask_mid, width)
        self.p3_error_near = self._moment_center_error(mask_near, width)

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

    @staticmethod
    def _moment_center_error(mask_roi: np.ndarray, width: int) -> float:
        m = cv2.moments(mask_roi)
        if m['m00'] <= 0:
            return 0.0
        cx = int(m['m10'] / m['m00'])
        dist = abs(cx - width / 2.0)
        force = ((width / 2.0 - dist) / (width / 2.0)) ** 3
        return float(force) if cx > width / 2.0 else -float(force)

    def control_loop(self):
        if self.latest_bgr is None:
            self.send_velocity_command(0.0, 0.0, 0.0)
            return

        if self.is_aligned_and_stopped:
            self.send_velocity_command(0.0, 0.0, 0.0)
            return

        if self.p3_s4_valid > 0.5:
            err_lat = self.p3_s4_lat
            err_yaw = self.p3_s4_yaw

            aligned = (
                abs(err_lat) < self.p3_align_lat_tol
                and abs(err_yaw) < self.p3_align_yaw_tol
            )
            if aligned:
                self.aligned_count += 1
            else:
                self.aligned_count = 0

            lateral_speed = clamp(
                err_lat * self.p3_align_lat_gain,
                -self.p3_align_lat_max,
                self.p3_align_lat_max,
            )
            turn_speed = clamp(
                err_yaw * self.p3_align_yaw_gain,
                -self.p3_align_yaw_max,
                self.p3_align_yaw_max,
            )

            if self.stop_when_aligned and self.aligned_count >= self.aligned_stable_frames:
                self.is_aligned_and_stopped = True
                self.send_velocity_command(0.0, 0.0, 0.0)
                self.get_logger().info(
                    f'[ALIGN DONE] stable_frames={self.aligned_count}, '
                    f'lat={err_lat:.4f}, yaw={err_yaw:.4f}'
                )
                return

            # 重点：第四赛段结束测试，不允许向前走，所以 vx 固定为 0。
            self.send_velocity_command(0.0, lateral_speed, turn_speed)
            self.get_logger().info(
                f'[ALIGN] valid lat={err_lat:.4f}, yaw={err_yaw:.4f}, '
                f'cmd=(0.000,{lateral_speed:.3f},{turn_speed:.3f}), '
                f'aligned_count={self.aligned_count}/{self.aligned_stable_frames}',
                throttle_duration_sec=0.5,
            )
        else:
            if self.use_source_search_when_lost:
                # 和源代码 P3_ALIGN_TRACK 保持一致：找不到有效轨迹时 vx + wz 搜索。
                self.send_velocity_command(self.p3_align_search_vx, 0.0, self.p3_align_search_wz)
                self.get_logger().info(
                    f'[SEARCH] no valid track, source search: '
                    f'cmd=({self.p3_align_search_vx:.3f},0.000,{self.p3_align_search_wz:.3f})',
                    throttle_duration_sec=0.5,
                )
            else:
                self.send_velocity_command(0.0, 0.0, 0.0)
                self.get_logger().info('[SEARCH] no valid track, hold still', throttle_duration_sec=0.5)

    def show_debug_window(self, frame: np.ndarray):
        try:
            vis = frame.copy()
            height, width = vis.shape[:2]

            crop_left = int(width * self.p3_crop_left_ratio)
            crop_right = int(width * self.p3_crop_right_ratio)
            mid_top = int(height * self.p3_mid_top_ratio)
            mid_bottom = int(height * self.p3_mid_bottom_ratio)
            near_top = int(height * self.p3_near_top_ratio)
            near_bottom = int(height * self.p3_near_bottom_ratio)
            near_y = int(height * self.p3_align_near_y_ratio)
            far_y = int(height * self.p3_align_far_y_ratio)
            roi_left = int(width * self.p3_align_roi_left_ratio)
            roi_right = int(width * self.p3_align_roi_right_ratio)

            vx, vy, wz = self.motion_cmd

            cv2.putText(vis, 'P4 FINAL P3 ALIGN TEST', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(vis, f's4_valid={self.p3_s4_valid:.1f} lat={self.p3_s4_lat:.4f} yaw={self.p3_s4_yaw:.4f}', (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(vis, f'cmd vx={vx:.3f} vy={vy:.3f} wz={wz:.3f}', (10, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(vis, f'tol lat={self.p3_align_lat_tol:.3f} yaw={self.p3_align_yaw_tol:.3f} stable={self.aligned_count}/{self.aligned_stable_frames}', (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(vis, 'vx is forced to 0.0 in this test', (10, 133), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)

            # 图像中心线
            cv2.line(vis, (width // 2, 0), (width // 2, height), (255, 255, 255), 1)

            # HSV crop 边界
            cv2.line(vis, (crop_left, 0), (crop_left, height), (0, 255, 255), 2)
            cv2.line(vis, (crop_right, 0), (crop_right, height), (0, 255, 255), 2)

            # mid / near mask 区域
            cv2.rectangle(vis, (0, mid_top), (width - 1, mid_bottom), (255, 0, 0), 1)
            cv2.rectangle(vis, (0, near_top), (width - 1, near_bottom), (0, 180, 255), 1)

            # P3_ALIGN_TRACK 用的 near/far 行和 ROI
            cv2.line(vis, (0, near_y), (width, near_y), (255, 255, 0), 2)
            cv2.line(vis, (0, far_y), (width, far_y), (255, 255, 0), 2)
            cv2.line(vis, (roi_left, 0), (roi_left, height), (255, 0, 255), 2)
            cv2.line(vis, (roi_right, 0), (roi_right, height), (255, 0, 255), 2)

            # near/far 中心点和连线
            if self.p3_align_near_center != -1:
                n = int(self.p3_align_near_center)
                cv2.circle(vis, (n, near_y), 7, (0, 0, 255), -1)
                cv2.putText(vis, f'near={n}', (n + 8, max(20, near_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if self.p3_align_far_center != -1:
                f = int(self.p3_align_far_center)
                cv2.circle(vis, (f, far_y), 7, (0, 0, 255), -1)
                cv2.putText(vis, f'far={f}', (f + 8, max(20, far_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if self.p3_align_near_center != -1 and self.p3_align_far_center != -1:
                cv2.line(
                    vis,
                    (int(self.p3_align_near_center), near_y),
                    (int(self.p3_align_far_center), far_y),
                    (0, 0, 255),
                    3,
                )

            cv2.imshow('p4_final_p3_align_origin', vis)
            if self.p3_latest_mask_mid is not None:
                cv2.imshow('p4_final_p3_align_mask_mid', self.p3_latest_mask_mid)
            if self.p3_latest_mask_near is not None:
                cv2.imshow('p4_final_p3_align_mask_near', self.p3_latest_mask_near)
            if self.p3_latest_mask is not None:
                cv2.imshow('p4_final_p3_align_mask_all', self.p3_latest_mask)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f'show_debug_window failed: {e}', throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = P4FinalP3AlignTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down, sending stop command...')
        try:
            node.send_stop_command()
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
