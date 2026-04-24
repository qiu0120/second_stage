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
        self.declare_parameter('initial_state', 'STAGE1_CRUISE_BALL_AND_YELLOW')
        # =========================
        # 状态机状态说明
        # =========================
        # STAGE1_CRUISE_BALL_AND_YELLOW:
        #   第一阶段巡航；沿两排球中间前进，同时看橙球和黄线。
        #   发现满足条件的橙球时转入 BALL_LATERAL_ALIGN；
        #   黄线达到第一阶段阈值时转入 STAGE1_ROTATE_LEFT_90。
        #
        # STAGE1_ROTATE_LEFT_90:
        #   第一阶段结束后的左跳转向状态。
        #   当前实现中执行一次原地左跳，完成后直接进入 STAGE2_CRUISE_YELLOW_ONLY。
        #
        # STAGE2_CRUISE_YELLOW_ONLY:
        #   第二阶段巡航；只看黄线，不处理橙球。
        #   黄线达到第二阶段阈值时转入 STAGE2_ROTATE_LEFT_90。
        #
        # STAGE2_ROTATE_LEFT_90:
        #   第二阶段结束后的左跳转向状态。
        #   当前实现中执行一次原地左跳，完成后先进入
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
        #   当前实现中连续执行两次原地左跳，近似代替 180° 掉头，
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
        #   最终出口前的小角度修正状态。
        #   通过 TF 朝向闭环左转约 30°，完成后进入 DONE。
        #
        # BALL_LATERAL_ALIGN:
        #   撞球子状态 1：横向对齐球。
        #   采用“小 vx + 主 vy”边前进边横移，把目标橙球送到机器狗正前方。
        #
        # BALL_HIT_CONFIRM_FORWARD:
        #   撞球子状态 2：确认后直接前冲撞击。
        #   进入时记录目标球深度，用“球深 + 额外前冲距离”生成撞击距离，
        #   并用 TF 位移判断撞击是否完成。
        #
        # BALL_POST_HIT_SIDE_SHIFT:
        #   撞球子状态 3：撞完后只做左右横移。
        #   撞左球则向右移，撞右球则向左移。
        #   横移固定距离完成后直接回到保存的巡航状态 ball_return_state。
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
        self.declare_parameter('yellow_min_contour_area', 600.0)

        self.declare_parameter('yellow_min_width_height_ratio', 2.0)
        self.declare_parameter('yellow_max_tilt_deg', 30.0)
        self.declare_parameter('yellow_center_tolerance_ratio', 0.28)
        self.declare_parameter('yellow_min_width_ratio', 0.18)

        self.declare_parameter('yellow_stop_line_y_ratio_stage1', 1.0)
        self.declare_parameter('yellow_stop_line_y_ratio_stage2', 0.77)
        self.declare_parameter('yellow_stop_line_y_ratio_stage3', 0.8)
        self.declare_parameter('yellow_stop_confirm_count', 1)

        self.declare_parameter('yellow_ratio_scan', 0.6)
        self.declare_parameter('yellow_ratio_final', 0.9)

        # =========================
        # 巡航 / 中线
        # =========================
        self.declare_parameter('stage1_cruise_forward_speed', 0.30)
        self.declare_parameter('stage2_cruise_forward_speed', 0.40)
        self.declare_parameter('stage3_cruise_ball_only_speed', 0.30)
        self.declare_parameter('stage3_go_scan_speed', 0.40)
        self.declare_parameter('stage3_go_final_speed', 0.40)

        # 黄线预触发减速区：先减速，再真正触发切状态
        self.declare_parameter('yellow_slowdown_ratio_stage1', 0.90)
        self.declare_parameter('yellow_slowdown_ratio_stage2', 0.68)
        self.declare_parameter('yellow_slowdown_ratio_stage3', 0.70)
        self.declare_parameter('yellow_slowdown_ratio_scan', 0.52)
        self.declare_parameter('yellow_slowdown_ratio_final', 0.80)

        self.declare_parameter('stage1_yellow_slow_speed', 0.15)
        self.declare_parameter('stage2_yellow_slow_speed', 0.15)
        self.declare_parameter('stage3_yellow_slow_speed', 0.15)
        self.declare_parameter('stage3_go_scan_yellow_slow_speed', 0.15)
        self.declare_parameter('stage3_go_final_yellow_slow_speed', 0.15)

        self.declare_parameter('turn_trigger_distance_m', 0.45)

        self.declare_parameter('center_cruise_vy_gain', 0.25)
        self.declare_parameter('center_cruise_vy_max', 0.3)
        self.declare_parameter('center_ok_px', 18.0)

        # =========================
        # 对齐球阶段：小 vx + 主 vy
        # =========================
        self.declare_parameter('lateral_align_forward_speed', 0.115)
        self.declare_parameter('lateral_align_vy_gain', 0.30)
        self.declare_parameter('lateral_align_vy_max', 0.30)
        self.declare_parameter('lateral_align_vy_min', 0.10)
        self.declare_parameter('lateral_align_px_tol', 20.0)
        self.declare_parameter('lateral_align_confirm_count', 1)

        # =========================
        # 撞击 / 回退
        # =========================
        self.declare_parameter('hit_forward_speed', 0.10)
        self.declare_parameter('hit_extra_distance_m', 0.05)

        self.declare_parameter('backoff_speed', 0.10)
        self.declare_parameter('backoff_distance_m', 0.10)

        # 撞完后先按左右无条件偏一段（前半段快，后半段慢）
        self.declare_parameter('post_hit_side_shift_distance_m', 0.30)
        self.declare_parameter('post_hit_side_shift_speed_fast', 0.15)
        self.declare_parameter('post_hit_side_shift_speed_slow', 0.15)
        self.declare_parameter('post_hit_side_shift_slowdown_ratio', 0.20)

        # =========================
        # 防重复撞同一颗球
        # =========================
        self.declare_parameter('ball_retrigger_cooldown_sec', 1.2)
        self.declare_parameter('ball_retrigger_min_travel_m', 0.25)

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
        self.declare_parameter('rotate_left_30_wz', 0.30)

        # =========================
        # 第二段左跳后按仿真时间前进
        # =========================
        self.declare_parameter('stage2_forward_after_left_jump_speed', 0.3)
        self.declare_parameter('stage2_forward_after_left_jump_duration_sec', 2.0)

        # =========================
        # 读取参数
        # =========================
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

        self.stage1_cruise_forward_speed = float(self.get_parameter('stage1_cruise_forward_speed').value)
        self.stage2_cruise_forward_speed = float(self.get_parameter('stage2_cruise_forward_speed').value)
        self.stage3_cruise_ball_only_speed = float(self.get_parameter('stage3_cruise_ball_only_speed').value)
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

        self.lateral_align_forward_speed = float(self.get_parameter('lateral_align_forward_speed').value)
        self.lateral_align_vy_gain = float(self.get_parameter('lateral_align_vy_gain').value)
        self.lateral_align_vy_max = float(self.get_parameter('lateral_align_vy_max').value)
        self.lateral_align_vy_min = float(self.get_parameter('lateral_align_vy_min').value)
        self.lateral_align_px_tol = float(self.get_parameter('lateral_align_px_tol').value)
        self.lateral_align_confirm_count = int(self.get_parameter('lateral_align_confirm_count').value)

        self.hit_forward_speed = float(self.get_parameter('hit_forward_speed').value)
        self.hit_extra_distance_m = float(self.get_parameter('hit_extra_distance_m').value)

        self.backoff_speed = float(self.get_parameter('backoff_speed').value)
        self.backoff_distance_m = float(self.get_parameter('backoff_distance_m').value)

        self.post_hit_side_shift_distance_m = float(self.get_parameter('post_hit_side_shift_distance_m').value)
        self.post_hit_side_shift_speed_fast = float(self.get_parameter('post_hit_side_shift_speed_fast').value)
        self.post_hit_side_shift_speed_slow = float(self.get_parameter('post_hit_side_shift_speed_slow').value)
        self.post_hit_side_shift_slowdown_ratio = float(self.get_parameter('post_hit_side_shift_slowdown_ratio').value)


        self.ball_retrigger_cooldown_sec = float(self.get_parameter('ball_retrigger_cooldown_sec').value)
        self.ball_retrigger_min_travel_m = float(self.get_parameter('ball_retrigger_min_travel_m').value)



        self.rotate_left_90_tolerance_rad = float(self.get_parameter('rotate_left_90_tolerance_rad').value)
        self.rotate_left_90_confirm_count = int(self.get_parameter('rotate_left_90_confirm_count').value)
        self.rotate_left_90_wz = float(self.get_parameter('rotate_left_90_wz').value)

        self.rotate_back_180_tolerance_rad = float(self.get_parameter('rotate_back_180_tolerance_rad').value)
        self.rotate_back_180_confirm_count = int(self.get_parameter('rotate_back_180_confirm_count').value)
        self.rotate_back_180_wz = float(self.get_parameter('rotate_back_180_wz').value)

        self.rotate_left_30_tolerance_rad = float(self.get_parameter('rotate_left_30_tolerance_rad').value)
        self.rotate_left_30_confirm_count = int(self.get_parameter('rotate_left_30_confirm_count').value)
        self.rotate_left_30_wz = float(self.get_parameter('rotate_left_30_wz').value)

        self.stage2_forward_after_left_jump_speed = float(self.get_parameter('stage2_forward_after_left_jump_speed').value)
        self.stage2_forward_after_left_jump_duration_sec = float(self.get_parameter('stage2_forward_after_left_jump_duration_sec').value)

        # =========================
        # 控制接口
        # =========================
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

        self.latest_yellow_result = {
            'has_line': False,
            'line_bottom_y': None,
            'line_center': None,
            'img_shape': None,
        }

        self.state = self.initial_state
        self.ball_return_state = self.initial_state

        self.yellow_stop_counter = 0
        self.rotate_counter = 0

        # 第一阶段黄线“到底后出图”逻辑
        self.stage1_yellow_touched_bottom = False
        self.stage1_yellow_disappear_counter = 0

        self.pre_turn_pose: Optional[Tuple[float, float, float]] = None
        self.last_ball_done_time_sec: Optional[float] = None
        self.last_ball_done_pose: Optional[Tuple[float, float, float]] = None

        self.rotation_target_yaw: Optional[float] = None
        self.stage2_forward_after_left_jump_start_time_sec: Optional[float] = None


        self.lateral_align_counter = 0

        self.hit_start_pose: Optional[Tuple[float, float, float]] = None
        self.hit_start_depth_m: Optional[float] = None
        self.hit_forward_target_distance_m: float = 0.0

        self.post_hit_side_shift_start_pose: Optional[Tuple[float, float, float]] = None
        self.last_hit_side: Optional[str] = None
        self.side_shift_done: bool = False

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

    # ============================================================
    # 基础工具
    # ============================================================
    def planar_distance(self, pose0: Tuple[float, float, float], pose1: Tuple[float, float, float]) -> float:
        x0, y0, _ = pose0
        x1, y1, _ = pose1
        return math.hypot(x1 - x0, y1 - y0)

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
        self.Ctrl.Wait_finish(16, 0)
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

        self.latest_ball_result = self.detect_ball_scene(frame)
        self.latest_yellow_result = self.detect_yellow_stop_line(frame)

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

    # ============================================================
    # 状态切换
    # ============================================================
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

            if new_state == 'STAGE1_CRUISE_BALL_AND_YELLOW':
                self.stage1_yellow_touched_bottom = False
                self.stage1_yellow_disappear_counter = 0

            if new_state in (
                'STAGE1_ROTATE_LEFT_90',
                'STAGE2_ROTATE_LEFT_90',
                'STAGE3_ROTATE_BACK_180',
                'STAGE3_ROTATE_LEFT_30'
            ):
                self.rotate_counter = 0
                self.rotation_target_yaw = None

            if new_state == 'STAGE2_MOVE_FORWARD_AFTER_LEFT_JUMP_TIME':
                self.stage2_forward_after_left_jump_start_time_sec = None

            if new_state == 'BALL_LATERAL_ALIGN':
                self.lateral_align_counter = 0

            if new_state == 'BALL_HIT_CONFIRM_FORWARD':
                self.hit_start_pose = None
                self.hit_start_depth_m = None
                self.hit_forward_target_distance_m = 0.0

            if new_state == 'BALL_POST_HIT_SIDE_SHIFT':
                self.post_hit_side_shift_start_pose = None
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

    # ============================================================
    # 巡航中线
    # ============================================================
    def send_center_cruise_command(self, ball: Dict, vx: float):
        if ball['has_center_reference'] and ball['center_error_px'] is not None:
            err_norm = ball['center_error_px'] / max(self.rgb_w * 0.5, 1.0)
            vy = clamp(
                -self.center_cruise_vy_gain * err_norm,
                -self.center_cruise_vy_max,
                self.center_cruise_vy_max
            )
            self.send_velocity_command(vx, vy, 0.0)
        else:
            self.send_velocity_command(vx, 0.0, 0.0)

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
    # 球子链
    # ============================================================
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
            if target is None:
                self.send_stop_command()
                self.set_state('BALL_POST_HIT_SIDE_SHIFT')
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

        # 2) 直接撞击：TF 位移结束
        if self.state == 'BALL_HIT_CONFIRM_FORWARD':
            if self.hit_start_pose is None:
                self.hit_start_pose = (x, y, yaw)

                if target is not None and target['depth_m'] is not None:
                    self.hit_start_depth_m = target['depth_m']
                    self.last_hit_side = target['side']
                else:
                    self.hit_start_depth_m = 0.20
                    self.last_hit_side = None

                self.hit_start_depth_m = clamp(self.hit_start_depth_m, 0.10, 0.80)

            raw_hit_forward_target_distance_m = self.hit_start_depth_m + self.hit_extra_distance_m

            if raw_hit_forward_target_distance_m > 0.3:
                self.get_logger().warn(
                    f'BALL_HIT_CONFIRM_FORWARD abnormal target distance: '
                    f'depth={self.hit_start_depth_m:.3f}, '
                    f'raw_target={raw_hit_forward_target_distance_m:.3f} > 0.400, '
                    f'override to 0.13'
                )
                self.hit_forward_target_distance_m = 0.13
            else:
                self.hit_forward_target_distance_m = raw_hit_forward_target_distance_m

                self.get_logger().info(
                    f'BALL_HIT_CONFIRM_FORWARD start: '
                    f'hit_start_depth_m={self.hit_start_depth_m:.3f}, '
                    f'hit_forward_target_distance_m={self.hit_forward_target_distance_m:.3f}, '
                    f'last_hit_side={self.last_hit_side}'
                )

            current_pose = (x, y, yaw)
            dist = self.planar_distance(self.hit_start_pose, current_pose)

            self.get_logger().info(
                f'BALL_HIT_CONFIRM_FORWARD dist={dist:.3f}/{self.hit_forward_target_distance_m:.3f}',
                throttle_duration_sec=0.2
            )

            if dist >= self.hit_forward_target_distance_m:
                self.set_state('BALL_POST_HIT_SIDE_SHIFT')
                return True

            self.send_velocity_command(self.hit_forward_speed, 0.0, 0.0)
            return True

        # 3) 撞后左右移动：完成后直接回巡航
        if self.state == 'BALL_POST_HIT_SIDE_SHIFT':
            if self.post_hit_side_shift_start_pose is None:
                self.post_hit_side_shift_start_pose = (x, y, yaw)
                self.get_logger().info(
                    f'BALL_POST_HIT_SIDE_SHIFT start, last_hit_side={self.last_hit_side}'
                )

            current_pose = (x, y, yaw)
            side_shift_dist = self.planar_distance(self.post_hit_side_shift_start_pose, current_pose)

            self.get_logger().info(
                f'BALL_POST_HIT_SIDE_SHIFT dist={side_shift_dist:.3f}/{self.post_hit_side_shift_distance_m:.3f}',
                throttle_duration_sec=0.2
            )

            if side_shift_dist >= self.post_hit_side_shift_distance_m:
                self.finish_ball_task_and_return(x, y, yaw)
                return True

            progress_ratio = 0.0
            if self.post_hit_side_shift_distance_m > 1e-6:
                progress_ratio = side_shift_dist / self.post_hit_side_shift_distance_m

            if progress_ratio >= self.post_hit_side_shift_slowdown_ratio:
                side_shift_speed = self.post_hit_side_shift_speed_slow
            else:
                side_shift_speed = self.post_hit_side_shift_speed_fast

            self.get_logger().info(
                f'BALL_POST_HIT_SIDE_SHIFT speed={side_shift_speed:.3f} | '
                f'progress={progress_ratio:.2f} | '
                f'slowdown_ratio={self.post_hit_side_shift_slowdown_ratio:.2f}',
                throttle_duration_sec=0.3
            )

            if self.last_hit_side == 'left':
                self.send_velocity_command(0.0, -abs(side_shift_speed), 0.0)
                return True

            if self.last_hit_side == 'right':
                self.send_velocity_command(0.0, abs(side_shift_speed), 0.0)
                return True

            self.finish_ball_task_and_return(x, y, yaw)
            return True

        return False

    # ============================================================
    # 主循环
    # ============================================================
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
                f"has_ball={ball['has_ball']} | ball_center={ball['ball_center']} | "
                f"ball_depth={ball['ball_depth_m']} | ball_radius={ball['ball_radius']} | "
                f"ball_error_x={ball['error_x']} | has_center_ref={ball['has_center_reference']} | "
                f"center_error_px={ball['center_error_px']} | "
                f"left_ref={'Y' if ball['left_ref'] is not None else 'N'} | "
                f"right_ref={'Y' if ball['right_ref'] is not None else 'N'} | "
                f"orange_cnt={len(ball['orange_balls'])} | blue_cnt={len(ball['blue_balls'])} | "
                f"yellow_has_line={yellow['has_line']} | yellow_bottom={yellow['line_bottom_y']}",
                throttle_duration_sec=0.6
            )

        if self.handle_ball_subchain(x, y, yaw):
            return

        if self.state == 'STAGE1_ROTATE_LEFT_90':
            self.execute_left_jump_turn(
                jump_count=1,
                next_state='STAGE2_CRUISE_YELLOW_ONLY'
            )
            return

        if self.state == 'STAGE2_ROTATE_LEFT_90':
            self.execute_left_jump_turn(
                jump_count=1,
                next_state='STAGE2_MOVE_FORWARD_AFTER_LEFT_JUMP_TIME'
            )
            return

        if self.state == 'STAGE3_ROTATE_BACK_180':
            self.execute_left_jump_turn(
                jump_count=2,
                next_state='STAGE3_FINAL_DECISION'
            )
            return

        if self.state == 'STAGE3_ROTATE_LEFT_30':
            if self.handle_rotation_state(
                yaw,
                math.pi / 6.0,
                self.rotate_left_30_tolerance_rad,
                self.rotate_left_30_confirm_count,
                self.rotate_left_30_wz,
                'DONE'
            ):
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
            self.send_center_cruise_command(ball, vx)
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
            self.send_velocity_command(vx, 0.0, 0.0)
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
            self.send_center_cruise_command(ball, vx)
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
            self.send_center_cruise_command(ball, vx)
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
            self.send_center_cruise_command(ball, vx)
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
