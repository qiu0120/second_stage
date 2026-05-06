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

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

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
        self.declare_parameter('initial_state', 'P1_STAND_WAIT')
        self.declare_parameter('second_stage_initial_state', 'STAGE1_CRUISE_BALL_AND_YELLOW')

        # OpenCV 可视化窗口：只用于调试，不参与控制逻辑
        self.declare_parameter('show_debug_vis', True)
        self.declare_parameter('show_yellow_mask', False)

        # =========================
        # 第一赛段参数（全部加 p1_ 前缀，避免和第二赛段变量冲突）
        # =========================
        self.declare_parameter('p1_stand_wait_sec', 0)
        self.declare_parameter('p1_stand_body_height', 0.28)

        self.declare_parameter('p1_stage1_max_duration_sec', 8.0)
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
        self.declare_parameter('p1_turn_yaw_vel', 0.50)

        self.declare_parameter('p1_blue_target_distance_m', 0.25)
        self.declare_parameter('p1_approach_blue_max_duration_sec', 6.0)
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
        self.declare_parameter('p1_stop_yellow_pixel_threshold', 1500)

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

        self.declare_parameter('yellow_min_width_height_ratio', 2.0)
        self.declare_parameter('yellow_max_tilt_deg', 30.0)
        self.declare_parameter('yellow_center_tolerance_ratio', 0.28)
        self.declare_parameter('yellow_min_width_ratio', 0.18)

        self.declare_parameter('yellow_stop_line_y_ratio_stage1', 1.0)
        self.declare_parameter('yellow_stop_line_y_ratio_stage2', 0.70)
        self.declare_parameter('yellow_stop_line_y_ratio_stage3', 0.8)
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
        self.declare_parameter('stage3_go_scan_speed', 0.40)
        self.declare_parameter('stage3_go_final_speed', 0.40)

        # 黄线预触发减速区：先减速，再真正触发切状态
        self.declare_parameter('yellow_slowdown_ratio_stage1', 0.90)
        self.declare_parameter('yellow_slowdown_ratio_stage2', 0.62)
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
        self.declare_parameter('center_cruise_vy_max', 0.3)    # 保留兼容，当前不再使用
        self.declare_parameter('center_ok_px', 10.0)
        self.declare_parameter('center_cruise_fixed_vy', 0.10)
        # 左右参考球深度差太大时，不再按两球图像中点做中线对齐，
        # 而是向距离更远的小球一侧给一个较小固定 vy。
        self.declare_parameter('center_depth_diff_disable_align_m', 0.50)
        self.declare_parameter('center_far_side_fixed_vy', 0.03)

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
        # 撞击 / 撞后移动
        # =========================
        # 撞击前冲：按仿真时间结束，不再用 TF 距离和 hit_extra_distance_m。
        self.declare_parameter('hit_forward_speed', 0.10)
        self.declare_parameter('hit_forward_duration_sec', 0.75)

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
        self.declare_parameter('p3_align_lat_tol', 0.08)
        self.declare_parameter('p3_align_yaw_tol', 0.08)
        self.declare_parameter('p3_align_lat_gain', 0.4)
        self.declare_parameter('p3_align_yaw_gain', 0.8)
        self.declare_parameter('p3_align_lat_max', 0.15)
        self.declare_parameter('p3_align_yaw_max', 0.30)
        self.declare_parameter('p3_align_search_vx', 0.10)
        self.declare_parameter('p3_align_search_wz', 0.10)

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

        self.stage2_forward_after_left_jump_speed = float(self.get_parameter('stage2_forward_after_left_jump_speed').value)
        self.stage2_forward_after_left_jump_duration_sec = float(self.get_parameter('stage2_forward_after_left_jump_duration_sec').value)

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

        self.lateral_align_counter = 0

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
            self.send_velocity_command(0.0, 0.0, 0.0, step_height=0.13)
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
            self.send_velocity_command(0.0, 0.0, turn_speed, step_height=0.13)
            return

        if self.state == 'P1_TURN_LEFT_TO_STAGE2':
            if elapsed >= self.p1_turn_duration_sec:
                self.get_logger().info('[P1] 左转结束，开始寻找蓝球并前进')
                self.set_state('P1_APPROACH_BLUE_BALL')
                return
            self.send_velocity_command(self.p1_turn_forward_vel, 0.0, self.p1_turn_yaw_vel, step_height=0.13)
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

            if new_state == 'STAGE3_ROTATE_LEFT_30':
                self.stage3_final_left_shift_start_time_sec = None

            if new_state == 'STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT':
                self.stage3_final_rotate_start_time_sec = None

            if new_state == 'STAGE2_MOVE_FORWARD_AFTER_LEFT_JUMP_TIME':
                self.stage2_forward_after_left_jump_start_time_sec = None

            if new_state == 'BALL_LATERAL_ALIGN':
                self.lateral_align_counter = 0
                # 锁定“开始对齐时”的目标球所在侧。
                # 后面对齐过程中目标球可能因为机器人横移跑到画面另一边，
                # 撞后横移方向仍然使用这里锁定的初始 side，不再在撞击前冲时覆盖。
                target = self.latest_ball_result.get('best_target_ball') if isinstance(self.latest_ball_result, dict) else None
                if target is not None:
                    self.last_hit_side = target.get('side')
                    self.get_logger().info(
                        f'BALL_LATERAL_ALIGN lock hit side at align start: '
                        f'last_hit_side={self.last_hit_side}, '
                        f'target_center={target.get("center")}, '
                        f'error_x={target.get("error_x")}, '
                        f'depth={target.get("depth_m")}'
                    )
                else:
                    self.last_hit_side = None
                    self.get_logger().warn('BALL_LATERAL_ALIGN start but target is None; last_hit_side=None')

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
            cv2.putText(vis, f'orange_hit_count={self.orange_hit_count}', (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

            # 画黄色 ROI
            roi_top = int(h * self.yellow_roi_top_ratio)
            roi_left = int(w * self.yellow_roi_left_ratio)
            roi_right = int(w * self.yellow_roi_right_ratio)
            roi_top = max(0, min(h - 1, roi_top))
            roi_left = max(0, min(w - 1, roi_left))
            roi_right = max(roi_left + 1, min(w, roi_right))
            cv2.rectangle(vis, (roi_left, roi_top), (roi_right, h - 1), (0, 255, 255), 1)
            cv2.putText(vis, 'yellow ROI', (roi_left + 3, max(18, roi_top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

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
            cv2.putText(vis, f'err_mid={self.p3_error_mid:.3f} err_near={self.p3_error_near:.3f}', (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(vis, f's4_valid={self.p3_s4_valid:.1f} lat={self.p3_s4_lat:.3f} yaw={self.p3_s4_yaw:.3f}', (10, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

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
                self.p3_send_velocity_command(self.p3_align_search_vx, 0.0, self.p3_align_search_wz, step_height=self.p3_align_step_height)
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
            self.send_velocity_command(vx, 0.0, wz)
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

def main(args=None):
    rclpy.init(args=args)
    node = CombinedStage1Stage2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down, sending stop command...')
        node.send_stop_command()
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
