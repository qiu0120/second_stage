#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import List

import cv2
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data

from cyberdog_msg.msg import YamlParam, ApplyForce

from second_stage.my_gait import Robot_Ctrl
from second_stage.robot_control_cmd_lcmt import robot_control_cmd_lcmt


class ControlParameterValueKind:
    kDOUBLE = 1
    kS64 = 2
    kVEC_X_DOUBLE = 3
    kMAT_X_DOUBLE = 4


class FifthStageBridgeNode(Node):
    """
    第五赛段：孤梁稳渡 单独调试节点

    当前版本：
    1. 使用状态机；
    2. P5_STEP_UP 使用仿真时间；
    3. P5_UP_SLOPE 使用右侧赛道黄线消失作为结束标志；
       右侧赛道黄线连续 lost N 帧后，不再左跳/恢复/右移，
       先用固定速度运行一段时间完成转向/位置调整，
       再设置右斜坡 body，然后直接进入右斜坡 1；
    4. P5_RIGHT_SLOPE_1 / 2 使用中间区域黄色消失作为提前结束提示；
       中间区域黄色消失后不立刻 stop，而是继续前进固定时间，然后直接衔接转向；
       P5_RIGHT_SLOPE_3 检测到中间黄色消失后，不再继续额外前进，
       先发送一次速度为 0 的速度命令，再进入 reset body / 右跳转向准备流程，然后执行右跳动作，
       最后固定时间前进，到时 stop 后进入离坡跳跃；
    5. 上坡/离坡跳跃等动作类状态使用 Ctrl.Wait_finish(mode, gait_id) 等待完成；
       右斜路段 1/2 之间的转向使用速度控制；右斜坡 3 结束后使用右跳动作离开/调整方向；
       离坡跳跃恢复站立后，最后再执行一次 16,6 跳远动作；
    6. 加入 OpenCV 可视化窗口；
    7. 结构尽量接近前四赛段整合代码，方便后续改成 FifthStageMixin。
    """

    # ============================================================
    # 第五赛段状态定义
    # ============================================================
    P5_RECOVERY_STAND = 'P5_RECOVERY_STAND'
    P5_SET_BODY_NORMAL = 'P5_SET_BODY_NORMAL'

    P5_STEP_UP = 'P5_STEP_UP'
    P5_UP_SLOPE = 'P5_UP_SLOPE'
    P5_AFTER_UP_SLOPE_VELOCITY_CONTROL = 'P5_AFTER_UP_SLOPE_VELOCITY_CONTROL'
    P5_SET_RIGHT_SLOPE_BODY = 'P5_SET_RIGHT_SLOPE_BODY'
    P5_RIGHT_SLOPE_1 = 'P5_RIGHT_SLOPE_1'
    P5_RIGHT_SLOPE_1_FORWARD_AFTER_CENTER_LOST = 'P5_RIGHT_SLOPE_1_FORWARD_AFTER_CENTER_LOST'
    P5_TURN_1 = 'P5_TURN_1'
    P5_RECOVERY_AFTER_TURN_1 = 'P5_RECOVERY_AFTER_TURN_1'

    P5_RIGHT_SLOPE_2 = 'P5_RIGHT_SLOPE_2'
    P5_RIGHT_SLOPE_2_FORWARD_AFTER_CENTER_LOST = 'P5_RIGHT_SLOPE_2_FORWARD_AFTER_CENTER_LOST'
    P5_TURN_2 = 'P5_TURN_2'
    P5_RECOVERY_AFTER_TURN_2 = 'P5_RECOVERY_AFTER_TURN_2'

    P5_RIGHT_SLOPE_3 = 'P5_RIGHT_SLOPE_3'
    P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP = 'P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP'
    P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP_2 = 'P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP_2'
    P5_RIGHT_JUMP_AFTER_RESET_BODY = 'P5_RIGHT_JUMP_AFTER_RESET_BODY'

    # 第三个转角右跳后：先前进并矫正；矫正完成后再固定前进，不再矫正。
    P5_FORWARD_AFTER_RESET_BODY = 'P5_FORWARD_AFTER_RESET_BODY'
    P5_FORWARD_NO_ALIGN_AFTER_RESET_BODY = 'P5_FORWARD_NO_ALIGN_AFTER_RESET_BODY'

    P5_JUMP_EXIT_SLOPE = 'P5_JUMP_EXIT_SLOPE'
    P5_RECOVERY_AFTER_JUMP_2 = 'P5_RECOVERY_AFTER_JUMP_2'
    P5_FINAL_LONG_JUMP = 'P5_FINAL_LONG_JUMP'
    P5_RESET_BODY = 'P5_RESET_BODY'

    P5_DONE = 'P5_DONE'

    def __init__(self):
        super().__init__('fifth_stage_bridge_node')

        # 强制使用 Gazebo /clock 仿真时间。
        # 注意：declare_parameter 的默认值可能被外部 launch/yaml/命令行覆盖，
        # 所以这里声明后再 set_parameters 强制设为 True，避免单独调试节点意外退回现实时间。
        try:
            self.declare_parameter('use_sim_time', True)
        except Exception as e:
            self.get_logger().warn(f'[P5_TIME] declare use_sim_time failed: {e}')

        try:
            self.set_parameters([
                Parameter('use_sim_time', Parameter.Type.BOOL, True)
            ])
            self.get_logger().warn(
                f'[P5_TIME] force use_sim_time={self.get_parameter("use_sim_time").value}'
            )
        except Exception as e:
            self.get_logger().error(f'[P5_TIME] force use_sim_time failed: {e}')

        self.Ctrl = Robot_Ctrl()
        self.Ctrl.run()

        self.msg = robot_control_cmd_lcmt()
        self.msg.life_count = 0
        self.msg.contact = 0

        self.bridge = CvBridge()
        self.latest_bgr = None
        self.latest_frame_seq = 0
        self.state_enter_frame_seq = 0

        self.latest_p5_yellow_result = {
            'has_line': False,
            'line_bottom_y': None,
            'line_center': None,
            'img_shape': None,
            'angle_deg': None,
            'abs_tilt_deg': None,
            'bbox': None,
            'width_ratio': None,
            'wh_ratio': None,
        }
        self.latest_p5_yellow_mask = None
        self.latest_p5_yellow_roi = None
        self.p5_yellow_stop_counter = 0

        # P5_RIGHT_SLOPE_1/2/3 专用：中间区域黄色存在/消失检测。
        # 右斜坡阶段相机画面会倾斜且抖动，不再强行识别“横向黄线形状”，
        # 只判断图像中间 ROI 内是否还有足够黄色像素。
        self.p5_center_yellow_absent_counter = 0
        self.latest_p5_center_yellow_presence_result = {
            'has_yellow': False,
            'yellow_pixels': 0,
            'roi_pixels': 0,
            'yellow_ratio': 0.0,
            'bbox': None,
            'roi': None,
            'img_shape': None,
            'reason': 'init',
        }

        # P5_UP_SLOPE 专用：右侧赛道黄线检测。
        # 注意：只有 bottom_ratio 靠近图像底部的黄线，才算“右侧赛道黄线还存在”。
        self.p5_right_side_yellow_lost_counter = 0
        self.latest_p5_right_side_yellow_result = {
            'has_line': False,
            'valid_bottom': False,
            'bbox': None,
            'center': None,
            'bottom_y': None,
            'bottom_ratio': None,
            'area': None,
            'height': None,
            'width': None,
            'roi': None,
            'candidates': [],
            'img_shape': None,
            'reason': 'init',
        }

        # P5_UP_SLOPE 专用：左右内侧黄线边缘检测，用于上坡角度/居中修正。
        self.latest_p5_inner_edge_result = {
            'mask': None,
            'left_roi': None,
            'right_roi': None,
            'left_edge': None,
            'right_edge': None,
            'has_left': False,
            'has_right': False,
            'has_both': False,
            'center_error': None,
            'heading_error': None,
            'common_valid': False,
            'common_reason': 'init',
            'cmd_vy_correction': 0.0,
            'cmd_wz_correction': 0.0,
        }

        # P5_RIGHT_SLOPE_1/2/3 专用：右侧黄线内侧边缘检测，用于三档 vy 修正。
        # 只检测右侧黄线；right_inner_x 太靠中间则加大 vy，太靠右则减小 vy。
        self.latest_p5_right_slope_right_edge_result = {
            'mask': None,
            'roi': None,
            'raw_points': [],
            'points': [],
            'valid': False,
            'reason': 'init',
            'right_inner_x': None,
            'right_inner_x_ratio': None,
            'too_center': False,
            'too_right': False,
            'cmd_vy': None,
            'base_vy': None,
            'action': 'init',
            'lost_extra_active': False,
            'lost_extra_direction': 'none',
            'too_center_count': 0,
            'too_right_count': 0,
            'record_ignore_active': False,
            'record_ignore_elapsed_s': 0.0,
            'record_ignore_duration_s': 0.0,
        }

        # P5_RIGHT_SLOPE_1/2/3 专用：
        # 右侧黄线危险趋势记忆。当前右斜坡段内，如果前面连续 too_center/too_right，
        # 后面即使右侧黄线无效/识别不到，也继续按最后确认的危险方向给额外 vy。
        # 注意：进入每一段新的 P5_RIGHT_SLOPE_x 时会清零，不跨右斜坡段继承。
        self.p5_right_slope_too_center_count = 0
        self.p5_right_slope_too_right_count = 0
        self.p5_right_slope_lost_extra_active = False
        self.p5_right_slope_lost_extra_direction = 'none'  # none / too_center / too_right

        # P5_FORWARD_AFTER_RESET_BODY 专用：右跳后前进并矫正；时间到仍未对齐则 vx=0 继续矫正。
        self.p5_forward_align_stable_counter = 0

        self.para_pub = self.create_publisher(YamlParam, 'yaml_parameter', 10)
        self.force_pub = self.create_publisher(ApplyForce, 'apply_force', 10)

        self._declare_params()
        self._load_params()

        self.rgb_sub = self.create_subscription(
            Image,
            self.p5_rgb_topic,
            self.rgb_callback,
            qos_profile_sensor_data
        )

        self.state = self.p5_initial_state
        self.state_start_time = self.now_sec()
        # 阻塞动作 / STOP 后切入新状态时，下一次真正运行该状态再重新开始计时。
        # 避免 Ctrl.Wait_finish 阻塞期间 /clock 未刷新，导致新状态一进入 elapsed 就偏大。
        self.state_timer_reset_after_blocking_wait = False
        self.action_sent = False

        self.timer = self.create_timer(self.control_period_s, self.control_loop)

        self.get_logger().info(
            f'[P5] fifth stage bridge node started, '
            f'initial_state={self.state}, '
            f'use_sim_time={self.get_parameter("use_sim_time").value}, '
            f'rgb_topic={self.p5_rgb_topic}'
        )


    # ============================================================
    # 参数声明
    # ============================================================
    def _declare_params(self):
        self.declare_parameter('p5_initial_state', self.P5_RECOVERY_STAND)
        self.declare_parameter('p5_control_period_s', 0.02)

        # RGB / 可视化
        self.declare_parameter('p5_rgb_topic', '/rgb_camera/rgb_camera/image_raw')
        self.declare_parameter('p5_show_debug_vis', True)
        # 可视化详细程度：
        # 0 = 不显示窗口；1 = 简洁模式，只显示关键控制信息；2 = 详细模式，显示所有调试点/ROI/原因。
        self.declare_parameter('p5_debug_vis_detail_level', 2)
        self.declare_parameter('p5_show_yellow_mask', False)

        # 启动时先设置 normal body
        self.declare_parameter('p5_body_normal_roll', 0.0)
        self.declare_parameter('p5_body_normal_height', 0.25)
        self.declare_parameter('p5_body_normal_wait_s', 0.3)

        # 第五赛段上台阶步态：这一段仍然按仿真时间
        self.declare_parameter('p5_step_up_vx', 0.40)
        self.declare_parameter('p5_step_up_vy', 0.0)
        self.declare_parameter('p5_step_up_wz', 0.0)
        self.declare_parameter('p5_step_up_step_height', 0.13)
        self.declare_parameter('p5_step_up_duration_s', 2.0)

        # 第五赛段上坡路段步态：视觉黄线结束
        self.declare_parameter('p5_up_slope_vx', 0.45)
        self.declare_parameter('p5_up_slope_vy', 0.0)
        self.declare_parameter('p5_up_slope_wz', 0.0)
        self.declare_parameter('p5_up_slope_step_height', 0.10)
        self.declare_parameter('p5_up_slope_roll', 0.0)
        self.declare_parameter('p5_up_slope_pitch', 0.20)

        # P5 上坡阶段：左右内侧黄线边缘矫正
        # 只在 P5_UP_SLOPE 中使用；右侧黄线消失仍然负责上坡结束。
        self.declare_parameter('p5_inner_edge_align_enabled', True)
        self.declare_parameter('p5_inner_edge_enable_vy', True)
        self.declare_parameter('p5_inner_edge_enable_wz', True)

        # ROI：左右下方区域。左 ROI 取黄色区域最右边缘，右 ROI 取黄色区域最左边缘。
        self.declare_parameter('p5_inner_edge_left_roi_x_min', 0.05)
        self.declare_parameter('p5_inner_edge_left_roi_x_max', 0.48)
        self.declare_parameter('p5_inner_edge_right_roi_x_min', 0.52)
        self.declare_parameter('p5_inner_edge_right_roi_x_max', 0.95)
        self.declare_parameter('p5_inner_edge_roi_y_min', 0.60)
        self.declare_parameter('p5_inner_edge_roi_y_max', 1.00)

        # 内侧边缘有效性判断
        self.declare_parameter('p5_inner_edge_min_points', 80)
        self.declare_parameter('p5_inner_edge_min_y_span', 80.0)
        self.declare_parameter('p5_inner_edge_bottom_min_ratio', 0.95)
        self.declare_parameter('p5_inner_edge_x_std_max', 90.0)
        self.declare_parameter('p5_inner_edge_use_bottom_connected_segment', True)
        self.declare_parameter('p5_inner_edge_max_y_gap', 8)
        self.declare_parameter('p5_inner_edge_min_common_y_span', 50.0)
        self.declare_parameter('p5_inner_edge_row_step', 1)
        self.declare_parameter('p5_inner_edge_top_bottom_band_ratio', 0.20)

        # 控制增益：center_error 像素 -> vy；heading_error 像素 -> wz。
        # 当前默认符号：center_error>0 表示赛道中心在图像右侧，给负 vy 向右修正；
        # heading_error>0 表示远端中心偏右，给负 wz 右转修正。
        self.declare_parameter('p5_inner_edge_center_k_vy', 0.0012)
        self.declare_parameter('p5_inner_edge_heading_k_wz', 0.0020)
        self.declare_parameter('p5_inner_edge_vy_max_correction', 0.08)
        self.declare_parameter('p5_inner_edge_wz_max_correction', 0.18)
        self.declare_parameter('p5_inner_edge_center_deadband_px', 8.0)
        self.declare_parameter('p5_inner_edge_heading_deadband_px', 6.0)

        # P5 上坡阶段：右侧赛道黄线消失判定
        self.declare_parameter('p5_right_side_yellow_roi_x_min', 0.50)
        self.declare_parameter('p5_right_side_yellow_roi_x_max', 1.00)
        self.declare_parameter('p5_right_side_yellow_roi_y_min', 0.50)
        self.declare_parameter('p5_right_side_yellow_roi_y_max', 1.00)

        # 右侧黄线检测不需要太严格，但必须接近图像底部才算有效
        self.declare_parameter('p5_right_side_yellow_min_area', 80.0)
        self.declare_parameter('p5_right_side_yellow_min_height', 20)
        self.declare_parameter('p5_right_side_yellow_min_width', 2)
        self.declare_parameter('p5_right_side_yellow_bottom_valid_ratio', 0.95)

        # 上坡黄线消失后的固定速度控制段
        self.declare_parameter('p5_after_up_slope_control_duration_s', 2.8)
        self.declare_parameter('p5_after_up_slope_control_vx', 0.075)
        self.declare_parameter('p5_after_up_slope_control_vy', 0.00)
        self.declare_parameter('p5_after_up_slope_control_wz', 0.60)
        self.declare_parameter('p5_after_up_slope_control_step_height', 0.04)

        # 右侧赛道黄线连续消失 N 帧后，不再左跳/恢复/右移，
        # 先固定速度运行一段时间完成转向/位置调整，再设置右斜坡 body。
        self.declare_parameter('p5_right_side_yellow_lost_confirm_count', 1)
        self.declare_parameter('p5_right_side_yellow_ignore_after_enter_s', 0.5)

        # 右斜坡 body
        self.declare_parameter('p5_right_slope_roll', -0.60)
        self.declare_parameter('p5_right_slope_height', 0.25)
        self.declare_parameter('p5_right_slope_body_wait_s', 0.3)

        # 右斜坡阶段：右侧黄线内侧边缘三档 vy 修正。
        # 只在 P5_RIGHT_SLOPE_1/2/3 主运动阶段使用，不参与转向和额外前进。
        # right_inner_x 太靠中间：认为机器狗往坡下滑，固定加大 vy；
        # right_inner_x 太靠右：认为机器狗往坡上爬，固定减小 vy。
        self.declare_parameter('p5_right_slope_right_edge_vy_adjust_enabled', True)

        self.declare_parameter('p5_right_slope_right_edge_roi_x_min', 0.45)
        self.declare_parameter('p5_right_slope_right_edge_roi_x_max', 1.00)
        self.declare_parameter('p5_right_slope_right_edge_roi_y_min', 0.60)
        self.declare_parameter('p5_right_slope_right_edge_roi_y_max', 1.00)

        self.declare_parameter('p5_right_slope_right_edge_row_step', 1)
        self.declare_parameter('p5_right_slope_right_edge_use_bottom_connected_segment', True)
        self.declare_parameter('p5_right_slope_right_edge_max_y_gap', 5)

        self.declare_parameter('p5_right_slope_right_edge_min_points', 100)
        self.declare_parameter('p5_right_slope_right_edge_min_y_span', 100.0)
        self.declare_parameter('p5_right_slope_right_edge_x_std_max', 20.0)
        self.declare_parameter('p5_right_slope_right_edge_bottom_min_ratio', 0.90)
        self.declare_parameter('p5_right_slope_right_edge_bottom_band_ratio', 0.20)

        self.declare_parameter('p5_right_slope_right_too_center_ratio', 0.50)
        self.declare_parameter('p5_right_slope_right_too_right_ratio', 0.65)
        self.declare_parameter('p5_right_slope_right_too_center_add_vy', 0.1)
        self.declare_parameter('p5_right_slope_right_too_right_reduce_vy', 0.1)

        # 右斜坡阶段：危险趋势确认后，如果后续右侧黄线识别不到，
        # 则持续给额外 vy，直到当前 P5_RIGHT_SLOPE_x 状态结束。
        self.declare_parameter('p5_right_slope_lost_extra_enabled', True)

        # 每段 P5_RIGHT_SLOPE_1/2/3 刚进入后的前一小段时间，
        # 只执行可见黄线 vy 修正，但不累计 too_center / too_right 危险次数，
        # 也不激活后续丢线持续补偿。
        self.declare_parameter('p5_right_slope_lost_extra_ignore_after_enter_s', 1.0)

        # 连续多少帧 too_center / too_right 后，才认为危险趋势成立。
        self.declare_parameter('p5_right_slope_lost_extra_confirm_count', 3)

        # 后续丢线时，如果之前确认的是 too_center，则叠加这个 vy。
        # 默认方向与“too_center 时加大 vy”一致。
        self.declare_parameter('p5_right_slope_lost_extra_too_center_vy', 0.035)

        # 后续丢线时，如果之前确认的是 too_right，则叠加这个 vy。
        # 默认方向与“too_right 时减小 vy”一致，所以是负数。
        # 如果实测方向反了，直接把这个参数改成正数。
        self.declare_parameter('p5_right_slope_lost_extra_too_right_vy', -0.035)


        # 右斜坡 1：视觉黄线结束
        self.declare_parameter('p5_right_slope_1_vx', 0.30)
        self.declare_parameter('p5_right_slope_1_vy', 0.1325)
        self.declare_parameter('p5_right_slope_1_wz', 0.0)
        self.declare_parameter('p5_right_slope_1_step_height', 0.04)
        self.declare_parameter('p5_right_slope_1_after_center_lost_duration_s', 0.0)
        self.declare_parameter('p5_right_slope_1_after_center_lost_vx', 0.30)
        self.declare_parameter('p5_right_slope_1_after_center_lost_vy', 0.13)
        self.declare_parameter('p5_right_slope_1_after_center_lost_wz', 0.0)
        self.declare_parameter('p5_right_slope_1_after_center_lost_step_height', 0.04)

        # 右斜赛段 1/2 之间的转向方式：
        # "velocity"   ：使用当前的速度控制转向参数 p5_turn_1_* / p5_turn_2_*
        # "right_jump" ：使用右跳动作 mode/gait 转向
        self.declare_parameter('p5_right_slope_turn_method', 'right_jump')
        self.declare_parameter('p5_right_slope_turn_1_jump_mode', 16)
        self.declare_parameter('p5_right_slope_turn_1_jump_gait', 3)
        self.declare_parameter('p5_right_slope_turn_2_jump_mode', 16)
        self.declare_parameter('p5_right_slope_turn_2_jump_gait', 3)
        self.declare_parameter('p5_right_slope_turn_jump_stop_after_finish', True)

        # 转弯 1：不再使用右跳动作，改为速度控制固定时间转向
        # 由于机器狗仍在右斜坡上，转向时保留较小 vx / vy，让它继续贴着坡面运动。
        self.declare_parameter('p5_turn_1_duration_s', 5.85)
        self.declare_parameter('p5_turn_1_vx', 0.05)
        self.declare_parameter('p5_turn_1_vy', 0.0)
        self.declare_parameter('p5_turn_1_wz', -0.70)
        self.declare_parameter('p5_turn_1_step_height', 0.04)

        # 右斜坡 2：视觉黄线结束
        self.declare_parameter('p5_right_slope_2_vx', 0.30)
        self.declare_parameter('p5_right_slope_2_vy', 0.1325)
        self.declare_parameter('p5_right_slope_2_wz', 0.0)
        self.declare_parameter('p5_right_slope_2_step_height', 0.04)
        self.declare_parameter('p5_right_slope_2_after_center_lost_duration_s', 0.0)
        self.declare_parameter('p5_right_slope_2_after_center_lost_vx', 0.30)
        self.declare_parameter('p5_right_slope_2_after_center_lost_vy', 0.13)
        self.declare_parameter('p5_right_slope_2_after_center_lost_wz', 0.0)
        self.declare_parameter('p5_right_slope_2_after_center_lost_step_height', 0.04)

        # 转弯 2：不再使用右跳动作，改为速度控制固定时间转向
        self.declare_parameter('p5_turn_2_duration_s', 5.9)
        self.declare_parameter('p5_turn_2_vx', 0.05)
        self.declare_parameter('p5_turn_2_vy', 0.00)
        self.declare_parameter('p5_turn_2_wz', -0.7)
        self.declare_parameter('p5_turn_2_step_height', 0.04)

        # 右斜坡 3：视觉黄线结束
        self.declare_parameter('p5_right_slope_3_vx', 0.30)
        self.declare_parameter('p5_right_slope_3_vy', 0.1325)
        self.declare_parameter('p5_right_slope_3_wz', 0.0)
        self.declare_parameter('p5_right_slope_3_step_height', 0.04)

        # 右斜坡 3 中间黄色消失后：不再执行额外前进段，直接进入 reset body / 右跳转向准备流程。

        # 右斜坡 3 额外前进并 reset body 后：先右移固定时间，再执行右跳动作
        self.declare_parameter('p5_right_shift_before_right_jump_duration_s', 0.75)
        self.declare_parameter('p5_right_shift_before_right_jump_vx', 0.10)
        self.declare_parameter('p5_right_shift_before_right_jump_vy', -0.30)
        self.declare_parameter('p5_right_shift_before_right_jump_wz', 0.0)
        self.declare_parameter('p5_right_shift_before_right_jump_step_height', 0.04)

        # 第一段右移结束后，再继续右移一段固定时间，然后才执行右跳动作。
        self.declare_parameter('p5_right_shift_before_right_jump_2_duration_s', 0.50)
        self.declare_parameter('p5_right_shift_before_right_jump_2_vx', 0.0)
        self.declare_parameter('p5_right_shift_before_right_jump_2_vy', -0.30)
        self.declare_parameter('p5_right_shift_before_right_jump_2_wz', 0.0)
        self.declare_parameter('p5_right_shift_before_right_jump_2_step_height', 0.04)

        self.declare_parameter('p5_right_jump_after_reset_body_mode', 16)
        self.declare_parameter('p5_right_jump_after_reset_body_gait', 3)

        # 第三个转角右跳后第一段：固定时间前进，同时使用内侧边缘做 vy/wz 矫正。
        # 如果时间到了还没对齐，则 vx=0，继续原地横移/转向矫正。
        self.declare_parameter('p5_forward_after_reset_body_duration_s', 1.5)
        self.declare_parameter('p5_forward_after_reset_body_vx', 0.50)
        self.declare_parameter('p5_forward_after_reset_body_vy', 0.0)
        self.declare_parameter('p5_forward_after_reset_body_wz', 0.0)
        self.declare_parameter('p5_forward_after_reset_body_step_height', 0.04)

        self.declare_parameter('p5_forward_after_reset_body_hold_align_enabled', True)
        self.declare_parameter('p5_forward_after_reset_body_align_center_done_px', 12.0)
        self.declare_parameter('p5_forward_after_reset_body_align_heading_done_px', 8.0)
        self.declare_parameter('p5_forward_after_reset_body_align_stable_frames', 5)
        # 0 表示不设额外最长矫正时间；如果担心卡死，可以设成 2.0 / 3.0 等。
        self.declare_parameter('p5_forward_after_reset_body_align_max_extra_s', 5.0)

        # 第三个转角右跳后第二段：矫正完成后，再固定前进一段时间，不再叠加视觉矫正。
        # 这一段结束后才 stop，然后进入右跳/跳远流程。
        self.declare_parameter('p5_forward_no_align_after_reset_body_duration_s', 1.5)
        self.declare_parameter('p5_forward_no_align_after_reset_body_vx', 0.50)
        self.declare_parameter('p5_forward_no_align_after_reset_body_vy', 0.0)
        self.declare_parameter('p5_forward_no_align_after_reset_body_wz', 0.0)
        self.declare_parameter('p5_forward_no_align_after_reset_body_step_height', 0.04)

        # 离开坡度区右跳
        self.declare_parameter('p5_jump_exit_slope_mode', 16)
        self.declare_parameter('p5_jump_exit_slope_gait', 3)

        # 最后跳远动作
        self.declare_parameter('p5_final_long_jump_mode', 16)
        self.declare_parameter('p5_final_long_jump_gait', 1)

        # 离开坡度区前 reset body
        self.declare_parameter('p5_reset_roll', 0.0)
        self.declare_parameter('p5_reset_height', 0.25)
        self.declare_parameter('p5_reset_body_wait_s', 0.3)

        # =========================
        # 严格前方横向黄线检测参数
        # =========================
        self.declare_parameter('p5_yellow_roi_top_ratio', 0.45)
        self.declare_parameter('p5_yellow_roi_left_ratio', 0.40)
        self.declare_parameter('p5_yellow_roi_right_ratio', 0.60)

        self.declare_parameter('p5_yellow_h_min', 15)
        self.declare_parameter('p5_yellow_h_max', 40)
        self.declare_parameter('p5_yellow_s_min', 80)
        self.declare_parameter('p5_yellow_s_max', 255)
        self.declare_parameter('p5_yellow_v_min', 80)
        self.declare_parameter('p5_yellow_v_max', 255)

        self.declare_parameter('p5_yellow_min_contour_area', 100.0)
        self.declare_parameter('p5_yellow_min_width_height_ratio', 2.0)
        self.declare_parameter('p5_yellow_max_tilt_deg', 30.0)
        self.declare_parameter('p5_yellow_center_tolerance_ratio', 0.28)
        self.declare_parameter('p5_yellow_min_width_ratio', 0.18)

        # 黄线到底判定
        self.declare_parameter('p5_yellow_stop_line_y_ratio', 0.95)
        self.declare_parameter('p5_yellow_stop_confirm_count', 1)

        # 进入新状态后，忽略刚进入时的旧帧/旧黄线一小段时间
        self.declare_parameter('p5_yellow_ignore_after_enter_s', 0.3)

        # =========================
        # 右斜坡阶段：中间区域黄色消失检测
        # =========================
        # 右斜坡阶段相机歪斜、抖动大，因此不再要求横线形状。
        # 只要中间 ROI 内黄色像素少于阈值，并连续确认 N 帧，就认为当前斜坡段结束。
        self.declare_parameter('p5_center_yellow_roi_x_min', 0.35)
        self.declare_parameter('p5_center_yellow_roi_x_max', 0.65)
        self.declare_parameter('p5_center_yellow_roi_y_min', 0.35)
        self.declare_parameter('p5_center_yellow_roi_y_max', 1.00)
        self.declare_parameter('p5_center_yellow_min_pixels', 80)
        self.declare_parameter('p5_center_yellow_min_ratio', 0.002)
        self.declare_parameter('p5_center_yellow_absent_confirm_count', 3)
        self.declare_parameter('p5_center_yellow_ignore_after_enter_s', 0.3)

        # 没有图像时是否继续走
        self.declare_parameter('p5_keep_moving_when_no_image', True)

        # 可选角度修正
        self.declare_parameter('p5_yellow_angle_align_enabled', True)
        self.declare_parameter('p5_yellow_angle_align_fixed_wz', 0.15)
        self.declare_parameter('p5_yellow_angle_align_deadband_deg', 0.5)

    # ============================================================
    # 参数读取
    # ============================================================
    def _load_params(self):
        gp = self.get_parameter

        self.p5_initial_state = str(gp('p5_initial_state').value)
        self.control_period_s = float(gp('p5_control_period_s').value)

        self.p5_rgb_topic = str(gp('p5_rgb_topic').value)
        self.p5_show_debug_vis = bool(gp('p5_show_debug_vis').value)
        self.p5_debug_vis_detail_level = int(gp('p5_debug_vis_detail_level').value)
        self.p5_show_yellow_mask = bool(gp('p5_show_yellow_mask').value)

        self.p5_body_normal_roll = float(gp('p5_body_normal_roll').value)
        self.p5_body_normal_height = float(gp('p5_body_normal_height').value)
        self.p5_body_normal_wait_s = float(gp('p5_body_normal_wait_s').value)

        self.p5_step_up_vx = float(gp('p5_step_up_vx').value)
        self.p5_step_up_vy = float(gp('p5_step_up_vy').value)
        self.p5_step_up_wz = float(gp('p5_step_up_wz').value)
        self.p5_step_up_step_height = float(gp('p5_step_up_step_height').value)
        self.p5_step_up_duration_s = float(gp('p5_step_up_duration_s').value)

        self.p5_up_slope_vx = float(gp('p5_up_slope_vx').value)
        self.p5_up_slope_vy = float(gp('p5_up_slope_vy').value)
        self.p5_up_slope_wz = float(gp('p5_up_slope_wz').value)
        self.p5_up_slope_step_height = float(gp('p5_up_slope_step_height').value)
        self.p5_up_slope_roll = float(gp('p5_up_slope_roll').value)
        self.p5_up_slope_pitch = float(gp('p5_up_slope_pitch').value)

        self.p5_inner_edge_align_enabled = bool(gp('p5_inner_edge_align_enabled').value)
        self.p5_inner_edge_enable_vy = bool(gp('p5_inner_edge_enable_vy').value)
        self.p5_inner_edge_enable_wz = bool(gp('p5_inner_edge_enable_wz').value)

        self.p5_inner_edge_left_roi_x_min = float(gp('p5_inner_edge_left_roi_x_min').value)
        self.p5_inner_edge_left_roi_x_max = float(gp('p5_inner_edge_left_roi_x_max').value)
        self.p5_inner_edge_right_roi_x_min = float(gp('p5_inner_edge_right_roi_x_min').value)
        self.p5_inner_edge_right_roi_x_max = float(gp('p5_inner_edge_right_roi_x_max').value)
        self.p5_inner_edge_roi_y_min = float(gp('p5_inner_edge_roi_y_min').value)
        self.p5_inner_edge_roi_y_max = float(gp('p5_inner_edge_roi_y_max').value)

        self.p5_inner_edge_min_points = int(gp('p5_inner_edge_min_points').value)
        self.p5_inner_edge_min_y_span = float(gp('p5_inner_edge_min_y_span').value)
        self.p5_inner_edge_bottom_min_ratio = float(gp('p5_inner_edge_bottom_min_ratio').value)
        self.p5_inner_edge_x_std_max = float(gp('p5_inner_edge_x_std_max').value)
        self.p5_inner_edge_use_bottom_connected_segment = bool(
            gp('p5_inner_edge_use_bottom_connected_segment').value
        )
        self.p5_inner_edge_max_y_gap = int(gp('p5_inner_edge_max_y_gap').value)
        self.p5_inner_edge_min_common_y_span = float(gp('p5_inner_edge_min_common_y_span').value)
        self.p5_inner_edge_row_step = int(gp('p5_inner_edge_row_step').value)
        self.p5_inner_edge_top_bottom_band_ratio = float(
            gp('p5_inner_edge_top_bottom_band_ratio').value
        )

        self.p5_inner_edge_center_k_vy = float(gp('p5_inner_edge_center_k_vy').value)
        self.p5_inner_edge_heading_k_wz = float(gp('p5_inner_edge_heading_k_wz').value)
        self.p5_inner_edge_vy_max_correction = abs(
            float(gp('p5_inner_edge_vy_max_correction').value)
        )
        self.p5_inner_edge_wz_max_correction = abs(
            float(gp('p5_inner_edge_wz_max_correction').value)
        )
        self.p5_inner_edge_center_deadband_px = abs(
            float(gp('p5_inner_edge_center_deadband_px').value)
        )
        self.p5_inner_edge_heading_deadband_px = abs(
            float(gp('p5_inner_edge_heading_deadband_px').value)
        )

        self.p5_right_side_yellow_roi_x_min = float(gp('p5_right_side_yellow_roi_x_min').value)
        self.p5_right_side_yellow_roi_x_max = float(gp('p5_right_side_yellow_roi_x_max').value)
        self.p5_right_side_yellow_roi_y_min = float(gp('p5_right_side_yellow_roi_y_min').value)
        self.p5_right_side_yellow_roi_y_max = float(gp('p5_right_side_yellow_roi_y_max').value)

        self.p5_right_side_yellow_min_area = float(gp('p5_right_side_yellow_min_area').value)
        self.p5_right_side_yellow_min_height = int(gp('p5_right_side_yellow_min_height').value)
        self.p5_right_side_yellow_min_width = int(gp('p5_right_side_yellow_min_width').value)
        self.p5_right_side_yellow_bottom_valid_ratio = float(
            gp('p5_right_side_yellow_bottom_valid_ratio').value
        )

        self.p5_right_side_yellow_lost_confirm_count = int(
            gp('p5_right_side_yellow_lost_confirm_count').value
        )
        self.p5_right_side_yellow_ignore_after_enter_s = float(
            gp('p5_right_side_yellow_ignore_after_enter_s').value
        )

        self.p5_after_up_slope_control_duration_s = float(
            gp('p5_after_up_slope_control_duration_s').value
        )
        self.p5_after_up_slope_control_vx = float(
            gp('p5_after_up_slope_control_vx').value
        )
        self.p5_after_up_slope_control_vy = float(
            gp('p5_after_up_slope_control_vy').value
        )
        self.p5_after_up_slope_control_wz = float(
            gp('p5_after_up_slope_control_wz').value
        )
        self.p5_after_up_slope_control_step_height = float(
            gp('p5_after_up_slope_control_step_height').value
        )

        self.p5_right_slope_roll = float(gp('p5_right_slope_roll').value)
        self.p5_right_slope_height = float(gp('p5_right_slope_height').value)
        self.p5_right_slope_body_wait_s = float(gp('p5_right_slope_body_wait_s').value)

        self.p5_right_slope_right_edge_vy_adjust_enabled = bool(
            gp('p5_right_slope_right_edge_vy_adjust_enabled').value
        )
        self.p5_right_slope_right_edge_roi_x_min = float(
            gp('p5_right_slope_right_edge_roi_x_min').value
        )
        self.p5_right_slope_right_edge_roi_x_max = float(
            gp('p5_right_slope_right_edge_roi_x_max').value
        )
        self.p5_right_slope_right_edge_roi_y_min = float(
            gp('p5_right_slope_right_edge_roi_y_min').value
        )
        self.p5_right_slope_right_edge_roi_y_max = float(
            gp('p5_right_slope_right_edge_roi_y_max').value
        )
        self.p5_right_slope_right_edge_row_step = int(
            gp('p5_right_slope_right_edge_row_step').value
        )
        self.p5_right_slope_right_edge_use_bottom_connected_segment = bool(
            gp('p5_right_slope_right_edge_use_bottom_connected_segment').value
        )
        self.p5_right_slope_right_edge_max_y_gap = int(
            gp('p5_right_slope_right_edge_max_y_gap').value
        )
        self.p5_right_slope_right_edge_min_points = int(
            gp('p5_right_slope_right_edge_min_points').value
        )
        self.p5_right_slope_right_edge_min_y_span = float(
            gp('p5_right_slope_right_edge_min_y_span').value
        )
        self.p5_right_slope_right_edge_x_std_max = float(
            gp('p5_right_slope_right_edge_x_std_max').value
        )
        self.p5_right_slope_right_edge_bottom_min_ratio = float(
            gp('p5_right_slope_right_edge_bottom_min_ratio').value
        )
        self.p5_right_slope_right_edge_bottom_band_ratio = float(
            gp('p5_right_slope_right_edge_bottom_band_ratio').value
        )
        self.p5_right_slope_right_edge_bottom_band_ratio = max(
            0.05, min(0.80, self.p5_right_slope_right_edge_bottom_band_ratio)
        )
        self.p5_right_slope_right_too_center_ratio = float(
            gp('p5_right_slope_right_too_center_ratio').value
        )
        self.p5_right_slope_right_too_right_ratio = float(
            gp('p5_right_slope_right_too_right_ratio').value
        )
        self.p5_right_slope_right_too_center_add_vy = abs(float(
            gp('p5_right_slope_right_too_center_add_vy').value
        ))
        self.p5_right_slope_right_too_right_reduce_vy = abs(float(
            gp('p5_right_slope_right_too_right_reduce_vy').value
        ))

        self.p5_right_slope_lost_extra_enabled = bool(
            gp('p5_right_slope_lost_extra_enabled').value
        )
        self.p5_right_slope_lost_extra_ignore_after_enter_s = max(
            0.0,
            float(gp('p5_right_slope_lost_extra_ignore_after_enter_s').value)
        )
        self.p5_right_slope_lost_extra_confirm_count = max(
            1,
            int(gp('p5_right_slope_lost_extra_confirm_count').value)
        )

        # 这里不要 abs，因为这两个参数需要允许正负号，方便直接调方向。
        self.p5_right_slope_lost_extra_too_center_vy = float(
            gp('p5_right_slope_lost_extra_too_center_vy').value
        )
        self.p5_right_slope_lost_extra_too_right_vy = float(
            gp('p5_right_slope_lost_extra_too_right_vy').value
        )


        self.p5_right_slope_1_vx = float(gp('p5_right_slope_1_vx').value)
        self.p5_right_slope_1_vy = float(gp('p5_right_slope_1_vy').value)
        self.p5_right_slope_1_wz = float(gp('p5_right_slope_1_wz').value)
        self.p5_right_slope_1_step_height = float(gp('p5_right_slope_1_step_height').value)
        self.p5_right_slope_1_after_center_lost_duration_s = float(
            gp('p5_right_slope_1_after_center_lost_duration_s').value
        )
        self.p5_right_slope_1_after_center_lost_vx = float(
            gp('p5_right_slope_1_after_center_lost_vx').value
        )
        self.p5_right_slope_1_after_center_lost_vy = float(
            gp('p5_right_slope_1_after_center_lost_vy').value
        )
        self.p5_right_slope_1_after_center_lost_wz = float(
            gp('p5_right_slope_1_after_center_lost_wz').value
        )
        self.p5_right_slope_1_after_center_lost_step_height = float(
            gp('p5_right_slope_1_after_center_lost_step_height').value
        )

        self.p5_right_slope_turn_method = str(
            gp('p5_right_slope_turn_method').value
        ).strip().lower()
        self.p5_right_slope_turn_1_jump_mode = int(
            gp('p5_right_slope_turn_1_jump_mode').value
        )
        self.p5_right_slope_turn_1_jump_gait = int(
            gp('p5_right_slope_turn_1_jump_gait').value
        )
        self.p5_right_slope_turn_2_jump_mode = int(
            gp('p5_right_slope_turn_2_jump_mode').value
        )
        self.p5_right_slope_turn_2_jump_gait = int(
            gp('p5_right_slope_turn_2_jump_gait').value
        )
        self.p5_right_slope_turn_jump_stop_after_finish = bool(
            gp('p5_right_slope_turn_jump_stop_after_finish').value
        )

        if self.p5_right_slope_turn_method not in ['velocity', 'right_jump']:
            self.get_logger().warn(
                f'[P5_PARAM] unknown p5_right_slope_turn_method='
                f'{self.p5_right_slope_turn_method}, fallback to velocity'
            )
            self.p5_right_slope_turn_method = 'velocity'

        self.p5_turn_1_duration_s = float(gp('p5_turn_1_duration_s').value)
        self.p5_turn_1_vx = float(gp('p5_turn_1_vx').value)
        self.p5_turn_1_vy = float(gp('p5_turn_1_vy').value)
        self.p5_turn_1_wz = float(gp('p5_turn_1_wz').value)
        self.p5_turn_1_step_height = float(gp('p5_turn_1_step_height').value)

        self.p5_right_slope_2_vx = float(gp('p5_right_slope_2_vx').value)
        self.p5_right_slope_2_vy = float(gp('p5_right_slope_2_vy').value)
        self.p5_right_slope_2_wz = float(gp('p5_right_slope_2_wz').value)
        self.p5_right_slope_2_step_height = float(gp('p5_right_slope_2_step_height').value)
        self.p5_right_slope_2_after_center_lost_duration_s = float(
            gp('p5_right_slope_2_after_center_lost_duration_s').value
        )
        self.p5_right_slope_2_after_center_lost_vx = float(
            gp('p5_right_slope_2_after_center_lost_vx').value
        )
        self.p5_right_slope_2_after_center_lost_vy = float(
            gp('p5_right_slope_2_after_center_lost_vy').value
        )
        self.p5_right_slope_2_after_center_lost_wz = float(
            gp('p5_right_slope_2_after_center_lost_wz').value
        )
        self.p5_right_slope_2_after_center_lost_step_height = float(
            gp('p5_right_slope_2_after_center_lost_step_height').value
        )

        self.p5_turn_2_duration_s = float(gp('p5_turn_2_duration_s').value)
        self.p5_turn_2_vx = float(gp('p5_turn_2_vx').value)
        self.p5_turn_2_vy = float(gp('p5_turn_2_vy').value)
        self.p5_turn_2_wz = float(gp('p5_turn_2_wz').value)
        self.p5_turn_2_step_height = float(gp('p5_turn_2_step_height').value)

        self.p5_right_slope_3_vx = float(gp('p5_right_slope_3_vx').value)
        self.p5_right_slope_3_vy = float(gp('p5_right_slope_3_vy').value)
        self.p5_right_slope_3_wz = float(gp('p5_right_slope_3_wz').value)
        self.p5_right_slope_3_step_height = float(gp('p5_right_slope_3_step_height').value)

        self.p5_right_shift_before_right_jump_duration_s = float(
            gp('p5_right_shift_before_right_jump_duration_s').value
        )
        self.p5_right_shift_before_right_jump_vx = float(
            gp('p5_right_shift_before_right_jump_vx').value
        )
        self.p5_right_shift_before_right_jump_vy = float(
            gp('p5_right_shift_before_right_jump_vy').value
        )
        self.p5_right_shift_before_right_jump_wz = float(
            gp('p5_right_shift_before_right_jump_wz').value
        )
        self.p5_right_shift_before_right_jump_step_height = float(
            gp('p5_right_shift_before_right_jump_step_height').value
        )

        self.p5_right_shift_before_right_jump_2_duration_s = float(
            gp('p5_right_shift_before_right_jump_2_duration_s').value
        )
        self.p5_right_shift_before_right_jump_2_vx = float(
            gp('p5_right_shift_before_right_jump_2_vx').value
        )
        self.p5_right_shift_before_right_jump_2_vy = float(
            gp('p5_right_shift_before_right_jump_2_vy').value
        )
        self.p5_right_shift_before_right_jump_2_wz = float(
            gp('p5_right_shift_before_right_jump_2_wz').value
        )
        self.p5_right_shift_before_right_jump_2_step_height = float(
            gp('p5_right_shift_before_right_jump_2_step_height').value
        )

        self.p5_right_jump_after_reset_body_mode = int(
            gp('p5_right_jump_after_reset_body_mode').value
        )
        self.p5_right_jump_after_reset_body_gait = int(
            gp('p5_right_jump_after_reset_body_gait').value
        )

        self.p5_forward_after_reset_body_duration_s = float(
            gp('p5_forward_after_reset_body_duration_s').value
        )
        self.p5_forward_after_reset_body_vx = float(
            gp('p5_forward_after_reset_body_vx').value
        )
        self.p5_forward_after_reset_body_vy = float(
            gp('p5_forward_after_reset_body_vy').value
        )
        self.p5_forward_after_reset_body_wz = float(
            gp('p5_forward_after_reset_body_wz').value
        )
        self.p5_forward_after_reset_body_step_height = float(
            gp('p5_forward_after_reset_body_step_height').value
        )
        self.p5_forward_after_reset_body_hold_align_enabled = bool(
            gp('p5_forward_after_reset_body_hold_align_enabled').value
        )
        self.p5_forward_after_reset_body_align_center_done_px = abs(float(
            gp('p5_forward_after_reset_body_align_center_done_px').value
        ))
        self.p5_forward_after_reset_body_align_heading_done_px = abs(float(
            gp('p5_forward_after_reset_body_align_heading_done_px').value
        ))
        self.p5_forward_after_reset_body_align_stable_frames = int(
            gp('p5_forward_after_reset_body_align_stable_frames').value
        )
        self.p5_forward_after_reset_body_align_max_extra_s = float(
            gp('p5_forward_after_reset_body_align_max_extra_s').value
        )

        self.p5_forward_no_align_after_reset_body_duration_s = float(
            gp('p5_forward_no_align_after_reset_body_duration_s').value
        )
        self.p5_forward_no_align_after_reset_body_vx = float(
            gp('p5_forward_no_align_after_reset_body_vx').value
        )
        self.p5_forward_no_align_after_reset_body_vy = float(
            gp('p5_forward_no_align_after_reset_body_vy').value
        )
        self.p5_forward_no_align_after_reset_body_wz = float(
            gp('p5_forward_no_align_after_reset_body_wz').value
        )
        self.p5_forward_no_align_after_reset_body_step_height = float(
            gp('p5_forward_no_align_after_reset_body_step_height').value
        )

        self.p5_jump_exit_slope_mode = int(gp('p5_jump_exit_slope_mode').value)
        self.p5_jump_exit_slope_gait = int(gp('p5_jump_exit_slope_gait').value)

        self.p5_final_long_jump_mode = int(gp('p5_final_long_jump_mode').value)
        self.p5_final_long_jump_gait = int(gp('p5_final_long_jump_gait').value)

        self.p5_reset_roll = float(gp('p5_reset_roll').value)
        self.p5_reset_height = float(gp('p5_reset_height').value)
        self.p5_reset_body_wait_s = float(gp('p5_reset_body_wait_s').value)

        self.p5_yellow_roi_top_ratio = float(gp('p5_yellow_roi_top_ratio').value)
        self.p5_yellow_roi_left_ratio = float(gp('p5_yellow_roi_left_ratio').value)
        self.p5_yellow_roi_right_ratio = float(gp('p5_yellow_roi_right_ratio').value)

        self.p5_yellow_h_min = int(gp('p5_yellow_h_min').value)
        self.p5_yellow_h_max = int(gp('p5_yellow_h_max').value)
        self.p5_yellow_s_min = int(gp('p5_yellow_s_min').value)
        self.p5_yellow_s_max = int(gp('p5_yellow_s_max').value)
        self.p5_yellow_v_min = int(gp('p5_yellow_v_min').value)
        self.p5_yellow_v_max = int(gp('p5_yellow_v_max').value)

        self.p5_yellow_min_contour_area = float(gp('p5_yellow_min_contour_area').value)
        self.p5_yellow_min_width_height_ratio = float(gp('p5_yellow_min_width_height_ratio').value)
        self.p5_yellow_max_tilt_deg = float(gp('p5_yellow_max_tilt_deg').value)
        self.p5_yellow_center_tolerance_ratio = float(gp('p5_yellow_center_tolerance_ratio').value)
        self.p5_yellow_min_width_ratio = float(gp('p5_yellow_min_width_ratio').value)

        self.p5_yellow_stop_line_y_ratio = float(gp('p5_yellow_stop_line_y_ratio').value)
        self.p5_yellow_stop_confirm_count = int(gp('p5_yellow_stop_confirm_count').value)
        self.p5_yellow_ignore_after_enter_s = float(gp('p5_yellow_ignore_after_enter_s').value)

        self.p5_center_yellow_roi_x_min = float(gp('p5_center_yellow_roi_x_min').value)
        self.p5_center_yellow_roi_x_max = float(gp('p5_center_yellow_roi_x_max').value)
        self.p5_center_yellow_roi_y_min = float(gp('p5_center_yellow_roi_y_min').value)
        self.p5_center_yellow_roi_y_max = float(gp('p5_center_yellow_roi_y_max').value)
        self.p5_center_yellow_min_pixels = int(gp('p5_center_yellow_min_pixels').value)
        self.p5_center_yellow_min_ratio = float(gp('p5_center_yellow_min_ratio').value)
        self.p5_center_yellow_absent_confirm_count = int(
            gp('p5_center_yellow_absent_confirm_count').value
        )
        self.p5_center_yellow_ignore_after_enter_s = float(
            gp('p5_center_yellow_ignore_after_enter_s').value
        )

        self.p5_keep_moving_when_no_image = bool(gp('p5_keep_moving_when_no_image').value)

        self.p5_yellow_angle_align_enabled = bool(gp('p5_yellow_angle_align_enabled').value)
        self.p5_yellow_angle_align_fixed_wz = abs(float(gp('p5_yellow_angle_align_fixed_wz').value))
        self.p5_yellow_angle_align_deadband_deg = float(gp('p5_yellow_angle_align_deadband_deg').value)

    # ============================================================
    # 时间与状态工具
    # ============================================================
    def now_sec(self) -> float:
        # use_sim_time=True 时，这里读取的是 Gazebo /clock 仿真时间。
        return self.get_clock().now().nanoseconds * 1e-9

    def state_elapsed_s(self) -> float:
        now = self.now_sec()

        # 刚切到仿真时间时，如果 /clock 还没来，now 可能是 0。
        # 这时不要让 elapsed 变成异常大值，也不要推进状态机。
        if now <= 0.0:
            return 0.0

        # 阻塞动作 / STOP 之后进入的新状态，不使用阻塞结束瞬间可能陈旧的 /clock。
        # 等下一次真正运行该状态时，用此刻已经刷新的仿真时间作为计时起点。
        if getattr(self, 'state_timer_reset_after_blocking_wait', False):
            self.state_start_time = now
            self.state_timer_reset_after_blocking_wait = False
            self.get_logger().info(
                f'[P5_TIME] reset timer after blocking wait on first active tick: '
                f'state={self.state}, sim_time={now:.3f}',
                throttle_duration_sec=1.0
            )
            return 0.0

        # 如果进入状态时 /clock 尚未有效，state_start_time 可能是 0。
        # 第一次拿到有效 /clock 后，从当前仿真时刻重新开始计时。
        if getattr(self, 'state_start_time', 0.0) <= 0.0:
            self.state_start_time = now
            self.get_logger().info(
                f'[P5_TIME] start state timer after /clock valid: '
                f'state={self.state}, sim_time={now:.3f}',
                throttle_duration_sec=1.0
            )
            return 0.0

        return max(0.0, now - self.state_start_time)

    def enter_state(self, new_state: str):
        now = self.now_sec()
        self.get_logger().info(f'[P5] ENTER STATE -> {new_state}, sim_time={now:.3f}')
        self.state = new_state
        self.state_start_time = now
        self.state_enter_frame_seq = self.latest_frame_seq
        self.action_sent = False
        self.p5_yellow_stop_counter = 0
        self.p5_center_yellow_absent_counter = 0
        self.p5_right_side_yellow_lost_counter = 0
        self.p5_forward_align_stable_counter = 0

        if new_state in [
            self.P5_RIGHT_SLOPE_1,
            self.P5_RIGHT_SLOPE_2,
            self.P5_RIGHT_SLOPE_3,
        ]:
            self.reset_p5_right_slope_lost_extra_state()

    def enter_state_after_blocking_wait(self, new_state: str):
        """
        用于 Ctrl.Wait_finish / send_stop_command 这类阻塞调用之后切状态。

        阻塞期间 rclpy 可能没有及时处理 /clock，直接用 enter_state() 里的 now_sec()
        作为新状态起点，可能导致下一轮 elapsed 一进来就偏大。
        所以这里先切状态，再把计时起点标记为“下一次 active tick 重新开始”。
        """
        self.enter_state(new_state)
        self.state_start_time = 0.0
        self.state_timer_reset_after_blocking_wait = True
        self.get_logger().info(
            f'[P5_TIME] state entered after blocking wait, '
            f'timer will reset on first active tick: state={new_state}'
        )

    def _inc_life_count(self):
        self.msg.life_count += 1
        if self.msg.life_count > 127:
            self.msg.life_count = 1

    # ============================================================
    # 图像回调
    # ============================================================
    def rgb_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'[P5_RGB] cv_bridge convert failed: {e}')
            return

        self.latest_bgr = frame
        self.latest_frame_seq += 1
        self.latest_p5_yellow_result = self.detect_p5_yellow_stop_line(frame)

        if self.p5_show_debug_vis:
            self.show_p5_debug_window(frame)

    # ============================================================
    # YAML 参数发布
    # ============================================================
    def publish_yaml_vecxd(self, name: str, values: List[float], is_user: int = 1):
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

    def set_body_roll_height(self, roll: float, height: float):
        values = [0.0] * 12
        values[0] = float(roll)
        values[2] = float(height)

        self.publish_yaml_vecxd('des_roll_pitch_height', values, is_user=1)

        self.get_logger().info(
            f'[P5_BODY] set des_roll_pitch_height: '
            f'roll={roll:.3f}, height={height:.3f}'
        )

    # ============================================================
    # 控制命令
    # ============================================================
    def send_stop_command(self):
        self.msg.mode = 12
        self.msg.gait_id = 0
        self.msg.vel_des = [0.0, 0.0, 0.0]
        self.msg.step_height = [0.0, 0.0]
        self.msg.rpy_des = [0.0, 0.0, 0.0]
        self.msg.pos_des = [0.0, 0.0, 0.0]

        self._inc_life_count()
        self.Ctrl.Send_cmd(self.msg)
        self.Ctrl.Wait_finish(12, 0)

        self.get_logger().info('[P5_CMD] STOP', throttle_duration_sec=1.0)

    def send_velocity_command(
        self,
        vx: float,
        vy: float,
        wz: float,
        step_height: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        body_height: float = 0.25,
    ):
        self.msg.mode = 11
        self.msg.gait_id = 3
        self.msg.vel_des = [float(vx), float(vy), float(wz)]
        self.msg.step_height = [float(step_height), float(step_height)]
        self.msg.rpy_des = [float(roll), float(pitch), float(yaw)]
        self.msg.pos_des = [0.0, 0.0, float(body_height)]

        self._inc_life_count()
        self.Ctrl.Send_cmd(self.msg)

    def send_action_once(self, mode: int, gait_id: int):
        self.msg.mode = int(mode)
        self.msg.gait_id = int(gait_id)
        self.msg.vel_des = [0.0, 0.0, 0.0]
        self.msg.step_height = [0.0, 0.0]
        self.msg.rpy_des = [0.0, 0.0, 0.0]

        self._inc_life_count()
        self.Ctrl.Send_cmd(self.msg)

        self.get_logger().info(
            f'[P5_ACTION] send once: '
            f'mode={mode}, gait_id={gait_id}, life_count={self.msg.life_count}'
        )

    # ============================================================
    # 严格前方黄线检测
    # ============================================================
    def is_p5_front_horizontal_yellow_line(self, cnt, roi_shape) -> bool:
        _, roi_w = roi_shape[:2]

        area = cv2.contourArea(cnt)
        if area < self.p5_yellow_min_contour_area:
            return False

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh <= 0:
            return False

        wh_ratio = bw / float(bh)
        if wh_ratio < self.p5_yellow_min_width_height_ratio:
            return False

        width_ratio = bw / float(max(roi_w, 1))
        if width_ratio < self.p5_yellow_min_width_ratio:
            return False

        cx = x + bw / 2.0
        roi_cx = roi_w / 2.0
        center_offset_ratio = abs(cx - roi_cx) / float(max(roi_w, 1))
        if center_offset_ratio > self.p5_yellow_center_tolerance_ratio:
            return False

        rect = cv2.minAreaRect(cnt)
        (_, _), (rw, rh), angle = rect

        if rw < rh:
            tilt_deg = abs(angle - 90.0)
        else:
            tilt_deg = abs(angle)

        if tilt_deg > 45.0:
            tilt_deg = abs(90.0 - tilt_deg)

        if tilt_deg > self.p5_yellow_max_tilt_deg:
            return False

        return True

    def get_signed_p5_yellow_line_angle_deg(self, cnt) -> float:
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

    def make_p5_yellow_mask_for_debug(self, frame):
        h, w = frame.shape[:2]

        roi_top = int(h * self.p5_yellow_roi_top_ratio)
        roi_left = int(w * self.p5_yellow_roi_left_ratio)
        roi_right = int(w * self.p5_yellow_roi_right_ratio)

        roi_top = max(0, min(h - 1, roi_top))
        roi_left = max(0, min(w - 1, roi_left))
        roi_right = max(roi_left + 1, min(w, roi_right))

        roi = frame[roi_top:h, roi_left:roi_right]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array(
            [self.p5_yellow_h_min, self.p5_yellow_s_min, self.p5_yellow_v_min],
            dtype=np.uint8
        )
        upper_yellow = np.array(
            [self.p5_yellow_h_max, self.p5_yellow_s_max, self.p5_yellow_v_max],
            dtype=np.uint8
        )

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask, (roi_left, roi_top, roi_right, h)

    def detect_p5_yellow_stop_line(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]

        roi_top = int(h * self.p5_yellow_roi_top_ratio)
        roi_left = int(w * self.p5_yellow_roi_left_ratio)
        roi_right = int(w * self.p5_yellow_roi_right_ratio)

        roi_top = max(0, min(h - 1, roi_top))
        roi_left = max(0, min(w - 1, roi_left))
        roi_right = max(roi_left + 1, min(w, roi_right))

        roi = frame[roi_top:h, roi_left:roi_right]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array(
            [self.p5_yellow_h_min, self.p5_yellow_s_min, self.p5_yellow_v_min],
            dtype=np.uint8
        )
        upper_yellow = np.array(
            [self.p5_yellow_h_max, self.p5_yellow_s_max, self.p5_yellow_v_max],
            dtype=np.uint8
        )

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        self.latest_p5_yellow_mask = mask
        self.latest_p5_yellow_roi = (roi_left, roi_top, roi_right, h)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.p5_yellow_min_contour_area:
                continue

            if not self.is_p5_front_horizontal_yellow_line(cnt, roi.shape):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            # 严格版：优先选最靠下的前方横线
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
            }

        x, y, bw, bh = cv2.boundingRect(best_contour)

        line_bottom_y = roi_top + y + bh
        cx = roi_left + x + bw // 2
        cy = roi_top + y + bh // 2

        angle_deg = self.get_signed_p5_yellow_line_angle_deg(best_contour)
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
            'bbox': (
                int(roi_left + x),
                int(roi_top + y),
                int(roi_left + x + bw),
                int(roi_top + y + bh)
            ),
            'width_ratio': float(width_ratio),
            'wh_ratio': float(wh_ratio),
        }

    def p5_yellow_reached_bottom(self, yellow_result: dict) -> bool:
        if yellow_result is None:
            self.p5_yellow_stop_counter = 0
            return False

        if yellow_result.get('img_shape') is None or not yellow_result.get('has_line', False):
            self.p5_yellow_stop_counter = 0
            return False

        h, _ = yellow_result['img_shape']
        stop_y_threshold = int(h * self.p5_yellow_stop_line_y_ratio)

        bottom_y = yellow_result.get('line_bottom_y')

        if bottom_y is not None and int(bottom_y) >= stop_y_threshold:
            self.p5_yellow_stop_counter += 1
        else:
            self.p5_yellow_stop_counter = 0

        self.get_logger().info(
            f'[P5_YELLOW] bottom={bottom_y}, '
            f'threshold={stop_y_threshold}, '
            f'counter={self.p5_yellow_stop_counter}/{self.p5_yellow_stop_confirm_count}',
            throttle_duration_sec=0.3
        )

        return self.p5_yellow_stop_counter >= self.p5_yellow_stop_confirm_count

    def compute_p5_yellow_angle_align_wz(self, yellow_result: dict) -> float:
        if not self.p5_yellow_angle_align_enabled:
            return 0.0

        if yellow_result is None or not yellow_result.get('has_line', False):
            return 0.0

        angle_deg = yellow_result.get('angle_deg', None)
        if angle_deg is None:
            return 0.0

        angle_deg = float(angle_deg)

        if abs(angle_deg) <= self.p5_yellow_angle_align_deadband_deg:
            return 0.0

        # 符号沿用前面赛段严格横线角度矫正的约定：
        # angle > 0 给负 wz，angle < 0 给正 wz。
        # 如果实测越修越歪，就把这里正负号对调。
        if angle_deg > 0.0:
            wz = -abs(self.p5_yellow_angle_align_fixed_wz)
        else:
            wz = abs(self.p5_yellow_angle_align_fixed_wz)

        self.get_logger().info(
            f'[P5_YELLOW_ALIGN] angle={angle_deg:.2f}deg, '
            f'deadband={self.p5_yellow_angle_align_deadband_deg:.2f}, '
            f'wz={wz:.3f}',
            throttle_duration_sec=0.3
        )

        return wz



    # ============================================================
    # P5_RIGHT_SLOPE：中间区域黄色消失检测
    # ============================================================
    def detect_p5_center_yellow_presence(self, frame: np.ndarray) -> dict:
        """
        右斜坡阶段专用：检测图像中间区域是否还有黄色。

        右斜坡上相机画面倾斜、抖动明显，所以这里不再要求黄线是横向、
        不计算角度，也不检查宽高比。只看中间 ROI 内黄色像素数量/比例。

        has_yellow=True  表示中间区域还有黄色，继续走；
        has_yellow=False 表示中间区域基本没有黄色，可以累计 absent counter。
        """
        h, w = frame.shape[:2]

        x1 = int(w * self.p5_center_yellow_roi_x_min)
        x2 = int(w * self.p5_center_yellow_roi_x_max)
        y1 = int(h * self.p5_center_yellow_roi_y_min)
        y2 = int(h * self.p5_center_yellow_roi_y_max)

        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))

        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array(
            [self.p5_yellow_h_min, self.p5_yellow_s_min, self.p5_yellow_v_min],
            dtype=np.uint8
        )
        upper_yellow = np.array(
            [self.p5_yellow_h_max, self.p5_yellow_s_max, self.p5_yellow_v_max],
            dtype=np.uint8
        )

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        yellow_pixels = int(cv2.countNonZero(mask))
        roi_pixels = int(mask.shape[0] * mask.shape[1])
        yellow_ratio = yellow_pixels / float(max(roi_pixels, 1))

        has_yellow = (
            yellow_pixels >= self.p5_center_yellow_min_pixels and
            yellow_ratio >= self.p5_center_yellow_min_ratio
        )

        bbox = None
        if yellow_pixels > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if cv2.contourArea(c) > 5.0]
            if valid_contours:
                all_pts = np.vstack(valid_contours)
                rx, ry, rw, rh = cv2.boundingRect(all_pts)
                bbox = (
                    int(x1 + rx),
                    int(y1 + ry),
                    int(x1 + rx + rw),
                    int(y1 + ry + rh)
                )

        if has_yellow:
            reason = 'center_yellow_present'
        else:
            reason = 'center_yellow_absent'

        return {
            'has_yellow': bool(has_yellow),
            'yellow_pixels': int(yellow_pixels),
            'roi_pixels': int(roi_pixels),
            'yellow_ratio': float(yellow_ratio),
            'bbox': bbox,
            'roi': (int(x1), int(y1), int(x2), int(y2)),
            'img_shape': (h, w),
            'reason': reason,
        }

    # ============================================================
    # P5_UP_SLOPE：右侧赛道黄线消失检测
    # ============================================================
    def detect_p5_right_side_yellow_line(self, frame: np.ndarray) -> dict:
        """
        P5_UP_SLOPE 专用：检测右侧赛道旁边的黄色边线。

        这里不做很严格的横线/竖线形状判断，只在右侧 ROI 内找黄色区域。
        但是有一个关键限制：检测到的黄色区域 bottom_ratio 必须接近图像底部，
        才算“右侧赛道黄线还存在”。

        这样可以避免上坡末尾时，把前方右侧黄线误认为当前右侧赛道黄线。
        """
        h, w = frame.shape[:2]

        x1 = int(w * self.p5_right_side_yellow_roi_x_min)
        x2 = int(w * self.p5_right_side_yellow_roi_x_max)
        y1 = int(h * self.p5_right_side_yellow_roi_y_min)
        y2 = int(h * self.p5_right_side_yellow_roi_y_max)

        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))

        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array(
            [self.p5_yellow_h_min, self.p5_yellow_s_min, self.p5_yellow_v_min],
            dtype=np.uint8
        )
        upper_yellow = np.array(
            [self.p5_yellow_h_max, self.p5_yellow_s_max, self.p5_yellow_v_max],
            dtype=np.uint8
        )

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.p5_right_side_yellow_min_area:
                continue

            rx, ry, rw, rh = cv2.boundingRect(cnt)

            if rw < self.p5_right_side_yellow_min_width:
                continue
            if rh < self.p5_right_side_yellow_min_height:
                continue

            bx1 = x1 + rx
            by1 = y1 + ry
            bx2 = bx1 + rw
            by2 = by1 + rh
            cx = bx1 + rw // 2
            cy = by1 + rh // 2

            bottom_ratio = by2 / float(max(h, 1))

            candidates.append({
                'bbox': (int(bx1), int(by1), int(bx2), int(by2)),
                'center': (int(cx), int(cy)),
                'area': float(area),
                'height': int(rh),
                'width': int(rw),
                'bottom_y': int(by2),
                'bottom_ratio': float(bottom_ratio),
                # 右侧赛道黄线应贴近图像底部，所以优先选 bottom 最大的黄色区域
                'score': float(5.0 * by2 + 0.01 * area + rh),
            })

        if len(candidates) == 0:
            return {
                'has_line': False,
                'valid_bottom': False,
                'bbox': None,
                'center': None,
                'bottom_y': None,
                'bottom_ratio': None,
                'area': None,
                'height': None,
                'width': None,
                'roi': (int(x1), int(y1), int(x2), int(y2)),
                'candidates': candidates,
                'img_shape': (h, w),
                'reason': 'no_candidate',
            }

        best = max(candidates, key=lambda c: c['score'])
        valid_bottom = best['bottom_ratio'] >= self.p5_right_side_yellow_bottom_valid_ratio

        if not valid_bottom:
            # 检测到了黄线，但最低点不在图像底部附近。
            # 很可能是前方右侧黄线，不算右侧赛道黄线还存在。
            return {
                'has_line': False,
                'valid_bottom': False,
                'bbox': best['bbox'],
                'center': best['center'],
                'bottom_y': best['bottom_y'],
                'bottom_ratio': best['bottom_ratio'],
                'area': best['area'],
                'height': best['height'],
                'width': best['width'],
                'roi': (int(x1), int(y1), int(x2), int(y2)),
                'candidates': candidates,
                'img_shape': (h, w),
                'reason': 'bottom_not_near_image_bottom',
            }

        return {
            'has_line': True,
            'valid_bottom': True,
            'bbox': best['bbox'],
            'center': best['center'],
            'bottom_y': best['bottom_y'],
            'bottom_ratio': best['bottom_ratio'],
            'area': best['area'],
            'height': best['height'],
            'width': best['width'],
            'roi': (int(x1), int(y1), int(x2), int(y2)),
            'candidates': candidates,
            'img_shape': (h, w),
            'reason': 'valid_right_side_line',
        }


    # ============================================================
    # P5_UP_SLOPE：左右内侧黄线边缘检测与矫正
    # ============================================================
    def clamp_p5_roi(self, roi, w: int, h: int):
        x1, y1, x2, y2 = roi
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(x1 + 1, min(w, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(y1 + 1, min(h, int(y2)))
        return x1, y1, x2, y2

    def get_p5_inner_edge_rois(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        y1 = int(h * self.p5_inner_edge_roi_y_min)
        y2 = int(h * self.p5_inner_edge_roi_y_max)

        left_roi = (
            int(w * self.p5_inner_edge_left_roi_x_min),
            y1,
            int(w * self.p5_inner_edge_left_roi_x_max),
            y2,
        )
        right_roi = (
            int(w * self.p5_inner_edge_right_roi_x_min),
            y1,
            int(w * self.p5_inner_edge_right_roi_x_max),
            y2,
        )

        return (
            self.clamp_p5_roi(left_roi, w, h),
            self.clamp_p5_roi(right_roi, w, h),
        )

    def make_p5_inner_edge_yellow_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array(
            [self.p5_yellow_h_min, self.p5_yellow_s_min, self.p5_yellow_v_min],
            dtype=np.uint8
        )
        upper_yellow = np.array(
            [self.p5_yellow_h_max, self.p5_yellow_s_max, self.p5_yellow_v_max],
            dtype=np.uint8
        )

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def keep_p5_bottom_connected_segment(self, edge_points):
        """
        只保留从图像底部往上的连续边缘段。
        如果相邻两个边缘点的 y 间隔过大，认为上方远处黄线与近处黄线断开。
        """
        if not edge_points:
            return []

        if not self.p5_inner_edge_use_bottom_connected_segment:
            return sorted(edge_points, key=lambda p: p[1])

        max_y_gap = max(1, int(self.p5_inner_edge_max_y_gap))

        pts_bottom_to_top = sorted(edge_points, key=lambda p: p[1], reverse=True)

        kept = [pts_bottom_to_top[0]]
        prev_y = pts_bottom_to_top[0][1]

        for x, y in pts_bottom_to_top[1:]:
            y_gap = abs(prev_y - y)
            if y_gap > max_y_gap:
                break

            kept.append((x, y))
            prev_y = y

        return sorted(kept, key=lambda p: p[1])

    def mean_p5_x_in_y_band(self, points, y_low: float, y_high: float):
        if not points:
            return None, None, 0

        pts = np.array(points, dtype=np.float32)
        band = pts[(pts[:, 1] >= y_low) & (pts[:, 1] <= y_high)]

        if len(band) == 0:
            return None, None, 0

        return float(np.mean(band[:, 0])), float(np.mean(band[:, 1])), int(len(band))

    def extract_p5_inner_edge_points(
        self,
        mask: np.ndarray,
        roi,
        side: str,
        image_h: int,
    ) -> dict:
        """
        side='left'  ：左侧黄线取每一行最右侧黄色像素，作为左侧赛道内侧边缘。
        side='right' ：右侧黄线取每一行最左侧黄色像素，作为右侧赛道内侧边缘。
        """
        x1, y1, x2, y2 = roi
        roi_mask = mask[y1:y2, x1:x2]

        row_step = max(1, int(self.p5_inner_edge_row_step))

        edge_points = []
        for local_y in range(0, roi_mask.shape[0], row_step):
            row = roi_mask[local_y, :]
            xs = np.where(row > 0)[0]

            if xs.size == 0:
                continue

            if side == 'left':
                local_x = int(np.max(xs))
            else:
                local_x = int(np.min(xs))

            edge_points.append((int(x1 + local_x), int(y1 + local_y)))

        if len(edge_points) == 0:
            return {
                'valid': False,
                'reason': 'no_yellow_points',
                'points': [],
                'top_x': None,
                'bottom_x': None,
                'top_y': None,
                'bottom_y': None,
                'point_count': 0,
                'raw_point_count': 0,
                'y_span': 0.0,
                'bottom_ratio': 0.0,
                'x_std': 0.0,
            }

        raw_point_count = len(edge_points)
        edge_points = self.keep_p5_bottom_connected_segment(edge_points)

        if len(edge_points) == 0:
            return {
                'valid': False,
                'reason': 'no_bottom_connected_segment',
                'points': [],
                'top_x': None,
                'bottom_x': None,
                'top_y': None,
                'bottom_y': None,
                'point_count': 0,
                'raw_point_count': int(raw_point_count),
                'y_span': 0.0,
                'bottom_ratio': 0.0,
                'x_std': 0.0,
            }

        pts = np.array(edge_points, dtype=np.float32)
        xs = pts[:, 0]
        ys = pts[:, 1]

        point_count = len(pts)
        y_min = float(np.min(ys))
        y_max = float(np.max(ys))
        y_span = y_max - y_min
        bottom_ratio = y_max / float(max(image_h, 1))
        x_std = float(np.std(xs))

        band_ratio = max(0.05, min(0.50, float(self.p5_inner_edge_top_bottom_band_ratio)))
        top_thr = y_min + band_ratio * max(y_span, 1.0)
        bottom_thr = y_max - band_ratio * max(y_span, 1.0)

        top_band = pts[pts[:, 1] <= top_thr]
        bottom_band = pts[pts[:, 1] >= bottom_thr]

        if len(top_band) == 0:
            top_band = pts
        if len(bottom_band) == 0:
            bottom_band = pts

        top_x = float(np.mean(top_band[:, 0]))
        bottom_x = float(np.mean(bottom_band[:, 0]))

        fail_reasons = []

        if point_count < self.p5_inner_edge_min_points:
            fail_reasons.append(f'points<{self.p5_inner_edge_min_points}')

        if y_span < self.p5_inner_edge_min_y_span:
            fail_reasons.append(f'y_span<{self.p5_inner_edge_min_y_span:.0f}')

        if bottom_ratio < self.p5_inner_edge_bottom_min_ratio:
            fail_reasons.append(f'bottom<{self.p5_inner_edge_bottom_min_ratio:.2f}')

        if x_std > self.p5_inner_edge_x_std_max:
            fail_reasons.append(f'x_std>{self.p5_inner_edge_x_std_max:.0f}')

        valid = len(fail_reasons) == 0

        return {
            'valid': bool(valid),
            'reason': 'ok' if valid else ','.join(fail_reasons),
            'points': edge_points,
            'top_x': top_x,
            'bottom_x': bottom_x,
            'top_y': y_min,
            'bottom_y': y_max,
            'point_count': int(point_count),
            'raw_point_count': int(raw_point_count),
            'y_span': float(y_span),
            'bottom_ratio': float(bottom_ratio),
            'x_std': float(x_std),
        }

    def detect_p5_inner_edges(self, frame: np.ndarray) -> dict:
        """
        P5_UP_SLOPE 专用：检测左右两侧赛道内侧黄线边缘。
        输出 center_error 和 heading_error，供上坡过程的 vy / wz 修正使用。
        """
        h, w = frame.shape[:2]
        mask = self.make_p5_inner_edge_yellow_mask(frame)

        left_roi, right_roi = self.get_p5_inner_edge_rois(frame)

        left_edge = self.extract_p5_inner_edge_points(
            mask=mask,
            roi=left_roi,
            side='left',
            image_h=h,
        )
        right_edge = self.extract_p5_inner_edge_points(
            mask=mask,
            roi=right_roi,
            side='right',
            image_h=h,
        )

        has_left = bool(left_edge['valid'])
        has_right = bool(right_edge['valid'])
        has_both = has_left and has_right

        result = {
            'mask': mask,
            'left_roi': left_roi,
            'right_roi': right_roi,
            'left_edge': left_edge,
            'right_edge': right_edge,
            'has_left': has_left,
            'has_right': has_right,
            'has_both': has_both,
            'bottom_center_x': None,
            'top_center_x': None,
            'bottom_center_y': None,
            'top_center_y': None,
            'center_error': None,
            'heading_error': None,
            'common_top_y': None,
            'common_bottom_y': None,
            'common_y_span': None,
            'common_valid': False,
            'common_reason': 'need_both_edges',
            'cmd_vy_correction': 0.0,
            'cmd_wz_correction': 0.0,
        }

        if has_both:
            common_top_y = max(float(left_edge['top_y']), float(right_edge['top_y']))
            common_bottom_y = min(float(left_edge['bottom_y']), float(right_edge['bottom_y']))
            common_y_span = common_bottom_y - common_top_y

            result['common_top_y'] = float(common_top_y)
            result['common_bottom_y'] = float(common_bottom_y)
            result['common_y_span'] = float(common_y_span)

            if common_y_span >= self.p5_inner_edge_min_common_y_span:
                band_ratio = max(0.05, min(0.50, float(self.p5_inner_edge_top_bottom_band_ratio)))

                top_band_high = common_top_y + band_ratio * common_y_span
                bottom_band_low = common_bottom_y - band_ratio * common_y_span

                left_top_x, left_top_y, left_top_n = self.mean_p5_x_in_y_band(
                    left_edge['points'], common_top_y, top_band_high
                )
                right_top_x, right_top_y, right_top_n = self.mean_p5_x_in_y_band(
                    right_edge['points'], common_top_y, top_band_high
                )
                left_bottom_x, left_bottom_y, left_bottom_n = self.mean_p5_x_in_y_band(
                    left_edge['points'], bottom_band_low, common_bottom_y
                )
                right_bottom_x, right_bottom_y, right_bottom_n = self.mean_p5_x_in_y_band(
                    right_edge['points'], bottom_band_low, common_bottom_y
                )

                enough_band_points = (
                    left_top_n > 0 and right_top_n > 0 and
                    left_bottom_n > 0 and right_bottom_n > 0
                )

                if enough_band_points:
                    bottom_center_x = (left_bottom_x + right_bottom_x) / 2.0
                    top_center_x = (left_top_x + right_top_x) / 2.0
                    bottom_center_y = (left_bottom_y + right_bottom_y) / 2.0
                    top_center_y = (left_top_y + right_top_y) / 2.0

                    image_center_x = w / 2.0

                    result['bottom_center_x'] = float(bottom_center_x)
                    result['top_center_x'] = float(top_center_x)
                    result['bottom_center_y'] = float(bottom_center_y)
                    result['top_center_y'] = float(top_center_y)
                    result['center_error'] = float(bottom_center_x - image_center_x)
                    result['heading_error'] = float(top_center_x - bottom_center_x)
                    result['common_valid'] = True
                    result['common_reason'] = 'ok'

                    result['left_common_top'] = (float(left_top_x), float(left_top_y), int(left_top_n))
                    result['right_common_top'] = (float(right_top_x), float(right_top_y), int(right_top_n))
                    result['left_common_bottom'] = (float(left_bottom_x), float(left_bottom_y), int(left_bottom_n))
                    result['right_common_bottom'] = (float(right_bottom_x), float(right_bottom_y), int(right_bottom_n))
                else:
                    result['common_reason'] = 'empty_top_or_bottom_band'
            else:
                result['common_reason'] = (
                    f'common_y_span<{self.p5_inner_edge_min_common_y_span:.0f}'
                )

        return result

    @staticmethod
    def clamp_value(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def compute_p5_up_slope_inner_edge_corrected_cmd(
        self,
        base_vy: float,
        base_wz: float,
        frame: np.ndarray,
    ):
        """
        根据上坡左右内侧边缘结果修正 vy/wz。
        只在 common_valid=True 时修正；否则返回基础速度。
        """
        if not self.p5_inner_edge_align_enabled:
            return base_vy, base_wz

        edge_result = self.detect_p5_inner_edges(frame)
        self.latest_p5_inner_edge_result = edge_result

        if not edge_result.get('common_valid', False):
            self.get_logger().info(
                f'[P5_INNER_EDGE_ALIGN] no correction: '
                f'left={edge_result.get("has_left")}, '
                f'right={edge_result.get("has_right")}, '
                f'reason={edge_result.get("common_reason")}, '
                f'L_reason={edge_result.get("left_edge", {}).get("reason")}, '
                f'R_reason={edge_result.get("right_edge", {}).get("reason")}',
                throttle_duration_sec=0.5
            )
            return base_vy, base_wz

        center_error = float(edge_result.get('center_error', 0.0))
        heading_error = float(edge_result.get('heading_error', 0.0))

        vy_corr = 0.0
        wz_corr = 0.0

        if self.p5_inner_edge_enable_vy and abs(center_error) > self.p5_inner_edge_center_deadband_px:
            vy_corr = -self.p5_inner_edge_center_k_vy * center_error
            vy_corr = self.clamp_value(
                vy_corr,
                -self.p5_inner_edge_vy_max_correction,
                self.p5_inner_edge_vy_max_correction
            )

        if self.p5_inner_edge_enable_wz and abs(heading_error) > self.p5_inner_edge_heading_deadband_px:
            wz_corr = self.p5_inner_edge_heading_k_wz * heading_error
            wz_corr = self.clamp_value(
                wz_corr,
                -self.p5_inner_edge_wz_max_correction,
                self.p5_inner_edge_wz_max_correction
            )

        cmd_vy = base_vy + vy_corr
        cmd_wz = base_wz + wz_corr

        edge_result['cmd_vy_correction'] = float(vy_corr)
        edge_result['cmd_wz_correction'] = float(wz_corr)
        edge_result['cmd_vy'] = float(cmd_vy)
        edge_result['cmd_wz'] = float(cmd_wz)
        self.latest_p5_inner_edge_result = edge_result

        self.get_logger().info(
            f'[P5_INNER_EDGE_ALIGN] '
            f'center_error={center_error:.1f}px, heading_error={heading_error:.1f}px, '
            f'vy_corr={vy_corr:.3f}, wz_corr={wz_corr:.3f}, '
            f'cmd_vy={cmd_vy:.3f}, cmd_wz={cmd_wz:.3f}, '
            f'common_span={float(edge_result.get("common_y_span", 0.0)):.1f}',
            throttle_duration_sec=0.3
        )

        return cmd_vy, cmd_wz

    def draw_p5_inner_edge_debug(self, vis: np.ndarray, result: dict):
        if result is None:
            return

        h, w = vis.shape[:2]

        left_roi = result.get('left_roi')
        right_roi = result.get('right_roi')

        if left_roi is not None:
            x1, y1, x2, y2 = left_roi
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                vis,
                'INNER LEFT ROI',
                (x1 + 3, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                1
            )

        if right_roi is not None:
            x1, y1, x2, y2 = right_roi
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                vis,
                'INNER RIGHT ROI',
                (x1 + 3, max(45, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1
            )

        left_edge = result.get('left_edge')
        right_edge = result.get('right_edge')

        if left_edge is not None:
            self.draw_p5_inner_edge_line(vis, left_edge, (255, 0, 0), 'L_INNER')
        if right_edge is not None:
            self.draw_p5_inner_edge_line(vis, right_edge, (0, 0, 255), 'R_INNER')

        if result.get('common_top_y') is not None and result.get('common_bottom_y') is not None:
            common_top_y = int(result['common_top_y'])
            common_bottom_y = int(result['common_bottom_y'])
            cv2.line(vis, (0, common_top_y), (w - 1, common_top_y), (0, 180, 255), 1)
            cv2.line(vis, (0, common_bottom_y), (w - 1, common_bottom_y), (0, 180, 255), 1)

        if result.get('common_valid', False):
            bottom_center_x = int(result['bottom_center_x'])
            top_center_x = int(result['top_center_x'])
            bottom_center_y = int(result['bottom_center_y'])
            top_center_y = int(result['top_center_y'])

            cv2.circle(vis, (bottom_center_x, bottom_center_y), 7, (0, 255, 255), -1)
            cv2.circle(vis, (top_center_x, top_center_y), 7, (0, 255, 255), -1)
            cv2.line(
                vis,
                (bottom_center_x, bottom_center_y),
                (top_center_x, top_center_y),
                (0, 255, 255),
                2
            )

            cv2.putText(
                vis,
                f'INNER center={result["center_error"]:.1f}px '
                f'heading={result["heading_error"]:.1f}px '
                f'vy_c={result.get("cmd_vy_correction", 0.0):.3f} '
                f'wz_c={result.get("cmd_wz_correction", 0.0):.3f}',
                (10, h - 124),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                2
            )
        else:
            cv2.putText(
                vis,
                f'INNER no correction: L={result.get("has_left")} '
                f'R={result.get("has_right")} reason={result.get("common_reason")}',
                (10, h - 124),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                2
            )

    def draw_p5_inner_edge_line(self, vis: np.ndarray, edge: dict, color, name: str):
        pts = edge.get('points', [])

        for i, (x, y) in enumerate(pts):
            if i % 2 == 0:
                cv2.circle(vis, (int(x), int(y)), 2, color, -1)

        if edge.get('top_x') is not None and edge.get('bottom_x') is not None:
            top_pt = (int(edge['top_x']), int(edge['top_y']))
            bottom_pt = (int(edge['bottom_x']), int(edge['bottom_y']))

            cv2.circle(vis, top_pt, 6, color, -1)
            cv2.circle(vis, bottom_pt, 6, color, -1)
            cv2.line(vis, top_pt, bottom_pt, color, 2)

        text = (
            f'{name} valid={edge.get("valid")} '
            f'pts={edge.get("point_count", 0)}/{edge.get("raw_point_count", 0)} '
            f'bot={edge.get("bottom_ratio", 0):.2f} '
            f'yspan={edge.get("y_span", 0):.0f} '
            f'reason={edge.get("reason")}'
        )
        text_y = 108 if name == 'L_INNER' else 134
        cv2.putText(
            vis,
            text,
            (10, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1
        )


    def draw_p5_right_slope_right_edge_debug(self, vis: np.ndarray, result: dict):
        if result is None:
            return

        h, w = vis.shape[:2]
        roi = result.get('roi')
        if roi is not None:
            x1, y1, x2, y2 = roi
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                vis,
                'RIGHT SLOPE RIGHT EDGE ROI',
                (x1 + 3, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1
            )

        center_thr = result.get('too_center_threshold_x')
        right_thr = result.get('too_right_threshold_x')
        if center_thr is not None:
            cx = int(center_thr)
            cv2.line(vis, (cx, 0), (cx, h - 1), (255, 255, 0), 1)
            cv2.putText(vis, 'too_center', (max(5, cx - 80), 154),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
        if right_thr is not None:
            rx = int(right_thr)
            cv2.line(vis, (rx, 0), (rx, h - 1), (0, 255, 255), 2)
            cv2.putText(vis, 'too_right', (max(5, rx - 80), 178),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        for i, (x, y) in enumerate(result.get('raw_points', [])):
            if i % 3 == 0:
                cv2.circle(vis, (int(x), int(y)), 1, (120, 120, 120), -1)

        for i, (x, y) in enumerate(result.get('points', [])):
            if i % 2 == 0:
                cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)

        if result.get('top_x') is not None and result.get('bottom_x') is not None:
            top_pt = (int(result['top_x']), int(result['top_y']))
            bottom_pt = (int(result['bottom_x']), int(result['bottom_y']))
            cv2.circle(vis, top_pt, 5, (0, 0, 255), -1)
            cv2.circle(vis, bottom_pt, 5, (0, 0, 255), -1)
            cv2.line(vis, top_pt, bottom_pt, (0, 0, 255), 2)

        if result.get('bottom_band_low_y') is not None:
            by = int(result['bottom_band_low_y'])
            cv2.line(vis, (0, by), (w - 1, by), (0, 180, 255), 1)

        if result.get('valid') and result.get('right_inner_x') is not None:
            ix = int(result['right_inner_x'])
            cv2.line(vis, (ix, 0), (ix, h - 1), (0, 255, 0), 2)

        cmd_vx = result.get('cmd_vx')
        cmd_vy = result.get('cmd_vy')
        cmd_wz = result.get('cmd_wz')
        base_vy = result.get('base_vy')
        if cmd_vx is None or cmd_vy is None or cmd_wz is None:
            cmd_text = 'SEND cmd=(None,None,None)'
        else:
            cmd_text = f'SEND cmd=({float(cmd_vx):.3f},{float(cmd_vy):.3f},{float(cmd_wz):.3f})'

        cv2.putText(
            vis,
            f'R_SLOPE_EDGE valid={result.get("valid")} '
            f'too_center={result.get("too_center")} too_right={result.get("too_right")} '
            f'action={result.get("action")}',
            (10, h - 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 255, 255),
            2
        )
        cv2.putText(
            vis,
            f'{cmd_text} base_vy={base_vy} final_vy={cmd_vy}',
            (10, h - 152),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 255, 255),
            2
        )
        cv2.putText(
            vis,
            f'right_x={result.get("right_inner_x")} '
            f'pts={result.get("point_count", 0)}/{result.get("raw_point_count", 0)} '
            f'yspan={result.get("y_span", 0):.0f} '
            f'xstd={result.get("x_std", 0):.1f} '
            f'reason={result.get("reason")}',
            (10, h - 124),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 255, 255),
            2
        )
        cv2.putText(
            vis,
            f'lost_active={result.get("lost_extra_active")} '
            f'dir={result.get("lost_extra_direction")} '
            f'cntC={result.get("too_center_count", 0)}/'
            f'{result.get("lost_extra_confirm_count", 0)} '
            f'cntR={result.get("too_right_count", 0)}/'
            f'{result.get("lost_extra_confirm_count", 0)}',
            (10, h - 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 255, 255),
            2
        )


    # ============================================================
    # 可视化
    # ============================================================
    def show_p5_compact_debug_window(self, frame: np.ndarray):
        """
        简洁可视化模式：只保留当前调试最需要看的信息。
        不画 raw_points / points / bottom band / 大量 reason，避免画面太乱。
        """
        try:
            vis = frame.copy()
            h, w = vis.shape[:2]

            # 1. 图像中心线
            image_center_x = w // 2
            cv2.line(vis, (image_center_x, 0), (image_center_x, h - 1), (255, 255, 255), 1)

            # 2. 优先从右斜坡边缘结果里拿实际发送命令；如果还没有，就从 msg 里拿当前 vel_des。
            right_edge = getattr(self, 'latest_p5_right_slope_right_edge_result', None)
            cmd_vx = cmd_vy = cmd_wz = None
            if isinstance(right_edge, dict):
                cmd_vx = right_edge.get('cmd_vx')
                cmd_vy = right_edge.get('cmd_vy')
                cmd_wz = right_edge.get('cmd_wz')

            if cmd_vx is None or cmd_vy is None or cmd_wz is None:
                try:
                    vel = list(getattr(self.msg, 'vel_des', [0.0, 0.0, 0.0]))
                    cmd_vx, cmd_vy, cmd_wz = float(vel[0]), float(vel[1]), float(vel[2])
                except Exception:
                    cmd_vx, cmd_vy, cmd_wz = 0.0, 0.0, 0.0

            # 3. 右斜坡右侧边缘：只画 ROI、两个阈值、right_inner_x，不画点云。
            if isinstance(right_edge, dict):
                roi = right_edge.get('roi')
                if roi is not None:
                    x1, y1, x2, y2 = roi
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

                center_thr = right_edge.get('too_center_threshold_x')
                right_thr = right_edge.get('too_right_threshold_x')
                if center_thr is not None:
                    cx = int(center_thr)
                    cv2.line(vis, (cx, 0), (cx, h - 1), (255, 255, 0), 2)
                    cv2.putText(vis, 'center_thr', (max(5, cx - 72), 62),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

                if right_thr is not None:
                    rx = int(right_thr)
                    cv2.line(vis, (rx, 0), (rx, h - 1), (0, 255, 255), 2)
                    cv2.putText(vis, 'right_thr', (max(5, rx - 65), 88),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                if right_edge.get('valid') and right_edge.get('right_inner_x') is not None:
                    ix = int(right_edge['right_inner_x'])
                    cv2.line(vis, (ix, 0), (ix, h - 1), (0, 255, 0), 3)
                    cv2.putText(vis, 'right_inner', (max(5, ix - 70), 114),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # 4. 上坡/右跳后前进阶段的内侧边缘矫正：只显示数值，不画左右大量点。
            inner_edge = getattr(self, 'latest_p5_inner_edge_result', None)
            inner_text = 'INNER: n/a'
            if isinstance(inner_edge, dict):
                ce = inner_edge.get('center_error')
                he = inner_edge.get('heading_error')
                vy_c = float(inner_edge.get('cmd_vy_correction', 0.0))
                wz_c = float(inner_edge.get('cmd_wz_correction', 0.0))
                if ce is None or he is None:
                    inner_text = f'INNER: valid={inner_edge.get("common_valid")} reason={inner_edge.get("common_reason")}'
                else:
                    inner_text = (
                        f'INNER: valid={inner_edge.get("common_valid")} '
                        f'ce={float(ce):.1f} he={float(he):.1f} '
                        f'vy_c={vy_c:.3f} wz_c={wz_c:.3f}'
                    )

            # 5. 右斜坡边缘状态文字：压缩为 1 行。
            if isinstance(right_edge, dict):
                ratio = right_edge.get('right_inner_x_ratio')
                ratio_text = 'None' if ratio is None else f'{float(ratio):.3f}'
                edge_text = (
                    f'R_EDGE: valid={right_edge.get("valid")} '
                    f'ratio={ratio_text} action={right_edge.get("action")} '
                    f'lost={right_edge.get("lost_extra_active")} '
                    f'dir={right_edge.get("lost_extra_direction")} '
                    f'C/R={right_edge.get("too_center_count", 0)}/'
                    f'{right_edge.get("too_right_count", 0)} '
                    f'reason={right_edge.get("reason")}'
                )
            else:
                edge_text = 'R_EDGE: n/a'

            # 6. 总文字区：固定只显示 4 行。
            cv2.putText(
                vis,
                f'STATE: {self.state}',
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2
            )
            cv2.putText(
                vis,
                f'CMD: vx={float(cmd_vx):.3f} vy={float(cmd_vy):.3f} wz={float(cmd_wz):.3f}',
                (10, h - 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2
            )
            cv2.putText(
                vis,
                edge_text[:115],
                (10, h - 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 255),
                2
            )
            cv2.putText(
                vis,
                inner_text[:115],
                (10, h - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 0),
                2
            )

            cv2.imshow('P5 Bridge Debug', vis)

            # 简洁模式默认不强制显示 mask；只有用户显式打开 p5_show_yellow_mask 才显示。
            if self.p5_show_yellow_mask:
                inner_edge = getattr(self, 'latest_p5_inner_edge_result', None)
                inner_mask = None if inner_edge is None else inner_edge.get('mask')
                if inner_mask is not None:
                    cv2.imshow('P5 Yellow Mask', inner_mask)
                elif self.latest_p5_yellow_mask is not None:
                    cv2.imshow('P5 Yellow Mask', self.latest_p5_yellow_mask)

            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().warn(f'[P5_VIS_COMPACT] show debug failed: {repr(e)}')

    def show_p5_debug_window(self, frame: np.ndarray):
        try:
            detail_level = int(getattr(self, 'p5_debug_vis_detail_level', 1))
            if detail_level <= 0:
                return
            if detail_level == 1:
                self.show_p5_compact_debug_window(frame)
                return

            # detail_level >= 2：保留原来的完整可视化。
            vis = frame.copy()
            h, w = vis.shape[:2]

            yellow = self.latest_p5_yellow_result

            image_center_x = w // 2
            cv2.line(vis, (image_center_x, 0), (image_center_x, h - 1), (255, 255, 255), 1)

            roi_top = int(h * self.p5_yellow_roi_top_ratio)
            roi_left = int(w * self.p5_yellow_roi_left_ratio)
            roi_right = int(w * self.p5_yellow_roi_right_ratio)

            roi_top = max(0, min(h - 1, roi_top))
            roi_left = max(0, min(w - 1, roi_left))
            roi_right = max(roi_left + 1, min(w, roi_right))

            cv2.rectangle(vis, (roi_left, roi_top), (roi_right, h - 1), (0, 255, 255), 2)
            cv2.putText(
                vis,
                'P5 strict front yellow ROI',
                (roi_left + 3, max(20, roi_top - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1
            )

            threshold_y = int(h * self.p5_yellow_stop_line_y_ratio)
            cv2.line(vis, (0, threshold_y), (w - 1, threshold_y), (0, 180, 255), 2)
            cv2.putText(
                vis,
                f'stop_y={threshold_y} ratio={self.p5_yellow_stop_line_y_ratio:.2f}',
                (10, max(25, threshold_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 180, 255),
                2
            )

            cv2.putText(
                vis,
                f'state={self.state}',
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2
            )

            cv2.putText(
                vis,
                f'frame_seq={self.latest_frame_seq} state_enter_seq={self.state_enter_frame_seq}',
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1
            )

            if yellow.get('has_line') and yellow.get('line_bottom_y') is not None:
                bottom_y = int(yellow['line_bottom_y'])
                bbox = yellow.get('bbox')
                center = yellow.get('line_center')
                angle = yellow.get('angle_deg')
                width_ratio = yellow.get('width_ratio')
                wh_ratio = yellow.get('wh_ratio')

                cv2.line(vis, (0, bottom_y), (w - 1, bottom_y), (0, 255, 255), 2)

                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

                if center is not None:
                    cx, cy = center
                    cv2.circle(vis, (cx, cy), 6, (0, 255, 255), -1)

                    if angle is not None:
                        length = 80
                        rad = math.radians(float(angle))
                        dx = int(math.cos(rad) * length)
                        dy = int(math.sin(rad) * length)
                        cv2.line(vis, (cx - dx, cy - dy), (cx + dx, cy + dy), (0, 0, 255), 2)

                    angle_text = 'None' if angle is None else f'{float(angle):.1f}deg'
                    cv2.putText(
                        vis,
                        f'YELLOW bottom={bottom_y} angle={angle_text}',
                        (max(5, cx - 120), max(18, cy - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 255, 255),
                        2
                    )

                cv2.putText(
                    vis,
                    f'counter={self.p5_yellow_stop_counter}/{self.p5_yellow_stop_confirm_count} '
                    f'width_ratio={width_ratio:.2f} wh={wh_ratio:.1f}',
                    (10, h - 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 255),
                    2
                )

            else:
                cv2.putText(
                    vis,
                    'P5 strict front yellow: NOT detected',
                    (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 255),
                    2
                )

            # P5_RIGHT_SLOPE 专用：中间区域黄色存在/消失可视化
            center_yellow = getattr(self, 'latest_p5_center_yellow_presence_result', None)
            if center_yellow is not None:
                roi = center_yellow.get('roi')
                if roi is not None:
                    cx1, cy1, cx2, cy2 = roi
                    color = (0, 255, 0) if center_yellow.get('has_yellow', False) else (0, 0, 255)
                    cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), color, 2)
                    cv2.putText(
                        vis,
                        'P5 center yellow presence ROI',
                        (cx1 + 3, max(20, cy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1
                    )

                bbox = center_yellow.get('bbox')
                if bbox is not None:
                    bx1, by1, bx2, by2 = bbox
                    cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

                cv2.putText(
                    vis,
                    f'CENTER_YELLOW has={center_yellow.get("has_yellow")} '
                    f'pixels={center_yellow.get("yellow_pixels")}/'
                    f'{self.p5_center_yellow_min_pixels} '
                    f'ratio={float(center_yellow.get("yellow_ratio", 0.0)):.4f}/'
                    f'{self.p5_center_yellow_min_ratio:.4f} '
                    f'absent={self.p5_center_yellow_absent_counter}/'
                    f'{self.p5_center_yellow_absent_confirm_count}',
                    (10, h - 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0) if center_yellow.get('has_yellow', False) else (0, 0, 255),
                    2
                )


            # P5_UP_SLOPE 专用：右侧赛道黄线可视化
            right_side = getattr(self, 'latest_p5_right_side_yellow_result', None)
            if right_side is not None:
                roi = right_side.get('roi')
                if roi is not None:
                    rx1, ry1, rx2, ry2 = roi
                    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 0, 255), 2)
                    cv2.putText(
                        vis,
                        'P5 right-side yellow ROI',
                        (rx1 + 3, max(20, ry1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 0, 255),
                        1
                    )

                for c in right_side.get('candidates', []):
                    bx1, by1, bx2, by2 = c['bbox']
                    cv2.rectangle(vis, (bx1, by1), (bx2, by2), (120, 120, 120), 1)

                bbox = right_side.get('bbox')
                if bbox is not None:
                    bx1, by1, bx2, by2 = bbox
                    color = (255, 0, 255) if right_side.get('has_line', False) else (0, 0, 255)
                    cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 3)

                    center = right_side.get('center')
                    if center is not None:
                        cx, cy = center
                        cv2.circle(vis, (cx, cy), 5, color, -1)

                ratio = right_side.get('bottom_ratio')
                ratio_text = 'None' if ratio is None else f'{float(ratio):.3f}'
                cv2.putText(
                    vis,
                    f'RIGHT_SIDE has={right_side.get("has_line")} '
                    f'reason={right_side.get("reason")} '
                    f'ratio={ratio_text}/{self.p5_right_side_yellow_bottom_valid_ratio:.2f} '
                    f'lost={self.p5_right_side_yellow_lost_counter}/'
                    f'{self.p5_right_side_yellow_lost_confirm_count}',
                    (10, h - 96),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 255),
                    2
                )

            # P5_UP_SLOPE 专用：左右内侧黄线边缘矫正可视化
            self.draw_p5_inner_edge_debug(
                vis,
                getattr(self, 'latest_p5_inner_edge_result', None)
            )

            # P5_RIGHT_SLOPE_1/2/3 专用：右侧黄线内侧边缘 vy 修正可视化
            self.draw_p5_right_slope_right_edge_debug(
                vis,
                getattr(self, 'latest_p5_right_slope_right_edge_result', None)
            )

            cv2.imshow('P5 Bridge Debug', vis)

            if self.p5_show_yellow_mask:
                inner_edge = getattr(self, 'latest_p5_inner_edge_result', None)
                inner_mask = None if inner_edge is None else inner_edge.get('mask')
                if inner_mask is not None:
                    cv2.imshow('P5 Yellow Mask', inner_mask)
                elif self.latest_p5_yellow_mask is not None:
                    cv2.imshow('P5 Yellow Mask', self.latest_p5_yellow_mask)

            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().warn(f'[P5_VIS] show debug failed: {repr(e)}')

    # ============================================================
    # 通用状态执行函数
    # ============================================================
    def run_timed_velocity_state(
        self,
        duration_s: float,
        vx: float,
        vy: float,
        wz: float,
        step_height: float,
        next_state: str,
        roll: float = 0.0,
        pitch: float = 0.0,
        body_height: float = 0.25,
        log_name: str = '',
    ):
        elapsed = self.state_elapsed_s()

        if elapsed < duration_s:
            self.send_velocity_command(
                vx=vx,
                vy=vy,
                wz=wz,
                step_height=step_height,
                roll=roll,
                pitch=pitch,
                body_height=body_height,
            )

            self.get_logger().info(
                f'[{log_name or self.state}] moving: '
                f'elapsed={elapsed:.3f}/{duration_s:.3f}s, '
                f'cmd=({vx:.3f},{vy:.3f},{wz:.3f}), '
                f'step_height={step_height:.3f}, '
                f'roll={roll:.3f}, pitch={pitch:.3f}, body_height={body_height:.3f}',
                throttle_duration_sec=0.5
            )
            return

        self.get_logger().info(
            f'[{log_name or self.state}] done, go {next_state}'
        )

        # 不在持续速度段之间自动 stop，保证 P5_STEP_UP -> P5_UP_SLOPE 连续衔接。
        self.enter_state(next_state)

    def run_yellow_stop_velocity_state(
        self,
        vx: float,
        vy: float,
        wz: float,
        step_height: float,
        next_state: str,
        roll: float = 0.0,
        pitch: float = 0.0,
        body_height: float = 0.25,
        log_name: str = '',
    ):
        elapsed = self.state_elapsed_s()

        # 刚进入状态时，先忽略旧帧/旧黄线，避免动作刚切完立刻用上一状态图像触发。
        if elapsed < self.p5_yellow_ignore_after_enter_s:
            self.send_velocity_command(
                vx=vx,
                vy=vy,
                wz=wz,
                step_height=step_height,
                roll=roll,
                pitch=pitch,
                body_height=body_height,
            )
            self.get_logger().info(
                f'[{log_name or self.state}] ignore yellow after enter: '
                f'elapsed={elapsed:.3f}/{self.p5_yellow_ignore_after_enter_s:.3f}, '
                f'cmd=({vx:.3f},{vy:.3f},{wz:.3f})',
                throttle_duration_sec=0.3
            )
            return

        # 如果当前状态还没有收到新图像，先继续走，避免使用旧帧误触发。
        if self.latest_frame_seq <= self.state_enter_frame_seq:
            if self.p5_keep_moving_when_no_image:
                self.send_velocity_command(
                    vx=vx,
                    vy=vy,
                    wz=wz,
                    step_height=step_height,
                    roll=roll,
                    pitch=pitch,
                    body_height=body_height,
                )
                self.get_logger().warn(
                    f'[{log_name or self.state}] no new image after state enter, keep moving: '
                    f'frame_seq={self.latest_frame_seq}, enter_seq={self.state_enter_frame_seq}',
                    throttle_duration_sec=0.5
                )
            else:
                self.send_stop_command()
                self.get_logger().warn(
                    f'[{log_name or self.state}] no new image after state enter, stop',
                    throttle_duration_sec=0.5
                )
            return

        yellow = self.latest_p5_yellow_result

        yellow_wz = self.compute_p5_yellow_angle_align_wz(yellow)
        cmd_wz = yellow_wz if abs(yellow_wz) > 1e-6 else wz

        if self.p5_yellow_reached_bottom(yellow):
            bottom_y = yellow.get('line_bottom_y')
            self.get_logger().info(
                f'[{log_name or self.state}] yellow reached bottom, stop and go {next_state}: '
                f'bottom={bottom_y}'
            )
            self.send_stop_command()
            self.enter_state(next_state)
            return

        self.send_velocity_command(
            vx=vx,
            vy=vy,
            wz=cmd_wz,
            step_height=step_height,
            roll=roll,
            pitch=pitch,
            body_height=body_height,
        )

        if yellow is None or not yellow.get('has_line', False):
            self.get_logger().info(
                f'[{log_name or self.state}] no strict front yellow, keep moving: '
                f'cmd=({vx:.3f},{vy:.3f},{cmd_wz:.3f})',
                throttle_duration_sec=0.5
            )
        else:
            angle = yellow.get('angle_deg')
            angle_text = 'None' if angle is None else f'{float(angle):.2f}'
            self.get_logger().info(
                f'[{log_name or self.state}] strict front yellow seen, keep moving: '
                f'bottom={yellow.get("line_bottom_y")}, '
                f'angle={angle_text}, '
                f'cmd=({vx:.3f},{vy:.3f},{cmd_wz:.3f})',
                throttle_duration_sec=0.5
            )


    # ============================================================
    # P5_RIGHT_SLOPE：右侧黄线内侧边缘检测与 vy 三档修正
    # ============================================================
    def keep_p5_right_slope_bottom_connected_segment(self, edge_points):
        """只保留从图像底部往上的连续右侧内侧边缘段。"""
        if not edge_points:
            return []

        if not self.p5_right_slope_right_edge_use_bottom_connected_segment:
            return sorted(edge_points, key=lambda p: p[1])

        max_y_gap = max(1, int(self.p5_right_slope_right_edge_max_y_gap))
        pts_bottom_to_top = sorted(edge_points, key=lambda p: p[1], reverse=True)

        kept = [pts_bottom_to_top[0]]
        prev_y = pts_bottom_to_top[0][1]

        for x, y in pts_bottom_to_top[1:]:
            y_gap = abs(prev_y - y)
            if y_gap > max_y_gap:
                break
            kept.append((x, y))
            prev_y = y

        return sorted(kept, key=lambda p: p[1])

    def detect_p5_right_slope_right_inner_edge(self, frame: np.ndarray) -> dict:
        """
        右斜坡阶段专用：只检测右侧黄线的内侧边缘。

        在右侧 ROI 内，每一行取最左侧黄色像素，得到右侧黄线内侧边缘点串。
        然后用点数、纵向跨度、x_std 和 bottom_ratio 判断是否有效。
        最后用底部 band 的平均 x 作为 right_inner_x。
        """
        h, w = frame.shape[:2]

        mask = self.make_p5_inner_edge_yellow_mask(frame)

        x1 = int(w * self.p5_right_slope_right_edge_roi_x_min)
        x2 = int(w * self.p5_right_slope_right_edge_roi_x_max)
        y1 = int(h * self.p5_right_slope_right_edge_roi_y_min)
        y2 = int(h * self.p5_right_slope_right_edge_roi_y_max)
        x1, y1, x2, y2 = self.clamp_p5_roi((x1, y1, x2, y2), w, h)

        roi_mask = mask[y1:y2, x1:x2]
        row_step = max(1, int(self.p5_right_slope_right_edge_row_step))

        raw_points = []
        for local_y in range(0, roi_mask.shape[0], row_step):
            row = roi_mask[local_y, :]
            xs = np.where(row > 0)[0]
            if xs.size == 0:
                continue

            # 右侧黄线取内侧边缘：每行最左侧黄色像素
            local_x = int(np.min(xs))
            raw_points.append((int(x1 + local_x), int(y1 + local_y)))

        points = self.keep_p5_right_slope_bottom_connected_segment(raw_points)

        result = {
            'mask': mask,
            'roi': (int(x1), int(y1), int(x2), int(y2)),
            'raw_points': raw_points,
            'points': points,
            'valid': False,
            'reason': 'init',
            'point_count': len(points),
            'raw_point_count': len(raw_points),
            'y_span': 0.0,
            'x_std': 0.0,
            'bottom_ratio': 0.0,
            'top_x': None,
            'top_y': None,
            'bottom_x': None,
            'bottom_y': None,
            'right_inner_x': None,
            'right_inner_x_ratio': None,
            'too_center_threshold_x': float(w * self.p5_right_slope_right_too_center_ratio),
            'too_right_threshold_x': float(w * self.p5_right_slope_right_too_right_ratio),
            'too_center': False,
            'too_right': False,
            'bottom_band_low_y': None,
            'cmd_vy': None,
        }

        if len(raw_points) == 0:
            result['reason'] = 'no_yellow_points'
            return result

        if len(points) == 0:
            result['reason'] = 'no_bottom_connected_segment'
            return result

        pts = np.array(points, dtype=np.float32)
        xs = pts[:, 0]
        ys = pts[:, 1]

        point_count = len(points)
        y_min = float(np.min(ys))
        y_max = float(np.max(ys))
        y_span = y_max - y_min
        x_std = float(np.std(xs))
        bottom_ratio = y_max / float(max(h, 1))

        top_band_high = y_min + 0.20 * max(y_span, 1.0)
        bottom_band_low = y_max - 0.20 * max(y_span, 1.0)
        top_band = pts[pts[:, 1] <= top_band_high]
        bottom_band = pts[pts[:, 1] >= bottom_band_low]
        if len(top_band) == 0:
            top_band = pts
        if len(bottom_band) == 0:
            bottom_band = pts

        top_x = float(np.mean(top_band[:, 0]))
        top_y = float(np.mean(top_band[:, 1]))
        bottom_x = float(np.mean(bottom_band[:, 0]))
        bottom_y = float(np.mean(bottom_band[:, 1]))

        # 用底部 band 计算 right_inner_x，更接近机器狗当前近处位置。
        bottom_band_low_y = y_max - self.p5_right_slope_right_edge_bottom_band_ratio * max(y_span, 1.0)
        inner_band = pts[pts[:, 1] >= bottom_band_low_y]
        if len(inner_band) == 0:
            inner_band = pts

        right_inner_x = float(np.mean(inner_band[:, 0]))
        right_inner_x_ratio = right_inner_x / float(max(w, 1))

        fail_reasons = []
        if point_count < self.p5_right_slope_right_edge_min_points:
            fail_reasons.append(f'points<{self.p5_right_slope_right_edge_min_points}')
        if y_span < self.p5_right_slope_right_edge_min_y_span:
            fail_reasons.append(f'y_span<{self.p5_right_slope_right_edge_min_y_span:.0f}')
        if x_std > self.p5_right_slope_right_edge_x_std_max:
            fail_reasons.append(f'x_std>{self.p5_right_slope_right_edge_x_std_max:.0f}')
        if bottom_ratio < self.p5_right_slope_right_edge_bottom_min_ratio:
            fail_reasons.append(f'bottom<{self.p5_right_slope_right_edge_bottom_min_ratio:.2f}')

        valid = len(fail_reasons) == 0

        too_center_threshold_x = w * self.p5_right_slope_right_too_center_ratio
        too_right_threshold_x = w * self.p5_right_slope_right_too_right_ratio
        too_center = bool(valid and right_inner_x < too_center_threshold_x)
        too_right = bool(valid and right_inner_x > too_right_threshold_x)

        result.update({
            'valid': bool(valid),
            'reason': 'ok' if valid else ','.join(fail_reasons),
            'point_count': int(point_count),
            'raw_point_count': int(len(raw_points)),
            'y_span': float(y_span),
            'x_std': float(x_std),
            'bottom_ratio': float(bottom_ratio),
            'top_x': float(top_x),
            'top_y': float(top_y),
            'bottom_x': float(bottom_x),
            'bottom_y': float(bottom_y),
            'right_inner_x': float(right_inner_x),
            'right_inner_x_ratio': float(right_inner_x_ratio),
            'too_center_threshold_x': float(too_center_threshold_x),
            'too_right_threshold_x': float(too_right_threshold_x),
            'too_center': bool(too_center),
            'too_right': bool(too_right),
            'bottom_band_low_y': float(bottom_band_low_y),
        })

        return result

    def reset_p5_right_slope_lost_extra_state(self):
        """
        清空右斜坡“危险后丢线持续补偿”状态。

        每次进入新的 P5_RIGHT_SLOPE_1/2/3 时调用，
        避免上一段右斜坡的危险方向影响下一段。
        """
        self.p5_right_slope_too_center_count = 0
        self.p5_right_slope_too_right_count = 0
        self.p5_right_slope_lost_extra_active = False
        self.p5_right_slope_lost_extra_direction = 'none'

        if isinstance(getattr(self, 'latest_p5_right_slope_right_edge_result', None), dict):
            self.latest_p5_right_slope_right_edge_result['lost_extra_active'] = False
            self.latest_p5_right_slope_right_edge_result['lost_extra_direction'] = 'none'
            self.latest_p5_right_slope_right_edge_result['too_center_count'] = 0
            self.latest_p5_right_slope_right_edge_result['too_right_count'] = 0
            self.latest_p5_right_slope_right_edge_result['record_ignore_active'] = True
            self.latest_p5_right_slope_right_edge_result['record_ignore_elapsed_s'] = 0.0
            self.latest_p5_right_slope_right_edge_result['record_ignore_duration_s'] = float(
                getattr(self, 'p5_right_slope_lost_extra_ignore_after_enter_s', 0.0)
            )
            self.latest_p5_right_slope_right_edge_result['action'] = 'reset_on_enter_right_slope'

        self.get_logger().info(
            '[P5_RIGHT_SLOPE_RIGHT_EDGE_VY] reset lost-extra state for new right-slope segment'
        )


    def compute_p5_right_slope_right_edge_corrected_vy(self, base_vy: float, frame: np.ndarray) -> float:
        """
        右斜坡阶段 vy 三档修正 + 危险后丢线持续补偿。

        当前看得到右侧黄线时：
        - right_inner_x 太靠中间：固定加大 vy；
        - right_inner_x 太靠右：固定减小 vy；
        - too_center / too_right 连续达到确认次数后，记录危险方向。

        后续当前右侧黄线无效 / 识别不到时：
        - 如果本段右斜坡前面已经确认过 too_center，则持续叠加 lost_extra_too_center_vy；
        - 如果本段右斜坡前面已经确认过 too_right，则持续叠加 lost_extra_too_right_vy；
        - 如果本段右斜坡前面没有确认过危险，则保持 base_vy。

        注意：lost_extra_active 一旦触发，会保持到当前 P5_RIGHT_SLOPE_x 状态结束；
        进入下一段 P5_RIGHT_SLOPE_1/2/3 时由 enter_state() 清零。
        """
        if not self.p5_right_slope_right_edge_vy_adjust_enabled:
            return base_vy
        if frame is None:
            return base_vy

        result = self.detect_p5_right_slope_right_inner_edge(frame)

        cmd_vy = base_vy
        action = 'base'

        valid = bool(result.get('valid', False))
        too_center = bool(result.get('too_center', False))
        too_right = bool(result.get('too_right', False))

        elapsed = self.state_elapsed_s()
        ignore_duration = float(getattr(self, 'p5_right_slope_lost_extra_ignore_after_enter_s', 0.0))
        record_ignore_active = elapsed < ignore_duration

        if valid:
            # 当前看得到右侧黄线：使用原来的三档修正，同时更新危险趋势计数。
            if too_center:
                cmd_vy = base_vy + self.p5_right_slope_right_too_center_add_vy

                if record_ignore_active:
                    # 刚进入当前右斜坡段的一小段时间：只修正，不记录危险次数，
                    # 避免切状态瞬间、机身未稳定或旧帧导致误触发 lost-extra。
                    self.p5_right_slope_too_center_count = 0
                    self.p5_right_slope_too_right_count = 0
                    action = 'visible_too_center_add_vy_ignore_record'
                else:
                    self.p5_right_slope_too_center_count += 1
                    self.p5_right_slope_too_right_count = 0
                    action = 'visible_too_center_add_vy'

                    if (
                        self.p5_right_slope_lost_extra_enabled
                        and self.p5_right_slope_too_center_count >= self.p5_right_slope_lost_extra_confirm_count
                    ):
                        self.p5_right_slope_lost_extra_active = True
                        self.p5_right_slope_lost_extra_direction = 'too_center'

            elif too_right:
                cmd_vy = base_vy - self.p5_right_slope_right_too_right_reduce_vy

                if record_ignore_active:
                    # 刚进入当前右斜坡段的一小段时间：只修正，不记录危险次数。
                    self.p5_right_slope_too_center_count = 0
                    self.p5_right_slope_too_right_count = 0
                    action = 'visible_too_right_reduce_vy_ignore_record'
                else:
                    self.p5_right_slope_too_right_count += 1
                    self.p5_right_slope_too_center_count = 0
                    action = 'visible_too_right_reduce_vy'

                    if (
                        self.p5_right_slope_lost_extra_enabled
                        and self.p5_right_slope_too_right_count >= self.p5_right_slope_lost_extra_confirm_count
                    ):
                        self.p5_right_slope_lost_extra_active = True
                        self.p5_right_slope_lost_extra_direction = 'too_right'

            else:
                # 看得到黄线且处在安全范围：只清当前连续计数。
                # 不清 lost_extra_active，因为用户需求是：
                # “本段右斜坡一旦前面超过危险次数，后面识别不到就一直补偿到本段结束”。
                self.p5_right_slope_too_center_count = 0
                self.p5_right_slope_too_right_count = 0
                cmd_vy = base_vy
                action = 'visible_safe_base'

        else:
            # 当前看不到/无效：只有“本段前面已经确认过危险方向”才继续补偿。
            if self.p5_right_slope_lost_extra_enabled and self.p5_right_slope_lost_extra_active:
                if self.p5_right_slope_lost_extra_direction == 'too_center':
                    cmd_vy = base_vy + self.p5_right_slope_lost_extra_too_center_vy
                    action = 'lost_hold_too_center_extra_vy'
                elif self.p5_right_slope_lost_extra_direction == 'too_right':
                    cmd_vy = base_vy + self.p5_right_slope_lost_extra_too_right_vy
                    action = 'lost_hold_too_right_extra_vy'
                else:
                    cmd_vy = base_vy
                    action = 'lost_active_but_no_direction'
            else:
                cmd_vy = base_vy
                action = 'lost_no_extra_base'

        result['cmd_vy'] = float(cmd_vy)
        result['base_vy'] = float(base_vy)
        result['action'] = action

        result['lost_extra_enabled'] = bool(self.p5_right_slope_lost_extra_enabled)
        result['lost_extra_active'] = bool(self.p5_right_slope_lost_extra_active)
        result['lost_extra_direction'] = str(self.p5_right_slope_lost_extra_direction)
        result['too_center_count'] = int(self.p5_right_slope_too_center_count)
        result['too_right_count'] = int(self.p5_right_slope_too_right_count)
        result['lost_extra_confirm_count'] = int(self.p5_right_slope_lost_extra_confirm_count)
        result['record_ignore_active'] = bool(record_ignore_active)
        result['record_ignore_elapsed_s'] = float(elapsed)
        result['record_ignore_duration_s'] = float(ignore_duration)
        result['lost_extra_too_center_vy'] = float(self.p5_right_slope_lost_extra_too_center_vy)
        result['lost_extra_too_right_vy'] = float(self.p5_right_slope_lost_extra_too_right_vy)

        self.latest_p5_right_slope_right_edge_result = result

        self.get_logger().info(
            f'[P5_RIGHT_SLOPE_RIGHT_EDGE_VY] '
            f'valid={valid}, reason={result.get("reason")}, '
            f'right_x={result.get("right_inner_x")}, '
            f'ratio={result.get("right_inner_x_ratio")}, '
            f'center_thr={result.get("too_center_threshold_x")}, '
            f'right_thr={result.get("too_right_threshold_x")}, '
            f'too_center={too_center}, too_right={too_right}, '
            f'cnt_center={self.p5_right_slope_too_center_count}/'
            f'{self.p5_right_slope_lost_extra_confirm_count}, '
            f'cnt_right={self.p5_right_slope_too_right_count}/'
            f'{self.p5_right_slope_lost_extra_confirm_count}, '
            f'ignore_record={record_ignore_active}({elapsed:.2f}/{ignore_duration:.2f}s), '
            f'lost_active={self.p5_right_slope_lost_extra_active}, '
            f'lost_dir={self.p5_right_slope_lost_extra_direction}, '
            f'action={action}, base_vy={base_vy:.3f}, cmd_vy={cmd_vy:.3f}',
            throttle_duration_sec=0.3
        )

        return cmd_vy


    def run_center_yellow_absence_velocity_state(
        self,
        vx: float,
        vy: float,
        wz: float,
        step_height: float,
        next_state: str,
        roll: float = 0.0,
        pitch: float = 0.0,
        body_height: float = 0.25,
        log_name: str = '',
        stop_before_next: bool = False,
    ):
        """
        P5_RIGHT_SLOPE_1/2/3 专用：
        中间 ROI 内还有黄色 -> 继续走；
        中间 ROI 内连续 N 帧没有黄色 -> 进入 next_state。

        默认不主动 stop，保持 1/2 段连续衔接转向；
        如果 stop_before_next=True，则在切入 next_state 前先发送一次速度为 0 的速度命令，
        主要用于 P5_RIGHT_SLOPE_3 结束后清掉上一段运动速度，再 reset body。
        注意：这里不调用 send_stop_command()，避免 Ctrl.Wait_finish(12,0) 阻塞。
        """
        elapsed = self.state_elapsed_s()

        if elapsed < self.p5_center_yellow_ignore_after_enter_s:
            self.send_velocity_command(
                vx=vx,
                vy=vy,
                wz=wz,
                step_height=step_height,
                roll=roll,
                pitch=pitch,
                body_height=body_height,
            )
            self.get_logger().info(
                f'[{log_name or self.state}] ignore center yellow after enter: '
                f'elapsed={elapsed:.3f}/{self.p5_center_yellow_ignore_after_enter_s:.3f}, '
                f'cmd=({vx:.3f},{vy:.3f},{wz:.3f})',
                throttle_duration_sec=0.3
            )
            return

        if self.latest_frame_seq <= self.state_enter_frame_seq:
            if self.p5_keep_moving_when_no_image:
                self.send_velocity_command(
                    vx=vx,
                    vy=vy,
                    wz=wz,
                    step_height=step_height,
                    roll=roll,
                    pitch=pitch,
                    body_height=body_height,
                )
                self.get_logger().warn(
                    f'[{log_name or self.state}] no new image after state enter, keep moving: '
                    f'frame_seq={self.latest_frame_seq}, enter_seq={self.state_enter_frame_seq}',
                    throttle_duration_sec=0.5
                )
            else:
                self.send_stop_command()
                self.get_logger().warn(
                    f'[{log_name or self.state}] no new image after state enter, stop',
                    throttle_duration_sec=0.5
                )
            return

        if self.latest_bgr is None:
            self.send_velocity_command(
                vx=vx,
                vy=vy,
                wz=wz,
                step_height=step_height,
                roll=roll,
                pitch=pitch,
                body_height=body_height,
            )
            self.get_logger().warn(
                f'[{log_name or self.state}] no image, keep moving',
                throttle_duration_sec=0.5
            )
            return

        result = self.detect_p5_center_yellow_presence(self.latest_bgr)
        self.latest_p5_center_yellow_presence_result = result

        # 右斜坡主运动阶段专用：根据右侧黄线内侧边缘，对 vy 做三档固定修正。
        # 只影响 P5_RIGHT_SLOPE_1/2/3，不影响转向和额外前进阶段。
        cmd_vy = self.compute_p5_right_slope_right_edge_corrected_vy(
            base_vy=vy,
            frame=self.latest_bgr,
        )

        # 给可视化保存当前右斜坡阶段实际准备发送的控制命令。
        # draw_p5_right_slope_right_edge_debug() 会把这个 cmd=(vx,vy,wz) 直接画出来。
        if isinstance(getattr(self, 'latest_p5_right_slope_right_edge_result', None), dict):
            self.latest_p5_right_slope_right_edge_result['cmd_vx'] = float(vx)
            self.latest_p5_right_slope_right_edge_result['cmd_vy'] = float(cmd_vy)
            self.latest_p5_right_slope_right_edge_result['cmd_wz'] = float(wz)
            self.latest_p5_right_slope_right_edge_result['cmd_step_height'] = float(step_height)
            self.latest_p5_right_slope_right_edge_result['control_state'] = str(self.state)

        if result.get('has_yellow', False):
            self.p5_center_yellow_absent_counter = 0
            self.send_velocity_command(
                vx=vx,
                vy=cmd_vy,
                wz=wz,
                step_height=step_height,
                roll=roll,
                pitch=pitch,
                body_height=body_height,
            )
            self.get_logger().info(
                f'[{log_name or self.state}] center yellow present, keep moving: '
                f'pixels={result.get("yellow_pixels")}/{self.p5_center_yellow_min_pixels}, '
                f'ratio={float(result.get("yellow_ratio", 0.0)):.4f}/'
                f'{self.p5_center_yellow_min_ratio:.4f}, '
                f'cmd=({vx:.3f},{cmd_vy:.3f},{wz:.3f})',
                throttle_duration_sec=0.5
            )
            return

        self.p5_center_yellow_absent_counter += 1

        self.get_logger().info(
            f'[{log_name or self.state}] center yellow absent: '
            f'pixels={result.get("yellow_pixels")}/{self.p5_center_yellow_min_pixels}, '
            f'ratio={float(result.get("yellow_ratio", 0.0)):.4f}/'
            f'{self.p5_center_yellow_min_ratio:.4f}, '
            f'counter={self.p5_center_yellow_absent_counter}/'
            f'{self.p5_center_yellow_absent_confirm_count}',
            throttle_duration_sec=0.3
        )

        if self.p5_center_yellow_absent_counter >= self.p5_center_yellow_absent_confirm_count:
            if stop_before_next:
                self.get_logger().info(
                    f'[{log_name or self.state}] center yellow disappeared, '
                    f'send one zero-velocity command before entering {next_state}'
                )

                # 这里只发送一次速度为 0 的 mode=11/gait=3 命令，清掉上一段右斜坡速度。
                # 不调用 send_stop_command()，避免 Ctrl.Wait_finish(12,0) 阻塞，
                # 也避免阻塞后 /clock 刷新导致下一个计时状态 elapsed 异常。
                self.send_velocity_command(
                    vx=0.0,
                    vy=0.0,
                    wz=0.0,
                    step_height=0.0,
                    roll=roll,
                    pitch=pitch,
                    body_height=body_height,
                )
            else:
                self.get_logger().info(
                    f'[{log_name or self.state}] center yellow disappeared, '
                    f'go {next_state}'
                )

            # 这里没有阻塞等待，所以直接 enter_state，不走 enter_state_after_blocking_wait。
            self.enter_state(next_state)
            return

        self.send_velocity_command(
            vx=vx,
            vy=cmd_vy,
            wz=wz,
            step_height=step_height,
            roll=roll,
            pitch=pitch,
            body_height=body_height,
        )

    def run_right_side_yellow_lost_velocity_state(
        self,
        vx: float,
        vy: float,
        wz: float,
        step_height: float,
        next_state: str,
        roll: float = 0.0,
        pitch: float = 0.0,
        body_height: float = 0.25,
        log_name: str = '',
    ):
        """
        P5_UP_SLOPE 专用：
        1. 右侧赛道黄线还在图像底部附近 -> 继续上坡；
        2. 上坡过程中，如果左右内侧边缘都有效，则用 center_error 修正 vy，用 heading_error 修正 wz；
        3. 右侧赛道黄线连续 lost N 帧 -> 进入 next_state。
        """
        elapsed = self.state_elapsed_s()

        if elapsed < self.p5_right_side_yellow_ignore_after_enter_s:
            self.send_velocity_command(
                vx=vx,
                vy=vy,
                wz=wz,
                step_height=step_height,
                roll=roll,
                pitch=pitch,
                body_height=body_height,
            )
            self.get_logger().info(
                f'[{log_name or self.state}] ignore right-side yellow after enter: '
                f'elapsed={elapsed:.3f}/{self.p5_right_side_yellow_ignore_after_enter_s:.3f}, '
                f'cmd=({vx:.3f},{vy:.3f},{wz:.3f})',
                throttle_duration_sec=0.3
            )
            return

        if self.latest_bgr is None:
            self.send_velocity_command(
                vx=vx,
                vy=vy,
                wz=wz,
                step_height=step_height,
                roll=roll,
                pitch=pitch,
                body_height=body_height,
            )
            self.get_logger().warn(
                f'[{log_name or self.state}] no image, keep moving',
                throttle_duration_sec=0.5
            )
            return

        result = self.detect_p5_right_side_yellow_line(self.latest_bgr)
        self.latest_p5_right_side_yellow_result = result

        # 上坡内侧边缘修正：不负责结束，只负责改 vy / wz。
        cmd_vy, cmd_wz = self.compute_p5_up_slope_inner_edge_corrected_cmd(
            base_vy=vy,
            base_wz=wz,
            frame=self.latest_bgr,
        )

        if result.get('has_line', False):
            self.p5_right_side_yellow_lost_counter = 0

            self.send_velocity_command(
                vx=vx,
                vy=cmd_vy,
                wz=cmd_wz,
                step_height=step_height,
                roll=roll,
                pitch=pitch,
                body_height=body_height,
            )

            self.get_logger().info(
                f'[{log_name or self.state}] right-side yellow valid, keep moving: '
                f'bottom={result.get("bottom_y")}, '
                f'ratio={float(result.get("bottom_ratio", 0.0)):.3f}/'
                f'{self.p5_right_side_yellow_bottom_valid_ratio:.3f}, '
                f'bbox={result.get("bbox")}, '
                f'cmd=({vx:.3f},{cmd_vy:.3f},{cmd_wz:.3f})',
                throttle_duration_sec=0.5
            )
            return

        self.p5_right_side_yellow_lost_counter += 1

        ratio = result.get('bottom_ratio')
        ratio_text = 'None' if ratio is None else f'{float(ratio):.3f}'

        self.get_logger().info(
            f'[{log_name or self.state}] right-side yellow invalid/lost: '
            f'reason={result.get("reason")}, '
            f'bottom={result.get("bottom_y")}, '
            f'ratio={ratio_text}, '
            f'counter={self.p5_right_side_yellow_lost_counter}/'
            f'{self.p5_right_side_yellow_lost_confirm_count}, '
            f'candidates={len(result.get("candidates", []))}',
            throttle_duration_sec=0.3
        )

        if self.p5_right_side_yellow_lost_counter >= self.p5_right_side_yellow_lost_confirm_count:
            self.get_logger().info(
                f'[{log_name or self.state}] right-side yellow disappeared, '
                f'go {next_state} directly'
            )
            self.enter_state(next_state)
            return

        self.send_velocity_command(
            vx=vx,
            vy=cmd_vy,
            wz=cmd_wz,
            step_height=step_height,
            roll=roll,
            pitch=pitch,
            body_height=body_height,
        )

    def p5_forward_inner_edge_aligned(self) -> bool:
        """
        判断右跳后平地前进阶段是否已经居中和角度对齐。
        注意：这里不重新检测，只读取最近一次 compute_p5_up_slope_inner_edge_corrected_cmd()
        更新后的 latest_p5_inner_edge_result。
        """
        result = self.latest_p5_inner_edge_result
        if not result or not result.get('common_valid', False):
            return False

        center_error = result.get('center_error', None)
        heading_error = result.get('heading_error', None)
        if center_error is None or heading_error is None:
            return False

        center_ok = abs(float(center_error)) <= self.p5_forward_after_reset_body_align_center_done_px
        heading_ok = abs(float(heading_error)) <= self.p5_forward_after_reset_body_align_heading_done_px
        return bool(center_ok and heading_ok)

    def run_timed_velocity_then_stop_state(
        self,
        duration_s: float,
        vx: float,
        vy: float,
        wz: float,
        step_height: float,
        next_state: str,
        roll: float = 0.0,
        pitch: float = 0.0,
        body_height: float = 0.25,
        log_name: str = '',
        use_inner_edge_align: bool = False,
        hold_after_duration_until_aligned: bool = False,
        stop_when_done: bool = True,
    ):
        """
        持续发送速度命令 duration_s 秒。

        普通模式：duration_s 到时后根据 stop_when_done 决定是否 stop，然后进入 next_state。

        hold_after_duration_until_aligned=True 时：
        - duration_s 之前：正常 vx 前进，同时可叠加内侧边缘 vy/wz 矫正；
        - duration_s 之后：vx 置 0，只保留 vy/wz 矫正；
        - 连续稳定若干帧满足 center_error / heading_error 阈值后，根据 stop_when_done 决定是否 stop，然后进入 next_state。
        """
        elapsed = self.state_elapsed_s()

        cmd_vy = vy
        cmd_wz = wz

        if use_inner_edge_align and self.latest_bgr is not None:
            cmd_vy, cmd_wz = self.compute_p5_up_slope_inner_edge_corrected_cmd(
                base_vy=vy,
                base_wz=wz,
                frame=self.latest_bgr,
            )
        elif use_inner_edge_align and self.latest_bgr is None:
            self.get_logger().warn(
                f'[{log_name or self.state}] no image for inner-edge align, use base vy/wz',
                throttle_duration_sec=0.5
            )

        if elapsed < duration_s:
            self.send_velocity_command(
                vx=vx,
                vy=cmd_vy,
                wz=cmd_wz,
                step_height=step_height,
                roll=roll,
                pitch=pitch,
                body_height=body_height,
            )

            self.get_logger().info(
                f'[{log_name or self.state}] timed velocity: '
                f'elapsed={elapsed:.3f}/{duration_s:.3f}s, '
                f'cmd=({vx:.3f},{cmd_vy:.3f},{cmd_wz:.3f}), '
                f'align={use_inner_edge_align}, '
                f'step_height={step_height:.3f}, '
                f'roll={roll:.3f}, pitch={pitch:.3f}, body_height={body_height:.3f}',
                throttle_duration_sec=0.3
            )
            return

        if hold_after_duration_until_aligned and use_inner_edge_align:
            # 3 秒前进结束后，不再继续向前冲。只用 vy/wz 做居中和角度矫正。
            aligned = self.p5_forward_inner_edge_aligned()
            if aligned:
                self.p5_forward_align_stable_counter += 1
            else:
                self.p5_forward_align_stable_counter = 0

            max_extra_s = self.p5_forward_after_reset_body_align_max_extra_s
            max_extra_reached = (
                max_extra_s > 0.0 and
                elapsed >= duration_s + max_extra_s
            )

            if self.p5_forward_align_stable_counter >= self.p5_forward_after_reset_body_align_stable_frames:
                self.get_logger().info(
                    f'[{log_name or self.state}] forward duration done and align stable, '
                    f'go {next_state}: '
                    f'stable={self.p5_forward_align_stable_counter}/'
                    f'{self.p5_forward_after_reset_body_align_stable_frames}, '
                    f'stop_when_done={stop_when_done}'
                )
                if stop_when_done:
                    self.send_stop_command()
                    self.enter_state_after_blocking_wait(next_state)
                else:
                    self.enter_state(next_state)
                return

            if max_extra_reached:
                self.get_logger().warn(
                    f'[{log_name or self.state}] align max extra time reached, '
                    f'go {next_state}: elapsed={elapsed:.3f}, '
                    f'duration={duration_s:.3f}, max_extra={max_extra_s:.3f}, '
                    f'stop_when_done={stop_when_done}'
                )
                if stop_when_done:
                    self.send_stop_command()
                    self.enter_state_after_blocking_wait(next_state)
                else:
                    self.enter_state(next_state)
                return

            self.send_velocity_command(
                vx=0.0,
                vy=cmd_vy,
                wz=cmd_wz,
                step_height=step_height,
                roll=roll,
                pitch=pitch,
                body_height=body_height,
            )

            edge_result = self.latest_p5_inner_edge_result
            self.get_logger().info(
                f'[{log_name or self.state}] duration done, hold vx=0 for align: '
                f'elapsed={elapsed:.3f}/{duration_s:.3f}s, '
                f'center_error={edge_result.get("center_error")}, '
                f'heading_error={edge_result.get("heading_error")}, '
                f'common_valid={edge_result.get("common_valid")}, '
                f'reason={edge_result.get("common_reason")}, '
                f'stable={self.p5_forward_align_stable_counter}/'
                f'{self.p5_forward_after_reset_body_align_stable_frames}, '
                f'cmd=(0.000,{cmd_vy:.3f},{cmd_wz:.3f})',
                throttle_duration_sec=0.3
            )
            return

        self.get_logger().info(
            f'[{log_name or self.state}] timed velocity done, '
            f'go {next_state}, stop_when_done={stop_when_done}'
        )
        if stop_when_done:
            self.send_stop_command()
            self.enter_state_after_blocking_wait(next_state)
        else:
            self.enter_state(next_state)

    def run_action_state(
        self,
        mode: int,
        gait_id: int,
        next_state: str,
        log_name: str = '',
        stop_after_finish: bool = False,
    ):
        if not self.action_sent:
            self.send_action_once(mode, gait_id)
            self.action_sent = True

            self.get_logger().info(
                f'[{log_name or self.state}] waiting by Ctrl.Wait_finish: '
                f'mode={mode}, gait_id={gait_id}'
            )

            self.Ctrl.Wait_finish(mode, gait_id)

            if stop_after_finish:
                self.get_logger().info(
                    f'[{log_name or self.state}] action finished, send STOP before go {next_state}: '
                    f'mode={mode}, gait_id={gait_id}'
                )
                self.send_stop_command()
            else:
                self.get_logger().info(
                    f'[{log_name or self.state}] action finished, go {next_state}: '
                    f'mode={mode}, gait_id={gait_id}'
                )

            self.enter_state_after_blocking_wait(next_state)
            return

    # ============================================================
    # 主状态机
    # ============================================================
    def control_loop(self):
        try:
            if self.state == self.P5_RECOVERY_STAND:
                self.run_action_state(
                    mode=12,
                    gait_id=0,
                    next_state=self.P5_SET_BODY_NORMAL,
                    log_name='P5_RECOVERY_STAND'
                )

            elif self.state == self.P5_SET_BODY_NORMAL:
                if not self.action_sent:
                    self.set_body_roll_height(
                        roll=self.p5_body_normal_roll,
                        height=self.p5_body_normal_height
                    )
                    self.action_sent = True

                if self.state_elapsed_s() >= self.p5_body_normal_wait_s:
                    self.enter_state(self.P5_STEP_UP)

            elif self.state == self.P5_STEP_UP:
                self.run_timed_velocity_state(
                    duration_s=self.p5_step_up_duration_s,
                    vx=self.p5_step_up_vx,
                    vy=self.p5_step_up_vy,
                    wz=self.p5_step_up_wz,
                    step_height=self.p5_step_up_step_height,
                    next_state=self.P5_UP_SLOPE,
                    body_height=self.p5_body_normal_height,
                    log_name='P5_STEP_UP'
                )

            elif self.state == self.P5_UP_SLOPE:
                self.run_right_side_yellow_lost_velocity_state(
                    vx=self.p5_up_slope_vx,
                    vy=self.p5_up_slope_vy,
                    wz=self.p5_up_slope_wz,
                    step_height=self.p5_up_slope_step_height,
                    next_state=self.P5_AFTER_UP_SLOPE_VELOCITY_CONTROL,
                    roll=self.p5_up_slope_roll,
                    pitch=self.p5_up_slope_pitch,
                    body_height=self.p5_body_normal_height,
                    log_name='P5_UP_SLOPE'
                )

            elif self.state == self.P5_AFTER_UP_SLOPE_VELOCITY_CONTROL:
                # 上坡右侧黄线消失后，先用速度控制固定时间完成转向/位置调整；
                # 结束后再设置右斜坡 body。
                self.run_timed_velocity_state(
                    duration_s=self.p5_after_up_slope_control_duration_s,
                    vx=self.p5_after_up_slope_control_vx,
                    vy=self.p5_after_up_slope_control_vy,
                    wz=self.p5_after_up_slope_control_wz,
                    step_height=self.p5_after_up_slope_control_step_height,
                    next_state=self.P5_SET_RIGHT_SLOPE_BODY,
                    body_height=self.p5_body_normal_height,
                    log_name='P5_AFTER_UP_SLOPE_VELOCITY_CONTROL'
                )

            elif self.state == self.P5_SET_RIGHT_SLOPE_BODY:
                if not self.action_sent:
                    self.set_body_roll_height(
                        roll=self.p5_right_slope_roll,
                        height=self.p5_right_slope_height
                    )
                    self.action_sent = True

                if self.state_elapsed_s() >= self.p5_right_slope_body_wait_s:
                    self.enter_state(self.P5_RIGHT_SLOPE_1)

            elif self.state == self.P5_RIGHT_SLOPE_1:
                self.run_center_yellow_absence_velocity_state(
                    vx=self.p5_right_slope_1_vx,
                    vy=self.p5_right_slope_1_vy,
                    wz=self.p5_right_slope_1_wz,
                    step_height=self.p5_right_slope_1_step_height,
                    next_state=self.P5_RIGHT_SLOPE_1_FORWARD_AFTER_CENTER_LOST,
                    roll=self.p5_right_slope_roll,
                    body_height=self.p5_right_slope_height,
                    log_name='P5_RIGHT_SLOPE_1'
                )

            elif self.state == self.P5_RIGHT_SLOPE_1_FORWARD_AFTER_CENTER_LOST:
                self.run_timed_velocity_state(
                    duration_s=self.p5_right_slope_1_after_center_lost_duration_s,
                    vx=self.p5_right_slope_1_after_center_lost_vx,
                    vy=self.p5_right_slope_1_after_center_lost_vy,
                    wz=self.p5_right_slope_1_after_center_lost_wz,
                    step_height=self.p5_right_slope_1_after_center_lost_step_height,
                    next_state=self.P5_TURN_1,
                    roll=self.p5_right_slope_roll,
                    body_height=self.p5_right_slope_height,
                    log_name='P5_RIGHT_SLOPE_1_FORWARD_AFTER_CENTER_LOST'
                )  # no stop: extra forward -> turn directly

            elif self.state == self.P5_TURN_1:
                if self.p5_right_slope_turn_method == 'right_jump':
                    self.run_action_state(
                        mode=self.p5_right_slope_turn_1_jump_mode,
                        gait_id=self.p5_right_slope_turn_1_jump_gait,
                        next_state=self.P5_RIGHT_SLOPE_2,
                        log_name='P5_TURN_1_RIGHT_JUMP',
                        stop_after_finish=self.p5_right_slope_turn_jump_stop_after_finish
                    )
                else:
                    self.run_timed_velocity_state(
                        duration_s=self.p5_turn_1_duration_s,
                        vx=self.p5_turn_1_vx,
                        vy=self.p5_turn_1_vy,
                        wz=self.p5_turn_1_wz,
                        step_height=self.p5_turn_1_step_height,
                        next_state=self.P5_RIGHT_SLOPE_2,
                        roll=self.p5_right_slope_roll,
                        body_height=self.p5_right_slope_height,
                        log_name='P5_TURN_1'
                    )  # no stop: turn -> right slope 2 directly

            elif self.state == self.P5_RECOVERY_AFTER_TURN_1:
                # 保留这个状态名是为了兼容旧流程；当前速度控制转向不会进入这里。
                self.enter_state(self.P5_RIGHT_SLOPE_2)

            elif self.state == self.P5_RIGHT_SLOPE_2:
                self.run_center_yellow_absence_velocity_state(
                    vx=self.p5_right_slope_2_vx,
                    vy=self.p5_right_slope_2_vy,
                    wz=self.p5_right_slope_2_wz,
                    step_height=self.p5_right_slope_2_step_height,
                    next_state=self.P5_RIGHT_SLOPE_2_FORWARD_AFTER_CENTER_LOST,
                    roll=self.p5_right_slope_roll,
                    body_height=self.p5_right_slope_height,
                    log_name='P5_RIGHT_SLOPE_2'
                )

            elif self.state == self.P5_RIGHT_SLOPE_2_FORWARD_AFTER_CENTER_LOST:
                self.run_timed_velocity_state(
                    duration_s=self.p5_right_slope_2_after_center_lost_duration_s,
                    vx=self.p5_right_slope_2_after_center_lost_vx,
                    vy=self.p5_right_slope_2_after_center_lost_vy,
                    wz=self.p5_right_slope_2_after_center_lost_wz,
                    step_height=self.p5_right_slope_2_after_center_lost_step_height,
                    next_state=self.P5_TURN_2,
                    roll=self.p5_right_slope_roll,
                    body_height=self.p5_right_slope_height,
                    log_name='P5_RIGHT_SLOPE_2_FORWARD_AFTER_CENTER_LOST'
                )  # no stop: extra forward -> turn directly

            elif self.state == self.P5_TURN_2:
                if self.p5_right_slope_turn_method == 'right_jump':
                    self.run_action_state(
                        mode=self.p5_right_slope_turn_2_jump_mode,
                        gait_id=self.p5_right_slope_turn_2_jump_gait,
                        next_state=self.P5_RIGHT_SLOPE_3,
                        log_name='P5_TURN_2_RIGHT_JUMP',
                        stop_after_finish=self.p5_right_slope_turn_jump_stop_after_finish
                    )
                else:
                    self.run_timed_velocity_state(
                        duration_s=self.p5_turn_2_duration_s,
                        vx=self.p5_turn_2_vx,
                        vy=self.p5_turn_2_vy,
                        wz=self.p5_turn_2_wz,
                        step_height=self.p5_turn_2_step_height,
                        next_state=self.P5_RIGHT_SLOPE_3,
                        roll=self.p5_right_slope_roll,
                        body_height=self.p5_right_slope_height,
                        log_name='P5_TURN_2'
                    )  # no stop: turn -> right slope 3 directly

            elif self.state == self.P5_RECOVERY_AFTER_TURN_2:
                # 保留这个状态名是为了兼容旧流程；当前速度控制转向不会进入这里。
                self.enter_state(self.P5_RIGHT_SLOPE_3)

            elif self.state == self.P5_RIGHT_SLOPE_3:
                self.run_center_yellow_absence_velocity_state(
                    vx=self.p5_right_slope_3_vx,
                    vy=self.p5_right_slope_3_vy,
                    wz=self.p5_right_slope_3_wz,
                    step_height=self.p5_right_slope_3_step_height,
                    next_state=self.P5_RESET_BODY,
                    roll=self.p5_right_slope_roll,
                    body_height=self.p5_right_slope_height,
                    log_name='P5_RIGHT_SLOPE_3',
                    stop_before_next=True
                )  # right slope 3 -> zero velocity once -> reset body / right-jump turn directly

            elif self.state == self.P5_RESET_BODY:
                if not self.action_sent:
                    self.set_body_roll_height(
                        roll=self.p5_reset_roll,
                        height=self.p5_reset_height
                    )
                    self.action_sent = True

                if self.state_elapsed_s() >= self.p5_reset_body_wait_s:
                    self.get_logger().info(
                        '[P5_RESET_BODY] reset body done, go P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP'
                    )
                    self.enter_state(self.P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP)

            elif self.state == self.P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP:
                # reset body 后，先执行第一段右移固定时间，再进入第二段右移。
                self.run_timed_velocity_state(
                    duration_s=self.p5_right_shift_before_right_jump_duration_s,
                    vx=self.p5_right_shift_before_right_jump_vx,
                    vy=self.p5_right_shift_before_right_jump_vy,
                    wz=self.p5_right_shift_before_right_jump_wz,
                    step_height=self.p5_right_shift_before_right_jump_step_height,
                    next_state=self.P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP_2,
                    roll=self.p5_reset_roll,
                    body_height=self.p5_reset_height,
                    log_name='P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP'
                )

            elif self.state == self.P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP_2:
                # 第一段右移后，继续右移一段固定时间，然后再执行右跳动作。
                self.run_timed_velocity_state(
                    duration_s=self.p5_right_shift_before_right_jump_2_duration_s,
                    vx=self.p5_right_shift_before_right_jump_2_vx,
                    vy=self.p5_right_shift_before_right_jump_2_vy,
                    wz=self.p5_right_shift_before_right_jump_2_wz,
                    step_height=self.p5_right_shift_before_right_jump_2_step_height,
                    next_state=self.P5_RIGHT_JUMP_AFTER_RESET_BODY,
                    roll=self.p5_reset_roll,
                    body_height=self.p5_reset_height,
                    log_name='P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP_2'
                )

            elif self.state == self.P5_RIGHT_JUMP_AFTER_RESET_BODY:
                # 右移后执行第三个转角右跳；动作完成后先 STOP 一次，再进入“前进 + 矫正”。
                self.run_action_state(
                    mode=self.p5_right_jump_after_reset_body_mode,
                    gait_id=self.p5_right_jump_after_reset_body_gait,
                    next_state=self.P5_FORWARD_AFTER_RESET_BODY,
                    log_name='P5_RIGHT_JUMP_AFTER_RESET_BODY',
                    stop_after_finish=True
                )

            elif self.state == self.P5_FORWARD_AFTER_RESET_BODY:
                # 第三转角右跳后第一段：先固定时间前进，同时根据内侧边缘做 vy/wz 矫正；
                # 如果时间到了还没对齐，则 vx=0 原地横移/转向继续矫正。
                # 矫正完成后不 stop，直接进入下一段“不矫正固定前进”。
                self.run_timed_velocity_then_stop_state(
                    duration_s=self.p5_forward_after_reset_body_duration_s,
                    vx=self.p5_forward_after_reset_body_vx,
                    vy=self.p5_forward_after_reset_body_vy,
                    wz=self.p5_forward_after_reset_body_wz,
                    step_height=self.p5_forward_after_reset_body_step_height,
                    next_state=self.P5_FORWARD_NO_ALIGN_AFTER_RESET_BODY,
                    roll=self.p5_reset_roll,
                    body_height=self.p5_reset_height,
                    log_name='P5_FORWARD_AFTER_RESET_BODY',
                    use_inner_edge_align=True,
                    hold_after_duration_until_aligned=self.p5_forward_after_reset_body_hold_align_enabled,
                    stop_when_done=False
                )

            elif self.state == self.P5_FORWARD_NO_ALIGN_AFTER_RESET_BODY:
                # 矫正完成后第二段：固定时间前进，不再叠加视觉矫正；
                # 结束后 stop，再进入离坡右跳/跳远流程。
                self.run_timed_velocity_then_stop_state(
                    duration_s=self.p5_forward_no_align_after_reset_body_duration_s,
                    vx=self.p5_forward_no_align_after_reset_body_vx,
                    vy=self.p5_forward_no_align_after_reset_body_vy,
                    wz=self.p5_forward_no_align_after_reset_body_wz,
                    step_height=self.p5_forward_no_align_after_reset_body_step_height,
                    next_state=self.P5_JUMP_EXIT_SLOPE,
                    roll=self.p5_reset_roll,
                    body_height=self.p5_reset_height,
                    log_name='P5_FORWARD_NO_ALIGN_AFTER_RESET_BODY',
                    use_inner_edge_align=False,
                    hold_after_duration_until_aligned=False,
                    stop_when_done=True
                )

            elif self.state == self.P5_JUMP_EXIT_SLOPE:
                self.run_action_state(
                    mode=self.p5_jump_exit_slope_mode,
                    gait_id=self.p5_jump_exit_slope_gait,
                    next_state=self.P5_RECOVERY_AFTER_JUMP_2,
                    log_name='P5_JUMP_EXIT_SLOPE'
                )

            elif self.state == self.P5_RECOVERY_AFTER_JUMP_2:
                self.run_action_state(
                    mode=12,
                    gait_id=0,
                    next_state=self.P5_FINAL_LONG_JUMP,
                    log_name='P5_RECOVERY_AFTER_JUMP_2',
                    stop_after_finish=True
                )

            elif self.state == self.P5_FINAL_LONG_JUMP:
                self.run_action_state(
                    mode=self.p5_final_long_jump_mode,
                    gait_id=self.p5_final_long_jump_gait,
                    next_state=self.P5_DONE,
                    log_name='P5_FINAL_LONG_JUMP'
                )

            elif self.state == self.P5_DONE:
                if not self.action_sent:
                    self.send_stop_command()
                    self.action_sent = True

                self.get_logger().info(
                    '[P5_DONE] fifth stage done, keep stop',
                    throttle_duration_sec=1.0
                )

            else:
                self.get_logger().error(
                    f'[P5] unknown state={self.state}, send stop'
                )
                self.send_stop_command()

        except Exception as e:
            self.get_logger().error(
                f'[P5] control_loop exception: {repr(e)}'
            )
            self.send_stop_command()
            raise

    def destroy_node(self):
        try:
            self.send_stop_command()
            self.Ctrl.quit()
            cv2.destroyAllWindows()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FifthStageBridgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('[P5] KeyboardInterrupt, stop and exit')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()