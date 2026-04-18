#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from second_stage.my_gait import Robot_Ctrl
from second_stage.robot_control_cmd_lcmt import robot_control_cmd_lcmt


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


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class TfForwardBackwardYawDriftTestNode(Node):
    def __init__(self):
        super().__init__('tf_forward_backward_yaw_drift_test_node')

        # =========================
        # 参数
        # =========================
        self.declare_parameter('global_frame', 'vodom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('control_hz', 30.0)

        self.declare_parameter('forward_speed', 0.08)
        self.declare_parameter('backward_speed', 0.08)
        self.declare_parameter('travel_distance_m', 1.0)

        self.declare_parameter('return_pos_tolerance_m', 0.05)
        self.declare_parameter('return_confirm_count', 3)

        # 后退时横向修正，避免越退越偏
        self.declare_parameter('backward_lateral_kp', 0.5)
        self.declare_parameter('backward_max_vy', 0.05)

        self.global_frame = self.get_parameter('global_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.control_hz = float(self.get_parameter('control_hz').value)

        self.forward_speed = float(self.get_parameter('forward_speed').value)
        self.backward_speed = float(self.get_parameter('backward_speed').value)
        self.travel_distance_m = float(self.get_parameter('travel_distance_m').value)

        self.return_pos_tolerance_m = float(self.get_parameter('return_pos_tolerance_m').value)
        self.return_confirm_count = int(self.get_parameter('return_confirm_count').value)

        self.backward_lateral_kp = float(self.get_parameter('backward_lateral_kp').value)
        self.backward_max_vy = float(self.get_parameter('backward_max_vy').value)

        # =========================
        # TF
        # =========================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # =========================
        # 控制
        # =========================
        self.Ctrl = Robot_Ctrl()
        self.Ctrl.run()
        self.msg = robot_control_cmd_lcmt()
        if not hasattr(self.msg, 'life_count'):
            self.msg.life_count = 0

        # =========================
        # 状态
        # =========================
        self.state = 'WAIT_INIT'

        self.initial_pose: Optional[Tuple[float, float, float]] = None
        self.forward_start_pose: Optional[Tuple[float, float, float]] = None
        self.forward_end_pose: Optional[Tuple[float, float, float]] = None

        self.initial_yaw: Optional[float] = None
        self.forward_end_yaw: Optional[float] = None
        self.final_return_yaw: Optional[float] = None

        self.return_ok_counter = 0

        self.control_timer = self.create_timer(1.0 / self.control_hz, self.control_loop)

        self.send_stop_command()
        self.Ctrl.Wait_finish(12, 0)

        self.get_logger().info('TfForwardBackwardYawDriftTestNode started')
        self.get_logger().info(f'tf: {self.global_frame} -> {self.base_frame}')

    # =========================
    # 控制命令
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
    # TF位姿
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

    # =========================
    # 主循环
    # =========================
    def control_loop(self):
        pose = self.get_current_pose()
        if pose is None:
            self.get_logger().warn('TF unavailable, stop for safety.', throttle_duration_sec=1.0)
            self.send_stop_command()
            return

        x, y, yaw = pose

        # 1. 初始化
        if self.state == 'WAIT_INIT':
            self.initial_pose = (x, y, yaw)
            self.forward_start_pose = (x, y, yaw)
            self.initial_yaw = yaw
            self.state = 'FORWARD'

            self.get_logger().info(
                f'[INIT] x={x:.3f}, y={y:.3f}, yaw={yaw:.6f} rad ({math.degrees(yaw):.3f} deg)'
            )
            return

        # 2. 前进
        if self.state == 'FORWARD':
            if self.forward_start_pose is None or self.initial_yaw is None:
                self.send_stop_command()
                return

            x0, y0, _ = self.forward_start_pose
            dist = math.hypot(x - x0, y - y0)
            yaw_diff = wrap_to_pi(yaw - self.initial_yaw)

            self.get_logger().info(
                f'[FORWARD YAW] dist={dist:.3f}/{self.travel_distance_m:.3f} m | '
                f'initial_yaw={self.initial_yaw:.6f} rad ({math.degrees(self.initial_yaw):.3f} deg) | '
                f'current_yaw={yaw:.6f} rad ({math.degrees(yaw):.3f} deg) | '
                f'yaw_diff_from_start={yaw_diff:.6f} rad ({math.degrees(yaw_diff):.3f} deg)',
                throttle_duration_sec=0.2
            )

            if dist >= self.travel_distance_m:
                self.send_stop_command()
                self.forward_end_pose = (x, y, yaw)
                self.forward_end_yaw = yaw
                self.state = 'BACKWARD'

                self.get_logger().info('=' * 60)
                self.get_logger().info(
                    f'[FORWARD END] dist={dist:.3f} m | '
                    f'forward_end_yaw={yaw:.6f} rad ({math.degrees(yaw):.3f} deg) | '
                    f'forward_yaw_diff={yaw_diff:.6f} rad ({math.degrees(yaw_diff):.3f} deg)'
                )
                self.get_logger().info('=' * 60)
                return

            self.send_velocity_command(self.forward_speed, 0.0, 0.0)
            return

        # 3. 后退回起点
        if self.state == 'BACKWARD':
            if self.initial_pose is None or self.initial_yaw is None:
                self.send_stop_command()
                return

            x_tar, y_tar, _ = self.initial_pose
            dx = x_tar - x
            dy = y_tar - y
            dist_to_start = math.hypot(dx, dy)
            yaw_diff = wrap_to_pi(yaw - self.initial_yaw)

            self.get_logger().info(
                f'[BACKWARD YAW] dist_to_start={dist_to_start:.3f} m | '
                f'initial_yaw={self.initial_yaw:.6f} rad ({math.degrees(self.initial_yaw):.3f} deg) | '
                f'current_yaw={yaw:.6f} rad ({math.degrees(yaw):.3f} deg) | '
                f'yaw_diff_from_start={yaw_diff:.6f} rad ({math.degrees(yaw_diff):.3f} deg)',
                throttle_duration_sec=0.2
            )

            if dist_to_start <= self.return_pos_tolerance_m:
                self.return_ok_counter += 1
                self.send_stop_command()

                if self.return_ok_counter >= self.return_confirm_count:
                    self.final_return_yaw = yaw
                    final_yaw_diff = wrap_to_pi(self.final_return_yaw - self.initial_yaw)

                    self.get_logger().info('=' * 60)
                    self.get_logger().info(
                        f'[RETURN END] final_return_yaw={self.final_return_yaw:.6f} rad '
                        f'({math.degrees(self.final_return_yaw):.3f} deg) | '
                        f'return_yaw_diff={final_yaw_diff:.6f} rad '
                        f'({math.degrees(final_yaw_diff):.3f} deg)'
                    )

                    if self.forward_end_yaw is not None:
                        forward_yaw_diff = wrap_to_pi(self.forward_end_yaw - self.initial_yaw)
                        self.get_logger().info(
                            f'[SUMMARY] forward_yaw_diff={forward_yaw_diff:.6f} rad '
                            f'({math.degrees(forward_yaw_diff):.3f} deg) | '
                            f'return_yaw_diff={final_yaw_diff:.6f} rad '
                            f'({math.degrees(final_yaw_diff):.3f} deg)'
                        )
                    self.get_logger().info('=' * 60)

                    self.state = 'DONE'
                return

            self.return_ok_counter = 0

            # 用 body frame 做一点横向修正
            x_body = math.cos(yaw) * dx + math.sin(yaw) * dy
            y_body = -math.sin(yaw) * dx + math.cos(yaw) * dy

            vx = -self.backward_speed
            vy = clamp(
                self.backward_lateral_kp * y_body,
                -self.backward_max_vy,
                self.backward_max_vy
            )

            self.send_velocity_command(vx, vy, 0.0)
            return

        # 4. 完成
        if self.state == 'DONE':
            self.send_stop_command()
            return


def main(args=None):
    rclpy.init(args=args)
    node = TfForwardBackwardYawDriftTestNode()
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