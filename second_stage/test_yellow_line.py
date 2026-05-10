#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试功能：
1. 机器狗按固定速度向前走。
2. 同时检测前方黄色横线，检测逻辑参考当前整合代码里的 detect_yellow_stop_line()，
   但前方横线判断只使用更严格的 bbox 条件，不再使用角度。
3. 如果识别到黄色轮廓，但不满足“前方横线”条件，会输出具体不满足原因。
4. OpenCV 可视化：
   - 蓝框：检测 ROI
   - 绿色框：满足前方横线条件的候选
   - 红色框：黄色但不满足前方横线条件的候选
   - 画面左上角显示当前最佳横线、bottom_y、threshold 等信息

运行示例：
python3 test_forward_yellow_horizontal_debug_bbox_only.py --ros-args \
  -p forward_speed:=0.20 \
  -p duration_sec:=20.0 \
  -p yellow_stop_line_y_ratio:=0.70
"""

import math
import time
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from second_stage.my_gait import Robot_Ctrl
from second_stage.robot_control_cmd_lcmt import robot_control_cmd_lcmt


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class ForwardYellowHorizontalDebugNode(Node):
    def __init__(self):
        super().__init__('forward_yellow_horizontal_debug_node')

        # 默认使用 Gazebo 仿真时间
        try:
            self.declare_parameter('use_sim_time', True)
        except Exception:
            pass
        try:
            self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        except Exception as e:
            self.get_logger().warn(f'failed to set use_sim_time=True: {e}')

        # =========================
        # 基础参数
        # =========================
        self.declare_parameter('rgb_topic', '/rgb_camera/rgb_camera/image_raw')
        self.declare_parameter('control_hz', 30.0)
        self.declare_parameter('forward_speed', 0.20)
        self.declare_parameter('vy', 0.0)
        self.declare_parameter('wz', 0.0)
        self.declare_parameter('duration_sec', 20.0)
        self.declare_parameter('stop_when_reached', False)
        self.declare_parameter('show_debug_vis', True)

        # =========================
        # 黄色检测参数：对齐你当前第二赛段逻辑
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

        # 前方横线判定条件
        self.declare_parameter('yellow_min_width_height_ratio', 2.5)
        # 不再使用角度判断；保留参数名会影响旧命令行兼容，但不参与判断。
        self.declare_parameter('yellow_max_tilt_deg', 30.0)
        self.declare_parameter('yellow_center_tolerance_ratio', 0.25)
        self.declare_parameter('yellow_min_width_ratio', 0.45)

        # 到达停止线判断：line_bottom_y >= image_height * ratio
        self.declare_parameter('yellow_stop_line_y_ratio', 0.70)
        self.declare_parameter('yellow_stop_confirm_count', 1)

        # 日志打印节流
        self.declare_parameter('print_reject_details', True)

        # =========================
        # 读取参数
        # =========================
        self.rgb_topic = str(self.get_parameter('rgb_topic').value)
        self.control_hz = float(self.get_parameter('control_hz').value)
        self.forward_speed = float(self.get_parameter('forward_speed').value)
        self.vy = float(self.get_parameter('vy').value)
        self.wz = float(self.get_parameter('wz').value)
        self.duration_sec = float(self.get_parameter('duration_sec').value)
        self.stop_when_reached = bool(self.get_parameter('stop_when_reached').value)
        self.show_debug_vis = bool(self.get_parameter('show_debug_vis').value)

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

        self.yellow_stop_line_y_ratio = float(self.get_parameter('yellow_stop_line_y_ratio').value)
        self.yellow_stop_confirm_count = int(self.get_parameter('yellow_stop_confirm_count').value)
        self.print_reject_details = bool(self.get_parameter('print_reject_details').value)

        self.bridge = CvBridge()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_result: Dict = {
            'has_line': False,
            'line_bottom_y': None,
            'img_shape': None,
            'accepted': [],
            'rejected': [],
            'mask': None,
        }

        self.yellow_stop_counter = 0
        self.start_time_sec: Optional[float] = None
        self.done = False

        self.Ctrl = Robot_Ctrl()
        self.Ctrl.run()
        self.msg = robot_control_cmd_lcmt()
        if not hasattr(self.msg, 'life_count'):
            self.msg.life_count = 0

        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            qos_profile_sensor_data,
        )
        self.timer = self.create_timer(1.0 / self.control_hz, self.control_loop)

        self.send_stop_command()
        self.get_logger().info(
            f'started: forward_speed={self.forward_speed:.3f}, duration={self.duration_sec:.2f}s, '
            f'rgb_topic={self.rgb_topic}'
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def inc_life_count(self):
        self.msg.life_count += 1
        if self.msg.life_count > 127:
            self.msg.life_count = 0

    def send_stop_command(self):
        self.msg.mode = 12
        self.msg.gait_id = 0
        self.inc_life_count()
        self.Ctrl.Send_cmd(self.msg)
        self.get_logger().info('[CMD] STOP', throttle_duration_sec=1.0)

    def send_velocity_command(self, vx: float, vy: float, wz: float):
        self.msg.mode = 11
        self.msg.gait_id = 3
        self.inc_life_count()
        self.msg.vel_des = [float(vx), float(vy), float(wz)]
        self.msg.step_height = [0.02, 0.02]
        self.msg.rpy_des = [0.0, 0.0, 0.0]
        self.Ctrl.Send_cmd(self.msg)
        self.get_logger().info(
            f'[CMD] vel_des=[{vx:.3f}, {vy:.3f}, {wz:.3f}]',
            throttle_duration_sec=0.5,
        )

    # ============================================================
    # 黄线检测调试
    # ============================================================
    def _signed_line_angle_deg(self, cnt) -> float:
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

    def check_front_horizontal_yellow_line_debug(self, cnt, roi_shape) -> Tuple[bool, List[str], Dict]:
        """
        返回：是否满足、失败原因列表、调试指标。

        当前版本不再使用角度判断，只用更严格的 bbox 条件判断“前方横线”：
        1. 面积够大，过滤小噪声；
        2. wh_ratio = bbox_width / bbox_height 足够大，说明它是横向长条；
        3. width_ratio = bbox_width / roi_width 足够大，说明它横跨了较大前方区域；
        4. center_offset_ratio 足够小，说明它靠近画面中间，不是旁边的黄线。
        """
        roi_h, roi_w = roi_shape[:2]
        reasons: List[str] = []
        metrics: Dict = {}

        area = float(cv2.contourArea(cnt))
        metrics['area'] = area
        if area < self.yellow_min_contour_area:
            reasons.append(f'area {area:.1f} < {self.yellow_min_contour_area:.1f}')

        x, y, bw, bh = cv2.boundingRect(cnt)
        metrics['bbox_roi'] = (int(x), int(y), int(bw), int(bh))

        if bh <= 0:
            reasons.append('bbox height <= 0')
            wh_ratio = 0.0
        else:
            wh_ratio = bw / float(bh)
        metrics['wh_ratio'] = float(wh_ratio)
        if wh_ratio < self.yellow_min_width_height_ratio:
            reasons.append(
                f'wh_ratio {wh_ratio:.2f} < {self.yellow_min_width_height_ratio:.2f}'
            )

        width_ratio = bw / float(max(roi_w, 1))
        metrics['width_ratio'] = float(width_ratio)
        if width_ratio < self.yellow_min_width_ratio:
            reasons.append(
                f'width_ratio {width_ratio:.2f} < {self.yellow_min_width_ratio:.2f}'
            )

        cx = x + bw / 2.0
        roi_cx = roi_w / 2.0
        center_offset_ratio = abs(cx - roi_cx) / float(max(roi_w, 1))
        metrics['center_offset_ratio'] = float(center_offset_ratio)
        if center_offset_ratio > self.yellow_center_tolerance_ratio:
            reasons.append(
                f'center_offset_ratio {center_offset_ratio:.2f} > {self.yellow_center_tolerance_ratio:.2f}'
            )

        # 角度只作为调试显示，不参与通过/拒绝判断。
        metrics['fit_angle_deg'] = self._signed_line_angle_deg(cnt)
        metrics['tilt_deg'] = metrics['fit_angle_deg']
        metrics['angle_used'] = False

        return len(reasons) == 0, reasons, metrics

    def detect_yellow_stop_line_debug(self, frame: np.ndarray) -> Dict:
        h, w = frame.shape[:2]
        roi_top = int(h * self.yellow_roi_top_ratio)
        roi_left = int(w * self.yellow_roi_left_ratio)
        roi_right = int(w * self.yellow_roi_right_ratio)
        roi_top = max(0, min(h - 1, roi_top))
        roi_left = max(0, min(w - 1, roi_left))
        roi_right = max(roi_left + 1, min(w, roi_right))

        roi = frame[roi_top:h, roi_left:roi_right]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array(
            [self.yellow_h_min, self.yellow_s_min, self.yellow_v_min], dtype=np.uint8
        )
        upper_yellow = np.array(
            [self.yellow_h_max, self.yellow_s_max, self.yellow_v_max], dtype=np.uint8
        )
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        accepted = []
        rejected = []

        for idx, cnt in enumerate(contours):
            ok, reasons, metrics = self.check_front_horizontal_yellow_line_debug(cnt, roi.shape)
            x, y, bw, bh = metrics.get('bbox_roi', cv2.boundingRect(cnt))
            bx1 = roi_left + x
            by1 = roi_top + y
            bx2 = bx1 + bw
            by2 = by1 + bh
            item = {
                'idx': idx,
                'bbox': (int(bx1), int(by1), int(bx2), int(by2)),
                'bottom_y': int(by2),
                'center': (int((bx1 + bx2) / 2), int((by1 + by2) / 2)),
                'reasons': reasons,
                'metrics': metrics,
            }
            if ok:
                accepted.append(item)
            else:
                rejected.append(item)

        best = None
        if accepted:
            # 和当前 detect_yellow_stop_line 一样：优先选最靠下的横线
            best = max(accepted, key=lambda d: d['bottom_y'])

        return {
            'has_line': best is not None,
            'best': best,
            'line_bottom_y': None if best is None else best['bottom_y'],
            'img_shape': frame.shape,
            'accepted': accepted,
            'rejected': rejected,
            'mask': mask,
            'roi': (roi_left, roi_top, roi_right, h),
        }

    def yellow_reached(self, result: Dict) -> bool:
        if not result.get('has_line'):
            self.yellow_stop_counter = 0
            return False

        h = result['img_shape'][0]
        bottom_y = result['line_bottom_y']
        threshold = int(h * self.yellow_stop_line_y_ratio)
        if bottom_y is not None and bottom_y >= threshold:
            self.yellow_stop_counter += 1
        else:
            self.yellow_stop_counter = 0

        self.get_logger().info(
            f'yellow line check: bottom={bottom_y}, threshold={threshold}, '
            f'counter={self.yellow_stop_counter}/{self.yellow_stop_confirm_count}',
            throttle_duration_sec=0.3,
        )
        return self.yellow_stop_counter >= self.yellow_stop_confirm_count

    def draw_debug(self, frame: np.ndarray, result: Dict):
        vis = frame.copy()
        roi_left, roi_top, roi_right, roi_bottom = result['roi']
        cv2.rectangle(vis, (roi_left, roi_top), (roi_right, roi_bottom), (255, 0, 0), 2)

        # 红色：黄色但不满足前方横线条件
        for item in result['rejected']:
            x1, y1, x2, y2 = item['bbox']
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            reason_text = '; '.join(item['reasons'][:2])
            cv2.putText(
                vis,
                f'REJ {item["idx"]}: {reason_text}',
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 0, 255),
                1,
            )

        # 绿色：满足前方横线条件
        for item in result['accepted']:
            x1, y1, x2, y2 = item['bbox']
            m = item['metrics']
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f'OK wh={m["wh_ratio"]:.1f} width={m["width_ratio"]:.2f} center={m["center_offset_ratio"]:.2f}',
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )

        h = frame.shape[0]
        threshold = int(h * self.yellow_stop_line_y_ratio)
        cv2.line(vis, (0, threshold), (frame.shape[1], threshold), (255, 255, 0), 2)

        if result['has_line']:
            best = result['best']
            cv2.putText(
                vis,
                f'BEST bottom={best["bottom_y"]} threshold={threshold}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                vis,
                f'NO FRONT HORIZONTAL LINE | rejected={len(result["rejected"])}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

        cv2.imshow('yellow_horizontal_debug', vis)
        if result.get('mask') is not None:
            cv2.imshow('yellow_mask_roi', result['mask'])
        cv2.waitKey(1)

    # ============================================================
    # ROS callbacks / control
    # ============================================================
    def rgb_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge convert failed: {e}')
            return

        self.latest_frame = frame
        result = self.detect_yellow_stop_line_debug(frame)
        self.latest_result = result

        if self.print_reject_details and result['rejected']:
            # 节流打印，不然每帧太多
            lines = []
            for item in result['rejected'][:5]:
                m = item['metrics']
                lines.append(
                    f'idx={item["idx"]}, bbox={item["bbox"]}, '
                    f'area={m.get("area", 0):.1f}, wh={m.get("wh_ratio", 0):.2f}, '
                    f'width={m.get("width_ratio", 0):.2f}, center_off={m.get("center_offset_ratio", 0):.2f}, '
                    f'angle_unused={m.get("fit_angle_deg", 0):.1f}, reasons={item["reasons"]}'
                )
            self.get_logger().info(
                'yellow contours rejected:\n' + '\n'.join(lines),
                throttle_duration_sec=1.0,
            )

        if result['has_line']:
            best = result['best']
            m = best['metrics']
            self.get_logger().info(
                f'front horizontal yellow OK: bbox={best["bbox"]}, bottom={best["bottom_y"]}, '
                f'wh={m["wh_ratio"]:.2f}, width={m["width_ratio"]:.2f}, '
                f'center_off={m["center_offset_ratio"]:.2f}, angle_unused={m.get("fit_angle_deg", 0):.1f}',
                throttle_duration_sec=0.3,
            )

        if self.show_debug_vis:
            self.draw_debug(frame, result)

    def control_loop(self):
        now = self.now_sec()
        if now <= 0.0:
            self.get_logger().warn('waiting for /clock...', throttle_duration_sec=1.0)
            return

        if self.start_time_sec is None:
            self.start_time_sec = now
            self.get_logger().info(f'move start by sim time: {now:.3f}')

        elapsed = now - self.start_time_sec
        reached = self.yellow_reached(self.latest_result)

        if self.done:
            self.send_stop_command()
            return

        if elapsed >= self.duration_sec:
            self.get_logger().info(f'test duration finished: {elapsed:.2f}s')
            self.done = True
            self.send_stop_command()
            return

        if self.stop_when_reached and reached:
            self.get_logger().info('yellow reached threshold, stop_when_reached=True -> STOP')
            self.done = True
            self.send_stop_command()
            return

        self.send_velocity_command(self.forward_speed, self.vy, self.wz)

    def destroy_node(self):
        try:
            self.send_stop_command()
            time.sleep(0.2)
            self.Ctrl.quit()
        except Exception:
            pass
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ForwardYellowHorizontalDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, stopping...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
