#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data


class OrangeBallDebugNode(Node):
    def __init__(self):
        super().__init__('orange_ball_debug_node')

        # ===== 参数 =====
        self.declare_parameter('rgb_topic', '/rgb_camera/rgb_camera/image_raw')
        self.declare_parameter('depth_topic', '/d435/depth/d435_depth/depth/image_raw')

        self.declare_parameter('orange_h_min', 5)
        self.declare_parameter('orange_h_max', 25)
        self.declare_parameter('orange_s_min', 100)
        self.declare_parameter('orange_s_max', 255)
        self.declare_parameter('orange_v_min', 80)
        self.declare_parameter('orange_v_max', 255)

        self.declare_parameter('orange_min_contour_area', 90.0)

        self.declare_parameter('depth_search_half', 12)
        self.declare_parameter('valid_min_depth_m', 0.05)
        self.declare_parameter('valid_max_depth_m', 10.0)

        self.declare_parameter('show_window', True)
        self.declare_parameter('print_every_frame', False)

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value

        self.orange_h_min = int(self.get_parameter('orange_h_min').value)
        self.orange_h_max = int(self.get_parameter('orange_h_max').value)
        self.orange_s_min = int(self.get_parameter('orange_s_min').value)
        self.orange_s_max = int(self.get_parameter('orange_s_max').value)
        self.orange_v_min = int(self.get_parameter('orange_v_min').value)
        self.orange_v_max = int(self.get_parameter('orange_v_max').value)

        self.orange_min_contour_area = float(self.get_parameter('orange_min_contour_area').value)

        self.depth_search_half = int(self.get_parameter('depth_search_half').value)
        self.valid_min_depth_m = float(self.get_parameter('valid_min_depth_m').value)
        self.valid_max_depth_m = float(self.get_parameter('valid_max_depth_m').value)

        self.show_window = bool(self.get_parameter('show_window').value)
        self.print_every_frame = bool(self.get_parameter('print_every_frame').value)

        # ===== 数据 =====
        self.bridge = CvBridge()
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_encoding: Optional[str] = None
        self.rgb_w = 640
        self.rgb_h = 480

        # ===== 订阅 =====
        self.rgb_sub = self.create_subscription(
            Image, self.rgb_topic, self.rgb_callback, qos_profile_sensor_data
        )
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, qos_profile_sensor_data
        )

        self.get_logger().info('OrangeBallDebugNode started')
        self.get_logger().info(f'rgb_topic={self.rgb_topic}')
        self.get_logger().info(f'depth_topic={self.depth_topic}')

    def depth_callback(self, msg: Image):
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'depth convert failed: {e}')
            return

        self.latest_depth = depth_img
        self.latest_depth_encoding = msg.encoding

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

    def detect_orange_balls(self, frame: np.ndarray) -> List[Dict]:
        h, w = frame.shape[:2]
        self.rgb_w = w
        self.rgb_h = h

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array(
            [self.orange_h_min, self.orange_s_min, self.orange_v_min], dtype=np.uint8
        )
        upper = np.array(
            [self.orange_h_max, self.orange_s_max, self.orange_v_max], dtype=np.uint8
        )

        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        balls: List[Dict] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.orange_min_contour_area:
                continue

            (cx_f, cy_f), r_circle = cv2.minEnclosingCircle(cnt)
            cx = int(cx_f)
            cy = int(cy_f)
            r_circle = float(r_circle)

            r_eq = math.sqrt(area / math.pi)

            depth_m, depth_center, depth_box = self.get_depth_for_rgb_point(cx, cy)

            balls.append({
                'center': (cx, cy),
                'area': area,
                'r_circle': r_circle,
                'r_eq': r_eq,
                'depth_m': depth_m,
                'depth_center': depth_center,
                'depth_box': depth_box,
                'contour': cnt,
            })

        return balls, mask

    def rgb_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'rgb convert failed: {e}')
            return

        balls, mask = self.detect_orange_balls(frame)

        if balls or self.print_every_frame:
            self.get_logger().info(f'orange_ball_count={len(balls)}')

        vis = frame.copy()

        for i, b in enumerate(balls):
            cx, cy = b['center']
            area = b['area']
            r_circle = b['r_circle']
            r_eq = b['r_eq']
            depth_m = b['depth_m']

            depth_str = 'None' if depth_m is None else f'{depth_m:.3f}m'

            self.get_logger().info(
                f'[ball {i}] center=({cx}, {cy}) | area={area:.1f} | '
                f'r_circle={r_circle:.2f} | r_eq={r_eq:.2f} | depth={depth_str}'
            )

            # 可视化
            cv2.circle(vis, (cx, cy), int(round(r_circle)), (0, 255, 0), 2)
            cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)

            text1 = f'({cx},{cy})'
            text2 = f'A={area:.0f} Rc={r_circle:.1f} Re={r_eq:.1f}'
            text3 = f'D={depth_str}'

            cv2.putText(vis, text1, (cx + 6, cy - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(vis, text2, (cx + 6, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(vis, text3, (cx + 6, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        if self.show_window:
            cv2.imshow('orange_ball_debug', vis)
            cv2.imshow('orange_mask', mask)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = OrangeBallDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()