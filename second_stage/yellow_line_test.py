#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data


class YellowLineDebugNode(Node):
    def __init__(self):
        super().__init__('yellow_line_debug_node')

        # =========================
        # 参数
        # =========================
        self.declare_parameter('rgb_topic', '/rgb_camera/rgb_camera/image_raw')

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
        self.declare_parameter('show_mask', True)

        self.rgb_topic = self.get_parameter('rgb_topic').value

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
        self.show_mask = bool(self.get_parameter('show_mask').value)

        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, self.rgb_topic, self.rgb_callback, qos_profile_sensor_data
        )

        self.get_logger().info('YellowLineDebugNode started')
        self.get_logger().info(f'rgb_topic={self.rgb_topic}')

    def detect_yellow_stop_line(self, frame):
        h, w = frame.shape[:2]

        roi_top = int(h * self.yellow_roi_top_ratio)
        roi_left = int(w * self.yellow_roi_left_ratio)
        roi_right = int(w * self.yellow_roi_right_ratio)

        roi = frame[roi_top:h, roi_left:roi_right]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array(
            [self.yellow_h_min, self.yellow_s_min, self.yellow_v_min],
            dtype=np.uint8
        )
        upper_yellow = np.array(
            [self.yellow_h_max, self.yellow_s_max, self.yellow_v_max],
            dtype=np.uint8
        )

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.yellow_min_contour_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            # 不做横线判断，只选最靠下的黄色区域
            score = y + bh

            if score > best_score:
                best_score = score
                best_contour = cnt

        if best_contour is None:
            return {
                'has_line': False,
                'line_bottom_y': None,
                'line_center': None,
                'line_bbox': None,
                'img_shape': (h, w),
                'roi_box': (roi_left, roi_top, roi_right, h - 1),
                'mask': mask,
            }

        x, y, bw, bh = cv2.boundingRect(best_contour)

        x1 = roi_left + x
        y1 = roi_top + y
        x2 = x1 + bw
        y2 = y1 + bh

        line_bottom_y = y2
        cx = x1 + bw // 2
        cy = y1 + bh // 2

        return {
            'has_line': True,
            'line_bottom_y': int(line_bottom_y),
            'line_center': (int(cx), int(cy)),
            'line_bbox': (int(x1), int(y1), int(x2), int(y2)),
            'img_shape': (h, w),
            'roi_box': (roi_left, roi_top, roi_right, h - 1),
            'mask': mask,
        }

    def rgb_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge convert failed: {e}')
            return

        yellow = self.detect_yellow_stop_line(frame)
        vis = frame.copy()

        roi_left, roi_top, roi_right, roi_bottom = yellow['roi_box']

        # 画 ROI
        cv2.rectangle(vis, (roi_left, roi_top), (roi_right, roi_bottom), (255, 255, 0), 1)

        if yellow['has_line']:
            x1, y1, x2, y2 = yellow['line_bbox']
            cx, cy = yellow['line_center']
            yb = yellow['line_bottom_y']

            # 画选中的黄色区域包围框
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # 画中心点
            cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

            # 画 yellow_bottom
            cv2.line(vis, (0, yb), (vis.shape[1] - 1, yb), (0, 255, 0), 2)

            cv2.putText(
                vis,
                f'yellow_bottom={yb}',
                (20, max(30, yb - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            self.get_logger().info(
                f'has_line=True | line_bbox=({x1},{y1},{x2},{y2}) | '
                f'center=({cx},{cy}) | yellow_bottom={yb}',
                throttle_duration_sec=0.5
            )
        else:
            self.get_logger().info(
                'has_line=False',
                throttle_duration_sec=0.5
            )

        cv2.imshow('yellow_line_debug', vis)

        if self.show_mask:
            cv2.imshow('yellow_mask', yellow['mask'])

        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YellowLineDebugNode()
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