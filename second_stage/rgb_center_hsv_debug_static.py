#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data


class RgbCenterHsvDebugStaticNode(Node):
    def __init__(self):
        super().__init__('rgb_center_hsv_debug_static_node')

        self.declare_parameter('rgb_topic', '/rgb_camera/rgb_camera/image_raw')
        self.rgb_topic = self.get_parameter('rgb_topic').value

        self.bridge = CvBridge()

        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info('RgbCenterHsvDebugStaticNode started')
        self.get_logger().info(f'rgb_topic={self.rgb_topic}')
        self.get_logger().info('Robot stays still. Only printing center 21x21 BGR / HSV.')

    def rgb_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge convert failed: {e}')
            return

        h, w = frame.shape[:2]
        cx = w // 2
        cy = h // 2

        half_size = 10  # 21x21
        x1 = max(0, cx - half_size)
        x2 = min(w, cx + half_size + 1)
        y1 = max(0, cy - half_size)
        y2 = min(h, cy + half_size + 1)

        center_roi_bgr = frame[y1:y2, x1:x2]
        if center_roi_bgr.size == 0:
            return

        center_roi_hsv = cv2.cvtColor(center_roi_bgr, cv2.COLOR_BGR2HSV)

        # 中心单像素 BGR / HSV
        center_pixel_bgr = frame[cy, cx]
        center_pixel_hsv = cv2.cvtColor(
            np.uint8([[center_pixel_bgr]]),
            cv2.COLOR_BGR2HSV
        )[0, 0]

        # BGR 统计
        b_mean = float(np.mean(center_roi_bgr[:, :, 0]))
        g_mean = float(np.mean(center_roi_bgr[:, :, 1]))
        r_mean = float(np.mean(center_roi_bgr[:, :, 2]))

        b_min = int(np.min(center_roi_bgr[:, :, 0]))
        g_min = int(np.min(center_roi_bgr[:, :, 1]))
        r_min = int(np.min(center_roi_bgr[:, :, 2]))

        b_max = int(np.max(center_roi_bgr[:, :, 0]))
        g_max = int(np.max(center_roi_bgr[:, :, 1]))
        r_max = int(np.max(center_roi_bgr[:, :, 2]))

        # HSV 统计
        h_mean = float(np.mean(center_roi_hsv[:, :, 0]))
        s_mean = float(np.mean(center_roi_hsv[:, :, 1]))
        v_mean = float(np.mean(center_roi_hsv[:, :, 2]))

        h_min = int(np.min(center_roi_hsv[:, :, 0]))
        s_min = int(np.min(center_roi_hsv[:, :, 1]))
        v_min = int(np.min(center_roi_hsv[:, :, 2]))

        h_max = int(np.max(center_roi_hsv[:, :, 0]))
        s_max = int(np.max(center_roi_hsv[:, :, 1]))
        v_max = int(np.max(center_roi_hsv[:, :, 2]))

        self.get_logger().info(
            f'[CENTER 21x21 STATIC] '
            f'roi=({x1}:{x2}, {y1}:{y2}) | '
            f'center_pixel_bgr=({int(center_pixel_bgr[0])}, {int(center_pixel_bgr[1])}, {int(center_pixel_bgr[2])}) | '
            f'center_pixel_hsv=({int(center_pixel_hsv[0])}, {int(center_pixel_hsv[1])}, {int(center_pixel_hsv[2])}) | '
            f'roi_mean_bgr=({b_mean:.1f}, {g_mean:.1f}, {r_mean:.1f}) | '
            f'roi_min_bgr=({b_min}, {g_min}, {r_min}) | '
            f'roi_max_bgr=({b_max}, {g_max}, {r_max}) | '
            f'roi_mean_hsv=({h_mean:.1f}, {s_mean:.1f}, {v_mean:.1f}) | '
            f'roi_min_hsv=({h_min}, {s_min}, {v_min}) | '
            f'roi_max_hsv=({h_max}, {s_max}, {v_max})',
            throttle_duration_sec=0.5
        )


def main(args=None):
    rclpy.init(args=args)
    node = RgbCenterHsvDebugStaticNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down...')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()