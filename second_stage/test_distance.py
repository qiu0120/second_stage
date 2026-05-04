#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
单独测试代码：检测当前 RGB 图像里的橙色 / 蓝色小球，并在 OpenCV 窗口显示距离。

功能：
1. 订阅 RGB 图像
2. 订阅深度图
3. 按当前第二赛段代码的 HSV + 轮廓逻辑检测橙球、蓝球
4. 对每个球在深度图对应位置附近取深度
5. 可视化显示：
   - 橙球 / 蓝球圆框
   - center
   - radius
   - depth_m
   - error_x
   - side
   - 左右最近参考球 left_ref / right_ref
   - 左右参考球深度差 depth_diff
   - 中线 lane_mid_x
"""

import math
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class TestOrangeBlueBallDepthVisNode(Node):
    def __init__(self):
        super().__init__('test_orange_blue_ball_depth_vis_node')

        # =========================
        # 话题
        # =========================
        self.declare_parameter('rgb_topic', '/rgb_camera/rgb_camera/image_raw')
        self.declare_parameter('depth_topic', '/d435/depth/d435_depth/depth/image_raw')

        # =========================
        # 橙球 HSV
        # =========================
        self.declare_parameter('orange_h_min', 5)
        self.declare_parameter('orange_h_max', 25)
        self.declare_parameter('orange_s_min', 100)
        self.declare_parameter('orange_s_max', 255)
        self.declare_parameter('orange_v_min', 80)
        self.declare_parameter('orange_v_max', 255)
        self.declare_parameter('orange_min_contour_area', 400.0)

        # =========================
        # 蓝球 HSV
        # =========================
        self.declare_parameter('blue_h_min', 90)
        self.declare_parameter('blue_h_max', 130)
        self.declare_parameter('blue_s_min', 80)
        self.declare_parameter('blue_s_max', 255)
        self.declare_parameter('blue_v_min', 50)
        self.declare_parameter('blue_v_max', 255)
        self.declare_parameter('blue_min_contour_area', 400.0)

        # =========================
        # 深度搜索
        # =========================
        self.declare_parameter('depth_search_half', 12)
        self.declare_parameter('valid_min_depth_m', 0.05)
        self.declare_parameter('valid_max_depth_m', 10.0)

        # =========================
        # 检测 / 可视化
        # =========================
        self.declare_parameter('show_mask', False)
        self.declare_parameter('prefer_nearest_ball', True)
        self.declare_parameter('min_ball_radius_to_trigger', 40.0)

        # 如果左右参考球深度差超过这个值，就在窗口提示 FAR_SIDE_BIAS
        self.declare_parameter('center_depth_diff_disable_align_m', 0.25)

        self.rgb_topic = str(self.get_parameter('rgb_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)

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

        self.depth_search_half = int(self.get_parameter('depth_search_half').value)
        self.valid_min_depth_m = float(self.get_parameter('valid_min_depth_m').value)
        self.valid_max_depth_m = float(self.get_parameter('valid_max_depth_m').value)

        self.show_mask = bool(self.get_parameter('show_mask').value)
        self.prefer_nearest_ball = bool(self.get_parameter('prefer_nearest_ball').value)
        self.min_ball_radius_to_trigger = float(self.get_parameter('min_ball_radius_to_trigger').value)
        self.center_depth_diff_disable_align_m = float(
            self.get_parameter('center_depth_diff_disable_align_m').value
        )

        self.bridge = CvBridge()

        self.latest_depth = None
        self.latest_depth_encoding = None

        self.rgb_w = 640
        self.rgb_h = 480

        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            qos_profile_sensor_data
        )
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info('TestOrangeBlueBallDepthVisNode started.')
        self.get_logger().info(f'rgb_topic={self.rgb_topic}')
        self.get_logger().info(f'depth_topic={self.depth_topic}')

    # ============================================================
    # 回调
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
            self.get_logger().error(f'RGB convert failed: {e}')
            return

        result = self.detect_ball_scene(frame)
        self.show_debug_window(frame, result)

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
            self.get_logger().warn(
                f'Unsupported depth encoding: {encoding}',
                throttle_duration_sec=1.0
            )
            return None, (depth_cx, depth_cy), (x1, y1, x2, y2)

        valid = patch_m[np.isfinite(patch_m)]
        valid = valid[(valid > self.valid_min_depth_m) & (valid < self.valid_max_depth_m)]

        if valid.size == 0:
            return None, (depth_cx, depth_cy), (x1, y1, x2, y2)

        # 和主代码一致：用较近一侧的 20 分位数，减少背景污染
        depth_m = float(np.percentile(valid, 20))
        return depth_m, (depth_cx, depth_cy), (x1, y1, x2, y2)

    # ============================================================
    # 球检测
    # ============================================================
    def make_color_mask(
        self,
        frame: np.ndarray,
        h_min: int, h_max: int,
        s_min: int, s_max: int,
        v_min: int, v_max: int,
    ) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

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

        mask = self.make_color_mask(frame, h_min, h_max, s_min, s_max, v_min, v_max)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_contour_area:
                continue

            (cx_f, cy_f), r_circle = cv2.minEnclosingCircle(cnt)
            cx = int(cx_f)
            cy = int(cy_f)
            r_circle = float(r_circle)

            # 面积等效半径
            r_eq = math.sqrt(area / math.pi)

            # 和主代码一致：只信较小的半径
            radius = min(r_circle, r_eq)

            depth_m, depth_center, depth_box = self.get_depth_for_rgb_point(cx, cy)

            image_center_x = w // 2
            error_x = cx - image_center_x
            side = 'left' if cx < image_center_x else 'right'

            # 测试代码里：即使深度为空也画出来，方便判断是颜色没问题还是深度没问题
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

        candidates.sort(key=lambda b: (999.0 if b['depth_m'] is None else b['depth_m']))
        return candidates

    def choose_side_reference_ball(self, balls: List[Dict]) -> Optional[Dict]:
        valid = [b for b in balls if b.get('depth_m') is not None]
        if len(valid) == 0:
            return None
        return min(valid, key=lambda b: b['depth_m'])

    def choose_best_target_orange_ball(self, orange_balls: List[Dict]) -> Optional[Dict]:
        valid = [b for b in orange_balls if b.get('depth_m') is not None]
        if len(valid) == 0:
            return None

        if self.prefer_nearest_ball:
            return min(valid, key=lambda b: b['depth_m'])

        return min(valid, key=lambda b: b['depth_m'] + 0.002 * abs(b['error_x']))

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
        lane_mid_x = None
        depth_diff = None
        far_side = None
        center_mode = 'NO_REF'

        if has_center_reference:
            left_cx = left_ref['center'][0]
            right_cx = right_ref['center'][0]
            lane_mid_x = 0.5 * (left_cx + right_cx)
            center_error_px = lane_mid_x - image_center_x

            left_depth = left_ref['depth_m']
            right_depth = right_ref['depth_m']
            depth_diff = abs(left_depth - right_depth)

            if depth_diff >= self.center_depth_diff_disable_align_m:
                center_mode = 'FAR_SIDE_BIAS'
                far_side = 'left' if left_depth > right_depth else 'right'
            else:
                center_mode = 'NORMAL_CENTER'

        best_target_ball = self.choose_best_target_orange_ball(orange_balls)

        return {
            'img_shape': (h, w),
            'orange_balls': orange_balls,
            'blue_balls': blue_balls,
            'all_balls': all_balls,
            'left_balls': left_balls,
            'right_balls': right_balls,
            'left_ref': left_ref,
            'right_ref': right_ref,
            'has_center_reference': has_center_reference,
            'center_error_px': center_error_px,
            'lane_mid_x': lane_mid_x,
            'depth_diff': depth_diff,
            'far_side': far_side,
            'center_mode': center_mode,
            'best_target_ball': best_target_ball,
        }

    # ============================================================
    # 可视化
    # ============================================================
    def draw_ball(self, vis: np.ndarray, b: Dict, idx: int):
        color_name = b['color']
        if color_name == 'orange':
            draw_color = (0, 140, 255)
            label = f'O{idx}'
        else:
            draw_color = (255, 0, 0)
            label = f'B{idx}'

        cx, cy = b['center']
        radius = int(max(3, round(b.get('radius', 3))))

        cv2.circle(vis, (cx, cy), radius, draw_color, 2)
        cv2.circle(vis, (cx, cy), 4, draw_color, -1)

        depth_m = b.get('depth_m')
        depth_text = 'None' if depth_m is None else f'{depth_m:.2f}m'

        text1 = f'{label} {color_name} d={depth_text}'
        text2 = (
            f'r={b.get("radius", 0):.1f} '
            f'rc={b.get("radius_circle", 0):.1f} '
            f'req={b.get("radius_eq", 0):.1f} '
            f'ex={b.get("error_x")}'
        )

        x_text = max(5, min(cx - 60, vis.shape[1] - 260))
        y_text = max(20, cy - radius - 22)

        cv2.putText(
            vis,
            text1,
            (x_text, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            draw_color,
            2
        )
        cv2.putText(
            vis,
            text2,
            (x_text, y_text + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            draw_color,
            1
        )

        # 深度搜索框映射回 RGB 图像，方便看深度取样位置
        box = b.get('depth_box')
        if box is not None and self.latest_depth is not None:
            dh, dw = self.latest_depth.shape[:2]
            x1, y1, x2, y2 = box
            rx1 = int(x1 * self.rgb_w / max(dw, 1))
            rx2 = int(x2 * self.rgb_w / max(dw, 1))
            ry1 = int(y1 * self.rgb_h / max(dh, 1))
            ry2 = int(y2 * self.rgb_h / max(dh, 1))
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), draw_color, 1)

    def show_debug_window(self, frame: np.ndarray, result: Dict):
        vis = frame.copy()
        h, w = vis.shape[:2]

        image_center_x = w // 2
        image_center_y = h // 2

        cv2.line(vis, (image_center_x, 0), (image_center_x, h - 1), (255, 255, 255), 1)
        cv2.line(vis, (0, image_center_y), (w - 1, image_center_y), (80, 80, 80), 1)

        orange_balls = result['orange_balls']
        blue_balls = result['blue_balls']

        cv2.putText(
            vis,
            f'orange_cnt={len(orange_balls)} blue_cnt={len(blue_balls)}',
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            vis,
            f'depth_encoding={self.latest_depth_encoding}',
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        for idx, b in enumerate(orange_balls):
            self.draw_ball(vis, b, idx)

        for idx, b in enumerate(blue_balls):
            self.draw_ball(vis, b, idx)

        left_ref = result.get('left_ref')
        right_ref = result.get('right_ref')

        if left_ref is not None:
            cx, cy = left_ref['center']
            cv2.circle(vis, (cx, cy), 10, (255, 255, 0), 3)
            cv2.putText(
                vis,
                f'LEFT_REF d={left_ref["depth_m"]:.2f}',
                (cx + 8, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 0),
                2
            )

        if right_ref is not None:
            cx, cy = right_ref['center']
            cv2.circle(vis, (cx, cy), 10, (255, 255, 0), 3)
            cv2.putText(
                vis,
                f'RIGHT_REF d={right_ref["depth_m"]:.2f}',
                (cx + 8, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 0),
                2
            )

        if result.get('has_center_reference'):
            lane_mid_x = int(result['lane_mid_x'])
            center_error_px = result['center_error_px']
            cv2.line(vis, (lane_mid_x, 0), (lane_mid_x, h - 1), (0, 255, 0), 2)

            depth_diff = result.get('depth_diff')
            far_side = result.get('far_side')
            center_mode = result.get('center_mode')

            info = (
                f'lane_mid={lane_mid_x} center_err={center_error_px:.1f} '
                f'diff={depth_diff:.2f} mode={center_mode} far={far_side}'
            )
            cv2.putText(
                vis,
                info,
                (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                vis,
                'center_ref: not enough valid left/right depth balls',
                (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 0),
                2
            )

        target = result.get('best_target_ball')
        if target is not None:
            cx, cy = target['center']
            radius = int(max(8, round(target.get('radius', 8))))
            cv2.circle(vis, (cx, cy), radius + 5, (0, 0, 255), 3)
            cv2.putText(
                vis,
                f'TARGET orange d={target["depth_m"]:.2f} side={target["side"]}',
                (max(5, cx - 80), min(h - 10, cy + radius + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 0, 255),
                2
            )

        cv2.imshow('test_orange_blue_ball_depth_vis', vis)

        if self.show_mask:
            orange_mask = self.make_color_mask(
                frame,
                self.orange_h_min, self.orange_h_max,
                self.orange_s_min, self.orange_s_max,
                self.orange_v_min, self.orange_v_max,
            )
            blue_mask = self.make_color_mask(
                frame,
                self.blue_h_min, self.blue_h_max,
                self.blue_s_min, self.blue_s_max,
                self.blue_v_min, self.blue_v_max,
            )
            cv2.imshow('test_orange_mask', orange_mask)
            cv2.imshow('test_blue_mask', blue_mask)

        cv2.waitKey(1)

        self.get_logger().info(
            f'orange_cnt={len(orange_balls)} blue_cnt={len(blue_balls)} '
            f'left_ref={"Y" if left_ref is not None else "N"} '
            f'right_ref={"Y" if right_ref is not None else "N"} '
            f'center_mode={result.get("center_mode")}',
            throttle_duration_sec=0.6
        )


def main(args=None):
    rclpy.init(args=args)
    node = TestOrangeBlueBallDepthVisNode()

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
