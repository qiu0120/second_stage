#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['QT_X11_NO_MITSHM'] = '1'
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import math
import threading
import sys
import traceback
import cv2
import numpy as np

# 导入底层 LCM 消息 (请确保路径正确)
sys.path.append('/home/cyberdog_sim/loco_hl_example/basic_motion')
try:
    from robot_control_cmd_lcmt import robot_control_cmd_lcmt
    from robot_control_response_lcmt import robot_control_response_lcmt
except ImportError:
    print("⚠️ 找不到 LCM 文件，请检查 sys.path 路径！")

from cyberdog_msg.msg import YamlParam

class ControlParameterValueKind:
    kVEC_X_DOUBLE = 3

class LcmController:
    def __init__(self):
        import lcm
        self.lc_r = lcm.LCM("udpm://239.255.76.67:7670?ttl=255")
        self.lc_s = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
        self.cmd_msg = robot_control_cmd_lcmt()
        self.rec_msg = robot_control_response_lcmt()
        self.send_lock = threading.Lock()
        self.delay_cnt = 0
        self.running = True

        self.rec_thread = threading.Thread(target=self.rec_response, daemon=True)
        self.send_thread = threading.Thread(target=self.send_publish, daemon=True)
        self.rec_thread.start()
        self.send_thread.start()

    def msg_handler(self, channel, data):
        self.rec_msg = robot_control_response_lcmt().decode(data)

    def rec_response(self):
        self.lc_r.subscribe("robot_control_response", self.msg_handler)
        while self.running:
            self.lc_r.handle()
            time.sleep(0.002)

    def send_publish(self):
        while self.running:
            with self.send_lock:
                if self.delay_cnt > 20: 
                    self.lc_s.publish("robot_control_cmd", self.cmd_msg.encode())
                    self.delay_cnt = 0
                self.delay_cnt += 1
            time.sleep(0.005)

    def send_cmd(self, msg):
        with self.send_lock:
            self.delay_cnt = 50
            self.cmd_msg = msg

    def quit(self):
        self.running = False
        self.rec_thread.join()
        self.send_thread.join()

class SneakController(Node):
    def __init__(self, controller: LcmController):
        super().__init__('sneak_controller')
        self.controller = controller
        # 👇👇👇 [核心修改]：强制当前节点使用 Gazebo 的仿真时间
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        # 👆👆👆 
        self.yaml_pub = self.create_publisher(YamlParam, "yaml_parameter", 10)
        
        # ========================================================
        # 👑 战术调参区 (重点关注这里)
        # ========================================================
        self.blind_march_time_s = 2       # 【盲走时间】起步先闭眼往前走几秒
        self.north_stop_dist_m = 0.45       # 【北墙刹车距离】(米)
        self.align_tolerance_deg = 1      # 【垂直容差】偏角小于几度算对齐了
        
        # 👇👇👇 转身90度调参区 👇👇👇
        self.turn_vyaw = -0.6               # 自转角速度 (负数是右转，保持不变即可)
        self.turn_time_s = 3.24              # 【调这里！】转身时间(秒)。慢慢改这个数字直到刚好转90度！
        # 👆👆👆 转身90度调参区 👆👆👆
        
       # 👇👇👇 东进潜入调参区 (绕过球走到死角) 👇👇👇
        self.eastward_vx = 0.25             # 【向东潜入】前进速度
        self.eastward_vy = 0.25             # 【往北的横移】贴墙避开球的侧滑速度
        self.eastward_vy_time_s = 1.8       # 【横移持续时间】往北侧滑几秒后停止横移
        
        # 👑【核心调参点】：越过球之后的盲走定位
        self.eastward_blind_after_lost_s = 2.0 # 调这里！球从视野消失(说明我们绕过它了)后，再往前走几秒刚好到达完美死角
        self.eastward_timeout_s = 20.0      # 【绝对保底机制】最多走20秒强制停
        # 👆👆👆 东进潜入调参区 👆👆👆
        
        # 👇👇👇 解围推球调参区 👇👇👇
        self.clear_turn_vyaw = -0.4         # 【解围自转】角速度 (负数右转，正数左转，您可以观察往哪边转更容易把球兜住)
        self.clear_turn_time_s = 2.3        # 【解围自转】持续时间 (秒)
        self.buffer_crab_vy = -0.05         # 【缓冲横移】极慢的右横移速度，用于拉开与左墙的间隙
        self.buffer_crab_time_s = 0.2       # 【缓冲横移】持续时间 (秒)，调大拉开的距离越远
        self.clear_crab_vy = -0.40          # 【加速右横移】侧滑速度 (负数代表向右横移！数值越小(如-0.3)横移越快)
        self.clear_crab_time_s = 3        # 【加速右横移】持续时间 (秒)
        # 👆👆👆 解围推球调参区 👆👆👆

      # 👇👇👇 终极解围后新战术：西墙校准与冲线调参区 👇👇👇
        # 1. 转西墙
        self.west_turn_vyaw = -0.6           # 【转西墙】角速度 (正数左转，负数右转，视狗当前姿态而定)
        self.west_turn_time_s = 5.2         # 【转西墙】时间(秒)。调这里让它大概正对西墙！
  
        
        
        self.west_visual_march_vx = 0.5    # 【视觉靠近西墙】边走边调姿态的速度
        self.west_stop_dist_m = 0.45        # 🎯【调这里！】距离西墙多近时刹车 (米)
        self.west_align_tolerance_deg = 1 # 【西墙校准】偏角小于几度算对齐西墙

        # 2. 转出口
        self.exit_turn_vyaw = 0.6           # 【转出口】角速度 (反向转回去面朝南方出口)
        self.exit_turn_time_s = 2.7         # 【转出口】时间(秒)。调这里让它刚好正对出口！
        
        # 👇👇👇 终极冲刺：三点一线推球调参区 👇👇👇
        self.push_vx = 0.30                 # 【推球】前进速度 (稳稳向前推)
        self.push_vy_kp = 0.30              # 【推球】横向追球敏锐度 (球偏了，狗往侧面滑去追的力度)
        self.push_vyaw_kp = 0.6             # 【推球】转头瞄准出口敏锐度 (头永远正对大门)
        

        self.cross_line_time_s = 0.01        # 🎯【进圈时间】大门从视野消失(或距离到达)后，闭眼再推几秒让后腿进圈！
        self.push_timeout_s = 20          # 【超时保底】最多推 15 秒强制结束
        # 👆👆👆 终极冲刺：三点一线推球调参区 👆👆👆
        
# 🆕 新增：出口冲线与趴下参数
     

        self.life_count_val = 0
        self.wall_angle_rad = 0.0
        self.wall_dist = -1.0

        # ========================================================
        # 视觉模块整合区：原 ball_vision_tracker + wall_vision_tracker
        # 不再发布/订阅 /vision/ball_info、/vision/wall_info、/vision/exit_info，
        # 直接在同一个节点内更新 self.ball_* / self.wall_* / self.exit_*。
        # ========================================================
        self.bridge = CvBridge()
        self.latest_depth = None

        self.wall_angle_rad = 0.0
        self.wall_dist = -1.0

        self.ball_offset_x = -999.0
        self.ball_dist = -1.0

        self.exit_offset_norm = -999.0
        self.exit_dist = -1.0

        self.show_vision = True
        self.vision_window_name = 'Sixth Stage Integrated Vision'
        self.vision_window_ready = False

        self.create_subscription(Image, '/rgb_camera/rgb_camera/image_raw', self.image_callback, qos_profile_sensor_data)
        self.create_subscription(Image, '/d435/depth/d435_depth/depth/image_raw', self.depth_callback, qos_profile_sensor_data)

        self.timer = self.create_timer(0.1, self.behavior_loop) # 10Hz
        
        # 将初始状态设为盲走
        self.state = 'BLIND_MARCH' 
        self.state_ticks = 0
        self.stable_counter = 0
        
        self.set_dynamic_shape(target_height=0.25, leg_offset=0.04)
        self.get_logger().info("=== 战术启动：幽灵潜入 ===")

    def publish_yaml_vecxd(self, name: str, values):
        msg = YamlParam()
        msg.name = name
        msg.kind = ControlParameterValueKind.kVEC_X_DOUBLE
        msg.vecxd_value = [float(v) for v in values]
        msg.is_user = 1
        self.yaml_pub.publish(msg)

    def set_dynamic_shape(self, target_height: float, leg_offset: float):
        values_h = [0.0] * 12
        values_h[2] = float(target_height)
        self.publish_yaml_vecxd("des_roll_pitch_height_motion", values_h)
        self.publish_yaml_vecxd("des_roll_pitch_height", values_h)
        
        values_y = [0.0] * 12
        values_y[0] = float(leg_offset)
        values_y[1] = -float(leg_offset)
        values_y[2] = 0.04
        values_y[3] = -0.04
        self.publish_yaml_vecxd("y_offset_trot", values_y)
        self.publish_yaml_vecxd("y_offset_trot_10_4", values_y)

    def depth_callback(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception:
            self.get_logger().warn('Depth image convert failed', throttle_duration_sec=1.0)

    def _median_depth_m(self, cx, cy, patch_radius=5, max_depth=5.0):
        """从深度图局部 patch 取中位数，返回米；无效时返回 -1.0。"""
        if self.latest_depth is None:
            return -1.0

        h, w = self.latest_depth.shape[:2]
        x = int(max(patch_radius, min(w - patch_radius - 1, cx)))
        y = int(max(patch_radius, min(h - patch_radius - 1, cy)))
        patch = self.latest_depth[y-patch_radius:y+patch_radius, x-patch_radius:x+patch_radius]

        if patch.dtype == np.uint16:
            patch = patch.astype(np.float32) / 1000.0
        else:
            patch = patch.astype(np.float32)

        valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < max_depth)]
        if valid.size == 0:
            return -1.0
        return float(np.median(valid))

    def _update_wall_vision(self, cv_image, hsv):
        """整合 wall_vision_tracker：检测最近内墙，更新 wall_angle_rad / wall_dist。"""
        height, width = cv_image.shape[:2]

        lower_yellow = np.array([10, 50, 40])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 原 wall_vision_tracker 的 ROI：只看画面中下部，减少选到远处墙线。
        roi_x_min = int(width * 0.20)
        roi_x_max = int(width * 0.80)
        roi_y_min = int(height * 0.70)
        roi_y_max = height

        mask[:roi_y_min, :] = 0
        mask[:, :roi_x_min] = 0
        mask[:, roi_x_max:] = 0

        cv2.rectangle(cv_image, (roi_x_min, roi_y_min), (roi_x_max, roi_y_max), (0, 0, 255), 2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.wall_angle_rad = 0.0
        self.wall_dist = -1.0
        valid_walls = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 300:
                continue

            vx, vy, x0, y0 = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = float(vx), float(vy)
            if vx < 0:
                vx, vy = -vx, -vy

            angle_rad = math.atan2(vy, vx)
            if abs(math.degrees(angle_rad)) < 30.0:
                valid_walls.append({
                    'area': area,
                    'angle_rad': angle_rad,
                    'cx': float(x0),
                    'cy': float(y0),
                    'vx': vx,
                    'vy': vy,
                })

        if not valid_walls:
            cv2.putText(cv_image, 'WALL: SEARCHING', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return

        # 选画面里位置最低的黄线，也就是最近内墙。
        best_wall = max(valid_walls, key=lambda item: item['cy'])
        angle_rad = best_wall['angle_rad']
        cx_contour, cy_contour = best_wall['cx'], best_wall['cy']
        vx, vy = best_wall['vx'], best_wall['vy']

        line_length = 500
        pt1 = (int(cx_contour - vx * line_length), int(cy_contour - vy * line_length))
        pt2 = (int(cx_contour + vx * line_length), int(cy_contour + vy * line_length))
        cv2.line(cv_image, pt1, pt2, (0, 255, 0), 2)

        center_x = width // 2
        if abs(vx) > 1e-5:
            center_y = int(cy_contour + (vy / vx) * (center_x - cx_contour))
        else:
            center_y = int(cy_contour)

        center_x = max(10, min(width - 11, center_x))
        center_y = max(10, min(height - 11, center_y))

        cv2.circle(cv_image, (center_x, center_y), 8, (0, 0, 255), -1)
        cv2.line(cv_image, (center_x, 0), (center_x, height), (0, 255, 255), 1)

        real_dist = self._median_depth_m(center_x, center_y, patch_radius=5, max_depth=5.0)
        self.wall_angle_rad = float(angle_rad)
        self.wall_dist = float(real_dist)

        if real_dist > 0:
            cv2.putText(cv_image, f'WALL Dist:{real_dist:.2f}m ANG:{math.degrees(angle_rad):.1f}deg',
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(cv_image, 'WALL DEPTH INVALID', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def _update_exit_vision(self, cv_image, hsv):
        """整合 ball_vision_tracker 的出口检测：检测黄色墙壁断口，更新 exit_offset_norm / exit_dist。"""
        height, width = cv_image.shape[:2]

        self.exit_offset_norm = -999.0
        self.exit_dist = -1.0

        lower_yellow = np.array([10, 50, 40])
        upper_yellow = np.array([40, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel_y = np.ones((9, 9), np.uint8)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel_y)
        yellow_contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_walls = []
        for c in yellow_contours:
            if cv2.contourArea(c) > 1500:
                x, y, w, h = cv2.boundingRect(c)
                valid_walls.append({'x': x, 'y': y, 'w': w, 'h': h, 'bottom_y': y + h, 'contour': c})

        if not valid_walls:
            cv2.putText(cv_image, 'EXIT: SEARCHING', (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return

        valid_walls.sort(key=lambda item: item['bottom_y'], reverse=True)
        nearest_bottom_y = valid_walls[0]['bottom_y']
        front_walls = [w for w in valid_walls if abs(w['bottom_y'] - nearest_bottom_y) < 350]
        front_walls.sort(key=lambda item: item['x'])

        for i, w in enumerate(front_walls):
            cv2.drawContours(cv_image, [w['contour']], -1, (0, 255, 255), 2)
            cv2.putText(cv_image, f'EW{i+1}', (w['x'], max(20, w['y'] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if len(front_walls) < 2:
            cv2.putText(cv_image, 'EXIT: NEED 2 WALLS', (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return

        max_gap = 0
        gap_cx = -1
        best_left_edge = None
        best_right_edge = None
        for i in range(len(front_walls) - 1):
            left_edge = front_walls[i]['x'] + front_walls[i]['w']
            right_edge = front_walls[i + 1]['x']
            gap = right_edge - left_edge
            if gap > max_gap:
                max_gap = gap
                gap_cx = left_edge + gap / 2.0
                best_left_edge = left_edge
                best_right_edge = right_edge

        if max_gap <= 20 or best_left_edge is None or best_right_edge is None:
            cv2.putText(cv_image, 'EXIT: GAP INVALID', (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return

        cv2.line(cv_image, (int(gap_cx), 0), (int(gap_cx), height), (255, 0, 255), 3)
        self.exit_offset_norm = float((gap_cx - width / 2.0) / (width / 2.0))

        exit_dist = -1.0
        px_l = int(max(5, min(width - 6, best_left_edge - 5)))
        px_r = int(max(5, min(width - 6, best_right_edge + 5)))
        py = int(max(5, min(height - 6, nearest_bottom_y - 10)))

        cv2.circle(cv_image, (px_l, py), 5, (255, 0, 0), -1)
        cv2.circle(cv_image, (px_r, py), 5, (255, 0, 0), -1)

        d_l = self._median_depth_m(px_l, py, patch_radius=5, max_depth=5.0)
        d_r = self._median_depth_m(px_r, py, patch_radius=5, max_depth=5.0)
        valid_dists = [d for d in (d_l, d_r) if d > 0]
        if valid_dists:
            exit_dist = float(min(valid_dists))
        self.exit_dist = exit_dist

        if exit_dist > 0:
            cv2.putText(cv_image, f'EXIT off:{self.exit_offset_norm:.2f} dist:{exit_dist:.2f}m',
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        else:
            cv2.putText(cv_image, f'EXIT off:{self.exit_offset_norm:.2f} depth invalid',
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def _update_ball_vision(self, cv_image, hsv):
        """整合 ball_vision_tracker 的白球检测，更新 ball_offset_x / ball_dist。"""
        height, width = cv_image.shape[:2]

        self.ball_offset_x = -999.0
        self.ball_dist = -1.0

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        kernel_white = np.ones((5, 5), np.uint8)
        mask_white = cv2.erode(mask_white, kernel_white, iterations=1)
        mask_white = cv2.dilate(mask_white, kernel_white, iterations=2)

        contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        for c in contours:
            ((cx, cy), cr) = cv2.minEnclosingCircle(c)
            if cr > 2:
                valid_contours.append(c)

        if not valid_contours:
            cv2.putText(cv_image, 'BALL: SEARCHING', (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return

        c_best = max(valid_contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c_best)

        cv2.circle(cv_image, (int(x), int(y)), int(max(radius, 2)), (0, 255, 0), 2)
        cv2.circle(cv_image, (int(x), int(y)), 3, (0, 0, 255), -1)

        self.ball_offset_x = float((x - width / 2.0) / (width / 2.0))
        self.ball_dist = self._median_depth_m(x, y, patch_radius=5, max_depth=5.0)

        if self.ball_dist > 0:
            cv2.putText(cv_image, f'BALL off:{self.ball_offset_x:.2f} dist:{self.ball_dist:.2f}m',
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(cv_image, f'BALL off:{self.ball_offset_x:.2f} depth invalid',
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # 一个图像回调里同时更新三类视觉结果。
            self._update_wall_vision(cv_image, hsv)
            self._update_exit_vision(cv_image, hsv)
            self._update_ball_vision(cv_image, hsv)

            cv2.putText(cv_image, f'STATE: {self.state}', (20, cv_image.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if self.show_vision:
                if not self.vision_window_ready:
                    try:
                        cv2.namedWindow(self.vision_window_name, cv2.WINDOW_NORMAL)
                        self.vision_window_ready = True
                    except Exception:
                        self.show_vision = False
                if self.show_vision:
                    cv2.imshow(self.vision_window_name, cv_image)
                    cv2.waitKey(1)

        except Exception:
            self.get_logger().error(f'Integrated vision failed:\n{traceback.format_exc()}')

    def destroy_vision_windows(self):
        try:
            if self.vision_window_ready:
                cv2.destroyWindow(self.vision_window_name)
            cv2.destroyAllWindows()
        except Exception:
            pass

    def behavior_loop(self):
        self.life_count_val = (self.life_count_val + 1) % 128
        msg = robot_control_cmd_lcmt()
        msg.life_count = self.life_count_val
        msg.duration = 0 
        msg.pos_des = [0.0, 0.0, 0.0]
        msg.rpy_des = [0.0, 0.0, 0.0]
        self.state_ticks += 1

        wall_visible = (self.wall_dist != -1.0)
      
        # ==========================================
        # 0. 盲走阶段
        # ==========================================
        if self.state == 'BLIND_MARCH':
            msg.mode = 11
            msg.gait_id = 3 
            msg.step_height = [0.05, 0.05] 
            
            if self.state_ticks >= int(self.blind_march_time_s * 10):
                self.get_logger().info("🦯 盲走结束，开启视觉雷达看北墙！")
                self.state = 'NORTHWARD_MARCH'
                self.state_ticks = 0
                return
                
            msg.vel_des = [0.5, 0.0, 0.0]
            self.controller.send_cmd(msg)

        # ==========================================
        # 1. 视觉北伐阶段
        # ==========================================
        elif self.state == 'NORTHWARD_MARCH':
            msg.mode = 11
            msg.gait_id = 3 
            msg.step_height = [0.05, 0.05] 
            
            vx = 0.5 
            vyaw = 0.0 
            
            if wall_visible:
                # 边走边粗调朝向
                vyaw = - (self.wall_angle_rad * 0.8) 
                vyaw = max(min(vyaw, 0.3), -0.3)
                
                if 0.0 < self.wall_dist < self.north_stop_dist_m:
                    self.stable_counter += 1
                    if self.stable_counter >= 2:
                        self.get_logger().info(f"🎯 抵达极近距离 ({self.wall_dist:.2f}m)！停车准备原地垂直校准。")
                        self.state = 'FINAL_ALIGN'
                        self.state_ticks = 0
                        self.stable_counter = 0
                       
                        msg.vel_des = [0.0, 0.0, 0.0]
                        
                        return
                else:
                    self.stable_counter = 0
                
            msg.vel_des = [vx, 0.0, vyaw]
            self.controller.send_cmd(msg)

        # ==========================================
        # 2. 原地绝对垂直校准
        # ==========================================
        elif self.state == 'FINAL_ALIGN':
            msg.mode = 11
            msg.gait_id = 3
            msg.step_height = [0.04, 0.04]
            
            if wall_visible:
                angle_deg = math.degrees(self.wall_angle_rad)
                
                # 对齐了！进入开环参数自转
                if abs(angle_deg) < self.align_tolerance_deg:
                    self.stable_counter += 1
                    if self.stable_counter >= 3:
                        self.get_logger().info(f"📐 北墙已绝对垂直 (偏角: {angle_deg:.1f}度)，开始自转90度！")
                        self.state = 'OPEN_LOOP_TURN'
                        self.state_ticks = 0
                        self.stable_counter = 0
                        
                        return
                else:
                    self.stable_counter = 0
                    
                # 原地强扭
                vyaw = - (self.wall_angle_rad * 1.5)
                vyaw = max(min(vyaw, 0.4), -0.4)
                msg.vel_des = [0.0, 0.0, vyaw]
            else:
                msg.vel_des = [0.0, 0.0, 0.0]
                
            # 超时保护(最多对齐3秒)
            if self.state_ticks > 35:
                self.get_logger().warn("⚠️ 校准超时，强制进入转身！")
                self.state = 'OPEN_LOOP_TURN'
                self.state_ticks = 0
                
            self.controller.send_cmd(msg)

        # ==========================================
        # 3. 参数开环转身 (调参区在上面)
        # ==========================================
        elif self.state == 'OPEN_LOOP_TURN':
            msg.mode = 11
            msg.gait_id = 3
            msg.step_height = [0.04, 0.04]
            
            if self.state_ticks >= int(self.turn_time_s * 10):
                self.get_logger().info("✅ 参数转向结束！准备收腿！")
                self.state = 'SHRINK_LEGS'
                self.state_ticks = 0
                
                return
                
            msg.vel_des = [0.0, 0.0, self.turn_vyaw]
            self.controller.send_cmd(msg)

        # ==========================================
        # 4. 收腿成刀片形态
        # ==========================================
        elif self.state == 'SHRINK_LEGS':
            if self.state_ticks == 1:
                self.set_dynamic_shape(target_height=0.25, leg_offset=0.04) 
            
            msg.mode = 11
            msg.gait_id = 0
            self.controller.send_cmd(msg)
            
            if self.state_ticks > 15: 
                self.get_logger().info("🔪 刀片形态就绪！向东进发！")
                self.state = 'EASTWARD_MARCH'
                self.state_ticks = 0

       # ==========================================
        # 5. 东进潜入 (北向纯定时横移 + 东向看球盲走定位)
        # ==========================================
        elif self.state == 'EASTWARD_MARCH':
            if self.state_ticks == 1:
                self.ball_lost_ticks = 0

            msg.mode = 11      
            msg.gait_id = 27   
            msg.step_height = [0.03, 0.03] 
            
            vyaw = 0.0 
            
            # ---------------------------------------------------------
            # 🟢 独立控制轴 A：北向横移 (vy) —— 纯靠绝对时间，与球无关！
            # ---------------------------------------------------------
            # 只要当前状态的总时间没达到您设定的横移时间，就一直往北挤
            if self.state_ticks < int(self.eastward_vy_time_s * 10):
                vy = self.eastward_vy
            else:
                if self.state_ticks == int(self.eastward_vy_time_s * 10):
                    self.get_logger().info("🛑 北向横移时间到！已贴紧北墙，停止横移！")
                vy = 0.0  # 时间一到，无论看不看得见球，北向移动绝对归零！
                
            # ---------------------------------------------------------
            # 🟢 独立控制轴 B：东向前进 (vx) —— 速度恒定，何时停车取决于球！
            # ---------------------------------------------------------
            vx = self.eastward_vx
            
            ball_visible = (self.ball_dist != -1.0)
            
            if ball_visible:
                # 能看见球，说明在东向上还没绕过去，盲走倒计时死死压在0
                self.ball_lost_ticks = 0
            else:
                # 球从视野边缘滑出消失（说明东向已经越过了球），开始累计东向的盲走时间！
                self.ball_lost_ticks += 1
            
            # 【东向的最终停止条件】：越过球之后，往东盲走的时间达到了您的设定值
            if self.ball_lost_ticks >= int(self.eastward_blind_after_lost_s * 10):
                self.get_logger().info(f"🏁 成功绕过球！东向盲走 {self.eastward_blind_after_lost_s}s 结束！完美抵达死角准备解围！")
                self.state = 'CLEAR_BALL_TURN' 
                self.state_ticks = 0
                return
                
            # 超时绝对保底（防止意外卡死）
            if self.state_ticks >= int(self.eastward_timeout_s * 10):
                self.get_logger().warn("⚠️ 东征潜入超时保底触发！强制进入解围！")
                self.state = 'CLEAR_BALL_TURN'
                self.state_ticks = 0
                return
            
            msg.vel_des = [vx, vy, vyaw] 
            self.controller.send_cmd(msg)

# ==========================================
        # 6. 解围第一步：轻微自转兜球
        # ==========================================
        elif self.state == 'CLEAR_BALL_TURN':
            msg.mode = 11
            msg.gait_id = 3
            msg.step_height = [0.04, 0.04]
            
            if self.state_ticks >= int(self.clear_turn_time_s * 10):
                self.get_logger().info("🔄 兜球自转结束！开始加速右侧螃蟹步解围！")
                self.state = 'BUFFER_CRAB'
                self.state_ticks = 0
                msg.mode = 11
                self.controller.send_cmd(msg)
                return
                
            msg.vel_des = [0.0, 0.0, self.clear_turn_vyaw]
            self.controller.send_cmd(msg)

        # ==========================================
        # 6.2. 插入新阶段：极慢速横移，腾出降底盘空间
        # ==========================================
        elif self.state == 'BUFFER_CRAB':
            msg.mode = 11
            msg.gait_id = 3
            msg.step_height = [0.03, 0.03]
            
            if self.state_ticks >= int(self.buffer_crab_time_s * 10):
                self.get_logger().info("🛡️ 缓冲空间已拉开，准备降低底盘贴地推球！")
                self.state = 'LOWER_BODY_FOR_CRAB'
                self.state_ticks = 0
                msg.mode = 11
                self.controller.send_cmd(msg)
                return
                
            # 注意：仅做 vy 的横移，vx 为 0
            msg.vel_des = [0.0, self.buffer_crab_vy, 0.0]
            self.controller.send_cmd(msg)

        # ==========================================
        # 6.5. 插入新阶段：极限压低底盘
        # ==========================================
        elif self.state == 'LOWER_BODY_FOR_CRAB':
            if self.state_ticks == 1:
                # 【核心黑魔法】：高度降到物理极限 0.14m，腿距保持刀片形态 0.0
                self.set_dynamic_shape(target_height=0.14, leg_offset=0.0) 
            
            msg.mode = 11  # 保持原地站立，等待身体降下去
            msg.gait_id = 0
            self.controller.send_cmd(msg)
            
            if self.state_ticks > 15: # 给物理引擎 1.5 秒的时间把身体压低
                self.get_logger().info("⬇️ 底盘已降至最低 0.14m！开启螃蟹步强力扫球！")
                self.state = 'CLEAR_BALL_CRAB'
                self.state_ticks = 0

        # ==========================================
        # 7. 解围第二步：加速右侧横移把球扫出来
        # ==========================================
        elif self.state == 'CLEAR_BALL_CRAB':
            msg.mode = 11
            msg.gait_id = 3  # 用普通 trot 步态横移，爆发力更强
            msg.step_height = [0.04, 0.04]
            
            if self.state_ticks >= int(self.clear_crab_time_s * 10):
                self.get_logger().info("🎉 解围完成！起立，准备找球！")
                self.state = 'RESTORE_POSTURE'    # <--- 修改这里
                self.state_ticks = 0
                msg.mode = 11
                self.controller.send_cmd(msg)
                return
                
            # 注意：向右横移，所以 vx=0，vy 为负数
            msg.vel_des = [0.0, self.clear_crab_vy, 0.0]
            self.controller.send_cmd(msg)

    
        # ==========================================
        # 8. 起立恢复正常姿态
        # ==========================================
        elif self.state == 'RESTORE_POSTURE':
            if self.state_ticks == 1:
                self.set_dynamic_shape(target_height=0.25, leg_offset=0.04) 
            
            msg.mode = 11
            msg.gait_id = 3
            msg.step_height = [0.04, 0.04]
            msg.vel_des = [0.0, 0.0, 0.0]
            self.controller.send_cmd(msg)
            
            if self.state_ticks > 25: 
                self.get_logger().info("🐕 姿态恢复完毕！开始开环定时转身面向西墙！")
                self.state = 'TURN_TO_WEST_WALL'  # <--- 修改这里
                self.state_ticks = 0
   # ==========================================
        # 9. 转向西墙 (纯调参开环)
        # ==========================================
        elif self.state == 'TURN_TO_WEST_WALL':
            msg.mode = 11
            msg.gait_id = 3
            msg.step_height = [0.04, 0.04]
            
            if self.state_ticks >= int(self.west_turn_time_s * 10):
                self.get_logger().info("✅ 转西墙结束！开始向前盲走靠近西墙！")
                self.state = 'WESTWARD_VISUAL_MARCH'  # <--- 修改这里：去盲走！
                self.state_ticks = 0
                return
                
            msg.vel_des = [0.0, 0.0, self.west_turn_vyaw]
            self.controller.send_cmd(msg)


        # ==========================================
        # 9.6. 视觉靠近西墙 (直到满足距离才停车！)
        # ==========================================
        elif self.state == 'WESTWARD_VISUAL_MARCH':
            msg.mode = 11
            msg.gait_id = 3 
            msg.step_height = [0.04, 0.04] 
            
            vx = self.west_visual_march_vx 
            vyaw = 0.0 
            
            if wall_visible:
                # 边走边粗调朝向
                vyaw = - (self.wall_angle_rad * 0.8) 
                vyaw = max(min(vyaw, 0.3), -0.3)
                
                # 👑 核心逻辑：离西墙达到指定距离才刹车！
                if 0.0 < self.wall_dist < self.west_stop_dist_m:
                    self.stable_counter += 1
                    if self.stable_counter >= 2:
                        self.get_logger().info(f"🎯 抵达西墙极近距离 ({self.wall_dist:.2f}m)！停车准备原地垂直校准。")
                        self.state = 'ALIGN_WEST_WALL'
                        self.state_ticks = 0
                        self.stable_counter = 0
                        msg.mode = 11 
                        msg.vel_des = [0.0, 0.0, 0.0]
                        self.controller.send_cmd(msg)
                        return
                else:
                    self.stable_counter = 0
                
            msg.vel_des = [vx, 0.0, vyaw]
            self.controller.send_cmd(msg)

        # ==========================================
        # 10. 原地垂直校准西墙
        # ==========================================
        elif self.state == 'ALIGN_WEST_WALL':
            msg.mode = 11
            msg.gait_id = 3
            msg.step_height = [0.04, 0.04]
            
            if wall_visible:
                angle_deg = math.degrees(self.wall_angle_rad)
                
                # 对齐了！进入下一步
                if abs(angle_deg) < self.west_align_tolerance_deg:
                    self.stable_counter += 1
                    if self.stable_counter >= 3:
                        self.get_logger().info(f"📐 西墙已绝对垂直 (偏角: {angle_deg:.1f}度)！开始转身面朝出口！")
                        self.state = 'TURN_TO_EXIT'
                        self.state_ticks = 0
                        self.stable_counter = 0
                        msg.mode = 11
                        self.controller.send_cmd(msg)
                        return
                else:
                    self.stable_counter = 0
                    
                # 原地强扭 (P控制)
                vyaw = - (self.wall_angle_rad * 1.5)
                vyaw = max(min(vyaw, 0.4), -0.4)
                msg.vel_des = [0.0, 0.0, vyaw]
            else:
                msg.vel_des = [0.0, 0.0, 0.0]
                
            # 超时保护(最多对齐4秒)
            if self.state_ticks > 35:
                self.get_logger().warn("⚠️ 西墙校准超时，强制进入转身出口阶段！")
                self.state = 'TURN_TO_EXIT'
                self.state_ticks = 0
                
            self.controller.send_cmd(msg)

        # ==========================================
        # 11. 转向出口 (纯调参开环)
        # ==========================================
        elif self.state == 'TURN_TO_EXIT':
            msg.mode = 11
            msg.gait_id = 3
            msg.step_height = [0.04, 0.04]
            
            if self.state_ticks >= int(self.exit_turn_time_s * 10):
                self.get_logger().info("✅ 面朝出口！准备降底盘！")
                self.state = 'LOWER_BODY_FINAL'
                self.state_ticks = 0
                msg.mode = 11
                self.controller.send_cmd(msg)
                return
                
            msg.vel_des = [0.0, 0.0, self.exit_turn_vyaw]
            self.controller.send_cmd(msg)

        # ==========================================
        # 12. 降底盘准备冲刺
        # ==========================================
        elif self.state == 'LOWER_BODY_FINAL':
            if self.state_ticks == 1:
                # 依然保持腿距 0.04，只降低高度，保证平稳行走
                self.set_dynamic_shape(target_height=0.14, leg_offset=0.04) 
            
            msg.mode = 11
            msg.gait_id = 1
            msg.vel_des = [0.0, 0.0, 0.0]
            self.controller.send_cmd(msg)
            
            if self.state_ticks > 15: # 1.5 秒丝滑降落
                self.get_logger().info("🚜 最终推土机形态就绪！开启三点一线推球冲刺！")
                # 👇 改这里：去执行三点一线推球！
                self.state = 'PUSH_TO_EXIT'
                self.state_ticks = 0

      # ==========================================
        # 12. 终极冲刺：纯视觉三点一线推球入洞
        # ==========================================
        elif self.state == 'PUSH_TO_EXIT':
            if self.state_ticks == 1:
                self.has_seen_exit = False  
                self.exit_lost_ticks = 0    
                
            msg.mode = 11
            msg.gait_id = 27  # 用 Trot_Slow 推球最稳
            msg.step_height = [0.02, 0.02] 
            
            exit_visible = (self.exit_offset_norm != -999.0)
            ball_visible = (self.ball_offset_x != -999.0 and self.ball_dist > 0.0)
            
            vx = self.push_vx 
            vy = 0.0
            vyaw = 0.0
            
            # 【绝技 1：只要看见紫线，就死磕到底绝不停车！】
            if exit_visible:
                self.has_seen_exit = True
                self.exit_lost_ticks = 0  
                # 视觉巡线：让狗的头永远对着缺口的中心
                vyaw = - (self.exit_offset_norm * self.push_vyaw_kp)
                vyaw = max(min(vyaw, 0.3), -0.3) 
                
                # 每秒打印一次状态，告诉您它还在死磕紫线
                if self.state_ticks % 10 == 0:
                    self.get_logger().info(f"⛳ 看到紫线！坚决推进中！vyaw: {vyaw:.2f}")
            else:
                # 【消失判定】：如果曾经看到过大门，现在紫线消失了，说明狗头已经穿过大门了！
                if self.has_seen_exit:
                    self.exit_lost_ticks += 1
                    self.get_logger().info(f"⛩️ 紫线消失！冲线判定中... ({self.exit_lost_ticks}/5)")
            
            # 门消失 0.5 秒后确认，瞬间切换到盲走进圈状态
            if self.exit_lost_ticks > 5:  
                self.get_logger().info(f"⛩️ 视野已彻底越过大门！开启盲走冲线，确保进圈！")
                self.state = 'CROSS_FINISH_LINE'
                self.state_ticks = 0
                return
            
            # 【绝技 2：看球侧滑】如果此时还能看见球，就用侧滑去包抄球！
            if ball_visible:  
                vy = - (self.ball_offset_x * self.push_vy_kp)
                vy = max(min(vy, 0.15), -0.15)   
                
            msg.vel_des = [vx, vy, vyaw]
            self.controller.send_cmd(msg)
            
            # 【超时保底】如果连紫线都没看到，就推满15秒强行结束
            if self.state_ticks > int(self.push_timeout_s * 10): 
                self.get_logger().info("⏱️ 推球超时！强行闭眼冲线！")
                self.state = 'CROSS_FINISH_LINE'
                self.state_ticks = 0

        # ==========================================
        # 14. 进圈冲刺：确保后腿完全越过终点线
        # ==========================================
        elif self.state == 'CROSS_FINISH_LINE':
            msg.mode = 11
            msg.gait_id = 27
            msg.step_height = [0.02, 0.02]
            
            # 闭着眼睛直直地往前推，不回头！
            msg.vel_des = [self.push_vx, 0.0, 0.0]
            self.controller.send_cmd(msg)
            
            if self.state_ticks >= int(self.cross_line_time_s * 10):
                self.get_logger().info("🎉 四腿已进圈！完美趴下！")
                self.state = 'MISSION_COMPLETE'
                self.state_ticks = 0

        # ==========================================
        # 15. 任务结束：受控趴下
        # ==========================================
        elif self.state == 'MISSION_COMPLETE':
            msg.mode = 7 
            msg.gait_id = 1 
            msg.vel_des = [0.0, 0.0, 0.0]
            self.controller.send_cmd(msg)

def main(args=None):
    rclpy.init(args=args)
    controller = LcmController()
    
    stand_msg = robot_control_cmd_lcmt()
    stand_msg.mode = 12 
    stand_msg.life_count = 1
    controller.send_cmd(stand_msg)
    time.sleep(2.0)
    
    node = SneakController(controller)
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        rest_msg = robot_control_cmd_lcmt()
        rest_msg.mode = 7 
        controller.send_cmd(rest_msg)
        time.sleep(0.5)
        controller.quit()
        try:
            node.destroy_vision_windows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
