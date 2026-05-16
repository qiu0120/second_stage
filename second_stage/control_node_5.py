import lcm
import sys
import os
import time
import math
from threading import Thread, Lock

from second_stage.robot_control_cmd_lcmt import robot_control_cmd_lcmt
from second_stage.robot_control_response_lcmt import robot_control_response_lcmt

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.parameter import Parameter
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException

from cyberdog_msg.msg import YamlParam, ApplyForce

from sensor_msgs.msg import Imu
# from rosgraph_msgs.msg import Clock



def main(args=None):
    Ctrl = Robot_Ctrl()
    Ctrl.run()
    msg = robot_control_cmd_lcmt()

    rclpy.init(args=args)
    node = Stage5Node(imu_topic = "/imu")

    try:
        values = [0.0] * 12
        values[0] = 0.0   # roll
        values[2] = 0.25  # height
        node.publish_yaml_vecxd("des_roll_pitch_height", values, is_user=1)

        if not startup_recovery_stand(Ctrl, msg, node):
            print("[startup] robot is not ready, stop before 上台阶")
            Ctrl.quit()
            return

        node.publish_yaml_vecxd("des_roll_pitch_height", values, is_user=1)
        wait_startup_settle(node, "startup settle before 上台阶")
        
        # 第五赛段上台阶步态
        msg.mode = 11
        msg.gait_id = 3
        msg.life_count += 1
        msg.vel_des = [0.6,0,0]
        msg.step_height = [0.2,0.2]
        msg.rpy_des = [0.0,0,0.0]
        Ctrl.Send_cmd(msg)
        ok = run_for_seconds(node, Ctrl, msg, 4.0, "上台阶", guard = False)
        if not ok:
            Ctrl.quit()
            return

        # 第五赛段上坡路段步态
        msg.mode = 11
        msg.gait_id = 3
        msg.life_count += 1
        msg.vel_des = [0.5,0,0]
        msg.step_height = [0.1,0.1]
        msg.rpy_des = [0.0,0.2,0.0]
        Ctrl.Send_cmd(msg)
        ok = run_until_distance(
            node,
            Ctrl,
            msg,
            target_dist=3.55,
            max_seconds=8.5,
            label="上坡路段",
            guard=False
        )
        if not ok:
            Ctrl.quit()
            return

        # 跳跃动作
        log_robot_state(node, "before jump gait_id=0", spin_count=1)
        msg.mode = 16
        msg.gait_id = 0
        msg.life_count += 1
        Ctrl.Send_cmd(msg)
        Ctrl.Wait_finish(16,0)
        log_robot_state(node, "after jump gait_id=0", spin_count=1)
        msg.mode = 12 # Recovery stand
        msg.gait_id = 0
        msg.life_count += 1 # Command will take effect when life_count update
        Ctrl.Send_cmd(msg)
        Ctrl.Wait_finish(12, 0)
        log_robot_state(node, "after recovery before right slope", spin_count=3)

        # 右斜坡路段步态
        values = [0.0] * 12
        values[0] = -0.6   # roll
        values[2] = 0.25  # height
        log_robot_state(node, "before publish roll=-0.6 height=0.25", spin_count=1)
        node.publish_yaml_vecxd("des_roll_pitch_height", values, is_user=1)
        log_robot_state(node, "after publish roll=-0.6 height=0.25", spin_count=1)
        
        msg.mode = 11
        msg.gait_id = 3
        msg.life_count += 1
        msg.vel_des = [0.3,0.135,0.0]
        msg.step_height = [0.04,0.04]
        msg.rpy_des = [0.0,0.0,0.0]
        msg.pos_des = [0.0,0,0.25]
        log_robot_state(node, "before right slope gait command", spin_count=1)
        Ctrl.Send_cmd(msg)
        ok = run_for_seconds(node, Ctrl, msg, 27.0, "右斜坡一", guard = False)
        if not ok:
            Ctrl.quit()
            return

        # 转弯部分
        msg.mode = 16
        msg.gait_id = 3
        msg.life_count += 1
        Ctrl.Send_cmd(msg)
        Ctrl.Wait_finish(16,3)
        msg.mode = 12 # Recovery stand
        msg.gait_id = 0
        msg.life_count += 1 # Command will take effect when life_count update
        Ctrl.Send_cmd(msg)
        Ctrl.Wait_finish(12, 0)

        # 重复一次
        msg.mode = 11
        msg.gait_id = 3
        msg.life_count += 1
        msg.vel_des = [0.3,0.13,0.0]
        msg.step_height = [0.04,0.04]
        msg.rpy_des = [0.0,0.0,0.0]
        msg.pos_des = [0.0,0,0.25]
        Ctrl.Send_cmd(msg)
        ok = run_for_seconds(node, Ctrl, msg, 31.0, "右斜坡二", guard = False)
        if not ok:
            Ctrl.quit()
            return
        
        # 转弯部分
        msg.mode = 16
        msg.gait_id = 3
        msg.life_count += 1
        Ctrl.Send_cmd(msg)
        Ctrl.Wait_finish(16,3)
        msg.mode = 12 # Recovery stand
        msg.gait_id = 0
        msg.life_count += 1 # Command will take effect when life_count update
        Ctrl.Send_cmd(msg)
        Ctrl.Wait_finish(12, 0)

        # 重复两次
        msg.mode = 11
        msg.gait_id = 3
        msg.life_count += 1
        msg.vel_des = [0.3,0.14,0.0]
        msg.step_height = [0.04,0.04]
        msg.rpy_des = [0.0,0.0,0.0]
        msg.pos_des = [0.0,0,0.25]
        Ctrl.Send_cmd(msg)
        ok = run_for_seconds(node, Ctrl, msg, 35.0, "右斜坡三", guard = False)
        if not ok:
            Ctrl.quit()
            return 

        # 平动离开坡度区
        msg.mode = 11
        msg.gait_id = 3
        msg.life_count += 1
        msg.vel_des = [0,-0.3,0]
        msg.step_height = [0.05,0.05]
        msg.rpy_des = [0.0,0.0,0.0]
        Ctrl.Send_cmd(msg)
        ok = run_for_seconds(node, Ctrl, msg, 5.0, "平动离开坡度区", guard = False)
        if not ok:
            Ctrl.quit()
            return 

        # 跳跃动作
        msg.mode = 16
        msg.gait_id = 3
        msg.life_count += 1
        Ctrl.Send_cmd(msg)
        Ctrl.Wait_finish(16,3)
        msg.mode = 12 # Recovery stand
        msg.gait_id = 0
        msg.life_count += 1 # Command will take effect when life_count update
        Ctrl.Send_cmd(msg)
        Ctrl.Wait_finish(12, 0)

        values = [0.0] * 12
        values[0] = 0.0   # roll
        values[2] = 0.25  # height
        node.publish_yaml_vecxd("des_roll_pitch_height", values, is_user=1)

        # 结尾平路
        msg.mode = 11
        msg.gait_id = 3
        msg.life_count += 1
        msg.vel_des = [0.5,0,0]
        msg.step_height = [0.05,0.05]
        msg.rpy_des = [0.0,0.0,0.0]
        Ctrl.Send_cmd(msg)
        ok = run_for_seconds(node, Ctrl, msg, 8.0, "结尾平路", guard = False)
        if not ok:
            Ctrl.quit()
            return

        # 跳远
        msg.mode = 16
        msg.gait_id = 1
        msg.life_count += 1
        Ctrl.Send_cmd(msg)
        Ctrl.Wait_finish(16,1)

        msg.mode = 12 # Recovery stand
        msg.gait_id = 0
        msg.life_count += 1 # Command will take effect when life_count update
        Ctrl.Send_cmd(msg)
        Ctrl.Wait_finish(12, 0)

    except KeyboardInterrupt:
        pass
    Ctrl.quit()
    sys.exit()


class Robot_Ctrl(object):
    def __init__(self):
        self.rec_thread = Thread(target=self.rec_responce)
        self.send_thread = Thread(target=self.send_publish)
        self.lc_r = lcm.LCM("udpm://239.255.76.67:7670?ttl=255")
        self.lc_s = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
        self.cmd_msg = robot_control_cmd_lcmt()
        self.rec_msg = robot_control_response_lcmt()
        self.send_lock = Lock()
        self.delay_cnt = 0
        self.mode_ok = 0
        self.gait_ok = 0
        self.runing = 1

    def run(self):
        self.lc_r.subscribe("robot_control_response", self.msg_handler)
        self.send_thread.start()
        self.rec_thread.start()

    def msg_handler(self, channel, data):
        self.rec_msg = robot_control_response_lcmt().decode(data)
        if(self.rec_msg.order_process_bar >= 95):
            self.mode_ok = self.rec_msg.mode
            self.gait_ok = self.rec_msg.gait_id
        else:
            self.mode_ok = 0          
            self.gait_ok = 0

    def rec_responce(self):
        while self.runing:
            self.lc_r.handle()
            time.sleep( 0.002 )

    def Wait_finish(self, mode, gait_id):
        count = 0
        while self.runing and count < 2000: #10s
            if self.mode_ok == mode and self.gait_ok == gait_id:
                return True
            else:
                time.sleep(0.005)
                count += 1

    def send_publish(self):
        while self.runing:
            self.send_lock.acquire()
            if self.delay_cnt > 20: # Heartbeat signal 10HZ, It is used to maintain the heartbeat when life count is not updated
                self.lc_s.publish("robot_control_cmd",self.cmd_msg.encode())
                self.delay_cnt = 0
            self.delay_cnt += 1
            self.send_lock.release()
            time.sleep( 0.005 )

    def Send_cmd(self, msg):
        self.send_lock.acquire()
        self.delay_cnt = 50
        self.cmd_msg = msg
        self.send_lock.release()

    def quit(self):
        self.runing = 0
        self.rec_thread.join()
        self.send_thread.join()

class ControlParameterValueKind:
    kDOUBLE = 1
    kS64 = 2
    kVEC_X_DOUBLE = 3
    kMAT_X_DOUBLE = 4

class yaml_pub(Node):
    def __init__(self):
        super().__init__("cyberdogmsg_node")

        self.para_pub = self.create_publisher(YamlParam, "yaml_parameter", 10)
        self.force_pub = self.create_publisher(ApplyForce, "apply_force", 10)

    def publish_yaml_kDOUBLE(self, name: str, value: float, is_user: int = 0):
        msg = YamlParam()
        msg.name = name
        msg.kind = ControlParameterValueKind.kDOUBLE
        msg.s64_value = float(value)
        msg.is_user = int(is_user)
        self.para_pub.publish(msg)

    def publish_yaml_s64(self, name: str, value: int, is_user: int = 0):
        msg = YamlParam()
        msg.name = name
        msg.kind = ControlParameterValueKind.kS64
        msg.s64_value = int(value)
        msg.is_user = int(is_user)
        self.para_pub.publish(msg)

    def publish_yaml_vecxd(self, name: str, values, is_user: int = 1):
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

    def publish_apply_force(self, link_name: str, rel_pos, force, duration: float):
        msg = ApplyForce()
        msg.link_name = link_name
        msg.rel_pos = [float(x) for x in rel_pos]
        msg.force = [float(x) for x in force]
        msg.time = float(duration)
        self.force_pub.publish(msg)
    def publish_apply_force(self, link_name: str, rel_pos, force, duration: float):
        msg = ApplyForce()
        msg.link_name = link_name
        msg.rel_pos = [float(x) for x in rel_pos]
        msg.force = [float(x) for x in force]
        msg.time = float(duration)
        self.force_pub.publish(msg)

def quat_to_euler(x, y, z, w):
    """
    四元数转 roll, pitch, yaw
    """
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class Stage5Node(yaml_pub):
    def __init__(self, imu_topic="/imu"):
        super().__init__()

        # 使用仿真时间。后面 run_for_seconds 会用 node.get_clock()。
        try:
            self.declare_parameter("use_sim_time", True)
        except Exception:
            pass

        try:
            self.set_parameters([
                Parameter("use_sim_time", Parameter.Type.BOOL, True)
            ])
        except Exception as e:
            self.get_logger().warn(f"set use_sim_time failed: {e}")

        self.imu_ready = False
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.wx = 0.0
        self.wy = 0.0
        self.wz = 0.0

        self.imu_sub = self.create_subscription(
            Imu,
            imu_topic,
            self.imu_callback,
            qos_profile_sensor_data
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pose_ready = False
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.seg_start_x = 0.0
        self.seg_start_y = 0.0
        self.seg_start_ready = False
        self.last_tf_error = ""

        self.get_logger().info(f"Stage5Node started, subscribing IMU topic: {imu_topic}")

    def imu_callback(self, msg):
        q = msg.orientation
        self.roll, self.pitch, self.yaw = quat_to_euler(q.x, q.y, q.z, q.w)

        self.wx = msg.angular_velocity.x
        self.wy = msg.angular_velocity.y
        self.wz = msg.angular_velocity.z

        self.imu_ready = True

    def update_pose(self):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                "vodom",
                "base_link",
                Time(),
                timeout=Duration(seconds=0.01)
            )
        except TransformException as e:
            self.pose_ready = False
            self.last_tf_error = str(e)
            return False

        t = tf_msg.transform.translation
        self.x = t.x
        self.y = t.y
        self.z = t.z
        self.pose_ready = True
        self.last_tf_error = ""
        return True

    def mark_segment_start(self):
        self.seg_start_x = self.x
        self.seg_start_y = self.y
        self.seg_start_ready = True

    def segment_distance(self):
        if not self.seg_start_ready:
            return 0.0
        return math.hypot(self.x - self.seg_start_x, self.y - self.seg_start_y)

def now_sec(node):
    return node.get_clock().now().nanoseconds / 1e9


def log_robot_state(node, label, spin_count=1):
    for _ in range(spin_count):
        if not rclpy.ok():
            break
        rclpy.spin_once(node, timeout_sec=0.01)
        node.update_pose()

    if node.pose_ready:
        pose_text = f"x={node.x:.3f}, y={node.y:.3f}, z={node.z:.3f}"
    else:
        pose_text = f"pose not ready, tf_error={node.last_tf_error}"

    if node.imu_ready:
        imu_text = (
            f"roll={node.roll:.3f}, pitch={node.pitch:.3f}, yaw={node.yaw:.3f}, "
            f"wx={node.wx:.3f}, wy={node.wy:.3f}, wz={node.wz:.3f}"
        )
    else:
        imu_text = "waiting imu"

    print(f"[DEBUG] {label}: sim_t={now_sec(node):.2f}s {pose_text}, {imu_text}")


def wait_robot_ready(
    node,
    label,
    timeout_sec=12.0,
    min_z=0.20,
    max_abs_roll=0.35,
    max_abs_pitch=0.45,
    stable_count_required=5,
    allow_tf_only=True,
    tf_only_min_z=0.23,
    tf_only_stable_count_required=25,
    print_interval=0.5
):
    start_wall = time.monotonic()
    last_print = -999.0
    stable_count = 0

    while rclpy.ok():
        elapsed = time.monotonic() - start_wall
        if elapsed >= timeout_sec:
            break

        rclpy.spin_once(node, timeout_sec=0.02)
        node.update_pose()

        pose_ok = node.pose_ready and node.z >= min_z
        imu_ok = (
            node.imu_ready
            and abs(node.roll) <= max_abs_roll
            and abs(node.pitch) <= max_abs_pitch
        )
        if node.imu_ready:
            ready_ok = pose_ok and imu_ok
            required_count = stable_count_required
        else:
            ready_ok = allow_tf_only and node.pose_ready and node.z >= tf_only_min_z
            required_count = tf_only_stable_count_required

        if ready_ok:
            stable_count += 1
        else:
            stable_count = 0

        if elapsed - last_print >= print_interval:
            if node.pose_ready:
                pose_text = f"x={node.x:.3f}, y={node.y:.3f}, z={node.z:.3f}"
            else:
                pose_text = f"pose not ready, tf_error={node.last_tf_error}"

            if node.imu_ready:
                imu_text = (
                    f"roll={node.roll:.3f}, pitch={node.pitch:.3f}, yaw={node.yaw:.3f}"
                )
            else:
                imu_text = "waiting imu"

            print(
                f"[{label}] readiness t={elapsed:.2f}s {pose_text}, {imu_text}, "
                f"stable={stable_count}/{required_count}"
            )
            last_print = elapsed

        if stable_count >= required_count:
            if node.imu_ready:
                print(
                    f"[{label}] ready: z={node.z:.3f}, "
                    f"roll={node.roll:.3f}, pitch={node.pitch:.3f}"
                )
            else:
                print(
                    f"[{label}] ready with TF only: z={node.z:.3f}, "
                    "IMU not ready"
                )
            return True

    if node.pose_ready:
        pose_text = f"x={node.x:.3f}, y={node.y:.3f}, z={node.z:.3f}"
    else:
        pose_text = f"pose not ready, tf_error={node.last_tf_error}"

    if node.imu_ready:
        imu_text = f"roll={node.roll:.3f}, pitch={node.pitch:.3f}, yaw={node.yaw:.3f}"
    else:
        imu_text = "waiting imu"

    print(f"[{label}] readiness timeout: {pose_text}, {imu_text}")
    return False


def startup_recovery_stand(Ctrl, msg, node, attempts=3, attempt_timeout_sec=8.0):
    for attempt in range(1, attempts + 1):
        msg.mode = 12
        msg.gait_id = 0
        msg.life_count += 1
        Ctrl.Send_cmd(msg)

        print(
            f"[startup] send recovery stand attempt {attempt}/{attempts}, "
            f"life_count={msg.life_count}"
        )

        if wait_robot_ready(
            node,
            f"startup recovery stand attempt {attempt}",
            timeout_sec=attempt_timeout_sec
        ):
            print(
                f"[startup] recovery stand ready on attempt {attempt}, "
                f"mode_ok={Ctrl.mode_ok}, gait_ok={Ctrl.gait_ok}"
            )
            return True

        print(
            f"[startup] recovery stand attempt {attempt} not ready, "
            f"mode_ok={Ctrl.mode_ok}, gait_ok={Ctrl.gait_ok}"
        )

    return False


def wait_startup_settle(node, label, seconds=0.6, print_interval=0.3):
    start = time.monotonic()
    last_print = -999.0

    while rclpy.ok():
        elapsed = time.monotonic() - start
        if elapsed >= seconds:
            break

        rclpy.spin_once(node, timeout_sec=0.02)
        node.update_pose()

        if elapsed - last_print >= print_interval:
            if node.pose_ready:
                pose_text = f"x={node.x:.3f}, y={node.y:.3f}, z={node.z:.3f}"
            else:
                pose_text = f"pose not ready, tf_error={node.last_tf_error}"

            if node.imu_ready:
                imu_text = (
                    f"roll={node.roll:.3f}, pitch={node.pitch:.3f}, yaw={node.yaw:.3f}"
                )
            else:
                imu_text = "waiting imu"

            print(f"[{label}] t={elapsed:.2f}s {pose_text}, {imu_text}")
            last_print = elapsed


def send_slow_stop(Ctrl, msg):
    """
    遇到姿态异常时，先让速度变成 0。
    这里仍然保持 mode=11，不直接 recovery stand，避免桥上突然大动作。
    """
    msg.vel_des = [0.0, 0.0, 0.0]
    msg.step_height = [0.04, 0.04]
    msg.life_count += 1
    Ctrl.Send_cmd(msg)


def run_for_seconds(
    node,
    Ctrl,
    msg,
    seconds,
    label="",
    print_interval=0.5,
    guard=False,
    max_abs_roll=0.55,
    max_abs_pitch=0.65,
    max_abs_wx=3.0,
    max_abs_wy=3.0
):
    """
    用仿真时间跑 seconds 秒，同时读取 IMU 和 TF 位姿。
    guard 默认关闭；打开后遇到姿态明显异常才发 0 速度，返回 False。
    """
    start = now_sec(node)
    last_print = -999.0
    node.seg_start_ready = False

    wait_count = 0
    while rclpy.ok() and start <= 0.0 and wait_count < 200:
        rclpy.spin_once(node, timeout_sec=0.01)
        node.update_pose()
        start = now_sec(node)
        wait_count += 1

    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.01)
        node.update_pose()
        if node.pose_ready:
            break

    if node.pose_ready:
        node.mark_segment_start()
        print(
            f"[{label}] segment start: "
            f"x={node.x:.3f}, y={node.y:.3f}, z={node.z:.3f}"
        )
    else:
        print(
            f"[{label}] segment start: pose not ready, "
            f"tf_error={node.last_tf_error}"
        )

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.01)
        node.update_pose()

        if node.pose_ready and not node.seg_start_ready:
            node.mark_segment_start()
            print(
                f"[{label}] segment start recovered: "
                f"x={node.x:.3f}, y={node.y:.3f}, z={node.z:.3f}"
            )

        elapsed = now_sec(node) - start

        if elapsed >= seconds:
            if node.pose_ready:
                print(
                    f"[{label}] finished, elapsed={elapsed:.2f}s "
                    f"dist={node.segment_distance():.3f}, "
                    f"x={node.x:.3f}, y={node.y:.3f}, z={node.z:.3f}"
                )
            else:
                print(
                    f"[{label}] finished, elapsed={elapsed:.2f}s "
                    f"pose not ready, tf_error={node.last_tf_error}"
                )
            return True

        if node.imu_ready and guard:
            bad_roll = abs(node.roll) > max_abs_roll
            bad_pitch = abs(node.pitch) > max_abs_pitch
            bad_w = abs(node.wx) > max_abs_wx or abs(node.wy) > max_abs_wy

            if bad_roll or bad_pitch or bad_w:
                print(
                    f"[{label}] IMU GUARD TRIGGERED: "
                    f"roll={node.roll:.3f}, pitch={node.pitch:.3f}, "
                    f"wx={node.wx:.3f}, wy={node.wy:.3f}"
                )
                send_slow_stop(Ctrl, msg)
                return False

        if elapsed - last_print >= print_interval:
            if node.pose_ready:
                pose_text = (
                    f"x={node.x:.3f}, y={node.y:.3f}, z={node.z:.3f}, "
                    f"dist={node.segment_distance():.3f}"
                )
            else:
                pose_text = f"pose not ready, tf_error={node.last_tf_error}"

            if node.imu_ready:
                print(
                    f"[{label}] t={elapsed:.2f}s "
                    f"{pose_text}, "
                    f"roll={node.roll:.3f}, pitch={node.pitch:.3f}, yaw={node.yaw:.3f}, "
                    f"wx={node.wx:.3f}, wy={node.wy:.3f}, wz={node.wz:.3f}"
                )
            else:
                print(f"[{label}] t={elapsed:.2f}s {pose_text}, waiting imu...")

            last_print = elapsed


def run_until_distance(
    node,
    Ctrl,
    msg,
    target_dist,
    max_seconds,
    label="",
    print_interval=0.5,
    guard=False,
    max_abs_roll=0.55,
    max_abs_pitch=0.65,
    max_abs_wx=3.0,
    max_abs_wy=3.0
):
    """
    以 TF 段内位移为主触发切换，仿真时间只做兜底。
    guard 默认关闭；当前阶段只记录，不主动干预步态。
    """
    start = now_sec(node)
    last_print = -999.0
    node.seg_start_ready = False

    wait_count = 0
    while rclpy.ok() and start <= 0.0 and wait_count < 200:
        rclpy.spin_once(node, timeout_sec=0.01)
        node.update_pose()
        start = now_sec(node)
        wait_count += 1

    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.01)
        node.update_pose()
        if node.pose_ready:
            break

    if node.pose_ready:
        node.mark_segment_start()
        print(
            f"[{label}] distance segment start: "
            f"x={node.x:.3f}, y={node.y:.3f}, z={node.z:.3f}, "
            f"target_dist={target_dist:.3f}, max_seconds={max_seconds:.2f}"
        )
    else:
        print(
            f"[{label}] distance segment start: pose not ready, "
            f"target_dist={target_dist:.3f}, max_seconds={max_seconds:.2f}, "
            f"tf_error={node.last_tf_error}"
        )

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.01)
        node.update_pose()

        if node.pose_ready and not node.seg_start_ready:
            node.mark_segment_start()
            print(
                f"[{label}] distance segment start recovered: "
                f"x={node.x:.3f}, y={node.y:.3f}, z={node.z:.3f}"
            )

        elapsed = now_sec(node) - start
        dist = node.segment_distance() if node.pose_ready else 0.0

        if node.pose_ready and node.seg_start_ready and dist >= target_dist:
            print(
                f"[{label}] distance_trigger, elapsed={elapsed:.2f}s "
                f"dist={dist:.3f}, target={target_dist:.3f}, "
                f"x={node.x:.3f}, y={node.y:.3f}, z={node.z:.3f}, "
                f"roll={node.roll:.3f}, pitch={node.pitch:.3f}, yaw={node.yaw:.3f}"
            )
            return True

        if elapsed >= max_seconds:
            if node.pose_ready:
                print(
                    f"[{label}] timeout_fallback, elapsed={elapsed:.2f}s "
                    f"dist={dist:.3f}, target={target_dist:.3f}, "
                    f"x={node.x:.3f}, y={node.y:.3f}, z={node.z:.3f}, "
                    f"roll={node.roll:.3f}, pitch={node.pitch:.3f}, yaw={node.yaw:.3f}"
                )
            else:
                print(
                    f"[{label}] timeout_fallback, elapsed={elapsed:.2f}s "
                    f"pose not ready, target={target_dist:.3f}, "
                    f"tf_error={node.last_tf_error}"
                )
            return True

        if node.imu_ready and guard:
            bad_roll = abs(node.roll) > max_abs_roll
            bad_pitch = abs(node.pitch) > max_abs_pitch
            bad_w = abs(node.wx) > max_abs_wx or abs(node.wy) > max_abs_wy

            if bad_roll or bad_pitch or bad_w:
                print(
                    f"[{label}] IMU GUARD TRIGGERED: "
                    f"roll={node.roll:.3f}, pitch={node.pitch:.3f}, "
                    f"wx={node.wx:.3f}, wy={node.wy:.3f}"
                )
                send_slow_stop(Ctrl, msg)
                return False

        if elapsed - last_print >= print_interval:
            if node.pose_ready:
                pose_text = (
                    f"x={node.x:.3f}, y={node.y:.3f}, z={node.z:.3f}, "
                    f"dist={dist:.3f}, target={target_dist:.3f}"
                )
            else:
                pose_text = f"pose not ready, target={target_dist:.3f}, tf_error={node.last_tf_error}"

            if node.imu_ready:
                print(
                    f"[{label}] t={elapsed:.2f}s "
                    f"{pose_text}, "
                    f"roll={node.roll:.3f}, pitch={node.pitch:.3f}, yaw={node.yaw:.3f}, "
                    f"wx={node.wx:.3f}, wy={node.wy:.3f}, wz={node.wz:.3f}"
                )
            else:
                print(f"[{label}] t={elapsed:.2f}s {pose_text}, waiting imu...")

            last_print = elapsed

# Main function
if __name__ == '__main__':
    main()
