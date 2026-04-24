import lcm
import sys
import os
import time
from threading import Thread, Lock

from second_stage.robot_control_cmd_lcmt import robot_control_cmd_lcmt
from second_stage.robot_control_response_lcmt import robot_control_response_lcmt

import rclpy
from rclpy.node import Node
from cyberdog_msg.msg import YamlParam, ApplyForce

def main(args=None):
    Ctrl = Robot_Ctrl()
    Ctrl.run()
    msg = robot_control_cmd_lcmt()

    rclpy.init(args=args)
    node = yaml_pub()

    try:
        msg.mode = 12 # Recovery stand
        msg.gait_id = 0
        msg.life_count += 1 # Command will take effect when life_count update
        Ctrl.Send_cmd(msg)
        Ctrl.Wait_finish(12, 0)

        # values = [0.0] * 12
        # values[0] = 0.0   # roll
        # values[2] = 0.25  # height
        # node.publish_yaml_vecxd("des_roll_pitch_height", values, is_user=1)
        
        # # 第五赛段上台阶步态
        # msg.mode = 11
        # msg.gait_id = 3
        # msg.life_count += 1
        # msg.vel_des = [0.6,0,0]
        # msg.step_height = [0.2,0.2]
        # msg.rpy_des = [0.0,0,0.0]
        # Ctrl.Send_cmd(msg)
        # time.sleep(5)

        # # 第五赛段上坡路段步态
        # msg.mode = 11
        # msg.gait_id = 3
        # msg.life_count += 1
        # msg.vel_des = [0.5,0,0]
        # msg.step_height = [0.1,0.1]
        # msg.rpy_des = [0.0,0.2,0.0]
        # Ctrl.Send_cmd(msg)
        # time.sleep(27)

        # # 跳跃动作
        # msg.mode = 16
        # msg.gait_id = 0
        # msg.life_count += 1
        # Ctrl.Send_cmd(msg)
        # Ctrl.Wait_finish(16,0)
        # msg.mode = 12 # Recovery stand
        # msg.gait_id = 0
        # msg.life_count += 1 # Command will take effect when life_count update
        # Ctrl.Send_cmd(msg)
        # Ctrl.Wait_finish(12, 0)

        # # 右斜坡路段步态
        # values = [0.0] * 12
        # values[0] = -0.6   # roll
        # values[2] = 0.25  # height
        # node.publish_yaml_vecxd("des_roll_pitch_height", values, is_user=1)
        
        # msg.mode = 11
        # msg.gait_id = 3
        # msg.life_count += 1
        # msg.vel_des = [0.3,0.135,0.0]
        # msg.step_height = [0.04,0.04]
        # msg.rpy_des = [0.0,0.0,0.0]
        # msg.pos_des = [0.0,0,0.25]
        # Ctrl.Send_cmd(msg)
        # time.sleep(36)

        # # 转弯部分
        # msg.mode = 16
        # msg.gait_id = 3
        # msg.life_count += 1
        # Ctrl.Send_cmd(msg)
        # Ctrl.Wait_finish(16,3)
        # msg.mode = 12 # Recovery stand
        # msg.gait_id = 0
        # msg.life_count += 1 # Command will take effect when life_count update
        # Ctrl.Send_cmd(msg)
        # Ctrl.Wait_finish(12, 0)

        # # 重复一次
        # msg.mode = 11
        # msg.gait_id = 3
        # msg.life_count += 1
        # msg.vel_des = [0.3,0.13,0.0]
        # msg.step_height = [0.04,0.04]
        # msg.rpy_des = [0.0,0.0,0.0]
        # msg.pos_des = [0.0,0,0.25]
        # Ctrl.Send_cmd(msg)
        # time.sleep(31)

        # # 转弯部分
        # msg.mode = 16
        # msg.gait_id = 3
        # msg.life_count += 1
        # Ctrl.Send_cmd(msg)
        # Ctrl.Wait_finish(16,3)
        # msg.mode = 12 # Recovery stand
        # msg.gait_id = 0
        # msg.life_count += 1 # Command will take effect when life_count update
        # Ctrl.Send_cmd(msg)
        # Ctrl.Wait_finish(12, 0)

        # # 重复两次
        # msg.mode = 11
        # msg.gait_id = 3
        # msg.life_count += 1
        # msg.vel_des = [0.3,0.14,0.0]
        # msg.step_height = [0.04,0.04]
        # msg.rpy_des = [0.0,0.0,0.0]
        # msg.pos_des = [0.0,0,0.25]
        # Ctrl.Send_cmd(msg)
        # time.sleep(35)

        # # 平动离开坡度区
        # msg.mode = 11
        # msg.gait_id = 3
        # msg.life_count += 1
        # msg.vel_des = [0,-0.3,0]
        # msg.step_height = [0.05,0.05]
        # msg.rpy_des = [0.0,0.0,0.0]
        # Ctrl.Send_cmd(msg)
        # time.sleep(5)

        # # 跳跃动作
        # msg.mode = 16
        # msg.gait_id = 3
        # msg.life_count += 1
        # Ctrl.Send_cmd(msg)
        # Ctrl.Wait_finish(16,3)
        # msg.mode = 12 # Recovery stand
        # msg.gait_id = 0
        # msg.life_count += 1 # Command will take effect when life_count update
        # Ctrl.Send_cmd(msg)
        # Ctrl.Wait_finish(12, 0)

        # values = [0.0] * 12
        # values[0] = 0.0   # roll
        # values[2] = 0.25  # height
        # node.publish_yaml_vecxd("des_roll_pitch_height", values, is_user=1)

        # # 结尾平路
        # msg.mode = 11
        # msg.gait_id = 3
        # msg.life_count += 1
        # msg.vel_des = [0.5,0,0]
        # msg.step_height = [0.05,0.05]
        # msg.rpy_des = [0.0,0.0,0.0]
        # Ctrl.Send_cmd(msg)
        # time.sleep(8)

        # # 跳远
        # msg.mode = 16
        # msg.gait_id = 1
        # msg.life_count += 1
        # Ctrl.Send_cmd(msg)
        # Ctrl.Wait_finish(16,1)

        # msg.mode = 12 # Recovery stand
        # msg.gait_id = 0
        # msg.life_count += 1 # Command will take effect when life_count update
        # Ctrl.Send_cmd(msg)
        # Ctrl.Wait_finish(12, 0)

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


# Main function
if __name__ == '__main__':
    main()