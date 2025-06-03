#!/usr/bin/env python3
import rclpy
from rclpy.time import Time
from rclpy.node import Node
from geometry_msgs.msg import Twist
import torch
# from skrl.agents.torch.ppo import PPO
import sys
sys.path.insert(0,'/home/jetson/osr_inference_ws/src/isaac_rover_physical_2.0/src/controller/controller')

from utils.loadpolicy import student_loader

from rover_msgs.msg import Exteroception, Proprioception

#Joystick is the publisher (sends velocities)Motor node is the subscriber.

#When the joystick has 0 lin_vel and 0 ang_vel the robot will turn the wheels to the initial position

class Physical_osr(Node):

    def __init__(self):
        super().__init__('phy_osr_node')
        self.get_logger().info('Physical_osr node ready!')
        
        self.subscription = self.create_subscription(Exteroception, '/exteroception', self.exteroception_cb,1000)
        self.subscription = self.create_subscription(Proprioception, '/proprioception', self.proprioception_cb,1000)
    
        # TODO : 변경해야함
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        frequency = 60
        timer_period = 1/frequency  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)   # 60Hz 주기로 실행
        self.goal_dist = 1
        self.goal_heading = 0.0
        self.sparse = [0.0] * 441
        self.dense = [0.0] * 676
        self.prev_actions = [0.0, 0.0]
        self.ex_last_time = Time()
        self.pr_last_time = Time()
        
        info = {
            "reset": 0,
            "actions": 2,
            "proprioceptive": 4,
            "sparse": 441,
            "dense": 676}
        
        self.student = student_loader(info, "model1")
        # self.teacher = teacher_loader(info, "model1")

    def timer_callback(self):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        # Concantenate observation
        combine = self.prev_actions + [self.goal_dist,self.goal_heading] + self.sparse + self.dense
        obs = torch.tensor(combine).unsqueeze(0)
        obs = obs.to('cuda:0')
        
        # Model inference
        actions = self.student.act(obs).squeeze()
        self.prev_actions = actions.tolist()
        self.get_logger().info(str(self.prev_actions))
        self.get_logger().info("Inference mode")
        
        # Set message to motors
        msg = Twist()
        msg.linear.x = actions[0].item()
        msg.angular.z = actions[1].item()
                    # If distance is below threshold, shift to manual mode.
        if self.goal_dist <= 0.5:
            msg_zero = Twist()
            msg_zero.linear.x = 0.0
            msg_zero.angular.z = 0.0
            self.publisher_.publish(msg_zero)
            self.get_logger().info('Arrived!!')
        
        # Publish message to motors
        self.publisher_.publish(msg)
    
    def exteroception_cb(self, msg):
        # Calculate frequency
        self.ex_first_time = Time()
        self.ex_time_diff = self.ex_first_time - self.ex_last_time
        self.ex_last_time = self.ex_first_time

        # Exteroception processing
        self.sparse = list(msg.sparse)
        self.dense = list(msg.dense)
        self.get_logger().info(str(self.ex_time_diff))

    def proprioception_cb(self, msg):
        # Calculate frequency
        self.pr_first_time = Time()
        self.pr_time_diff = self.pr_first_time - self.pr_last_time
        self.pr_last_time = self.pr_first_time

        # Proprioception processing
        self.goal_dist = msg.distance
        self.goal_heading = msg.heading
        self.get_logger().info(str(self.pr_time_diff))
        pass


def main(args=None):
    rclpy.init(args=args)
    
    phy_osr = Physical_osr()
    rclpy.spin(phy_osr)

    phy_osr.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()