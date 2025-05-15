#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from stable_baselines3 import TD3, PPO, SAC
import tf.transformations as tf_trans
from env import RobotNavEnv
import math
import argparse
# Global variables for goal tracking
goal_received = False
current_goal = None

def goal_callback(msg):
    """
    Callback function for processing RViz goal poses.
    
    Args:
        msg (PoseStamped): Goal pose message from RViz
    """
    global goal_received, current_goal
    if msg.header.frame_id == "origin":
        x = msg.pose.position.x
        y = msg.pose.position.y

        # Extract yaw from quaternion
        orientation_q = msg.pose.orientation
        quaternion = [
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w,
        ]
        _, _, yaw = tf_trans.euler_from_quaternion(quaternion)

        current_goal = [x, y, yaw]
        rospy.loginfo(f"Received valid goal: x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.2f}°")
        goal_received = True
    else:
        rospy.logwarn(f"Ignoring goal from frame '{msg.header.frame_id}' (expecting 'origin')")

def run_rl(eval_env, goal):
    """
    Execute the trained policy to reach the specified goal.
    
    Args:
        eval_env (RobotNavEnv): Environment instance
        goal (list): [x, y, yaw] target pose
    """
    global model
    obs = eval_env.reset(goal)
    done = False
    total_reward = 0

    while not done and not rospy.is_shutdown():
        # Get action from trained policy
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        total_reward += reward

    rospy.loginfo(f"Task finished. Total reward: {total_reward}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RL navigation with ROS')
    parser.add_argument('--model-path', type=str, default="../../models/td3.zip",
                       help='Path to the trained model')
    args = parser.parse_args()

    # Initialize ROS node
    rospy.init_node('rl_goal_runner', anonymous=False)

    # Load the trained TD3 model
    model = TD3.load(args.model_path)
    print(f"Model loaded from {args.model_path}")

    # Subscribe to RViz goal topic
    rospy.Subscriber('/move_base_simple/goal', PoseStamped, goal_callback)

    # Initialize environment and main loop
    rate = rospy.Rate(1)  # 1 Hz
    eval_env = RobotNavEnv()

    while not rospy.is_shutdown():
        if goal_received:
            goal_received = False
            run_rl(eval_env, current_goal)
        rate.sleep()
