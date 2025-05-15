#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
import tf
import time

def reduce_lidar_scan(scan_array):
    """
    Reduce LIDAR scan data dimensionality by averaging groups of 4 readings.
    
    Args:
        scan_array (numpy.ndarray): Raw LIDAR scan data
        
    Returns:
        numpy.ndarray: Reduced LIDAR scan data
    """
    # Make sure the length is divisible by 4, otherwise trim the excess elements
    if len(scan_array) % 4 != 0:
        scan_array = scan_array[:-(len(scan_array) % 4)]

    # Reshape the array into chunks of 4 elements
    reshaped_array = np.array(scan_array).reshape(-1, 4)

    # Compute the average for each chunk, excluding zeros
    result = []
    for chunk in reshaped_array:
        non_zero_values = chunk[chunk != 0]
        if len(non_zero_values) == 0:
            result.append(10)  # All zeros, use 10
        else:
            result.append(np.mean(non_zero_values))  # Take the mean of non-zero values

    return np.array(result)

def constrain_lidar_scan(bot_pos, yaw, angles, lidar_ranges, box_limits):
    """
    Constrain LIDAR readings to stay within specified box limits.
    This is usefull if your arena lack proper boundaries.
    
    Args:
        bot_pos (tuple): (x, y) robot position
        yaw (float): Robot orientation
        angles (numpy.ndarray): LIDAR beam angles
        lidar_ranges (numpy.ndarray): LIDAR range readings
        box_limits (tuple): (min_x, max_x, min_y, max_y) environment boundaries
        
    Returns:
        numpy.ndarray: Constrained LIDAR ranges
    """
    bot_x, bot_y = bot_pos
    min_x, max_x, min_y, max_y = box_limits
    
    constrained_ranges = np.empty_like(lidar_ranges)

    for i, (angle, lidar_range) in enumerate(zip(angles, lidar_ranges)):
        adjusted_angle = angle + yaw
        ray_dx = np.cos(adjusted_angle)
        ray_dy = np.sin(adjusted_angle)

        distances = []

        # Check intersections with vertical boundaries
        if ray_dx != 0:
            t1 = (min_x - bot_x) / ray_dx
            t2 = (max_x - bot_x) / ray_dx
            distances.extend([t for t in [t1, t2] if t > 0])

        # Check intersections with horizontal boundaries
        if ray_dy != 0:
            t3 = (min_y - bot_y) / ray_dy
            t4 = (max_y - bot_y) / ray_dy
            distances.extend([t for t in [t3, t4] if t > 0])

        if distances:
            min_boundary_dist = min(distances)
            constrained_ranges[i] = min(lidar_range, min_boundary_dist)
        else:
            constrained_ranges[i] = lidar_range

    return constrained_ranges

class REAL_ENV:
    """
    Real robot environment class that interfaces with ROS.
    
    This class handles:
    - LIDAR data processing
    - Robot motion control
    - State tracking and goal progress
    - Collision detection
    
    Attributes:
        scan_sub (rospy.Subscriber): LIDAR scan subscriber
        tf_listener (tf.TransformListener): TF listener for pose tracking
        cmd_vel_pub (rospy.Publisher): Velocity command publisher
        latest_scan (list): Most recent LIDAR scan data
        robot_pose (list): Current robot position [x, y]
        robot_yaw (float): Current robot orientation
        collision (bool): Collision flag
        goal_reached (bool): Goal reached flag
        robot_goal (list): Target pose [x, y, yaw]
    """
    
    def __init__(self, goal_pose=None):
        """Initialize the real robot environment."""
        # Subscribers
        self.scan_sub = rospy.Subscriber('/lidar_scan', LaserScan, self.scan_callback)
        self.tf_listener = tf.TransformListener()

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # State variables
        self.latest_scan = []
        self.robot_pose = 0.0
        self.robot_yaw = 0.0
        self.collision = False
        self.goal_reached = False
        self.robot_goal = goal_pose
        
        # Performance metrics
        self.start_time = time.time()
        self.path_length = 0
        self.linear_vel_sum = 0
        self.angular_vel_sum = 0
        self.timestep = 0
        self.prev_pose = None

        rospy.sleep(1)  # Initialization wait

    def scan_callback(self, data):
        """
        Process incoming LIDAR scan data.
        
        Args:
            data (LaserScan): Raw LIDAR scan message
        """
        latest_scan = reduce_lidar_scan(data.ranges)
 
        bot_position = self.robot_pose
        bot_yaw = self.robot_yaw
        lidar_offset = 0.15  # LIDAR is 0.15m ahead of robot center

        # Calculate LIDAR position
        lidar_x = bot_position[0] + lidar_offset * np.cos(self.robot_yaw)
        lidar_y = bot_position[1] + lidar_offset * np.sin(self.robot_yaw)
        
        # Generate LIDAR beam angles
        lidar_angles = np.linspace(0, 2 * np.pi, num=420)
        box_limits = (0, 6, 0, 6)  # Environment boundaries

        # Constrain LIDAR readings to environment boundaries
        latest_scan = constrain_lidar_scan(
            [lidar_x, lidar_y], 
            bot_yaw, 
            lidar_angles, 
            latest_scan, 
            box_limits
        )
        
        # Rotate scan data to align with robot orientation
        self.latest_scan = np.roll(latest_scan, int(len(latest_scan) * 1/2))
    
        # Check for collisions
        self.collision = min(self.latest_scan) < 0.15  # 15cm collision threshold
        if self.collision:
            cmd = Twist()
            cmd.linear.x = 0
            cmd.angular.z = 0
            self.cmd_vel_pub.publish(cmd)
            print("Collision detected!")

    def get_robot_pose_from_tf(self):
        """Get current robot pose from TF."""
        self.tf_listener.waitForTransform("origin", "base_link", rospy.Time(0), rospy.Duration(0.1))
        (trans, rot) = self.tf_listener.lookupTransform("origin", "base_link", rospy.Time(0))
        
        self.robot_pose = [trans[0], trans[1]]
        _, _, self.robot_yaw = euler_from_quaternion(rot)

        print("TF Pose:", self.robot_pose, self.robot_yaw)

    def step(self, lin_velocity=0.0, ang_velocity=0.1):
        """
        Execute one step in the environment.
        
        Args:
            lin_velocity (float): Linear velocity command
            ang_velocity (float): Angular velocity command
            
        Returns:
            tuple: (scan_data, distance, cos, sin, collision, goal, diff_rad, action, reward)
        """
        self.timestep += 1
        self.get_robot_pose_from_tf()
        
        if self.robot_pose is None:
            rospy.logwarn("Waiting for AMCL pose...")
            rospy.sleep(0.1)
            return None

        # Publish velocity command
        cmd = Twist()
        cmd.linear.x = lin_velocity
        cmd.angular.z = ang_velocity
        if not self.collision and not self.goal_reached:
            self.cmd_vel_pub.publish(cmd)

        rospy.sleep(0.1)  # Allow time for motion

        # Compute goal vector and progress
        goal_vector = [
            self.robot_goal[0] - self.robot_pose[0],
            self.robot_goal[1] - self.robot_pose[1],
        ]

        # Calculate angle difference to goal
        diff_rad = float(((-self.robot_yaw + self.robot_goal[2] + np.pi) % (2 * np.pi)) - np.pi)
        distance = np.linalg.norm(goal_vector)
        goal = (distance < 0.15 and abs(diff_rad) < 0.15)  # 15cm position and 0.15rad angle threshold

        # Update path length
        if self.prev_pose is not None:
            delta = np.sqrt((self.robot_pose[0] - self.prev_pose[0])**2 +
                         (self.robot_pose[1] - self.prev_pose[1])**2)
            self.path_length += delta
        self.prev_pose = self.robot_pose

        # Update velocity sums for averaging in metrics
        self.linear_vel_sum += abs(lin_velocity)
        self.angular_vel_sum += abs(ang_velocity)

        if goal:
            print(self.robot_yaw, self.robot_goal[2])
            rospy.loginfo("Goal reached!")
            cmd = Twist()
            cmd.linear.x = 0
            cmd.angular.z = 0
            self.cmd_vel_pub.publish(cmd)
            self.goal_reached = True
            
            # Log performance metrics
            print("Time Taken:", time.time() - self.start_time)
            print("Distance:", distance)
            print("Ang_diff:", diff_rad)
            print("Path Length:", self.path_length)
            print('Avg Linear:', self.linear_vel_sum/self.timestep)
            print('Avg Ang:', self.angular_vel_sum/self.timestep)

        # Compute observation components
        pose_vector = [np.cos(self.robot_yaw), np.sin(self.robot_yaw)]
        cos, sin = self.cossin(pose_vector, goal_vector)
        action = [lin_velocity, ang_velocity]
        reward = 0  # Made for inference, not used

        return self.latest_scan, distance, cos, sin, self.collision, goal, diff_rad, action, reward

    def reset(self, goal_pose=None):
        """
        Reset the environment with a new goal.
        
        Args:
            goal_pose (list): [x, y, yaw] target pose
            
        Returns:
            tuple: Initial environment state
        """
        # Reset metrics
        self.start_time = time.time()
        self.path_length = 0
        self.timestep = 0
        self.linear_vel_sum = 0
        self.angular_vel_sum = 0
        
        # Get initial pose
        self.get_robot_pose_from_tf()
        self.prev_pose = None

        rospy.loginfo("Manually reset the robot and localization if needed.")
        rospy.sleep(2)  # Wait for manual reset or AMCL reinitialization

        # Set new goal
        self.robot_goal = goal_pose
        self.collision = False
        self.goal_reached = False

        # Take initial step
        action = [0.0, 0.0]
        return self.step(lin_velocity=action[0], ang_velocity=action[1])

    @staticmethod
    def cossin(vec1, vec2):
        """
        Compute cosine and sine of angle between two vectors.
        
        Args:
            vec1 (list): First vector [x, y]
            vec2 (list): Second vector [x, y]
            
        Returns:
            tuple: (cosine, sine) of angle between vectors
        """
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        cos = np.dot(vec1, vec2)
        sin = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        return cos, sin
    
if __name__ == "__main__":
    # Test the environment
    rospy.init_node('real_robot_env', anonymous=True)
    env = REAL_ENV(goal_pose=[0,0,0])
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        step_result = env.step(0.0, 0.0)
        if step_result:
            scan, distance, cos, sin, collision, goal, diff_rad, action, reward = step_result
            rospy.loginfo(f"Distance: {distance:.2f}, Reward: {reward:.2f}")
            if collision or goal:
                break
        rate.sleep()