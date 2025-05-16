import numpy as np
import random
import shapely
from irsim.lib.handler.geometry_handler import GeometryFactory
from irsim.env import EnvBase


class SIM_ENV:
    """
    Simulator environment wrapper for robot navigation.
    
    This class provides a high-level interface for robot navigation simulation,
    handling state tracking, reward calculation, and environment interactions.
    """
    
    def __init__(self, world_file="robot_world.yaml", render=False):
        """
        Initialize the simulation environment.
        
        Args:
            world_file (str): Path to the world configuration file
            render (bool): Whether to enable visualization
        """
        # Initialize environment
        self.env = EnvBase(world_file, display=render, disable_all_plot=not render)
        self.robot_goal = self.env.get_robot_info(0).goal
        
        # Initialize tracking variables
        self._reset_tracking()

    def _reset_tracking(self):
        """Reset tracking variables for distance and angle differences."""
        self.prev_distance = None
        self.prev_diff_rad = None

    def _calculate_robot_metrics(self, robot_state):
        """
        Calculate robot-related metrics including goal vector, distance, and orientation.
        
        Args:
            robot_state: Current state of the robot [x, y, theta]
            
        Returns:
            tuple: (goal_vector, distance, cos, sin, diff_rad)
        """
        # Calculate goal vector
        goal_vector = [
            self.robot_goal[0].item() - robot_state[0].item(),
            self.robot_goal[1].item() - robot_state[1].item(),
        ]
        
        # Calculate angle difference between robot orientation and goal orientation
        diff_rad = float(((-robot_state[2] + self.env.robot.goal[2] + np.pi) % (2 * np.pi)) - np.pi)
        
        # Calculate distance and pose
        distance = np.linalg.norm(goal_vector)
        pose_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
        cos, sin = self._calculate_cossin(pose_vector, goal_vector)
        
        return goal_vector, distance, cos, sin, diff_rad

    @staticmethod
    def _calculate_cossin(vec1, vec2):
        """
        Calculate cosine and sine between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            tuple: (cosine, sine) of the angle between vectors
        """
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        cos = np.dot(vec1, vec2)
        sin = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        return cos, sin

    def _calculate_reward(self, goal, collision, distance_delta, action, laser_scan, delta_rad):
        """
        Calculate reward based on various factors including goal achievement,
        collision avoidance, and movement efficiency.
        
        Args:
            goal (bool): Whether goal is reached
            collision (bool): Whether collision occurred
            distance_delta (float): Change in distance to goal
            action (list): [linear_velocity, angular_velocity]
            laser_scan (list): Laser scan readings
            delta_rad (float): Change in angle difference
            
        Returns:
            float: Calculated reward
        """
        if goal:
            return 100.0
        elif collision:
            return -100.0
        
        # Reward components
        progress_reward = distance_delta * 10  # Reward for moving closer to goal
        dir_progress = delta_rad * 1  # Reward for aligning with goal
        time_penalty = -0.65  # Penalty for each time step
        rotation_penalty = -abs(action[1]) * 0.4  # Penalty for excessive rotation
        
        # Obstacle avoidance
        safe_distance = 1.35
        min_dist = min(laser_scan)
        obstacle_penalty = -(safe_distance - min_dist) if min_dist < safe_distance else 0
        
        return progress_reward + time_penalty + obstacle_penalty + dir_progress + rotation_penalty

    def step(self, lin_velocity=0.0, ang_velocity=0.1):
        """
        Execute one simulation step.
        
        Args:
            lin_velocity (float): Linear velocity
            ang_velocity (float): Angular velocity
            
        Returns:
            tuple: (laser_scan, distance, cos, sin, collision, goal, diff_rad, action, reward)
        """
        # Step simulation
        self.env.step(action_id=0, action=np.array([[lin_velocity], [ang_velocity]]))
        if self.env.display:
            self.env.render()

        # Get sensor data
        scan = self.env.get_lidar_scan()
        robot_state = self.env.get_robot_state()
        
        # Calculate metrics
        goal_vector, distance, cos, sin, diff_rad = self._calculate_robot_metrics(robot_state)
        
        # Calculate deltas
        if self.prev_distance is None:
            distance_delta = delta_rad = 0
        else:
            distance_delta = self.prev_distance - distance
            delta_rad = abs(self.prev_diff_rad) - abs(diff_rad)
        
        # Update tracking
        self.prev_distance = distance
        self.prev_diff_rad = diff_rad
        
        # Get status and calculate reward
        goal = self.env.robot.arrive
        collision = self.env.robot.collision
        action = [lin_velocity, ang_velocity]
        reward = self._calculate_reward(goal, collision, distance_delta, action, scan["ranges"], delta_rad)
        
        if goal:
            print("Goal reached")

        return scan["ranges"], distance, cos, sin, collision, goal, diff_rad, action, reward

    def reset(self, robot_state=None, robot_goal=None, random_obstacles=True):
        """
        Reset the simulation environment.
        
        Args:
            robot_state (list): Initial robot state [x, y, theta]
            robot_goal (list): Goal position [x, y, theta]
            random_obstacles (bool): Whether to place random obstacles
            
        Returns:
            tuple: Initial state information
        """
        # Initialize robot state
        if robot_state is None:
            robot_state = [[random.uniform(0.5, 5.5)], 
                          [random.uniform(0.5, 5.5)], 
                          [0]]

        self.env.robot.set_state(state=np.array(robot_state), init=True)

        # Place obstacles
        if random_obstacles:
            self.env.random_obstacle_position(
                range_low=[0, 0, -3.14],
                range_high=[6, 6, 3.14],
                ids=list(range(1, 7)),
                non_overlapping=True
            )

        # Set goal
        if robot_goal is None:
            robot_goal = self._generate_valid_goal()
        
        self.env.robot.set_goal(np.array(robot_goal), init=True)
        self.env.reset()
        self.robot_goal = self.env.robot.goal
        self._reset_tracking()
        
        # Get initial state
        action = [0.0, 0.0]
        return self.step(lin_velocity=action[0], ang_velocity=action[1])

    def _generate_valid_goal(self):
        """
        Generate a valid goal position that doesn't overlap with obstacles.
        
        Returns:
            list: Valid goal position [x, y, theta]
        """
        while True:
            goal = [[random.uniform(0.5, 5.5)], 
                   [random.uniform(0.5, 5.5)], 
                   [random.uniform(-3.14, 3.14)]]
            
            # Check if goal overlaps with obstacles
            shape = {"name": "circle", "radius": 0.4}
            state = [goal[0], goal[1], goal[2]]
            gf = GeometryFactory.create_geometry(**shape)
            geometry = gf.step(np.c_[state])
            
            if not any(shapely.intersects(geometry, obj._geometry) for obj in self.env.obstacle_list):
                return goal