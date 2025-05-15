import gym
from gym import spaces
import numpy as np
from real_env import REAL_ENV


class RobotNavEnv(gym.Env):
    """
    Custom Gym environment that wraps the REAL_ENV simulator for robot navigation.
    
    This environment:
    - Converts the simulator's outputs into a fixed-size observation space
    - Defines a continuous action space for linear and angular velocities
    - Handles state normalization and preprocessing
    - Manages episode termination conditions
    
    Attributes:
        action_space (gym.spaces.Box): Continuous action space for linear and angular velocities
        observation_space (gym.spaces.Box): Fixed-size observation space
        state_dim (int): Dimension of the state vector
        sim (REAL_ENV): Instance of the real environment simulator
        time (int): Step counter for episode termination
    """
    def __init__(self):
        super(RobotNavEnv, self).__init__()
        # Action space: [linear_velocity, angular_velocity]
        # Linear velocity range: [-0.6, 0.6] m/s
        # Angular velocity range: [-1.2, 1.2] rad/s
        self.action_space = spaces.Box(
            low=np.array([-0.6, -1.2]), 
            high=np.array([0.6, 1.2]), 
            dtype=np.float32
        )
        
        # Observation space: 49-dimensional vector containing:
        # - Binned LIDAR scan data (42 dimensions)
        # - Distance to goal (1 dimension)
        # - Goal direction cos/sin (2 dimensions)
        # - Current linear/angular velocities (2 dimensions)
        # - Goal angle difference cos/sin (2 dimensions)
        self.state_dim = 49
        self.observation_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        
        # Initialize simulator with default goal
        self.goal = [0, 0, 0]
        self.sim = REAL_ENV(goal_pose=self.goal)
        self.time = 0

    def prepare_state(self, data):
        """
        Process raw environment data into a normalized state vector.
        
        Args:
            data (tuple): Raw environment data containing:
                - LIDAR scan data
                - Distance to goal
                - Goal direction cos/sin
                - Collision flag
                - Goal reached flag
                - Angle difference
                - Last action
                - Reward
        
        Returns:
            tuple: (normalized_state, terminal_flag)
        """
        latest_scan, distance, cos, sin, collision, goal, diff_rad, action, reward = data
        latest_scan = np.array(latest_scan)

        # Handle infinite values in LIDAR data
        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 10

        # Bin LIDAR data to reduce dimensionality
        max_bins = self.state_dim - 7
        bin_size = int(np.ceil(len(latest_scan) / max_bins))
        min_values = []
        
        for i in range(0, len(latest_scan), bin_size):
            bin = latest_scan[i : i + min(bin_size, len(latest_scan) - i)]
            # Find the minimum value in the current bin and append it to the min_values list
            min_values.append(min(bin) / 10)

        # Normalize distance and velocities
        distance /= 10
        lin_vel = (action[0] + 0.6) / 1.2
        ang_vel = (action[1] + 1.2) / 2.4

        # Convert angle difference to cos/sin representation
        rad_cos = np.cos(diff_rad)
        rad_sin = np.sin(diff_rad)

        # Combine all state components
        state = min_values + [distance, cos, sin] + [lin_vel, ang_vel] + [rad_cos, rad_sin]
        assert len(state) == self.state_dim

        terminal = 1 if collision or goal else 0
        return state, terminal

    def reset(self, goal):
        """
        Reset the environment with a new goal.
        
        Args:
            goal (list): [x, y, yaw] target pose
            
        Returns:
            numpy.ndarray: Initial observation
        """
        self.goal = goal
        sim_data = self.sim.reset(goal_pose=self.goal)
        obs, _ = self.prepare_state(sim_data)
        self.current_obs = obs
        self.time = 0
        return obs

    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (numpy.ndarray): [linear_velocity, angular_velocity]
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        lin_velocity, ang_velocity = action

        # Execute action in simulator
        sim_data = self.sim.step(lin_velocity=lin_velocity, ang_velocity=ang_velocity)
        obs, terminal = self.prepare_state(sim_data)
        reward = sim_data[-1]
        
        # Check termination conditions
        done = terminal
        self.time += 1
        if self.time >= 5000:  # Time limit
            done = True
            reward = 0
            print("Episode terminated due to time limit")
        
        info = {}
        self.current_obs = obs
        return obs, reward, done, info

