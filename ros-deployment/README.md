# ROS Interface for RL Navigation

This code provides an interface to deploy trained reinforcement learning policies on real ground robots like QBot. It implements a Gym-compatible environment that wraps ROS communication for robot control and sensor data processing.

<div style="display: flex; justify-content: space-between; width: 100%;">
    <img src="../images/single_obs.gif" alt="Robot Navigation Demo" width="48%"/>
    <img src="../images/dynamic_obs.gif" alt="Real Robot Demo" width="48%"/>
</div>
<br>

## Dependencies

- ROS (tested with ROS Noetic)
- Python 3.x (tested with 3.10)

## File Structure

The package contains the following files:

- `env.py`: Gym environment wrapper for real robot
  - Implements observation/action space definitions
  - Handles state normalization and preprocessing
  - Manages episode termination conditions
- `real_env.py`: Robot environment for communicating with bot
  - Processes LIDAR data
  - Handles robot motion control
  - Tracks state and goal progress
  - Implements collision detection
- `run.py`: Main entry point for running the navigation system
  - Loads trained RL models
  - Processes RViz goal commands
  - Executes navigation tasks

## Usage

1. Start your ROS core:
```bash
roscore
```

2. Launch AMCL node for robot localization (tuned according to your bot):
```bash
roslaunch your_robot_navigation amcl.launch
```

3. Run the navigation system:
```bash
python3 run.py --model-path ../../models/td3.zip
```

4. Set goals using RViz:
   - Open RViz
   - Use the "2D Nav Goal" tool to set goals
   - The robot will automatically navigate to the specified goal

## Notes

- The environment uses a 15cm collision threshold
- Goals are considered reached when:
  - Position error < 15cm
  - Angular error < 0.15 radians
- LIDAR data is processed to handle infinite values and environment boundaries. This is usefull when arena does not have predefined physical boundaries.
- The system automatically stops the robot when collisions are detected
