# Robot Navigation with Reinforcement Learning

This project implements a reinforcement learning-based robot navigation system that enables autonomous navigation in complex environments with obstacles. The work was done as part of my Master Thesis Submission. 

The code uses [IRSim](https://github.com/ir-sim/ir-sim) to train reinforcement learning policies (using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)) which can be deployed both in the simulator and on real hardware. We have deployed a policy on the physical [QBot](https://www.quanser.com/products/qbot-platform/) platform. The hardware implementation of ROS-based code can be found in the `ros-deployment/` folder.

<p align="center"> <img src="images/multi_obs.gif" alt="Real Robot Demo" width="88%"> </p>

The system enables a robot to navigate from a start position to a goal while avoiding obstacles. The reinforcement learning agent learns to output linear and angular velocities based on laser scan observations. The trained policy demonstrates robust navigation behavior in both simulated and real-world environments.

<p align="center"> <img src="images/sim.gif" alt="Simulation Demo" width="55%"> </p>




## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/harshmahesheka/rl-nav
cd rl-nav
```

2. Install the required dependencies (The code was tested with python 3.10):
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Training

To train a new model:

```bash
python train.py --num-envs 7 \
                --total-timesteps 200000 \
                --model-path models/td3_robot_nav_model \
                --tensorboard-log ./td3_robot_nav_tensorboard/ \
                --eval-episodes 10 \
                --render
```

### Evaluation

To evaluate a trained model:

```bash
python run.py --model-path models/your_model.zip \
              --num-episodes 10 \
```


## 📁 Project Structure

- `train.py`: Contains the custom Gym environment for training
- `sim.py`: Core simulation environment wrapper
- `run.py`: Training and evaluation scripts
- `models/`: Directory for storing trained models
- `robot_world.yaml`: World configuration file
- `ros-deployment/`: Package for deploying trained policies on physical robot

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for the RL algorithms
- [Gym](https://github.com/openai/gym) for the environment interface  
- [IRSim](https://github.com/intelligent-robotics/irsim) for the simulation environment
- [DRL-IRSim](https://github.com/reiniscimurs/DRL-robot-navigation-IR-SIM) for code inspiration
