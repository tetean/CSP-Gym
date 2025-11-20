import os, sys

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.core import RenderFrame
from lammps import lammps
from typing import Type
from abc import ABC, abstractmethod
import numpy as np


def scale_adjusted(x, steepness=5.0):
    t = x * steepness
    scaled = np.tanh(t)
    return scaled


def run_lammps_simulation(lattice_params, frac_coords, num_atoms):
    a = lattice_params[0]

    lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])  # 取消终端输出

    commands = [
        "units metal",
        "atom_style atomic",
        "dimension 3",
        "boundary p p p",
        f"region box block 0 {a} 0 {a} 0 {a}",
        "create_box 1 box",
        "mass 1 39.948",
        "pair_style lj/cut 10.0",
        "pair_coeff 1 1 0.0104 3.405"
    ]

    for cmd in commands:
        lmp.command(cmd)

    for i in range(num_atoms):
        x, y, z = frac_coords[i]
        if x < 0 or x > 1 or y < 0 or y > 1 or z < 0 or z > 1:
            print(f'illegal x, y, z: {x, y, z}')
        lmp.command(f"create_atoms 1 single {x * a} {y * a} {z * a} units box")

    lmp.command("min_style cg")
    lmp.command("minimize 1.0e-8 1.0e-10 1000 10000")

    energy = lmp.get_thermo("pe")
    lmp.close()

    return energy


def compute_energy_custom_positions(positions, box_size=5.41):
    positions = np.array(positions)
    if positions.shape != (4, 3):
        raise ValueError("必须提供4个原子的3D坐标，形状应为 (4, 3)")

    lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])  # 取消终端输出

    lammps_script = f"""
    units metal
    atom_style atomic
    dimension 3
    boundary p p p
    region box block 0 {box_size} 0 {box_size} 0 {box_size}
    create_box 1 box
    """

    for line in lammps_script.strip().split('\n'):
        lmp.command(line.strip())

    for i, (x, y, z) in enumerate(positions):
        lmp.command(f"create_atoms 1 single {x} {y} {z} units box")

    setup_commands = """
    mass 1 39.948
    pair_style lj/cut 10.0
    pair_coeff 1 1 0.0104 3.405
    minimize 1.0e-8 1.0e-10 1000 10000
    """
    for line in setup_commands.strip().split('\n'):
        lmp.command(line.strip())

    energy = lmp.get_thermo("pe")
    lmp.close()

    return energy


class BaseReward(ABC):
    @property
    @abstractmethod
    def get_reward(self):
        pass


class CSPEnergyReward(BaseReward):
    def __init__(self, env):
        self.env = env

    def get_reward(self, observation):
        energy = self.env._compute_energy(observation)
        scaled_energy = scale_adjusted(energy, 0.5)
        return -scaled_energy, energy


class CSPEnv(gym.Env):
    def __init__(self, num_atoms: int, reward_class: Type[BaseReward], max_episode_steps: int = 100,
                 min_distance: float = 0.01):
        super(CSPEnv, self).__init__()
        self.num_atoms = num_atoms
        self.state_dim = 1 + 3 * num_atoms
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.min_distance = min_distance

        low = np.concatenate([
            np.array([5.0]),
            np.full((3 * num_atoms,), 0.0)
        ])
        high = np.concatenate([
            np.array([6.0]),
            np.full((3 * num_atoms,), 1.0)
        ])

        self.observation_space = spaces.Box(
            low=low, high=high, shape=(self.state_dim,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(self.state_dim,), dtype=np.float64
        )
        self.state = None
        self.reward_class = reward_class(self)

    def _check_overlap(self, frac_coords):
        for i in range(len(frac_coords)):
            for j in range(i + 1, len(frac_coords)):
                dist = np.linalg.norm(frac_coords[i] - frac_coords[j])
                if dist < self.min_distance:
                    return True
        return False

    def _generate_non_overlapping_coords(self):
        while True:
            fractional_coords = np.array([
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5]
            ])
            perturbation = np.random.uniform(-0.1, 0.1, (self.num_atoms, 3))
            fractional_coords += perturbation
            fractional_coords = np.clip(fractional_coords, 0.0, 1.0)

            if not self._check_overlap(fractional_coords):
                return fractional_coords.flatten()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        if seed is not None:
            np.random.seed(seed)

        lattice_params = np.array([np.random.uniform(5.0, 6.0)])
        fractional_coords = self._generate_non_overlapping_coords()
        self.state = np.concatenate((lattice_params, fractional_coords))
        return self.state, {}

    def step(self, action):
        self.current_step += 1

        new_state = self.state + action
        new_state[0] = np.clip(new_state[0], 5.0, 6.0)
        new_frac_coords = np.clip(new_state[1:], 0.0, 1.0).reshape(-1, 3)

        if self._check_overlap(new_frac_coords):
            reward, energy = self.reward_class.get_reward(self.state)
        else:
            self.state = np.concatenate([new_state[:1], new_frac_coords.flatten()])
            reward, energy = self.reward_class.get_reward(self.state)

        done = False
        truncated = self.current_step >= self.max_episode_steps

        return self.state, reward, done, truncated, {'energy': energy}

    def _compute_energy(self, state):
        lattice_params = np.array([state[0], state[0], state[0], 90.0, 90.0, 90.0])
        frac_coords = state[1:].reshape(-1, 3)
        positions = frac_coords * state[0]
        return compute_energy_custom_positions(positions, state[0])

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        print('state: ', self.state)

    def close(self):
        pass


if __name__ == '__main__':
    import gymnasium as gym
    import matplotlib.pyplot as plt
    from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG
    from stable_baselines3.common.monitor import Monitor
    from callbacks.reward_callback import SaveOnBestTrainingRewardCallback
    from stable_baselines3.common.results_plotter import plot_results
    from stable_baselines3.common import results_plotter
    from datetime import datetime


    def train_model(env, algorithm, total_timesteps, callback):
        if algorithm == "PPO":
            model = PPO("MlpPolicy", env, verbose=1)
        elif algorithm == "SAC":
            model = SAC("MlpPolicy", env, verbose=1)
        elif algorithm == "TD3":
            model = TD3("MlpPolicy", env, verbose=1)
        elif algorithm == "A2C":
            model = A2C("MlpPolicy", env, verbose=1)
        elif algorithm == "DDPG":
            model = DDPG("MlpPolicy", env, verbose=1)
        else:
            raise ValueError("Invalid algorithm")
        model.learn(total_timesteps=total_timesteps, callback=callback)
        return model

    algorithms = ['PPO', 'SAC', 'TD3', 'A2C', 'DDPG']
    timesteps = int(5e5)

    for _ in range(3):
        for algorithm in algorithms:
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = f'results/{algorithm}/{current_time}'
            callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir, verbose=1)

            os.makedirs(log_dir, exist_ok=True)
            env = CSPEnv(num_atoms=4, reward_class=CSPEnergyReward)

            env = Monitor(env, log_dir)
            train_model(env, algorithm, timesteps, callback)

            plt.clf()
            plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "total rewards comparison")
            plt.savefig(os.path.join(log_dir, f'res_fig.png'))