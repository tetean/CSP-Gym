<h1>
<p align="center">
    <img src="properties/cover.png" alt="CSP-gym logo" width="800"/>
</p>
</h1>

<h4 align="center">


[![Static Badge](https://img.shields.io/badge/Docs-cspgym.tetean.com-purple?style=for-the-badge)](https://cspgym.tetean.com)
[![Static Badge](https://img.shields.io/badge/Python-3.10%2B-yellow?style=for-the-badge)](https://python.org/downloads)

</h4>


# CSP Gym: A Reinforcement Learning Environment for Crystal Structure Prediction

CSP Gym is a benchmark environment for applying reinforcement learning to crystal structure prediction (CSP). The goal is to find the lowest energy configuration of a crystal by adjusting lattice parameters and atomic positions.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/tetean/CSP-Gym.git
    cd CSP-Gym
    ```

2. Install dependencies:

    ```bash
    pip install uv
    uv venv .venv --python 3.10 
    source .venv/bin/activate
    uv pip install -e .
    ```

3. Usage

   See the `examples/train.py` script for an example of training a PPO agent using Stable-Baselines3.

4. Citation

   If you use CSP Gym in your research, please cite:

    ```
    @article{wang2025cspgym,
      title={No-Data-Driven Crystal Structure Prediction via Model-Free Reinforcement Learning},
      author={Xiean Wang, Pin Chen, Qingsong Zou},
      year={2025}
    }
    ```
