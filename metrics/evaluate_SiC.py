import os
import gymnasium as gym
from environments.SiC import CSPEnergyReward
import numpy as np
from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG
import pandas as pd
from lammps import lammps
from datetime import datetime


def lattice_vectors(lattice):
    a, b, c, alpha_deg, beta_deg, gamma_deg = lattice
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)
    gamma = np.deg2rad(gamma_deg)
    A = np.array([a, 0.0, 0.0])
    B = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])
    Cx = c * np.cos(beta)
    Cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    sqrt_term = 1 - np.cos(beta) ** 2 - ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)) ** 2
    if sqrt_term < 0:
        sqrt_term = 0
    Cz = c * np.sqrt(sqrt_term)
    C = np.array([Cx, Cy, Cz])
    return A, B, C


def frac_to_cart(frac_coords, lattice):
    A, B, C = lattice_vectors(lattice)
    lattice_matrix = np.array([A, B, C]).T
    cart_coords = np.dot(frac_coords.reshape(-1, 3), lattice_matrix)
    return cart_coords


def compute_rmsd(s1, species1, s2, species2):
    lattice1 = s1[:6]
    frac_coords1 = s1[6:].reshape(-1, 3)
    lattice2 = s2[:6]
    frac_coords2 = s2[6:].reshape(-1, 3)
    cart_coords1 = frac_to_cart(frac_coords1, lattice1)
    cart_coords2 = frac_to_cart(frac_coords2, lattice2)

    if len(species1) != len(species2) or sorted(species1) != sorted(species2):
        raise ValueError("Atom counts or species do not match between structures.")

    reordered_coords2 = np.zeros_like(cart_coords2)
    used_indices = set()

    for i, specie1 in enumerate(species1):
        min_dist = float('inf')
        best_idx = None
        for j, specie2 in enumerate(species2):
            if j in used_indices or specie1 != specie2:
                continue
            dist = np.linalg.norm(cart_coords1[i] - cart_coords2[j])
            if dist < min_dist:
                min_dist = dist
                best_idx = j
        if best_idx is not None:
            reordered_coords2[i] = cart_coords2[best_idx]
            used_indices.add(best_idx)
        else:
            raise ValueError(f"No matching atom found for {specie1} at index {i}")

    diffs = cart_coords1 - reordered_coords2
    rmsd = np.sqrt(np.mean(np.sum(diffs ** 2, axis=1)))
    return rmsd


def compute_lattice_params(lx, ly, lz, xy, xz, yz):
    a = lx
    b = np.sqrt(ly ** 2 + xy ** 2)
    c = np.sqrt(lz ** 2 + xz ** 2 + yz ** 2)
    gamma = np.arccos(xy / b) * 180 / np.pi if b != 0 else 90.0
    beta = np.arccos(xz / c) * 180 / np.pi if c != 0 else 90.0
    alpha = np.arccos(yz / c) * 180 / np.pi if c != 0 else 90.0
    return a, b, c, alpha, beta, gamma


def find_reference_structure_with_lammps():
    lmp = lammps()

    commands = [
        "units metal",
        "atom_style atomic",
        "dimension 3",
        "boundary p p p",

        "lattice custom 4.36 a1 1.0 0.0 0.0 a2 0.0 1.0 0.0 a3 0.0 0.0 1.0 "
        "basis 0.0 0.0 0.0 basis 0.25 0.25 0.25 "
        "basis 0.5 0.5 0.0 basis 0.75 0.75 0.25",
        "region box block 0 1 0 1 0 1",
        "create_box 2 box",
        "create_atoms 1 single 0.0 0.0 0.0 units lattice",  # Si at (0,0,0)
        "create_atoms 1 single 0.5 0.5 0.0 units lattice",  # Si at (0.5,0.5,0)
        "create_atoms 2 single 0.25 0.25 0.25 units lattice",  # C at (0.25,0.25,0.25)
        "create_atoms 2 single 0.75 0.75 0.25 units lattice",  # C at (0.75,0.75,0.25)

        "mass 1 28.0855",  # Si
        "mass 2 12.0107",  # C

        "pair_style tersoff",
        "pair_coeff * * potentials/SiC.tersoff Si C",

        "fix 1 all box/relax iso 0.0 vmax 0.005",
        "minimize 1.0e-8 1.0e-10 1000 10000",

        "thermo 1",
        "thermo_style custom step pe ke etotal press vol lx ly lz xy xz yz",
        "run 0"
    ]

    for cmd in commands:
        lmp.command(cmd)

    E_true = lmp.get_thermo("pe")  # 势能（eV）

    lx = lmp.get_thermo("lx")
    ly = lmp.get_thermo("ly")
    lz = lmp.get_thermo("lz")
    xy = lmp.get_thermo("xy")
    xz = lmp.get_thermo("xz")
    yz = lmp.get_thermo("yz")
    a, b, c, alpha, beta, gamma = compute_lattice_params(lx, ly, lz, xy, xz, yz)

    natoms = lmp.get_natoms()  # 应为4（2 Si, 2 C）
    coords = np.zeros((natoms, 3))
    for i in range(natoms):
        atom = lmp.extract_atom("x", 3)[i]
        coords[i, 0] = atom[0]
        coords[i, 1] = atom[1]
        coords[i, 2] = atom[2]

    lattice_matrix = np.array(lattice_vectors([a, b, c, alpha, beta, gamma])).T
    inv_matrix = np.linalg.inv(lattice_matrix)
    frac_coords = np.dot(coords, inv_matrix).flatten()

    s_true = np.concatenate(([a, b, c, alpha, beta, gamma], frac_coords))

    lmp.close()

    print(f"LAMMPS Optimized a: {a:.4f}, b: {b:.4f}, c: {c:.4f} Å")
    print(f"LAMMPS Optimized angles: α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}°")
    print(f"LAMMPS Optimized s_true: {s_true}")
    print(f"LAMMPS Optimized E_true: {E_true:.4f} eV")

    return s_true, E_true

def evaluate_pretrained_model(algorithm, rec_path, num_atoms, reward_class, s_true, E_true, num_eval_episodes=1000,
                              epsilon=3.5, epsilon_E=0.001):
    env = gym.make('CSPEnv-SiC-v0', num_atoms=num_atoms, reward_class=reward_class)
    model_path = os.path.join(rec_path, 'best_model.zip')

    if algorithm == "PPO":
        model = PPO.load(model_path)
    elif algorithm == "SAC":
        model = SAC.load(model_path)
    elif algorithm == "TD3":
        model = TD3.load(model_path)
    elif algorithm == "A2C":
        model = A2C.load(model_path)
    elif algorithm == "DDPG":
        model = DDPG.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    N = num_atoms
    EE_list = []
    SS_list = []
    CR_list = []
    CR_energy_list = []

    data = []


    print(f"\nDebugging {algorithm}:")
    for i in range(num_eval_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        info = {}
        while not done and not truncated:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
        E_pred = info['energy']
        s_pred = np.concatenate((np.array([obs[0], obs[0], obs[0], 90, 90, 90]), obs[1:]))
        EE = abs(E_pred - E_true) / N
        SS = compute_rmsd(s_pred, ['Si', 'Si', 'C', 'C'], s_true, ['Si', 'Si', 'C', 'C'])

        print(f"Episode {i + 1}: E_pred={E_pred:.4f}, E_true={E_true:.4f}, EE={EE:.4f}, SS={SS:.4f}")

        data.append({
            'Algorithm': algorithm,
            'Episode': i + 1,
            'E_pred': E_pred,
            'E_true': E_true,
            'EE': EE,
            'SS': SS
        })

        CR_list.append(1 if SS <= epsilon else 0)
        CR_energy_list.append(1 if EE <= epsilon_E else 0)

        EE_list.append(EE)
        SS_list.append(SS)


    avg_EE = np.mean(EE_list)
    avg_SS = np.mean(SS_list)
    CR = np.mean(CR_list)
    CR_energy = np.mean(CR_energy_list)

    monitor_path = os.path.join(rec_path, 'monitor.csv')
    if os.path.exists(monitor_path):
        df = pd.read_csv(monitor_path)
        training_time = df.iloc[-1, -1]
        CC = float(training_time) / 3600
    else:
        CC = float('nan')

    print(f"{algorithm} - Average EE: {avg_EE:.4f} eV/atom")
    print(f"{algorithm} - Average SS: {avg_SS:.4f} Å")
    print(f"{algorithm} - Convergence Rate (SS-based): {CR:.2f}")
    print(f"{algorithm} - Convergence Rate (Energy-based): {CR_energy:.2f}")
    print(f"{algorithm} - Computational Cost: {CC:.2f} GPU-hours")

    with open(os.path.join(rec_path, 'evaluation_metrics.txt'), 'w') as f:
        f.write(f"Average EE: {avg_EE:.4f} eV/atom\n")
        f.write(f"Average SS: {avg_SS:.4f} Å\n")
        f.write(f"Convergence Rate (SS-based): {CR:.2f}\n")
        f.write(f"Convergence Rate (Energy-based): {CR_energy:.2f}\n")
        f.write(f"Computational Cost: {CC:.2f} GPU-hours\n")

    return {'Algorithm': algorithm, 'EE': avg_EE, 'SS': avg_SS, 'CR': CR, 'CR_energy': CR_energy, 'CC': CC}, pd.DataFrame(data)


def evaluate_all_algorithms(model_paths, num_atoms, reward_class, s_true, E_true, num_eval_episodes=1000, epsilon=2.5,
                            epsilon_E=0.5, output_dir='evaluation_results'):
    algorithms = ['PPO', 'SAC', 'TD3', 'A2C', 'DDPG']
    results = []
    episode_results = pd.DataFrame()

    os.makedirs(output_dir, exist_ok=True)

    for algo in algorithms:
        if algo in model_paths:
            print(f"\nEvaluating {algo}...")


            result, episode_result = evaluate_pretrained_model(algo, model_paths[algo], num_atoms, reward_class, s_true, E_true,
                                               num_eval_episodes, epsilon, epsilon_E)
            results.append(result)
            episode_results = pd.concat([episode_results, episode_result], ignore_index=True)
        else:
            print(f"Skipping {algo}: Model path not provided.")

    df = pd.DataFrame(results)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    csv_path = os.path.join(output_dir, f'evaluation_summary_SiC_{current_time}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nEvaluation summary saved to {csv_path}")

    episode_csv_path = os.path.join(output_dir, f'episode_rec_SiC{current_time}.csv')
    episode_results.to_csv(episode_csv_path, index=False)
    return df

if __name__ == "__main__":
    s_true, E_true = find_reference_structure_with_lammps()

    model_paths = {
        'PPO': '/home/wangxiean/PycharmProjects/CSP-gym/results/SiC/PPO/20250327_132129',
        'SAC': '/home/wangxiean/PycharmProjects/CSP-gym/results/SiC/SAC/20250328_034706',
        'TD3': '/home/wangxiean/PycharmProjects/CSP-gym/results/SiC/TD3/20250327_234626',
        'A2C': '/home/wangxiean/PycharmProjects/CSP-gym/results/SiC/A2C/20250328_074902',
        'DDPG': '/home/wangxiean/PycharmProjects/CSP-gym/results/SiC/DDPG/20250328_083527'
    }

    evaluate_all_algorithms(model_paths, num_atoms=4, reward_class=CSPEnergyReward, s_true=s_true, E_true=E_true)