import os
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from lammps import lammps

def find_reference_structure_with_lammps():
    lmp = lammps()

    commands = [
        "units metal",  # Å, eV
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

    E_true = lmp.get_thermo("pe")

    lx = lmp.get_thermo("lx")
    ly = lmp.get_thermo("ly")
    lz = lmp.get_thermo("lz")
    xy = lmp.get_thermo("xy")
    xz = lmp.get_thermo("xz")
    yz = lmp.get_thermo("yz")
    a, b, c, alpha, beta, gamma = compute_lattice_params(lx, ly, lz, xy, xz, yz)

    natoms = lmp.get_natoms()
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

def compute_lattice_params(lx, ly, lz, xy, xz, yz):
    a = lx
    b = np.sqrt(ly ** 2 + xy ** 2)
    c = np.sqrt(lz ** 2 + xz ** 2 + yz ** 2)
    gamma = np.arccos(xy / b) * 180 / np.pi if b != 0 else 90.0
    beta = np.arccos(xz / c) * 180 / np.pi if c != 0 else 90.0
    alpha = np.arccos(yz / c) * 180 / np.pi if c != 0 else 90.0
    return a, b, c, alpha, beta, gamma


def extract_structure_from_cif(cif_file):
    parser = CifParser(cif_file)
    structure = parser.parse_structures(primitive=False)[0]  # 获取第一个结构
    lattice = structure.lattice
    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
    frac_coords = np.array([site.frac_coords for site in structure.sites]).flatten()
    s_pred = np.concatenate(([a, b, c, alpha, beta, gamma], frac_coords))
    num_atoms = len(structure.sites)
    species = [site.specie.symbol for site in structure.sites]
    return s_pred, num_atoms, species

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


def compute_energy_custom_positions(positions, lattice_params):
    """
    计算给定原子位置和晶格参数的能量。

    参数:
    positions (np.ndarray): 形状为 (4, 3) 的原子坐标数组，表示 4 个原子的 x, y, z 坐标。
    lattice_params (tuple): (a, b, c, alpha, beta, gamma)，晶格参数，单位为 Å 和度。

    返回:
    float: 体系的势能 (eV)。
    """
    positions = np.array(positions)
    if positions.shape != (4, 3):
        raise ValueError("必须提供 4 个原子的 3D 坐标，形状应为 (4, 3)")

    a, b, c, alpha, beta, gamma = lattice_params
    if not (0 < alpha < 180 and 0 < beta < 180 and 0 < gamma < 180):
        raise ValueError("晶格角度 α, β, γ 必须在 (0, 180) 度之间")

    lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])

    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    gamma_rad = np.deg2rad(gamma)

    lx = a
    xy = b * np.cos(gamma_rad)
    xz = c * np.cos(beta_rad)
    ly = np.sqrt(b ** 2 - xy ** 2)
    yz = (b * c * np.cos(alpha_rad) - xy * xz) / ly
    lz = np.sqrt(c ** 2 - xz ** 2 - yz ** 2)

    lammps_script = f"""
    units metal
    atom_style atomic
    dimension 3
    boundary p p p
    region box prism 0 {lx} 0 {ly} 0 {lz} {xy} {xz} {yz} units box
    create_box 2 box
    """

    for line in lammps_script.strip().split('\n'):
        lmp.command(line.strip())

    for i, (x, y, z) in enumerate(positions):
        atom_type = 1 if i < 2 else 2
        lmp.command(f"create_atoms {atom_type} single {x} {y} {z} units box")

    setup_commands = """
    mass 1 28.0855
    mass 2 12.0107
    pair_style tersoff
    pair_coeff * * potentials/SiC.tersoff Si C
    minimize 1.0e-8 1.0e-10 1000 10000
    """
    for line in setup_commands.strip().split('\n'):
        lmp.command(line.strip())

    energy = lmp.get_thermo("pe")
    lmp.close()

    return energy

def evaluate_cif_with_lammps_reference(cif_file):
    s_true, E_true = find_reference_structure_with_lammps()

    s_pred, _, species = extract_structure_from_cif(cif_file)

    SS = compute_rmsd(s_pred, species, s_true, ['Si', 'Si', 'C', 'C'])

    E_pred = compute_energy_custom_positions(s_pred[6:].reshape(4, 3), s_pred[:6])
    EE = abs(E_pred - E_true) / 4

    print(f"\nEvaluation Results for CIF File: {cif_file}")
    print(f"Predicted Energy (E_pred): {E_pred:.4f} eV")
    print(f"Reference Energy (E_true): {E_true:.4f} eV")
    print(f"Energy Error (EE): {EE:.4f} eV/atom")
    print(f"Structure Shift (SS): {SS:.4f} Å")

    return {"EE": EE, "SS": SS}


# example
if __name__ == "__main__":
    cif_file = "mattergen_Si2C2.cif"

    system_type = "SiC"

    metrics = evaluate_cif_with_lammps_reference(cif_file)