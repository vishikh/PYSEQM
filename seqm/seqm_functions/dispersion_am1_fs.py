import os
import math
import torch
from .constants import a0


def dispersion_am1_fs1(mol, P):
    # Dispersion corrected AM1 method called AM1-FS1
    # Foster, Michael E., and Karl Sohlberg. "A new empirical correction to the AM1 method for macromolecular complexes." Journal of chemical theory and computation 6.7 (2010): 2153-2166.
    # https://doi.org/10.1021/ct100177u
    # There is a small discrpency in how the vdw radii are calculated as claimed by the authors in their paper and in their implementation. We follow their implementation
    # in order to match the results given in their paper.
    # The Fortran implementation of AM1-FS1 can be found in Appendix F of the PhD thesis of Michael E. Foster:
    # Foster, M. E. (2011). The development of an empirically corrected semi-empirical method and its
    # application to macromolecular complexes [Drexel University]. https://doi.org/10.17918/etd-3517
    # Downloaded On 2025/02/24 14:20:30 -0500

    # E_disp = \sum_ij (i<j) -C_6^ij/r^6*f_damp(r_ij)
    # where f_damp(r_ij) = 1/(1+exp(-d(r_ij/(S_R*R_vdw)-1)))
    # C_6^ij = sqrt(C_6^i C_6^j) and R_vdw = (R_i + R_j)

    # get C6 parameters
    # C6 is in J nm^6/mol, R_van is in Ang
    C6, R_van = get_c6_r0_params(mol, mol.seqm_parameters['elements'])
    C6ij = torch.sqrt(C6[mol.ni] * C6[mol.nj])
    # get van der Walls radii R_van
    R_vdw = (
        R_van[mol.ni] + R_van[mol.nj]
    )  # The original paper claims that R_vdw = 0.5*(Ri + Rj), but in their actual implementation R_vdw = Ri + Rj
    S_R = 1.1058892
    d = 1000.0

    eV_per_atom_per_Joul_per_mol = 1.036410e-5  # from the nctu.edu website's energy conversion table

    exp_arg = d * (a0 * mol.rij / (S_R * R_vdw) - 1.0)
    f_damp = 1.0 / (1.0 + torch.exp(-exp_arg))
    # The original AM1-FS1 code corrects the f_damp function for numerical extreme values
    d_tol = 12.0 * math.log(10.0)  # tolerance for the exponential argument for the damping function
    f_damp[exp_arg > d_tol] = 1.0
    f_damp[exp_arg < -d_tol] = 0.0

    E_disp_pair = -C6ij * torch.pow(a0 * mol.rij, -6.0) * f_damp  # J/mol nm^6/Ang^6

    E_disp = torch.zeros((mol.nmol, ), dtype=mol.rij.dtype, device=mol.rij.device)
    E_disp.index_add_(0, mol.pair_molid, E_disp_pair)
    E_disp = E_disp * eV_per_atom_per_Joul_per_mol * 1e6  # nm^6/Ang^6 = 1e6; now the energy is in eV/atom

    print(f'Dispersion correction to the total energy is {E_disp}')
    E_hbonding = hbond_correction_am1fs1(mol, P, R_van)
    E_disp = E_disp + E_hbonding

    return E_disp


def get_c6_r0_params(mol, elements):
    # Parameters taken from Grimme, S. Semiempirical GGA-Type Density Functional Constructed with a Long-Range Dispersion Correction. J. Com- put. Chem. 2006, 27, 1787–1799.
    file_path = os.path.join(os.path.dirname(__file__), "../params/grimme_2006_b97-d.csv")

    m = max(elements)
    C_6 = torch.zeros(m + 1, device=mol.rij.device)  # m+1 because indexing starts from 1 for atomic number
    R_0 = torch.zeros(m + 1, device=mol.rij.device)

    # Open file and read line by line
    with open(file_path, "r") as f:
        _ = f.readline()  # Read the header line

        for line in f:
            values = line.strip().replace(' ', '').split(",")  # Split CSV row
            at_no = int(values[0])  # Convert at_no to int

            if at_no in elements:  # Check if at_no is in the target set
                C_6[at_no] = float(values[2])  # Store C6 directly
                R_0[at_no] = float(values[3])  # Store R0 directly

    return C_6, R_0


def hbond_correction_am1fs1(mol, P, R_van):
    # Works only for a single molecule.
    if mol.nmol > 1:
        raise Exception("Hbonding correction currently implemented for one molecule only")

    species = mol.species[0]
    coords = mol.coordinates[0]

    # Step 1. Identify hydrogen and heavy atoms (O, N, F)
    hydrogen_mask = (species == 1)
    heavy_mask = (species == 7) | (species == 8) | (species == 9)
    hydrogen_indices = torch.where(hydrogen_mask)[0]
    heavy_indices = torch.where(heavy_mask)[0]

    # Step 2. For each hydrogen, find its nearest neighbor among all atoms.
    hydrogen_coords = coords[hydrogen_indices]  # shape: (n_H, 3)
    distances = torch.cdist(hydrogen_coords, coords)  # shape: (n_H, N)

    # Exclude self-distance in a vectorized way:
    rows = torch.arange(hydrogen_coords.shape[0])
    distances[rows, hydrogen_indices] = float('inf')

    # For each hydrogen, get the index of its nearest neighbor
    _, nn_indices = distances.min(dim=1)
    nn_species = species[nn_indices]
    # Select hydrogens whose nearest neighbor is O, N, or F.
    sel_mask = (nn_species == 7) | (nn_species == 8) | (nn_species == 9)
    selected_hydrogen_indices = hydrogen_indices[sel_mask]
    selected_nn_indices = nn_indices[sel_mask]

    # Step 3. Compute connecting vectors: from hydrogen to its nearest heavy neighbor.
    vector_H_to_nn = coords[selected_nn_indices] - coords[selected_hydrogen_indices]

    # Step 4. Compute vectors from each selected hydrogen to all heavy atoms.
    sel_h_coords = coords[selected_hydrogen_indices]  # shape: (n_sel, 3)
    heavy_coords = coords[heavy_indices]  # shape: (n_heavy, 3)
    all_vectors = heavy_coords.unsqueeze(0) - sel_h_coords.unsqueeze(1)  # shape: (n_sel, n_heavy, 3)

    # Exclude the nearest neighbor from each list.
    # Create a mask comparing heavy_indices (broadcasted) with selected_nn_indices.
    mask_exclude_nn = heavy_indices.unsqueeze(0) != selected_nn_indices.unsqueeze(1)
    all_vectors = all_vectors[mask_exclude_nn].view(sel_h_coords.shape[0], -1, 3)
    # Record the corresponding heavy atom indices.
    heavy_excl_nn = heavy_indices.unsqueeze(0).expand(sel_h_coords.shape[0], -1)
    heavy_excl_nn = heavy_excl_nn[mask_exclude_nn].view(sel_h_coords.shape[0], -1)

    # Step 5. Compute cosine similarities between each hydrogen's nearest neighbor vector and
    # its vectors to the other heavy atoms.
    nn_norm = vector_H_to_nn / vector_H_to_nn.norm(dim=-1, keepdim=True)
    other_norm = all_vectors / all_vectors.norm(dim=-1, keepdim=True)
    cos_angles = (nn_norm.unsqueeze(1) * other_norm).sum(dim=-1)

    # Filter for angles greater than 90° (cosine < 0)
    angle_mask = cos_angles < 0

    # Flatten filtered heavy indices and vectors.
    filtered_heavy_indices = heavy_excl_nn[angle_mask]

    # Also retrieve the corresponding hydrogen indices.
    h_expanded = selected_hydrogen_indices.unsqueeze(1).expand_as(heavy_excl_nn)
    filtered_hydrogen_indices = h_expanded[angle_mask]

    # Step 6. Compute distances between the corresponding hydrogen and heavy atoms.
    h_coords_filtered = coords[filtered_hydrogen_indices]
    heavy_coords_filtered = coords[filtered_heavy_indices]
    pair_vectors = heavy_coords_filtered - h_coords_filtered
    rij = pair_vectors.norm(dim=1)

    # Get atomic charges
    atomic_charge = mol.const.tore[species] - P.diagonal(dim1=1, dim2=2).reshape(mol.molsize, -1).sum(dim=1)

    bohr_to_ang = 0.52917724924
    ha_to_eV = 27.2114079527
    alpha1, alpha2, alpha3, alpha4 = 0.4882, 0.6211, 0.3344, 1.5451

    # Determine the species for the heavy atoms in the filtered list.
    heavy_species = species[filtered_heavy_indices]

    # Compute van der Waals radius (R_vdw) for heavy atoms and hydrogen pairs. This is multiplied by 2 because each of the R_van have to be multiplied by 2 (idk why!)
    R_vdw = 2.0 * (R_van[heavy_species]**3 + R_van[1].unsqueeze(0)**3) / (R_van[heavy_species]**2 +
                                                                          R_van[1].unsqueeze(0)**2)

    # Compute the damping function.
    f_damp = torch.exp(-((rij - alpha2 * R_vdw)**2) / (alpha3 * (bohr_to_ang + alpha4 * (rij - alpha2 * R_vdw)))**2)

    # Calculate the hydrogen bonding correction energy.
    E_hbonding = ha_to_eV * alpha1 * torch.sum(
        atomic_charge[filtered_heavy_indices] * atomic_charge[filtered_hydrogen_indices] / rij * bohr_to_ang *
        (cos_angles[angle_mask]**2) * f_damp)

    return E_hbonding
