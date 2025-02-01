import os
import torch
from .constants import a0

def dispersion_am1_fs1(mol):

    # E_disp = \sum_ij (i<j) -C_6^ij/r^6*f_damp(r_ij)
    # where f_damp(r_ij) = 1/(1+exp(-d(r_ij/(S_R*R_vdw)-1)))
    # C_6^ij = sqrt(C_6^i C_6^j) and R_vdw = (R_i + R_j)/2

    # get C6 parameters
    C6, R_van = get_c6_r0_params(mol,mol.seqm_parameters['elements'])
    C6ij = torch.sqrt(C6[mol.ni] * C6[mol.nj])
    # get van der Walls radii R_van
    R_vdw = 0.5 * (R_van[mol.ni] + R_van[mol.nj])
    S_R = 1.1059 # TODO: make sure it was a misprint in the original paper and the authors did not mean to say S_6 = 1.1059
    d = 1000.0

    # C6 is in J nm^6/mol
    eV_per_atom_per_Joul_per_mol = 1.036410e-5 # from the nctu.edu website's energy conversion table
    # C6ij = C6ij * eV_per_atom_per_Joul_per_mol * 1e-6  # eV/Ang^6/atom
    f_damp = 1.0/(1.0+torch.exp(-d*(a0*mol.rij/(S_R*R_vdw)-1.0)))
    E_disp_pair = -C6ij*torch.pow(a0*mol.rij,-6.0)*f_damp
    E_disp = torch.zeros((mol.nmol,),dtype=mol.rij.dtype, device=mol.rij.device)
    E_disp.index_add_(0,mol.pair_molid, E_disp_pair)
    E_disp = E_disp * eV_per_atom_per_Joul_per_mol * 1e6  # eV/atom Ang^6
    print(f'Dispersion correction to the total energy is {E_disp}')

    return E_disp


def get_c6_r0_params(mol,elements):
    # Parameters taken from Grimme, S. Semiempirical GGA-Type Density Functional Constructed with a Long-Range Dispersion Correction. J. Com- put. Chem. 2006, 27, 1787–1799.
    file_path = os.path.join(os.path.dirname(__file__), "../params/grimme_2006_b97-d.csv")

    m = max(elements)
    C_6 = torch.zeros(m+1, device=mol.rij.device) # m+1 because indexing starts from 1 for atomic number
    R_0 = torch.zeros(m+1, device=mol.rij.device)

    # Open file and read line by line
    with open(file_path, "r") as f:
        _ = f.readline() # Read the header line

        for line in f:
            values = line.strip().replace(' ', '').split(",")  # Split CSV row
            at_no = int(values[0])   # Convert at_no to int
            
            if at_no in elements:  # Check if at_no is in the target set
                C_6[at_no] = float(values[2])  # Store C6 directly
                R_0[at_no] = float(values[3])  # Store R0 directly

    return C_6, R_0
