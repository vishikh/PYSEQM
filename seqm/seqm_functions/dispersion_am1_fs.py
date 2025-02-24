import os
import math
import torch
from .constants import a0

def dispersion_am1_fs1(mol):
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
    C6, R_van = get_c6_r0_params(mol,mol.seqm_parameters['elements'])
    C6ij = torch.sqrt(C6[mol.ni] * C6[mol.nj])
    # get van der Walls radii R_van
    R_vdw = (R_van[mol.ni] + R_van[mol.nj]) # The original paper claims that R_vdw = 0.5*(Ri + Rj), but in their actual implementation R_vdw = Ri + Rj
    S_R = 1.1058892 
    d = 1000.0

    eV_per_atom_per_Joul_per_mol = 1.036410e-5 # from the nctu.edu website's energy conversion table

    exp_arg = d*(a0*mol.rij/(S_R*R_vdw)-1.0)
    f_damp = 1.0/(1.0+torch.exp(-exp_arg))
    # The original AM1-FS1 code corrects the f_damp function for numerical extreme values
    d_tol = 12.0*math.log(10.0) # tolerance for the exponential argument for the damping function
    f_damp[exp_arg > d_tol] = 1.0
    f_damp[exp_arg < -d_tol] = 0.0

    E_disp_pair = -C6ij*torch.pow(a0*mol.rij,-6.0)*f_damp # J/mol nm^6/Ang^6

    E_disp = torch.zeros((mol.nmol,),dtype=mol.rij.dtype, device=mol.rij.device)
    E_disp.index_add_(0,mol.pair_molid, E_disp_pair)
    E_disp = E_disp * eV_per_atom_per_Joul_per_mol * 1e6  # nm^6/Ang^6 = 1e6; now the energy is in eV/atom

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
