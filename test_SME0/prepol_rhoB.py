from pyscf import qmmm, scf
from pyscf import gto, scf, tools, lib
import numpy as np
from utilities import *
import os

def prepol_B(frags, basis):
    scn = gto.M(
        atom=frags["A"],
        basis=basis,
        charge=-1,
        unit="Angstrom",
    )
    
    mf = scf.RHF(scn).run()          # ground-state HF wave-function
    
    # --- CHELPG fit (PySCF ≥2.3 has the ESP-fitting helpers) ---
    #xyz_grid, esp = tools.esp.get_esp(scn, mf.make_rdm1(), grid_spacing=0.25)
    #chelpg = tools.esp.fit_charges(
    #    scn, xyz_grid, esp, method="CHELPG"
    #)                                 # returns per-atom charges
    
    #q_S, q_C, q_N = -0.7046,0.3730,-0.6684           # save these three numbers
    #coor_S, coor_C, coor_N = scn.atom_coords(unit="Angstrom")
    #
    #
    ## coordinates (Å) and CHELPG charges we just computed
    #coords = np.vstack([coor_S, coor_C, coor_N])   # shape (3,3)
    #charges = np.array([q_S, q_C, q_N])            # S, C, N charges
    # Extract all atomic coordinates at once
    coords = scn.atom_coords(unit="Angstrom")  # shape (3,3)
    
    # Assign CHELPG charges manually (must match atom order!)
    charges = np.array([-0.7046, 0.3730, -0.6684])
    wat = gto.M(
        atom=frags["B"],
        basis=basis,
        charge=0,
        unit="Angstrom",
    )
    mf = scf.RHF(wat)                              # or scf.ROHF/DFT…
    mf = qmmm.mm_charge(mf, coords, charges, unit="Angstrom")
    
    mf.conv_tol = 1e-9      # tighten if you need highly converged ρ_B
    mf.kernel()               # <-- this gives you the density ρ_B
    rho_B = mf.make_rdm1()    # 1-body density matrix
    np.savetxt("rho_B.txt",rho_B,delimiter= '\n')

    return rho_B

if __name__ == "__main__":
# --- geometry of the isolated anion (Å) ---
    root = os.getcwd()
    zr_file = find_file(root,extension="zr")
    frags = zr_frag(zr_file)
    # Define arguments
    basis = '6-31+g*'
    prepol_B(frags, basis)
