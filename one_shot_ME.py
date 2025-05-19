"""Example on how to run a KS Embedding calculation.

Author: Cristina E. Gonzalez-Espinoza
Date: Sept. 2020

"""

import numpy as np
from pyscf import gto, scf, dft, mp
from pyscf.dft import gen_grid
import os
from utilities import *
from prepol_rhoB import prepol_B
from vemb import eval_vemb_ao
#############################################################
# Define Molecules of each fragment
#############################################################
root = os.getcwd()
zr_file = find_file(root,extension="zr")
frags = zr_frag(zr_file)
# Define arguments
basis = '6-31+g*'
xc_code = 'LDA,VWN5'
plambda = 50
charge_a = -1
charge_b = 0
mola = gto.M(atom=frags["A"], basis = basis, charge=charge_a)
molb = gto.M(atom=frags["B"], basis=basis,verbose=3,charge=charge_b)
#############################################################
# Get reference densities
#############################################################
# TIP: For HF you only need: scfres1 = scf.RHF(mol)
# Fragment A
scfres = scf.RHF(mola)
scfres.conv_tol = 1e-9
scfres.kernel()
dma = scfres.make_rdm1()
#fragA_mp2 = mp.MP2(scfres).run()
#dma_mp1 = fragA_mp2.make_rdm1(ao_repr = True)
# Fragment B
#scfres1 = scf.RHF(molb)
print("getting rho_B---")
#scfres1 = scf.addons.remove_linear_dep_(scfres1).run()
dmb = prepol_B(frags, basis)
print("rho_b fini")
#############################################################
# Make embedding potential 
#############################################################
# Construct grid for integration
# This could be any grid, but we use the Becke 
# Create supersystem
newatom = '\n'.join([mola.atom, molb.atom])
system = gto.M(atom=newatom, basis=basis, charge=charge_a+charge_b)
# Construct grid for complex
grids = gen_grid.Grids(system)
grids.level = 3
grids.build()
# call here vemb by the request functional
functional = "NDCS"
vemb_mat = eval_vemb_ao(mola, dma, molb, dmb, grids, xc_code, functional) 
np.savetxt("FDE_vembmat.txt",vemb_mat,delimiter= '\n')
atoms = []
for i in range(mola.natm):
    atoms.append(int(mola._atm[i][0])) #check mol._atm
atoms = np.array(atoms,dtype=int)
print(atoms)
from order_pyscf.reorder import reorder_matrix
vemb = reorder_matrix(vemb_mat,'pyscf','qchem',basis, atoms,"./order_pyscf/translation.json")
np.savetxt("FDE_vembmat.txt",vemb,delimiter='\n')
