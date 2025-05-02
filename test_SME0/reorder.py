"""Example to generate a embedding potential.

Author: Cristina E. Gonzalez-Espinoza
Date: Sept. 2020

"""

import numpy as np

from pyscf import gto, scf, dft, mp
from pyscf.dft import gen_grid
from pyscf.dft.numint import eval_ao, eval_rho, eval_mat
from taco.embedding.pyscf_emb_pot import get_charges_and_coords
from taco.embedding.pyscf_wrap_single import get_density_from_dm
from taco.translate.tools import reorder_matrix
from functionals import _kinetic_ndsd2_potential
from functionals import compute_ldacorr_pyscf
import os
import CCJob as ccj


def replace_char(list_obj):
    for i in list_obj:
        if i[0] == "O":
            i[0] = "X0"
        if i[0] == "H":
            i[0] = "X1"
        elif i[0] == "C":
            i[0] = "X2"
    return list_obj

def zr_pyscf(fname):
    with open(fname) as f:
        rl = f.readlines()
    line_s = 0
    frags = {}
    for i, line in enumerate(rl):
        if "----" in line:
            line_s = i
    frags["A"] = list(map(str.split, rl[0:line_s]))
    frags["B"] = list(map(str.split, rl[line_s+1:]))
    frags["AB"] = frags["A"] + frags["B"]
    frags["BA"] = frags["B"] + frags["A"]
    frags_Bgh = replace_char(list(map(str.split, rl[line_s+1:])))
    frags_Agh = replace_char(list(map(str.split, rl[0:line_s])))
    frags["AB_ghost"] = frags["A"] + frags_Bgh#list(map(lambda x: x[0] = "X0" if x[0] == "O" else x[0] = "X1",frags["B"]))
    frags["BA_ghost"] = frags["B"] + frags_Agh#list(map(lambda x: x[0] = "X0" if x[0] == "O" else x[0] = "X1",frags["B"]))
    for key in frags:
        frags[key] = "\n".join(["   ".join(s) for s in frags[key]])
    return frags
def compute_attraction_potential(mol0, mol1):
    """Compute the nuclei-electron attraction potentials.

    Returns
    -------
    v0nuc1 : np.ndarray(NAO,NAO)
        Attraction potential between electron density of fragment0
        and nuclei of fragment1.
    v1nuc0 : np.ndarray(NAO,NAO)
        Attraction potential between electron density of fragment0
        and nuclei of fragment1.

    """
    # Nuclear-electron attraction integrals
    mol0_charges, mol0_coords = get_charges_and_coords(mol0)
    mol1_charges, mol1_coords = get_charges_and_coords(mol1)
    v0_nuc1 = 0
    for i, q in enumerate(mol1_charges):
        mol0.set_rinv_origin(mol1_coords[i])
        v0_nuc1 += mol0.intor('int1e_rinv') * -q
    v1_nuc0 = 0
    for i, q in enumerate(mol0_charges):
        mol1.set_rinv_origin(mol0_coords[i])
        v1_nuc0 += mol1.intor('int1e_rinv') * -q
    return v0_nuc1, v1_nuc0


def compute_coulomb_potential(mol0, mol1, dm1):
    """Compute the electron-electron repulsion potential.

    Returns
    -------
    v_coulomb : np.ndarray(NAO,NAO)
        Coulomb repulsion potential.

    """
    mol1234 = mol1 + mol1 + mol0 + mol0
    shls_slice = (0, mol1.nbas,
                  mol1.nbas, mol1.nbas+mol1.nbas,
                  mol1.nbas+mol1.nbas, mol1.nbas+mol1.nbas+mol0.nbas,
                  mol1.nbas+mol1.nbas+mol0.nbas, mol1234.nbas)
    eris = mol1234.intor('int2e', shls_slice=shls_slice)
    v_coulomb = np.einsum('ab,abcd->cd', dm1, eris)
    return v_coulomb


def compute_nad_potential(mola, dma, molb, dmb, points, xc_code, plambda=50):
    # Evaluate electron densities and derivatives
    # rho[0] = rho
    # rho[1-3] = gradxyz
    # rho[4] = lap. rho
    rhoa_devs = get_density_from_dm(mola, dma, points, deriv=3, xctype='meta-GGA')
    rhob_devs = get_density_from_dm(molb, dmb, points, deriv=3, xctype='meta-GGA')
    # DFT nad potential
    rho_tot = rhoa_devs[0] + rhob_devs[0]
    # XC term
    exc_tot, vxc_tot = compute_ldacorr_pyscf(rho_tot, xc_code)
    exc_a, vxc_a = compute_ldacorr_pyscf(rhoa_devs[0], xc_code)
    exc_b, vxc_b = compute_ldacorr_pyscf(rhob_devs[0], xc_code)
    vxc_nad = vxc_tot - vxc_a
    # Ts term
    vts_nad = _kinetic_ndsd2_potential(rhoa_devs, rhob_devs)
#   # TF kitetic term
#   ets_tot, vts_tot = compute_ldacorr_pyscf(rho_tot, xc_code='LDA_K_TF')
#   ets_a, vts_a = compute_ldacorr_pyscf(rhoa_devs[0], xc_code='LDA_K_TF')
#   ets_b, vts_b = compute_ldacorr_pyscf(rhob_devs[0], xc_code='LDA_K_TF')
#   vts_nad = vts_tot - vts_a

    vnad_tot = vxc_nad + vts_nad
    return vnad_tot


#############################################################
# Define Molecules of each fragment
#############################################################
root = os.getcwd()
frags = zr_pyscf("d2h.zr")
# Define arguments
#with open('/home/fum/distance/c2h4_h2o/actz/grid6/basis/aug-cc-pVDZ.nwchem', 'r') as bfile:
#    ibasis = bfile.read()
basis = 'aug-cc-pvdz'

xc_code = 'LDA,VWN5'
mola = gto.M(atom=frags["A"], basis=basis, verbose=3)
molb= gto.M(atom=frags["B"], basis=basis, verbose=3)
nao_mol0 = mola.nao_nr()
nao_mol1 = molb.nao_nr()
print("number of AO  A:",nao_mol0,"number of AO B:",nao_mol1)
# import dm
dma = np.loadtxt("FDE_vembmat.txt")
dma = dma.reshape(nao_mol0,nao_mol0)
atoms0 = []
for i in range(mola.natm):
    atoms0.append(int(mola._atm[i][0])) #check mol._atm
print(atoms0)
atoms0 = np.array(atoms0, dtype=int)
dma = reorder_matrix(dma,'pyscf','qchem','aug-cc-pVDZ',atoms0)
np.savetxt("FDE_vembmat.txt",dma,delimiter='\n')
