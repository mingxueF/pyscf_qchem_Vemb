"""Example on how to run a KS Embedding calculation.

Author: Cristina E. Gonzalez-Espinoza
Date: Sept. 2020

"""

import numpy as np

from pyscf import gto, scf, dft, mp
from pyscf.dft import gen_grid
from pyscf.dft.numint import eval_ao, eval_rho, eval_mat
from taco.embedding.pyscf_emb_pot import get_charges_and_coords
from taco.embedding.pyscf_wrap_single import get_density_from_dm, get_coulomb_repulsion
from functionals import _kinetic_ndsd2_potential, compute_kinetic_ndsd
from functionals import compute_ldacorr_pyscf
import os
from utilities import *
from prepol_rhoB import prepol_B


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
zr_file = find_file(root,extension="zr")
frags = zr_frag(zr_file)
# Define arguments
basis = '6-31+g*'
xc_code = 'LDA,VWN5'
plambda = 50
mola = gto.M(atom=frags["A"], basis = basis, charge=-1)
molb = gto.M(atom=frags["B"], basis=basis,verbose=3,charge=0)

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
system = gto.M(atom=newatom, basis=basis, charge=-1)
# Construct grid for complex
grids = gen_grid.Grids(system)
grids.level = 3
grids.build()
vnad_pot = compute_nad_potential(mola, dma, molb, dmb, grids.coords, xc_code)
ao_mola = eval_ao(mola, grids.coords, deriv=0)
rhoa_devs = get_density_from_dm(mola, dma, grids.coords, deriv=3, xctype='meta-GGA')
vemb_mat = eval_mat(mola, ao_mola, grids.weights, rhoa_devs[0], vnad_pot, xctype='LDA')
# Other integrals
v_coul = compute_coulomb_potential(mola, molb, dmb)
v_nuc0, v_nuc1 = compute_attraction_potential(mola, molb)
vemb_mat += v_coul + v_nuc0
#print("intergrated vemb over rhoA",1/2*np.einsum('ab,ba',vemb_mat,dma))
np.savetxt("FDE_vembmat.txt",vemb_mat,delimiter= '\n')
