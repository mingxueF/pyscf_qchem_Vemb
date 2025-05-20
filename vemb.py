
import numpy as np
from pyscf import gto, scf, dft, mp
from pyscf.dft import gen_grid
from pyscf.dft.numint import eval_ao, eval_rho, eval_mat
from functionals import _kinetic_ndsd2_potential
from functionals import compute_ldacorr_pyscf

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


def compute_nad_potential(mola, dma, molb, dmb, points, xc_code, functional):
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
    if functional.lower() not in {"tf", "ndcs","lda"}:
        raise NotImplementedError(
            f"functional '{functional}' is not implemented; use 'tf' or 'ndcs'."
        )
    if functional.lower() == "ndcs":
        # Ts term
        vts_nad = _kinetic_ndsd2_potential(rhoa_devs, rhob_devs)
    elif functional.lower() == "tf" or "lda":
        # TF kitetic term
        ets_tot, vts_tot = compute_ldacorr_pyscf(rho_tot, xc_code='LDA_K_TF')
        ets_a, vts_a = compute_ldacorr_pyscf(rhoa_devs[0], xc_code='LDA_K_TF')
        ets_b, vts_b = compute_ldacorr_pyscf(rhob_devs[0], xc_code='LDA_K_TF')
        vts_nad = vts_tot - vts_a

    vnad_tot = vxc_nad + vts_nad

    return vnad_tot

def get_charges_and_coords(mol): 
    """Return arrays with charges and coordinates.""" 
    bohr2a = 0.529177210903 
    coords = [] 
    charges = [] 
    for i in range(mol.natm): 
        # Avoid ghost atoms 
        if mol._atm[i][0] == 0: 
            next 
        else: 
            if isinstance(mol.atom, str): 
                atm_str = mol.atom.split() 
                if mol.unit == 'Bohr': 
                    tmp = [float(f) for f in atm_str[i*4+1:(i*4)+4]] 
                else: 
                    tmp = [float(f)/bohr2a for f in atm_str[i*4+1:(i*4)+4]] 
            else: 
                if mol.unit == 'Bohr': 
                    tmp = [mol.atom[i][1][j] for j in range(3)] 
                else: 
                    tmp = [mol.atom[i][1][j]/bohr2a for j in range(3)] 
            coords.append(tmp) 
            charges.append(mol._atm[i][0])
    coords = np.array(coords)
    charges = np.array(charges, dtype=int)
    return charges, coords

def get_density_from_dm(mol, dm, points, xctype='LDA', deriv=0):
    """Compute density on a grid from the density matrix.

    Parameters
    ----------
    mol : gto.M
        Molecule PySCF object.
    dm : np.ndarray
        Density matrix corresponding to mol.
    points : np.ndarray
        Grid points where the density is evaluated.
    xctype : str
        Type of functional it's used later. This defines
        how many derivatives or rho will be computed.
    deriv : int
        Number of derivatives needed for orbitals and density.
    
    Returns
    ------- 
    rho : np.ndarray(npoints, dtype=float)
        Density on a grid.

    """
    ao_mol = eval_ao(mol, points, deriv=deriv)
    rho = eval_rho(mol, ao_mol, dm, xctype=xctype)
    return rho

def eval_vemb_ao(mola, dma, molb, dmb, grids, xc_code, functional):
    "evaluate embedding potentail matrix in AO basis"
    vnad_pot = compute_nad_potential(mola, dma, molb, dmb, grids.coords, xc_code,functional)
    ao_mola = eval_ao(mola, grids.coords, deriv=0)
    rhoa_devs = get_density_from_dm(mola, dma, grids.coords, deriv=3, xctype='meta-GGA')
    vemb_mat = eval_mat(mola, ao_mola, grids.weights, rhoa_devs[0], vnad_pot, xctype='LDA')
    # Other integrals
    v_coul = compute_coulomb_potential(mola, molb, dmb)
    v_nuc0, v_nuc1 = compute_attraction_potential(mola, molb)
    vemb_mat += v_coul + v_nuc0

    return vemb_mat
