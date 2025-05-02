def pyscf_to_qchem_permutation(pyscf_labels):
    """
    Given a list of AO labels from PySCF for a single atom, return the permutation
    that reorders them into Q-Chem AO order.

    Args:
        pyscf_labels (List[str]): List of AO labels like '1s', '2px', '3dxy' for a single atom

    Returns:
        List[int]: Indices that map PySCF AO ordering to Q-Chem AO ordering
    """
    import re

    # Helper to parse AO label like '3dxy' → (n=3, l='d', m='xy')
    def parse_label(label):
        match = re.match(r'(\d)([spdf])([xyz]*)', label)
        if match:
            n, l, m = match.groups()
            return int(n), l, m
        else:
            raise ValueError(f"Unrecognized AO label: {label}")

    # Q-Chem prefers px, py, pz (same as PySCF for Cartesian), but:
    # For d orbitals in Cartesian, Q-Chem prefers: dxx, dxy, dyy, dxz, dyz, dzz
    d_order_qchem = ['xx', 'xy', 'yy', 'xz', 'yz', 'zz']

    # Group by shell: s, p, d, f by principal quantum number
    shells = {'s': {}, 'p': {}, 'd': {}, 'f': {}}

    for i, label in enumerate(pyscf_labels):
        n, l, m = parse_label(label)
        if n not in shells[l]:
            shells[l][n] = []
        shells[l][n].append((m, i))  # store m-value and index

    # Construct Q-Chem order
    permutation = []

    # Q-Chem groups shells like: s(1s), s(2s), p(2p), s(3s), p(3p), ...
    for n in sorted(set(k for l in shells for k in shells[l])):
        # s-functions (1 per shell)
        if n in shells['s']:
            indices = [idx for _, idx in sorted(shells['s'][n])]
            permutation.extend(indices)

        # p-functions (px, py, pz)
        if n in shells['p']:
            p_order = ['x', 'y', 'z']
            p_map = {m: idx for m, idx in shells['p'][n]}
            permutation.extend([p_map[axis] for axis in p_order])

        # d-functions (Cartesian)
        if n in shells['d']:
            d_map = {m: idx for m, idx in shells['d'][n]}
            permutation.extend([d_map[m] for m in d_order_qchem if m in d_map])

    return permutation

if __name__ == "__main__":
    pyscf_labels = [
        '1s', '2s', '3s', '4s', '5s',
        '2px', '2py', '2pz',
        '3px', '3py', '3pz',
        '4px', '4py', '4pz',
        '5px', '5py', '5pz',
        '3dxx', '3dxy', '3dxz', '3dyy', '3dyz', '3dzz'
    ]
    
    perm = pyscf_to_qchem_permutation(pyscf_labels)
    print(perm)
