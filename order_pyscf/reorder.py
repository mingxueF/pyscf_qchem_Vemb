import numpy as np

def get_order_lists(atoms, basis_dict):
    """Get list of orders for matrix re-ordering.

    Parameters
    ----------
    atoms : np.darray(int)
        Atoms in molecule/fragment.
    basis_dict : dict
        Known orders for each row/group in the periodic table.

    Returns
    -------
    orders : list[list[],]
        List with order for each atom.
    """
    if not isinstance(atoms, np.ndarray):
        raise TypeError("`atoms` must be provided in a np.darray.")
    if atoms.dtype != int:
        raise NotImplementedError('For now, atomic numbers are accepted only.')
    orders = []
    for atom in atoms:
        if atom < 3:
            orders.append(basis_dict['first'])
        elif 2 < atom < 11:
            orders.append(basis_dict['second'])
        elif 11 < atom < 19:
            orders.append(basis_dict['third'])
        else:
            raise NotImplementedError('At the moment only first and second row elements are available.')
    return orders



def transform(inparr, natoms, orders):
    """Perform symmetric transformation of a matrix.
    The symmetric transformation of a matrix :math:`\\mathbf{X}` uses a
    rearranged identity matrix :math:`\\mathbf{P}` such that the working
    equation is:
    :math:`\\mathbf{P}^{T}\\cdot \\mathbf{X} \\cdot\\mathbf{P}`

    Parameters:
    -----------
    inparr : np.ndarray
        Input matrix :math:`\\mathbf{X}`.

    Returns:
    --------
    Q : np.ndarray
        Transformed matrix according to target format.

    """
    # -------------------
    # Q = P.T * X * P
    # -------------------
    sort_list = get_sort_list(natoms, orders)
    nAO = inparr.shape[0]
    idarr = np.identity(nAO)
    # Do transformation
    P = idarr[:, sort_list]  # black magic: rearrange columns of ID matrix
    M = np.dot(P.T, inparr)
    Q = np.dot(M, P)
    return Q

def reorder_matrix(inmat, inprog, outprog, basis, atoms, jsonfn):
    """Re-order matrix to fit some other program format.

    Parameters
    ---------
    inmat : np.ndarray((n,n))
        Square symmetric matrix to be re-ordered.
    inprog, outprog :  str
        Name of the programs to connect, all lowercase.
    basis : str
        Basis set name.
    atoms : np.ndarray
        Atomic numbers of molecule/fragment.
    jsonfn : path to json permutation list file

    Returns
    -------
    ordered : np.ndarray((n,n))
        Re-ordered square symmetric matrix.
    """
    if not isinstance(inmat, np.ndarray):
        raise TypeError("`inmat` must be a np.ndarray object.")
    if not isinstance(inprog, str):
        raise TypeError("`inprog` must be a string.")
    if not isinstance(outprog, str):
        raise TypeError("`outprog` must be a string.")
    if not isinstance(basis, str):
        raise TypeError("`basis` must be a string.")
    if atoms.dtype != int:
        raise TypeError("`atoms` must be an array with integer numbers.")
    # Get data from json file in data folder
    with open(jsonfn, 'r') as jf:
        formatdata = json.load(jf)
    natoms = len(atoms)
    transkey = inprog+'2'+outprog
    if not formatdata[transkey]:
        raise KeyError("No translation information available for %s SCF program." % inprog)
    # check that there is info for the basis requested
    if not formatdata[transkey][basis]:
        raise KeyError("The information for %s basis is missing." % basis)
    orders = get_order_lists(atoms, formatdata[transkey][basis])
    ordered = transform(inmat, natoms, orders)
    return ordered
