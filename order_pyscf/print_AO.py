"""Example on how to run a KS Embedding calculation.

Author: Cristina E. Gonzalez-Espinoza
Date: Sept. 2020

"""

import numpy as np
from pyscf import gto
import os
import glob

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
    frags_Bgh = replace_char(list(map(str.split, rl[line_s+1:])))
    frags_Agh = replace_char(list(map(str.split, rl[0:line_s])))
    frags["AB_ghost"] = frags["A"] + frags_Bgh#list(map(lambda x: x[0] = "X0" if x[0] == "O" else x[0] = "X1",frags["B"]))
    frags["BA_ghost"] = frags["B"] + frags_Agh#list(map(lambda x: x[0] = "X0" if x[0] == "O" else x[0] = "X1",frags["B"]))
    frags["AB"] = frags["A"] + frags["B"]
    for key in frags:
        frags[key] = "\n".join(["   ".join(s) for s in frags[key]])
    return frags

def find_file(directory, extension="in"):
    """ Find input file in a directory.
    
    Returns
    -------
     : str or int
        Path to file or 0 in case of AssertionError.
    """
    dir_list = [fn for fn in glob.glob(directory+"/*."+extension)]
    try:
        assert len(dir_list) == 1
        return dir_list[0]
    except AssertionError:
        print("AssertionError: Could not determine single "+ extension +
              "file in " + directory + " !")
        return 0

#############################################################
# Define Molecules of each fragment
#############################################################
root = os.getcwd()
zr_file = find_file(root,extension="zr")
frags = zr_pyscf(zr_file)
# Define arguments
with open("/home/mingxue/projects/unige/pyscf_qchem/SME_bases/SME0.txt", 'r') as bfile:
    ibasis = bfile.read()
xc_code = 'LDA,VWN5'
xc_code = 'LDA,VWN5'
molab = gto.M(atom=frags["AB"], basis=ibasis, charge=-1, verbose=3, cart=True)
ao_labels = molab.ao_labels()
for i, label in enumerate(ao_labels):
    print(f"{i}: {label}")
