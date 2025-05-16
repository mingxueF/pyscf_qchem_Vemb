# pyscf_qchem_Vemb

## order of basis function
There is difference in the order of basis function between pyscf and qchem,e.g.
for d shell, qchem's order:dxx, dxy, dyy, dxz, dyz, dzz, while in pyscf it reads:
dxx, dxy, dxz, dyy, dyz, dzz. A easy check is to confime the number of electrons are the same when importing density matrix 
from qchem to pyscf or vice versa.

the folder order_pyscf is responsible to reorder embedding potential v_emb in AO basis obtained from pyscf to the expected order in qchem software.

## importing v_emb to qchem (development version)
A embedding potentail matrix v_emb in AO basis generated from pyscf where NDCS functional has been implenmted can be read by qchem current in development version.

A typical input for reading external potentail in AO basis subsquently by a ADC(2) calculation is in the following format:

