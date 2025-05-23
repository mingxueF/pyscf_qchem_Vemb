# pyscf\_qchem\_Vemb

## Runnning FDET calculations in **Q-chem** 
With availbale XC, T functional supported by **Q-chem**, a sample input for runnning a ADC(2) calculation within the FDET implementation looks like this:

```text
$rem
  sym_ignore       true
  method           adc(2)
  basis            6-31+G*
  ee_states            1
  fde              true
  PURECART         2
  adc_davidson_maxiter 900
  adc_print            3
  adc_prop_es          true
  gen_scfman           false
  molden_format        true
  mem_static       1024
  mem_total        50000
$end

$molecule
  -1 1
  --
  -1 1
  S  0.70955  -0.28088  -1.53333
  C  0.46967  -0.20843   0.13142
  N  0.36180  -0.12531   1.30404
  --
  0 1
  O      -1.4840500000    -2.4338300000    -1.1651400000
  H      -1.0121100000    -3.3058300000    -1.4099100000 
  H      -0.8123400000    -1.7394400000    -1.4038700000
$end

$fde
T_Func TF
X_Func Slater
C_Func VWN5
expansion ME
rhoB_method hf
rhoA_method mp
PrintLevel 3
debug true
$end
```

## Reading external potential from **PySCF** to **Q-Chem**
*pyscf\_qchem\_Vemb* serves as a bridge between **PySCF** and the **development version of Q‑Chem** developed in Wesolowski Group that imports embedding potential matrix `v_emb` (AO basis) generated from PySCF to qchem for further desired calculations.  It provides helper scripts for:

* generating AO‑basis embedding potentials (`v_emb`) in PySCF,
* re‑ordering those matrices to match Q‑Chem’s AO ordering, and
* importing them into Q‑Chem for post‑SCF methods such as ADC(2).

This README summarises the basis‑order issues and shows a minimal, copy‑pasteable Q‑Chem input template.

## 1. Ordering of Basis Functions

Before exchanging data between **PySCF** and **Q‑Chem**, confirm that *both* codes use the same basis‑function representation—Cartesian or spherical. If they differ, array‑size mismatches will occur.

Even with the same representation, PySCF and Q‑Chem store **d‑shell** functions in a different order. For example,

| Code   | Order                   |
| ------ | ----------------------- |
| Q‑Chem | dxx dxy dyy dxz dyz dzz |
| PySCF  | dxx dxy dxz dyy dyz dzz |

A quick consistency check is to verify that the **total electron count** is unchanged after importing a density matrix between the two programs.

The folder **`order_pyscf`** contains a helper script that **re‑orders the embedding potential matrix `v_emb` (AO basis) generated by PySCF** so that it matches Q‑Chem’s ordering.

---

## 2. Importing `v_emb` into Q‑Chem (development version)

The current development version of Q‑Chem can read an external embedding potential (`v_emb`) in the AO basis—e.g., one produced in PySCF with the **NDCS** functional.

### 2.1  Minimal Q‑Chem input for reading an external potential

```text
$rem
  sym_ignore       true
  method           hf
  basis            mixed
  fde              true
  PURECART         2
  mem_static       1024
  mem_total        50000
$end

$molecule
  -1 1
  S  0.70955  -0.28088  -1.53333
  C  0.46967  -0.20843   0.13142
  N  0.36180  -0.12531   1.30404
$end

$fde
  import_vmat  true
  debug        true
$end

$basis
  ...  (basis set in Q‑Chem format) ...
$end
```

> **Tip**  Use the helper **`nwchem_to_qchem_format()`** (provided in this repo) to convert a basis written in PySCF/NWChem style to Q‑Chem format, then paste it into the `$basis` section.

### 2.2  Example: follow‑up ADC(2) calculation

```text
@@@

$rem
  sym_ignore           true
  method               adc(2)
  scf_guess            read
  fde_read_pot         true
  ee_states            1
  basis                mixed
  PURECART             2
  mem_static           1024
  mem_total            250000
  adc_davidson_maxiter 900
  adc_print            3
  adc_prop_es          true
  gen_scfman           false
  molden_format        true
  fde_export_density   true
$end

$molecule
  ...  (same as above) ...
$end

$basis
  ...  (same basis block) ...
$end
```

---

