from rdkit.Chem.QED import qed
import tensorflow as tf
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm
from random import shuffle

from MolGrad.loss import preprocess_bond_noise, s_diffusion, invsigmoid
from data import get_logp, get_qed, get_sas

def SmilesFromGraph(node_list, adjacency_matrix):
    mol = Chem.RWMol()
    node_to_idx = {}
    for i, atom in enumerate(node_list):
        molIdx = mol.AddAtom(Chem.Atom(atom))
        node_to_idx[i] = molIdx

    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):
            if iy <= ix or bond == 0:
                continue
            bond_type = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE][bond]
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    mol = mol.GetMol()
    if mol is None:
        return False, None

    try:
        smiles = Chem.MolToSmiles(mol)
        canonical_smiles = Chem.CanonSmiles(smiles)
        return True, canonical_smiles
    except Exception as e:
        print(f"Error generating SMILES from graph: {e}")
        return False, None

def check_valid(smiles, unstable=['OO', 'O(O)', 'NNN', 'N(N)N', 'NN(N)']):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol and '.' not in smiles and not any(u in smiles for u in unstable):
            return True
    except Exception as e:
        # More verbose logging can be done here if needed
        pass
    return False

def check_novel(smiles, dataset):
    return smiles not in dataset

def check_unique(smiles, covered):
    # 'covered' is a list of SMILES strings considered as "already existing"
    return smiles not in covered

def nuvfilter(smiles_list, dataset=[], nuv=[True, True, True]):
    """
    NUV filter => (N)ovel, (U)nique, (V)alid
    `nuv=[True,True,True]` means apply all three checks.
    """
    smiles_result = []
    invalid_smiles = []

    for smiles in smiles_list:
        try:
            canonical_smiles = Chem.CanonSmiles(smiles)
            # Valid check
            if nuv[0] and not check_valid(canonical_smiles):
                continue
            # Novel check
            if nuv[2] and not check_novel(canonical_smiles, dataset):
                continue
            smiles_result.append(canonical_smiles)
        except Exception as e:
            invalid_smiles.append(smiles)

    # Uniqueness check
    if nuv[1]:
        smiles_result = list(set(smiles_result))

    if invalid_smiles:
        print(f"Skipped {len(invalid_smiles)} invalid SMILES strings in nuvfilter.")

    return smiles_result

def get_properties(smiles_list):
    """
    Returns a dictionary with keys 'logP', 'QED', 'SAS'.
    Each value is a numpy array (same length as smiles_list).
    Invalid SMILES get None for their properties.
    """
    logps, qeds, sass = [], [], []
    for smiles in smiles_list:
        try:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule:
                logps.append(get_logp(molecule))
                qeds.append(get_qed(molecule))
                sass.append(get_sas(molecule))
            else:
                logps.append(None)
                qeds.append(None)
                sass.append(None)
        except Exception as e:
            logps.append(None)
            qeds.append(None)
            sass.append(None)

    return {
        'logP': np.array(logps, dtype=object),
        'QED': np.array(qeds, dtype=object),
        'SAS': np.array(sass, dtype=object)
    }

def get_failures(smiles_list, threshold=6):
    """
    Example function showing how you might filter out molecules
    with synthetic accessibility score above a certain threshold.
    """
    properties = get_properties(smiles_list)
    failure_smiles = [
        smiles for smiles, sas in zip(smiles_list, properties['SAS']) 
        if sas is not None and sas > threshold
    ]
    return failure_smiles

class MolPlotter:
    def __init__(self, smiles_list):
        shuffle(smiles_list)
        self.mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s)]
        self.len = len(self.mols)
        self.i = 0

    def __call__(self, nperrow=10, nrows=2, size=150):
        n = nperrow * nrows
        mols = self.mols[self.i:self.i + n]
        self.i = (self.i + n) % self.len
        return Draw.MolsToGridImage(mols, molsPerRow=nperrow, maxMols=n, useSVG=True, subImgSize=(size, size))



# === <<< EDIT THESE PATHS ONLY >>> ===================================== #
GENERATED_MOLS_DIR = r""           # folder with AKT1.csv, AKT2.csv, …
LIGANDS_FILE_PATH  = r"ExcapeDB_28targets_actives_edit.csv"      # known‑ligands master CSV
OUTPUT_DIR         = r"max_similarity_results"                  # where results will be written
# ======================================================================= #

# -----------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load full ligand table
ligands_df = pd.read_csv(LIGANDS_FILE_PATH)

# Quick sanity‑check
required_cols = {"Gene_Symbol", "SMILES"}
if not required_cols.issubset(ligands_df.columns):
    raise ValueError(f"Ligand file must contain columns {required_cols}. Found: {ligands_df.columns.tolist()}")

summary_rows = []  # collect one row per generated molecule across all targets

for fname in os.listdir(GENERATED_MOLS_DIR):
    if not fname.lower().endswith(".csv"):
        continue  # skip non‑CSV files

    target = fname.rsplit(".", 1)[0].strip()  # "AKT1.csv" -> "AKT1"
    fpath  = os.path.join(GENERATED_MOLS_DIR, fname)

    # ---- read generated SMILES file ----
    gdf = pd.read_csv(fpath)
    gdf.columns = gdf.columns.str.strip().str.lower()  # normalise header names

    smiles_col = "generated_smiles" if "generated_smiles" in gdf.columns else gdf.columns[0]
    gen_smiles_list = gdf[smiles_col].dropna().astype(str).tolist()

    if not gen_smiles_list:
        print(f"[WARN] No SMILES found for target {target} in {fname}. Skipping.")
        continue

    # ---- pull ligands for this target ----
    lig_smiles_list = ligands_df.loc[ligands_df["Gene_Symbol"].str.upper() == target.upper(), "SMILES"].dropna().unique().tolist()
    lig_mols  = [Chem.MolFromSmiles(s) for s in lig_smiles_list if Chem.MolFromSmiles(s)]

    if not lig_mols:
        print(f"[WARN] No valid ligand SMILES for target {target}. Skipping.")
        continue

    lig_fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in lig_mols]

    # ---- compute similarities ----
    results = []
    for smi in gen_smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        max_sim = max(DataStructs.BulkTanimotoSimilarity(fp, lig_fps))
        results.append({
            "Target Protein": target,
            "Generated SMILES": smi,
            "Maximum Tanimoto Similarity": max_sim
        })

    if not results:
        print(f"[WARN] No valid generated SMILES for target {target}. Skipping write.")
        continue

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUTPUT_DIR, f"{target}_max_similarity.csv"), index=False)
    summary_rows.extend(results)
    print(f"✓  Saved {len(res_df)} rows for {target} → {target}_max_similarity.csv")

# ---- combined summary ----
if summary_rows:
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, "all_targets_max_similarity.csv"), index=False)
    print("\nAll‑target summary written to all_targets_max_similarity.csv")
else:
    print("\nNo results produced.")
