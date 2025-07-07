import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, MolToSmiles, rdmolops

# ------------------------------
# 1️⃣ Load Single-Target Data (Preprocessing Required)
# ------------------------------
chembl_single_df = pd.read_csv("single_target_inhibitors.csv")
excape_single_df = pd.read_csv("excape_single_target.csv")

# Merge & Remove Duplicates
single_target_df = pd.concat([chembl_single_df, excape_single_df], ignore_index=True).drop_duplicates(subset=["SMILES"])

# Assign Label for Contrastive Learning
single_target_df["Label"] = 0  # Negative Class (Single-Target)

# ------------------------------
# 2️⃣ Canonicalize & Preprocess Single-Target Data
# ------------------------------
def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None  # Skip invalid SMILES

single_target_df["Canonical_SMILES"] = single_target_df["SMILES"].apply(canonicalize_smiles)

# Remove Invalid Entries
single_target_df.dropna(subset=["Canonical_SMILES"], inplace=True)

# ------------------------------
# 3️⃣ Augment Dual-Target Data (5 Times)
# ------------------------------
def randomize_smiles(smi, seed=None):
    """Generate randomized SMILES using restricted atom ordering."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    atom_indices = list(range(mol.GetNumAtoms()))
    random.seed(seed)
    random.shuffle(atom_indices)
    
    # Ensure restricted atom ordering (using RDKit built-in methods)
    new_mol = rdmolops.RenumberAtoms(mol, atom_indices)
    return Chem.MolToSmiles(new_mol, canonical=False)

def augment_data(df, num_augmentations=5):
    """Generate randomized SMILES for each molecule in the dataset."""
    augmented_smiles = []
    for smi in df["Canonical_SMILES"]:
        for _ in range(num_augmentations):
            new_smi = randomize_smiles(smi)
            if new_smi:
                augmented_smiles.append(new_smi)

    # Create augmented DataFrame
    augmented_df = pd.DataFrame({"Canonical_SMILES": augmented_smiles})
    augmented_df["Label"] = 1  # Label for multi-target or dual-target data
    return augmented_df

# Load dual-target data (already processed)
dual_target_df = pd.read_csv("final_dual_target_finetuning_v2.csv")  # ✅ Use processed dataset

# Augment the dual-target dataset (5 augmentations for each SMILES)
augmented_dual_target_df = augment_data(dual_target_df, num_augmentations=5)

# ------------------------------
# 4️⃣ Load Multi-Target Data and Augment
# ------------------------------
multi_target_df = pd.read_csv("final_multi_target_finetuning_v2.csv")  # ✅ Use processed dataset

# Augment the multi-target dataset (5 augmentations for each SMILES)
augmented_multi_target_df = augment_data(multi_target_df, num_augmentations=10)

# ------------------------------
# 5️⃣ Merge All Data for Contrastive Learning Dataset v5
# ------------------------------
# Merge Single-Target Data (Label=0)
single_target_df = single_target_df[["Canonical_SMILES", "Label"]]

# Merge Augmented Multi-Target Data (Label=1)
augmented_multi_target_df = augmented_multi_target_df[["Canonical_SMILES", "Label"]]

# Merge Augmented Dual-Target Data (Label=1)
augmented_dual_target_df = augmented_dual_target_df[["Canonical_SMILES", "Label"]]

# Combine all data into the final contrastive dataset
final_contrastive_df = pd.concat([single_target_df, augmented_multi_target_df, augmented_dual_target_df], ignore_index=True)

# ------------------------------
# 6️⃣ Save the Final Contrastive Learning Dataset v5
# ------------------------------
final_contrastive_df.to_csv("contrastive_learning_dataset_v5.csv", index=False)

# ------------------------------
# 7️⃣ Print Summary
# ------------------------------
print(f"✅ New Contrastive Learning Dataset v5 with Augmented Data saved!")
print(f"   → {len(single_target_df)} single-target molecules (Label 0)")
print(f"   → {len(augmented_multi_target_df)} augmented multi-target molecules (Label 1)")
print(f"   → {len(augmented_dual_target_df)} augmented dual-target molecules (Label 1)")
print(f"   → {len(final_contrastive_df)} total molecules in the new dataset v5")
