import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, MolToSmiles

# ------------------------------
# 1️⃣ Load Active Molecules Dataset
# ------------------------------
df = pd.read_csv("active_molecules.csv")

# Auto-detect SMILES column
smiles_column = None
for col in df.columns:
    if "smiles" in col.lower():
        smiles_column = col
        break

if smiles_column is None:
    raise ValueError("❌ No SMILES column found in the dataset!")

print(f"✅ Using column '{smiles_column}' for SMILES data.")

# ------------------------------
# 2️⃣ Canonicalize SMILES (Ensuring Consistency)
# ------------------------------
def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None  # Skip invalid SMILES

df["Canonical_SMILES"] = df[smiles_column].apply(canonicalize_smiles)

# Remove invalid SMILES
df.dropna(subset=["Canonical_SMILES"], inplace=True)

# Remove duplicates (Ensure unique molecules)
df.drop_duplicates(subset=["Canonical_SMILES"], inplace=True)

# ------------------------------
# 3️⃣ Compute Molecular Descriptors (For Filtering)
# ------------------------------
def compute_mol_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {
            "MolWeight": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol)
        }
    return None

# Compute descriptors
df = df.join(pd.DataFrame(df["Canonical_SMILES"].apply(compute_mol_descriptors).tolist()))

# Remove NaN descriptors
df.dropna(subset=["MolWeight"], inplace=True)

# ------------------------------
# 4️⃣ Apply Molecular Filtering (Lipinski's Rule)
# ------------------------------
def apply_filter(df):
    return df[
        (df["MolWeight"] >= 200) & (df["MolWeight"] <= 700) &
        (df["HBD"] <= 5) & (df["HBA"] <= 10) &
        (df["LogP"] <= 5)
    ].copy()

filtered_df = apply_filter(df)

# ------------------------------
# 5️⃣ Save Processed Pretraining Dataset
# ------------------------------
filtered_df = filtered_df[["Canonical_SMILES"]]  # Keep only needed columns
filtered_df.to_csv("pretraining_dataset.csv", index=False)

# ------------------------------
# 6️⃣ Print Summary
# ------------------------------
print(f"\n✅ Pretraining Dataset Saved!")
print(f"   → {len(filtered_df)} molecules prepared for pretraining.")


# ------------------------------
# 1️⃣ Load Adjusted Pretraining Dataset
# ------------------------------
pretraining_df = pd.read_csv("pretraining_dataset.csv")

# ------------------------------
# 2️⃣ Split into 90% Train & 10% Validation
# ------------------------------
train_df, val_df = train_test_split(pretraining_df, test_size=0.1, random_state=42, shuffle=True)

# ------------------------------
# 3️⃣ Save Train & Validation Splits
# ------------------------------
train_df.to_csv("pretraining_train.csv", index=False)
val_df.to_csv("pretraining_validation.csv", index=False)

# ------------------------------
# 4️⃣ Print Summary
# ------------------------------
print(f"✅ Pretraining Dataset Split Completed!")
print(f"   → Training Set: {len(train_df)} molecules")
print(f"   → Validation Set: {len(val_df)} molecules")