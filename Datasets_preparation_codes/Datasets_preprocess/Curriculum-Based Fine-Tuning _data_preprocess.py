import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
dual_df = pd.read_csv("final_dual_target_finetuning_v2.csv")
multi_df = pd.read_csv("final_multi_target_finetuning_v2.csv")

# Define filtering functions
def filter_dual(df):
    return df[
        (df["MolWeight"] >= 300) & (df["MolWeight"] <= 450) &
        (df["LogP"] >= 2) & (df["LogP"] <= 4) &
        (df["HBD"] <= 3) &
        (df["HBA"] >= 2) & (df["HBA"] <= 7)
    ]

def filter_multi(df):
    return df[
        (df["MolWeight"] >= 350) & (df["MolWeight"] <= 500) &
        (df["LogP"] >= 2.5) & (df["LogP"] <= 4.5) &
        (df["HBD"] <= 3) &
        (df["HBA"] >= 3) & (df["HBA"] <= 8)
    ]

# Apply the filters
filtered_dual = filter_dual(dual_df)
filtered_multi = filter_multi(multi_df)

# Save the filtered results (if desired)
filtered_dual.to_csv("filtered_dual_target.csv", index=False)
filtered_multi.to_csv("filtered_multi_target.csv", index=False)

# Define a function to plot property distributions (original vs filtered)
def plot_property_distributions(original, filtered, property_name, title):
    plt.figure(figsize=(8, 5))
    plt.hist(original[property_name].dropna(), bins=30, alpha=0.5, label="Original")
    plt.hist(filtered[property_name].dropna(), bins=30, alpha=0.5, label="Filtered")
    plt.xlabel(property_name)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot for Dual-Target Set
for prop in ["MolWeight", "LogP", "HBD", "HBA"]:
    plot_property_distributions(dual_df, filtered_dual, prop,
        f"Dual-Target: {prop} Distribution (Original vs. Filtered)")

# Plot for Multi-Target Set
for prop in ["MolWeight", "LogP", "HBD", "HBA"]:
    plot_property_distributions(multi_df, filtered_multi, prop,
        f"Multi-Target: {prop} Distribution (Original vs. Filtered)")

# -------------------------------
# Randomize SMILES Function
# -------------------------------
def randomize_smiles(smi, seed=None, random_type="restricted"):
    """
    Generate randomized SMILES by shuffling atom indices.
    The 'random_type' parameter indicates whether to apply the RDKit 
    ordering fixes ("restricted") or not ("unrestricted").
    For the restricted variant (the default), we rely on RDKit's 
    internal methods by calling MolToSmiles with canonical=False.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    atom_indices = list(range(mol.GetNumAtoms()))
    random.seed(seed)
    random.shuffle(atom_indices)
    new_mol = rdmolops.RenumberAtoms(mol, atom_indices)
    # For the restricted variant, RDKit's fixes are applied by default
    # when canonical=False. For unrestricted, you might bypass further processing.
    # Here we return the same result in both cases.
    return Chem.MolToSmiles(new_mol, canonical=False)

# -------------------------------
# Extraction Function
# -------------------------------
def extract_by_target_combination(input_csv, target_combination, output_csv):
    """
    Extract rows with a specific target combination from the input CSV
    and save to output CSV.
    """
    df = pd.read_csv(input_csv)
    extracted = df[df["Target_Combination"] == target_combination].copy()
    extracted.to_csv(output_csv, index=False)
    print(f"Extracted {len(extracted)} molecules with target combination '{target_combination}' from {input_csv} and saved to {output_csv}")
    return extracted

# -------------------------------
# Augmentation Function
# -------------------------------
def augment_smiles(df, num_augmentations=5, smiles_column="Canonical_SMILES", random_type="restricted"):
    """
    For each molecule in the dataframe, generate a number of randomized SMILES.
    Returns an augmented dataframe.
    """
    augmented_rows = []
    for idx, row in df.iterrows():
        original = row[smiles_column]
        for i in range(num_augmentations):
            new_smi = randomize_smiles(original, seed=random.randint(1, 10000), random_type=random_type)
            if new_smi:
                new_row = row.copy()
                new_row[smiles_column] = new_smi
                augmented_rows.append(new_row)
    augmented_df = pd.DataFrame(augmented_rows)
    return augmented_df

# -------------------------------
# Main Processing Pipeline
# -------------------------------
def main():
    # Create output directories if needed
    os.makedirs("extracted", exist_ok=True)
    os.makedirs("augmented_outputs", exist_ok=True)

    # Extract dual-target (PIK3CA+AKT1) and multi-target (PIK3CA+AKT1+MTOR)
    dual_extracted = extract_by_target_combination("filtered_dual_target.csv", "PIK3CA+AKT1", "extracted/PIK3CA_AKT1_dual.csv")
    multi_extracted = extract_by_target_combination("filtered_multi_target.csv", "PIK3CA+AKT1+MTOR", "extracted/PIK3CA_AKT1_MTOR_multi.csv")
    
    # Print count before augmentation
    print(f"Dual-target molecules before augmentation: {len(dual_extracted)}")
    print(f"Multi-target molecules before augmentation: {len(multi_extracted)}")
    
    # Augment: 20 times for dual-target, 10 times for multi-target
    dual_augmented = augment_smiles(dual_extracted, num_augmentations=20, smiles_column="Canonical_SMILES", random_type="restricted")
    multi_augmented = augment_smiles(multi_extracted, num_augmentations=10, smiles_column="Canonical_SMILES", random_type="restricted")
    
    # Save augmented datasets
    dual_augmented.to_csv("augmented_outputs/augmented_dual_target.csv", index=False)
    multi_augmented.to_csv("augmented_outputs/augmented_multi_target.csv", index=False)
    
    # Print count after augmentation
    print(f"Dual-target molecules after augmentation: {len(dual_augmented)}")
    print(f"Multi-target molecules after augmentation: {len(multi_augmented)}")

if __name__ == "__main__":
    main()
