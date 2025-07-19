import numpy as np
import pandas as pd
from datetime import datetime

# ------------------------------
# 1️⃣ Load the Pretraining Data
# ------------------------------
train_path = "pretraining_train.csv"
val_path = "pretraining_validation.csv"

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

# Extract SMILES strings
train_smiles = train_df["Canonical_SMILES"].tolist()
val_smiles = val_df["Canonical_SMILES"].tolist()


# ------------------------------
# 3️⃣ Define One-Hot Tokenizer
# ------------------------------
class OneHotTokenizer:
    def __init__(self, charset=CHARSET, fixed_length=100):
        self.charset = charset
        self.fixed_length = fixed_length

    @staticmethod
    def get_one_hot_vector(idx, N):
        """Create a one-hot vector of size N with 1 at index idx."""
        return [1 if i == idx else 0 for i in range(N)]

    def get_one_hot_index(self, char):
        """Get the index of a character in CHARSET."""
        return self.charset.index(char) if char in self.charset else None

# ------------------------------
# 4️⃣ Tokenize Train & Validation Data
# ------------------------------
tokenizer = OneHotTokenizer(charset=CHARSET, fixed_length=100)

# Tokenize SMILES
train_array_tokenized = tokenizer.tokenize(train_smiles)
val_array_tokenized = tokenizer.tokenize(val_smiles)

# Print shapes
print(f"✅ Train Tokenized Shape: {train_array_tokenized.shape}")
print(f"✅ Validation Tokenized Shape: {val_array_tokenized.shape}")

# ------------------------------
# 5️⃣ Save Tokenized Data
# ------------------------------
output_dir = "./"  # Modify this path if needed

np.savez_compressed(output_dir + "", arr=train_array_tokenized)
np.savez_compressed(output_dir + "", arr=val_array_tokenized)

print(f"✅ Tokenized datasets saved successfully!")
