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
# 2️⃣ Create a New CHARSET Based on SMILES Data
# ------------------------------
def get_CHARSET(list_smiles):
    """Extract unique characters from the dataset."""
    charset = set(' ')  # Include a space for padding
    for smiles in list_smiles:
        for char in smiles:
            charset.add(char)
    return sorted(list(charset))  # Sort for consistency

# Generate CHARSET from both train & validation data
CHARSET = get_CHARSET(train_smiles + val_smiles)
print(f"✅ New CHARSET: {CHARSET}")

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

    def pad_smiles(self, smiles):
        """Pad or truncate SMILES to a fixed length."""
        return smiles.ljust(self.fixed_length)[:self.fixed_length]

    def encode_one_hot(self, smiles):
        """Convert SMILES into a one-hot encoded matrix."""
        one_hot_indexes = [self.get_one_hot_index(char) for char in self.pad_smiles(smiles)]
        one_hot_vectors = [self.get_one_hot_vector(idx, len(self.charset)) for idx in one_hot_indexes]
        return np.array(one_hot_vectors)

    def tokenize(self, list_smiles):
        """Tokenize a list of SMILES strings."""
        return np.array([self.encode_one_hot(smiles) for smiles in list_smiles])

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
