import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import sys
from datetime import datetime
import math

# MOSES for evaluation
import moses

# --- Assumed to be in the same directory or accessible via sys.path ---
MODEL_DEFINITION_PATH = ""
sys.path.append(MODEL_DEFINITION_PATH)
from C10_vae_gt_r_b import VAE, OneHotTokenizer, decode_smiles_from_indexes

# ############################################################################
# ## CONFIGURATION FOR STAGE 2 EVALUATION
# ############################################################################
.

# --- 2. SAMPLING PARAMETERS ---
NUM_SAMPLES = 10000
EVAL_BATCH_SIZE = 128
SAMPLING_TEMP = 0.7
SAMPLING_TOP_K = 20
LATENT_DIM = 292

# ############################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_from_model(model, tokenizer, checkpoint_path, num_samples, batch_size):
    """Loads a checkpoint and generates SMILES strings in mini-batches."""
    print("-" * 50)
    print(f"Loading checkpoint: {checkpoint_path}")
    
    try:
        # For Stage 2, we load the entire model's state_dict directly
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load checkpoint {checkpoint_path}. Error: {e}")
        return None

    model.eval()
    all_smiles = []
    with torch.no_grad():
        print(f"Generating {num_samples} molecules in batches of {batch_size}...")
        num_batches = math.ceil(num_samples / batch_size)
        
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - len(all_smiles))
            if current_batch_size <= 0:
                break
                
            z_sample_batch = torch.randn(current_batch_size, LATENT_DIM).to(device)
            z_gpt_input = F.selu(model.fc_3(z_sample_batch))
            x_gen_logits = model.decode(z_gpt_input, temperature=SAMPLING_TEMP, top_k=SAMPLING_TOP_K)
            
            x_gen_probs = torch.softmax(x_gen_logits, dim=-1)
            sampled_indices = torch.multinomial(x_gen_probs.view(-1, x_gen_probs.shape[-1]), 1).view(x_gen_probs.shape[0], -1)
            
            smiles_batch = [decode_smiles_from_indexes(s.cpu().numpy(), tokenizer.charset) for s in sampled_indices]
            
            all_smiles.extend(smiles_batch)
            
            print(f"  Batch {i+1}/{num_batches} done. Total generated: {len(all_smiles)}", end='\r')

    print("\nGeneration complete.")
    return all_smiles


def load_training_smiles(csv_path):
    """Loads SMILES from the training data csv for novelty calculation."""
    try:
        df = pd.read_csv(csv_path)
        smiles = df['Canonical_SMILES'].tolist()
        print(f"Loaded {len(smiles)} training SMILES for novelty calculation from {csv_path}")
        return smiles
    except Exception as e:
        print(f"ERROR: Could not load training SMILES from {csv_path}. Error: {e}")
        return []

def main():
    """Main function to run the evaluation workflow."""
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = f"./final_evaluation_stage2_{TIMESTAMP}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Using device: {device}")
    
    print("Initializing VAE model...")
    model = VAE().to(device)

    all_results = {}

    for run_name, config in CHECKPOINT_CONFIG.items():
        print("\n" + "=" * 80)
        print(f"PROCESSING RUN: {run_name}")
        print(f"DESCRIPTION: {config['description']}")
        print("=" * 80)
        
        training_smiles = load_training_smiles(config['train_data_path'])
        if not training_smiles:
            continue

        tokenizer = OneHotTokenizer(fixed_length=100)
        generated_smiles = sample_from_model(model, tokenizer, config['checkpoint_path'], NUM_SAMPLES, EVAL_BATCH_SIZE)
        
        if generated_smiles is None:
            continue

        smiles_save_path = os.path.join(OUTPUT_DIR, f"{run_name.replace(' ', '_')}_10k_samples.txt")
        with open(smiles_save_path, 'w') as f:
            for smiles in generated_smiles:
                f.write(f"{smiles}\n")
        print(f"\nSaved {len(generated_smiles)} generated SMILES to {smiles_save_path}")

        print("Computing MOSES metrics... (This may take a while)")
        
        metrics = moses.get_all_metrics(
            generated_smiles,
            train=training_smiles,
            test=training_smiles,
            test_scaffolds=training_smiles,
            n_jobs=os.cpu_count()
        )
        
        print("MOSES metrics computed.")

        all_results[run_name] = metrics
        
        metrics_save_path = os.path.join(OUTPUT_DIR, f"{run_name.replace(' ', '_')}_metrics.json")
        with open(metrics_save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved metrics to {metrics_save_path}")
        
        print("\n--- METRIC SUMMARY ---")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("-" * 22 + "\n")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print(f"All results saved in directory: {OUTPUT_DIR}")
    print("=" * 80)

    print("\n--- FINAL COMPARISON SUMMARY ---")
    results_df = pd.DataFrame(all_results).T
    if not results_df.empty:
        display_cols = ['valid', 'FCD/Test', 'SNN/Test', 'Frag/Test']
        if 'unique@10000' in results_df.columns:
            display_cols.insert(1, 'unique@10000')
        if 'Novelty' in results_df.columns:
             display_cols.insert(2, 'Novelty')

        display_cols = [col for col in display_cols if col in results_df.columns]
        print(results_df[display_cols].round(4))

if __name__ == '__main__':
    main()
