import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import sys
import logging

sys.path.append("/") 
from C10_vae_gt_r_b import VAE, OneHotTokenizer

# Add MolGrad to path and import its evaluation module
current_script_dir = os.path.dirname(os.path.abspath(__file__))
molgrad_path = os.path.join(current_script_dir, 'MolGrad')
if os.path.isdir(molgrad_path):
    sys.path.append(current_script_dir)
else:
    sys.path.append(os.path.join(os.getcwd(), 'MolGrad'))

try:
    from MolGrad import evaluation as molgrad_evaluation
    print("✅ MolGrad.evaluation module imported successfully.")
except ImportError as e:
    print(f"❌ Error importing MolGrad.evaluation: {e}")
    molgrad_evaluation = None
except Exception as e_gen:
    print(f"❌ An unexpected error occurred during MolGrad import: {e_gen}")
    molgrad_evaluation = None

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = OneHotTokenizer(fixed_length=100)

class TripletTargetDataset(Dataset):
    def __init__(self, csv_file, logger_obj=None):
        self.df = pd.read_csv(csv_file)
        self.df['is_valid_rdkit'] = self.df['Canonical_SMILES'].apply(lambda s: Chem.MolFromSmiles(s) is not None)
        self.df = self.df[self.df['is_valid_rdkit']]
        self.smiles = self.df['Canonical_SMILES'].values
        if 'Target_Combination' in self.df.columns:
            self.labels = self.df['Target_Combination'].astype('category').cat.codes.values
        else:
            if logger_obj: logger_obj.warning("'Target_Combination' column not found in CSV. Using dummy labels.")
            else: print("Warning: 'Target_Combination' column not found in CSV. Using dummy labels.")
            self.labels = np.zeros(len(self.smiles), dtype=int)
        self.data = [tokenizer.encode_one_hot(s) for s in self.smiles]
        log_msg = f"Loaded {len(self.data)} valid SMILES from {csv_file}"
        if logger_obj: logger_obj.info(log_msg)
        else: print(log_msg)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]).float(), torch.tensor(self.labels[idx]).long()

def vae_loss_components(x_recon, x, z_mean, z_logvar):
    recon_loss = F.binary_cross_entropy(input=x_recon, target=x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return recon_loss, kl_div

def compute_validity(smiles_list):
    if not smiles_list: return 0.0
    return sum(1 for s in smiles_list if isinstance(s, str) and MolFromSmiles(s) is not None) / len(smiles_list)

def compute_uniqueness(smiles_list):
    valid_smiles = [s for s in smiles_list if isinstance(s, str) and MolFromSmiles(s)]
    if not valid_smiles: return 0.0
    return len(set(valid_smiles)) / len(valid_smiles)

def compute_novelty(generated_smiles, training_smiles):
    valid_generated = {s for s in generated_smiles if isinstance(s,str) and MolFromSmiles(s)}
    if not valid_generated: return 0.0
    novel_molecules = valid_generated - training_smiles
    return len(novel_molecules) / len(valid_generated)

def train_stage2(csv_path, save_path, checkpoint_load_path, 
                 batch_size=32, epochs=120, lr=1e-5, patience=30, 
                 kl_weight_param=0.02, kl_warmup_epochs=50, 
                 gradient_clip_norm=1.0,
                 sampling_temp=0.7, sampling_top_k=20,
                 samples_to_generate_epochwise=500, sample_every_n_epochs=10):
    
    os.makedirs(save_path, exist_ok=True) 
    
    logger_name = f"train_stage2_{os.path.basename(save_path)}" 
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(save_path, "training_run_log.txt"), mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler(sys.stdout) 
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"✅ Stage 2 Training Initialized. Logs will be written to {save_path}")
    logger.info(f"Device: {device}")

    dataset = TripletTargetDataset(csv_path, logger_obj=logger) 
    if len(dataset) < 10: 
        logger.error("Dataset too small to split. Exiting.")
        return

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == 'cuda'), num_workers=2 if device.type == 'cuda' else 0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=(device.type == 'cuda'), num_workers=2 if device.type == 'cuda' else 0)

    model = VAE().to(device)
    
    if checkpoint_load_path and os.path.exists(checkpoint_load_path):
        try:
            # Stage 2 starts by loading the *entire model state* from stage 1, not just encoder/decoder
            checkpoint = torch.load(checkpoint_load_path, map_location=device)
            model.load_state_dict(checkpoint) # Assumes the checkpoint is a model's state_dict
            logger.info(f"Successfully loaded Stage 1 checkpoint from: {checkpoint_load_path}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}. Starting with a fresh model.")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_load_path}. Starting with a fresh model.")

    logger.info("Applying Stage 2 layer freezing strategy (unfreezing only final layers):")
    for name, param in model.named_parameters():
        if name.startswith(('fc_4', 'smiles_gpt.lm_head')): 
            param.requires_grad = True
            logger.info(f"  Unfrozen: {name}")
        else:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=patience//2, verbose=False)

    best_val_loss = float('inf')
    patience_counter = 0

    train_smiles_set = {s for s in dataset.smiles}
    logger.info(f"Full training dataset has {len(train_smiles_set)} unique SMILES for novelty calculation.")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        current_kl_weight = kl_weight_param * min(1.0, epoch / kl_warmup_epochs) if kl_warmup_epochs > 0 else kl_weight_param

        for batch_data, _ in train_loader:
            batch_data = batch_data.to(device)
            x_recon, z_mean, z_logvar = model(batch_data)
            
            recon_loss, kl_div = vae_loss_components(x_recon, batch_data, z_mean, z_logvar)
            loss = (recon_loss + current_kl_weight * kl_div) / batch_data.size(0) 

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), gradient_clip_norm)
            optimizer.step()
            total_train_loss += loss.item() * batch_data.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_data, _ in val_loader:
                batch_data = batch_data.to(device)
                x_recon, z_mean, z_logvar = model(batch_data)
                recon_loss, kl_div = vae_loss_components(x_recon, batch_data, z_mean, z_logvar)
                loss = (recon_loss + current_kl_weight * kl_div) / batch_data.size(0)
                total_val_loss += loss.item() * batch_data.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        scheduler.step(avg_val_loss)
        
        logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
            patience_counter = 0
            logger.info(f"New best model saved with Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}.")
                break
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'checkpoint_epoch{epoch}.pt'))

# Main execution block
if __name__ == '__main__':
    # --- Experiment Configuration ---
    # This block defines the two experiments you want to run for comparison.
    
    experiments = {
        "stage2_from_base": {
            "load_checkpoint": "/best_model.pt",
            "save_dir": "/stage2_from_base_model",
            "description": "Stage 2 fine-tuning on triplet data, starting from the base model checkpoint."
        },
        "stage2_from_contrastive": {
            "load_checkpoint": "/best_model.pt", # Assuming this is your SuperCon stage 1 best model
            "save_dir": "/stage2_from_contrastive_model",
            "description": "Stage 2 fine-tuning on triplet data, starting from the contrastive learning checkpoint."
        }
    }

    # Common parameters for both experiments to ensure a fair comparison
    triplet_data_path = 'augmented_triplet_PIK3CA_AKT1_MTOR.csv'
    
    for exp_name, params in experiments.items():
        print("\n" + "="*80)
        print(f"STARTING EXPERIMENT: {exp_name}")
        print(f"DESCRIPTION: {params['description']}")
        print("="*80 + "\n")
        
        train_stage2(
            csv_path=triplet_data_path, 
            save_path=params['save_dir'], 
            checkpoint_load_path=params['load_checkpoint'], 
            epochs=150, 
            lr=1e-5,       
            patience=25, 
            kl_weight_param=0.05, 
            kl_warmup_epochs=25,
            gradient_clip_norm=1.0
        )
        
        print(f"\nFINISHED EXPERIMENT: {exp_name}")
        print("-" * 80 + "\n")

