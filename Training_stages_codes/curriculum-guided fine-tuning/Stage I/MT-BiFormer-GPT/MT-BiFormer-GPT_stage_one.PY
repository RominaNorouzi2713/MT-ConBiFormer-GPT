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
sys.path.append("") 

# Add MolGrad to path and import its evaluation module
# Assuming MolGrad directory is in the same directory as this script, or adjust path
current_script_dir = os.path.dirname(os.path.abspath(__file__))
molgrad_path = os.path.join(current_script_dir, 'MolGrad') # Assuming MolGrad folder is here
if os.path.isdir(molgrad_path):
    sys.path.append(current_script_dir) # Add current_script_dir to import MolGrad if it's a package there
else: # Fallback if MolGrad is in the CWD when script is run from elsewhere
    sys.path.append(os.path.join(os.getcwd(), 'MolGrad'))

try:
    from MolGrad import evaluation as molgrad_evaluation
    print("✅ MolGrad.evaluation module imported successfully.")
except ImportError as e:
    print(f"❌ Error importing MolGrad.evaluation: {e}")
    print("Please ensure the MolGrad directory is correctly placed and its __init__.py is set up if it's a package.")
    molgrad_evaluation = None
except Exception as e_gen:
    print(f"❌ An unexpected error occurred during MolGrad import: {e_gen}")
    molgrad_evaluation = None


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = OneHotTokenizer(fixed_length=100) #

class DualTargetDataset(Dataset):
    def __init__(self, csv_file, logger_obj=None):
        self.df = pd.read_csv(csv_file)
        self.df['is_valid_rdkit'] = self.df['Canonical_SMILES'].apply(lambda s: Chem.MolFromSmiles(s) is not None)
        self.df = self.df[self.df['is_valid_rdkit']]
        self.smiles = self.df['Canonical_SMILES'].values
        if 'Target_Combination' in self.df.columns:
            self.labels = self.df['Target_Combination'].astype('category').cat.codes.values
        else:
            if logger_obj: logger_obj.warning("'Target_Combination' column not found in CSV. Using dummy labels (zeros).")
            else: print("Warning: 'Target_Combination' column not found in CSV. Using dummy labels (zeros).")
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
    valid_count = 0
    for s in smiles_list:
        if isinstance(s, str) and MolFromSmiles(s) is not None:
            valid_count +=1
    return valid_count / len(smiles_list) if smiles_list else 0.0

def compute_uniqueness(smiles_list):
    if not smiles_list: return 0.0
    valid_smiles = [s for s in smiles_list if isinstance(s, str) and MolFromSmiles(s)]
    if not valid_smiles: return 0.0
    return len(set(valid_smiles)) / len(valid_smiles)

def compute_novelty(generated_smiles_list, training_smiles_set):
    if not generated_smiles_list: return 0.0
    valid_generated = {s for s in generated_smiles_list if isinstance(s,str) and MolFromSmiles(s)}
    if not valid_generated: return 0.0
    novel_molecules = valid_generated - training_smiles_set
    return len(novel_molecules) / len(valid_generated) if valid_generated else 0.0

# Training function
def train_dual_target(csv_path, save_path, checkpoint_load_path, 
                      batch_size=32, epochs=120, lr=1e-5, patience=30, 
                      kl_weight_param=0.02, kl_warmup_epochs=50, 
                      gradient_clip_norm=1.0,
                      sampling_temp=0.7, sampling_top_k=20,
                      samples_to_generate_epochwise=500, sample_every_n_epochs=10):
    
    os.makedirs(save_path, exist_ok=True) 
    
    run_log_file = os.path.join(save_path, "training_run_log.txt")
    # Ensure new logger for each call if function is called multiple times in one script
    logger_name = f"train_dual_target_{os.path.basename(save_path)}" 
    logger = logging.getLogger(logger_name)
    if not logger.handlers: # Setup handlers only if not already configured
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(run_log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler(sys.stdout) 
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"✅ Logs for this run will be written to {run_log_file}")
    logger.info(f"Device: {device}")

    dataset = DualTargetDataset(csv_path, logger_obj=logger) 
    if len(dataset) == 0: return

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    if train_size == 0 or val_size == 0:
        logger.error(f"Too few samples for train/val split. Train: {train_size}, Val: {val_size}")
        return

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    num_workers_val = 2 if device.type == 'cuda' else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True if device.type == 'cuda' else False, num_workers=num_workers_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True if device.type == 'cuda' else False, num_workers=num_workers_val)

    model = VAE().to(device) #
    
    logger.info(f"Attempting to load checkpoint from: {checkpoint_load_path}")
    checkpoint = None
    if checkpoint_load_path and os.path.exists(checkpoint_load_path):
        try:
            checkpoint = torch.load(checkpoint_load_path, map_location=device)
            logger.info(f"Checkpoint loaded successfully. Keys: {checkpoint.keys()}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}. Model will use its default initialization.")
    elif checkpoint_load_path:
        logger.error(f"Checkpoint file not found at {checkpoint_load_path}. Model will use its default initialization.")
    else:
        logger.info("No checkpoint_load_path provided. Model will use its default initialization.")

    if checkpoint:
        load_report = None
        if 'encoder' in checkpoint and 'decoder' in checkpoint:
            logger.info("Found 'encoder' and 'decoder' keys, attempting SupCon-style load.")
            cleaned_state_dict = {}
            for k, v in checkpoint['encoder'].items(): cleaned_state_dict[k.replace('vae.', '', 1)] = v
            for k, v in checkpoint['decoder'].items(): cleaned_state_dict[k.replace('vae.', '', 1)] = v
            load_report = model.load_state_dict(cleaned_state_dict, strict=False)
        elif 'model' in checkpoint:
            logger.info("Found 'model' key, attempting full VAE model load.")
            load_report = model.load_state_dict(checkpoint['model'], strict=False)
        elif 'model_state_dict' in checkpoint:
            logger.info("Found 'model_state_dict' key, attempting full VAE model load.")
            load_report = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            logger.warning("Checkpoint does not contain expected keys. Model may use defaults.")
        
        if load_report:
            logger.info(f"Load report: Missing: {load_report.missing_keys}, Unexpected: {load_report.unexpected_keys}")
            if not load_report.missing_keys and not load_report.unexpected_keys:
                logger.info("Checkpoint weights loaded perfectly (strict=False check).")
            
    logger.info("Applying layer freezing strategy:")
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if name.startswith(('fc_3', 'smiles_gpt.transformer', 'fc_4', 'smiles_gpt.lm_head')): 
            param.requires_grad = True
            logger.info(f"  Unfrozen: {name}")
            unfrozen_count +=1
        else:
            param.requires_grad = False
    logger.info(f"Total unfrozen parameter groups/layers: {unfrozen_count}")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=patience//2, verbose=False)

    best_val_loss = float('inf')
    patience_counter = 0

    training_smiles_list = []
    for i in range(len(train_dataset)):
        data_item, _ = train_dataset[i] 
        decoded_smiles_list = tokenizer.decode_one_hot([data_item.cpu().numpy()])
        if decoded_smiles_list and decoded_smiles_list[0] and isinstance(decoded_smiles_list[0][0], str):
            training_smiles_list.append(decoded_smiles_list[0][0].strip())
    train_smiles_set = set(s for s in training_smiles_list if s)
    logger.info(f"Number of unique training SMILES for novelty calculation: {len(train_smiles_set)}")
    
    if train_smiles_set:
        training_smiles_df = pd.DataFrame(list(train_smiles_set), columns=['Training_SMILES'])
        training_smiles_save_path = os.path.join(save_path, 'training_split_smiles_set.csv')
        try:
            training_smiles_df.to_csv(training_smiles_save_path, index=False)
            logger.info(f"Saved training split SMILES to {training_smiles_save_path}")
        except Exception as e:
            logger.error(f"Could not save training split SMILES: {e}")

    target_kl_weight = kl_weight_param 
    logger.info(f"KL Annealing: Target KL weight {target_kl_weight:.4f} over {kl_warmup_epochs} warmup epochs.")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss_sum_total = 0
        epoch_recon_loss_sum_train = 0
        epoch_kl_loss_sum_train = 0

        if epoch <= kl_warmup_epochs:
            current_epoch_kl_weight = target_kl_weight * (epoch / kl_warmup_epochs) if kl_warmup_epochs > 0 else target_kl_weight
        else:
            current_epoch_kl_weight = target_kl_weight
        
        for batch_idx, (batch_data, _) in enumerate(train_loader):
            batch_data = batch_data.to(device)
            x_recon, z_mean, z_logvar = model(batch_data)
            
            if epoch == 1 and batch_idx == 0:
                logger.info(f"--- Initial Latent Variable Stats (Epoch 1, Batch 0) ---")
                logger.info(f"z_mean - Shape: {z_mean.shape}, Mean: {z_mean.mean().item():.4f}, Std: {z_mean.std().item():.4f}, Min: {z_mean.min().item():.4f}, Max: {z_mean.max().item():.4f}")
                logger.info(f"z_logvar - Shape: {z_logvar.shape}, Mean: {z_logvar.mean().item():.4f}, Std: {z_logvar.std().item():.4f}, Min: {z_logvar.min().item():.4f}, Max: {z_logvar.max().item():.4f}")
                sigma = torch.exp(0.5 * z_logvar)
                logger.info(f"sigma (latent stddev) - Shape: {sigma.shape}, Mean: {sigma.mean().item():.4f}, Std: {sigma.std().item():.4f}, Min: {sigma.min().item():.4f}, Max: {sigma.max().item():.4f}")
                initial_recon_loss, initial_kl_div = vae_loss_components(x_recon, batch_data, z_mean, z_logvar)
                initial_total_loss_for_batch = (initial_recon_loss + current_epoch_kl_weight * initial_kl_div) / batch_data.size(0)
                logger.info(f"Initial Components for First Batch:")
                logger.info(f"  Recon Loss (sum): {initial_recon_loss.item():.4f} -> Avg: {(initial_recon_loss.item()/batch_data.size(0)):.4f}")
                logger.info(f"  KL Div (sum, unweighted): {initial_kl_div.item():.4f} -> Avg: {(initial_kl_div.item()/batch_data.size(0)):.4f}")
                logger.info(f"  KL Div (weighted, current_epoch_kl_weight={current_epoch_kl_weight:.4f}, sum): {current_epoch_kl_weight * initial_kl_div.item():.4f} -> Avg: {(current_epoch_kl_weight * initial_kl_div.item()/batch_data.size(0)):.4f}")
                logger.info(f"  Total VAE Loss (avg for batch): {initial_total_loss_for_batch.item():.4f}")
            
            recon_loss_train, kl_div_train = vae_loss_components(x_recon, batch_data, z_mean, z_logvar)
            loss = (recon_loss_train + current_epoch_kl_weight * kl_div_train) / batch_data.size(0) 

            optimizer.zero_grad()
            loss.backward()
            if gradient_clip_norm is not None:
                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()
            
            epoch_train_loss_sum_total += recon_loss_train.item() + current_epoch_kl_weight * kl_div_train.item()
            epoch_recon_loss_sum_train += recon_loss_train.item()
            epoch_kl_loss_sum_train += kl_div_train.item()

        avg_train_loss = epoch_train_loss_sum_total / len(train_loader.dataset)
        avg_recon_loss_train = epoch_recon_loss_sum_train / len(train_loader.dataset)
        avg_kl_loss_train = epoch_kl_loss_sum_train / len(train_loader.dataset)

        model.eval()
        epoch_val_loss_sum_total = 0
        epoch_val_recon_loss_sum = 0
        epoch_val_kl_loss_sum = 0
        last_val_z_mean_for_sampling = None

        with torch.no_grad():
            for batch_data, _ in val_loader:
                batch_data = batch_data.to(device)
                x_recon, z_mean, z_logvar = model(batch_data)
                recon_loss_val, kl_div_val = vae_loss_components(x_recon, batch_data, z_mean, z_logvar)
                
                epoch_val_loss_sum_total += recon_loss_val.item() + current_epoch_kl_weight * kl_div_val.item()
                epoch_val_recon_loss_sum += recon_loss_val.item()
                epoch_val_kl_loss_sum += kl_div_val.item()
                
                if last_val_z_mean_for_sampling is None: 
                    last_val_z_mean_for_sampling = z_mean

        avg_val_loss = epoch_val_loss_sum_total / len(val_loader.dataset)
        avg_recon_loss_val = epoch_val_recon_loss_sum / len(val_loader.dataset)
        avg_kl_loss_val = epoch_val_kl_loss_sum / len(val_loader.dataset)
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        log_msg = (f"Epoch {epoch}/{epochs} - LR: {current_lr:.2e} - KL_w: {current_epoch_kl_weight:.4f} - "
                   f"Train Loss: {avg_train_loss:.4f} (Recon: {avg_recon_loss_train:.4f}, KL_unw: {avg_kl_loss_train:.4f}) | "
                   f"Val Loss: {avg_val_loss:.4f} (Recon: {avg_recon_loss_val:.4f}, KL_unw: {avg_kl_loss_val:.4f})")
        logger.info(log_msg)
        # print(log_msg) # This is now handled by logger if console_handler is added

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
            patience_counter = 0
            logger.info(f"Best model updated. Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'checkpoint_epoch{epoch}.pt'))

        # --- Modified Evaluation Block using MolGrad ---
        if epoch % sample_every_n_epochs == 0 and last_val_z_mean_for_sampling is not None:
            model.eval() 
            with torch.no_grad():
                all_generated_smiles_raw = []
                latent_dim = model.fc_1.out_features 
                generation_batch_size = last_val_z_mean_for_sampling.size(0)

                for i in range(samples_to_generate_epochwise // generation_batch_size + 1):
                    z_sample_batch = torch.randn(generation_batch_size, latent_dim).to(device)
                    z_decoded = model.fc_3(z_sample_batch) 
                    
                    x_gen_probs = None
                    try:
                        x_gen_probs = model.decode(z_decoded, temperature=sampling_temp, top_k=sampling_top_k) 
                    except TypeError: 
                        try:
                            x_gen_probs = model.decode(z_decoded, temperature=sampling_temp)
                        except TypeError: 
                            x_gen_probs = model.decode(z_decoded)  
                    
                    generated_indices = torch.multinomial(x_gen_probs.view(-1, x_gen_probs.size(-1)), 
                                                          num_samples=1).view(x_gen_probs.size(0), x_gen_probs.size(1))
                    
                    temp_one_hot_batch = np.zeros((generated_indices.size(0), 
                                                   generated_indices.size(1), 
                                                   x_gen_probs.size(-1)), dtype=np.float32)
                    
                    for i_batch_idx in range(generated_indices.size(0)):
                        for i_seq_idx in range(generated_indices.size(1)):
                            char_idx = generated_indices[i_batch_idx, i_seq_idx].item()
                            if 0 <= char_idx < x_gen_probs.size(-1):
                                temp_one_hot_batch[i_batch_idx, i_seq_idx, char_idx] = 1.0
                            else:
                                logger.warning(f"Invalid char_idx {char_idx} at epoch {epoch} during sampling.")
                    
                    smiles_from_batch = [s[0] for s in tokenizer.decode_one_hot(temp_one_hot_batch)]
                    all_generated_smiles_raw.extend(smiles_from_batch)

                    if len(all_generated_smiles_raw) >= samples_to_generate_epochwise:
                        break
                all_generated_smiles_raw = all_generated_smiles_raw[:samples_to_generate_epochwise]

                df_gen_raw = pd.DataFrame({'SMILES': all_generated_smiles_raw})
                df_gen_raw.to_csv(os.path.join(save_path, f'samples_epoch{epoch}_raw.csv'), index=False)
                logger.info(f"Saved {len(all_generated_smiles_raw)} raw generated SMILES to samples_epoch{epoch}_raw.csv")

                # Calculate standard V/U/N on raw generated SMILES
                std_validity = compute_validity(all_generated_smiles_raw)
                std_uniqueness = compute_uniqueness(all_generated_smiles_raw) 
                std_novelty = compute_novelty(all_generated_smiles_raw, train_smiles_set)
                logger.info(f"Epoch {epoch} Standard Metrics (on raw {len(all_generated_smiles_raw)} samples): "
                            f"Validity: {std_validity:.4f}, Uniqueness: {std_uniqueness:.4f}, Novelty: {std_novelty:.4f}")

                # Now, use MolGrad for additional metrics on the RDKit-valid subset of these raw generated SMILES
                if molgrad_evaluation: # Check if MolGrad was imported successfully
                    rdkit_valid_smiles_for_molgrad = [s for s in all_generated_smiles_raw if MolFromSmiles(s)]
                    if rdkit_valid_smiles_for_molgrad:
                        logger.info(f"Epoch {epoch}: Evaluating {len(rdkit_valid_smiles_for_molgrad)} RDKit-valid SMILES with MolGrad...")
                        try:
                           # Corrected line
                            molgrad_metrics_df = molgrad_evaluation.evaluate_molecules(
                                rdkit_valid_smiles_for_molgrad, # First argument: generated SMILES
                                list(train_smiles_set)         # Second argument: training SMILES
                            )
                            
                            avg_qed = molgrad_metrics_df['QED'].mean() if 'QED' in molgrad_metrics_df.columns else float('nan')
                            avg_sas = molgrad_metrics_df['SAS'].mean() if 'SAS' in molgrad_metrics_df.columns else float('nan')
                            avg_logp = molgrad_metrics_df['logP'].mean() if 'logP' in molgrad_metrics_df.columns else float('nan')
                            # MolGrad's evaluate_molecules also returns aggregate 'Validity', 'Novelty', 'Uniqueness'
                            # These might be calculated differently (e.g. Uniqueness@k), so good to log them separately
                            mg_validity = molgrad_metrics_df.get('Validity', pd.Series(float('nan'))).iloc[0] if not molgrad_metrics_df.empty and 'Validity' in molgrad_metrics_df else std_validity # Fallback or check structure
                            mg_uniqueness = molgrad_metrics_df.get('Uniqueness', pd.Series(float('nan'))).iloc[0] if not molgrad_metrics_df.empty and 'Uniqueness' in molgrad_metrics_df else std_uniqueness
                            mg_novelty = molgrad_metrics_df.get('Novelty', pd.Series(float('nan'))).iloc[0] if not molgrad_metrics_df.empty and 'Novelty' in molgrad_metrics_df else std_novelty


                            logger.info(f"Epoch {epoch} MolGrad Aggregate Metrics:")
                            logger.info(f"  MolGrad Validity: {mg_validity:.4f} (Note: on pre-filtered RDKit-valid inputs)")
                            logger.info(f"  MolGrad Uniqueness: {mg_uniqueness:.4f}")
                            logger.info(f"  MolGrad Novelty: {mg_novelty:.4f}")
                            logger.info(f"  Avg QED: {avg_qed:.4f}, Avg SAS: {avg_sas:.4f}, Avg logP: {avg_logp:.4f}")

                            molgrad_metrics_df.to_csv(os.path.join(save_path, f'molgrad_detailed_metrics_epoch{epoch}.csv'), index=False)
                            logger.info(f"Saved MolGrad detailed metrics to molgrad_detailed_metrics_epoch{epoch}.csv")

                            # Update metrics text file
                            with open(os.path.join(save_path, f'metrics_epoch{epoch}.txt'), 'w') as f:
                                f.write(f"Epoch {epoch}\n")
                                f.write(f"Samples Generated (total attempts): {len(all_generated_smiles_raw)}\n")
                                f.write(f"RDKit Validity (this script): {std_validity:.4f}\n")
                                f.write(f"RDKit Uniqueness (of valid, this script): {std_uniqueness:.4f}\n")
                                f.write(f"RDKit Novelty (of unique valid, this script): {std_novelty:.4f}\n")
                                f.write(f"--- MolGrad Metrics (on {len(rdkit_valid_smiles_for_molgrad)} RDKit-valid inputs) ---\n")
                                f.write(f"  MolGrad Validity: {mg_validity:.4f}\n")
                                f.write(f"  MolGrad Uniqueness: {mg_uniqueness:.4f}\n")
                                f.write(f"  MolGrad Novelty: {mg_novelty:.4f}\n")
                                f.write(f"  Avg QED: {avg_qed:.4f}\n")
                                f.write(f"  Avg SAS: {avg_sas:.4f}\n")
                                f.write(f"  Avg logP: {avg_logp:.4f}\n")

                        except Exception as e_molgrad:
                            logger.error(f"Error during MolGrad evaluation for epoch {epoch}: {e_molgrad}")
                    else:
                        logger.info(f"Epoch {epoch} MolGrad Metrics: No RDKit-valid SMILES to evaluate with MolGrad.")
                else:
                    logger.warning(f"Epoch {epoch}: MolGrad.evaluation module not imported. Skipping MolGrad metrics.")
        elif epoch % sample_every_n_epochs == 0:
             logger.warning(f"Epoch {epoch}: Sampling skipped as last_val_z_mean_for_sampling is None.")


# Example usage
if __name__ == '__main__':
    main_save_path = '' 
    
    # Global logging setup - this logger is for the script itself, not the train_dual_target function.
    # The train_dual_target function sets up its own file handler.
    global_log_file_main = os.path.join(main_save_path, "MAIN_SCRIPT_EXECUTION_LOG.txt")
    os.makedirs(main_save_path, exist_ok=True) 
    
    # Configure the root logger - this will be inherited by the function-specific logger
    # if the function logger doesn't clear handlers or is a child and propagates.
    # To ensure the function logger is independent, it should get its own name.
    logging.basicConfig(filename=global_log_file_main, level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode='w')
    
    print(f"✅ Main script execution log (global): {global_log_file_main}")
    logging.info(f"Script started. Experiment save path: {main_save_path}")
    logging.info(f"Global logger configured. train_dual_target will use its own file handler in: {os.path.join(main_save_path, 'training_run_log.txt')}")

    train_dual_target(
        csv_path='/augmented_dual_PIK3CA_AKT1.csv', 
        save_path=main_save_path, 
        checkpoint_load_path='/best_checkpoint.pt', 
        epochs=200, 
        lr=5e-6,       
        patience=30, 
        kl_weight_param=0.05, 
        kl_warmup_epochs=30, # Matching log from last run
        gradient_clip_norm=1.0,
        sampling_temp=0.7, 
        sampling_top_k=20,
        samples_to_generate_epochwise=200, 
        sample_every_n_epochs=1 # Sample every epoch like last log
    )
    logging.info("Script finished.")