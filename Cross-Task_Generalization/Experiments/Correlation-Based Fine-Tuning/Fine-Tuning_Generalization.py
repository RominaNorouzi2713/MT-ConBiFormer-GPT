import os
import sys
import numpy as np
import pandas as pd
import argparse
import logging
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rdkit import Chem

from adopt import ADOPT
from vae_gt_r_b import VAE, OneHotTokenizer, CHARSET  # VAE, tokenizer, charset
from vae_gt_r_b import vae_loss  # loss function

# Add MolGrad directory to path
sys.path.append(os.path.join(os.getcwd(), 'MolGrad'))
from MolGrad import evaluation
from MolGrad.loss import preprocess_bond_noise, s_diffusion, invsigmoid

##############################################################################
# Helper Function for Per-Target Logger
##############################################################################
def get_target_logger(base_log_dir, target_name):
    """
    Creates a logger that writes to a file under:
      {base_log_dir}/{target_name}/finetuning_{target_name}.txt
    and also to the console.
    """
    target_log_dir = os.path.join(base_log_dir, target_name)
    os.makedirs(target_log_dir, exist_ok=True)

    log_path = os.path.join(target_log_dir, f'finetuning_{target_name}.txt')
    logger_name = f'finetuning_{target_name}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Avoid duplicating handlers
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

##############################################################################
# Setup Paths & Random Seeds
##############################################################################
torch.manual_seed(42)
np.random.seed(42)

path_project = os.path.abspath(os.getcwd())  # Base project directory
checkpoint_path = os.path.join(path_project, '')
best_checkpoint_path = os.path.join(checkpoint_path, 'best_checkpoint.pt')

log_dir = os.path.join(path_project, '')
tokenized_target_dir = os.path.join(path_project, '')
finetune_base_dir = os.path.join(path_project, '')

os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(tokenized_target_dir, exist_ok=True)
os.makedirs(finetune_base_dir, exist_ok=True)
os.makedirs(os.path.join(finetune_base_dir, 'real_data'), exist_ok=True)
os.makedirs(os.path.join(finetune_base_dir, 'generated_data'), exist_ok=True)
os.makedirs(os.path.join(finetune_base_dir, 'evaluation'), exist_ok=True)

##############################################################################
# Load Pretrained Model
##############################################################################
def load_pretrained_model(checkpoint_file, logger):
    model = VAE()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model'])
        logger.info(f"Loaded model weights from {checkpoint_file}")
    else:
        logger.info(f"No checkpoint found at {checkpoint_file}. Using a fresh model.")
    return model, device

def get_last_checkpoint(target_name):
    pattern = f"checkpoint_{target_name}_epoch_"
    files = [f for f in os.listdir(checkpoint_path) if f.startswith(pattern) and f.endswith(".pt")]
    if not files:
        return None
    epochs = []
    for f in files:
        try:
            epoch_str = f.split("_")[-1].split(".")[0]  
            epochs.append(int(epoch_str))
        except:
            continue
    if not epochs:
        return None
    latest_epoch = max(epochs)
    return os.path.join(checkpoint_path, f"checkpoint_{target_name}_epoch_{latest_epoch:03d}.pt")

##############################################################################
# Fine-Tune Function
##############################################################################
def fine_tune_target(target_name_to_process, expression, path_project, resume=False):
    """
    Fine-tune the VAE model on a chosen target. 
    Each target gets its own logger & log file.

    This version uses conservative hyperparameters:
    - GPT LR = 5e-7
    - VAE LR = 1e-5
    - Weight Decay = 1e-4
    - KL Weight = 0.1
    - Up to 20 epochs, patience of 15.
    """

    # 1) Read the CSV based on expression type
    if expression == "inhibit":
        df_gen = pd.read_csv(
            os.path.join(path_project, "software", "data", "target_profile",
                         "mean_target_profile", "L1000_inhibit_mcf7_mean.csv"),
            index_col=0
        )
    elif expression == "overexpression":
        df_gen = pd.read_csv(
            os.path.join(path_project, "software", "data", "target_profile",
                         "mean_target_profile", "L1000_overexpression_mcf7_mean.csv"),
            index_col=0
        )
    else:
        raise ValueError("expression should be either 'inhibit' or 'overexpression'.")

    target_names = df_gen.index.to_list()

    for target_name in target_names:
        if target_name != target_name_to_process:
            continue

        logger = get_target_logger(log_dir, target_name)
        logger.info(f"Fine-tuning for target: {target_name}, expression: {expression}")

        # 2) Load correlation file
        correlation_file_path = os.path.join(
            path_project,
            f'software/data/correlation_output/{expression}/{target_name}/correlation_{target_name}.txt'
        )
        if not os.path.exists(correlation_file_path):
            logger.warning(f"Correlation file not found for target: {target_name}. Skipping.")
            continue

        try:
            tmp = pd.read_csv(correlation_file_path, sep='\t')
            if 'Correlation' in tmp.columns:
                # e.g. top 3 by correlation
                tmp = tmp.sort_values(by='Correlation', ascending=False).head(3)
            else:
                logger.warning(f"'Correlation' column not found in {correlation_file_path}. Skipping.")
                continue
        except Exception as e:
            logger.error(f"Error processing {correlation_file_path}: {e}")
            continue

        # 3) Tokenize SMILES
        tokenizer = OneHotTokenizer(charset=CHARSET, fixed_length=100)
        smiles_encoded = []
        real_smiles = []
        for _, row in tmp.iterrows():
            smiles = row['SMILES']
            real_smiles.append(smiles)
            padded_smiles = tokenizer.pad_smiles(smiles)
            smiles_encoded.append(tokenizer.encode_one_hot(padded_smiles))

        # Save data
        smiles_array = np.array(smiles_encoded)
        np.save(os.path.join(tokenized_target_dir, f'smiles_tokenized_target_{target_name}.npy'), smiles_array)
        np.save(os.path.join(finetune_base_dir, 'real_data', f'{target_name}.npy'), real_smiles)
        logger.info(f"Saved real SMILES for {target_name}.")

        # 4) Prepare DataLoaders
        chembl_smiles = np.load(os.path.join(tokenized_target_dir, f'smiles_tokenized_target_{target_name}.npy')).astype(np.float32)
        fine_tune_batch_size = min(512, chembl_smiles.shape[0])

        train_data = torch.utils.data.TensorDataset(torch.from_numpy(chembl_smiles))
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=fine_tune_batch_size, shuffle=True)

        val_data_path = os.path.join(path_project, 'data2', 'validation_l_smiles_tokenized.npz')
        if not os.path.exists(val_data_path):
            logger.warning(f"Validation data not found at {val_data_path}. Skipping.")
            continue

        val_data = np.load(val_data_path)['arr'].astype(np.float32)
        val_data = torch.utils.data.TensorDataset(torch.from_numpy(val_data))
        val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=fine_tune_batch_size, shuffle=False)

        # 5) Decide checkpoint to start from
        if resume:
            last_ckpt = get_last_checkpoint(target_name)
            if last_ckpt is not None:
                model, device = load_pretrained_model(last_ckpt, logger)
            else:
                logger.info(f"No existing checkpoint for {target_name}, using best checkpoint.")
                model, device = load_pretrained_model(best_checkpoint_path, logger)
        else:
            model, device = load_pretrained_model(best_checkpoint_path, logger)

        # 6) Prepare optimizer (GPT & VAE param groups)
        gpt_params, non_gpt_params = model.get_parameter_groups()

        # GPT -> 5e-7, VAE -> 1e-5, weight_decay=1e-4
        optimizer = ADOPT([
            {'params': gpt_params, 'lr': 5e-7, 'weight_decay': 1e-4},
            {'params': non_gpt_params, 'lr': 1e-5, 'weight_decay': 1e-4}
        ])

        # 7) Scheduler & Early stopping
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        num_epochs = 20

        # Folders for generated molecules & logs
        target_gen_dir = os.path.join(finetune_base_dir, 'generated_data', target_name)
        os.makedirs(target_gen_dir, exist_ok=True)
        target_eval_dir = os.path.join(finetune_base_dir, 'evaluation', target_name)
        os.makedirs(target_eval_dir, exist_ok=True)

        # ------------------------
        # TRAINING LOOP
        # ------------------------
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0

            for batch_idx, data in enumerate(train_data_loader):
                data = data[0].to(device)
                optimizer.zero_grad()

                output, z_mean, z_logvar = model(data)

                # KL weight = 0.1 (lower => easier to match data distribution)
                loss = vae_loss(output, data, z_mean, z_logvar, kl_weight=0.1)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_data_loader.dataset)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in val_data_loader:
                    data = data[0].to(device)
                    output, z_mean, z_logvar = model(data)
                    loss = vae_loss(output, data, z_mean, z_logvar, kl_weight=0.1)
                    val_loss += loss.item()

            val_loss /= len(val_data_loader.dataset)
            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}. No improvement in val loss.")
                    break

            # Save checkpoint
            ckpt_data = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'scheduler': scheduler.state_dict()
            }
            ckpt_file = os.path.join(checkpoint_path, f'checkpoint_{target_name}_epoch_{epoch:03d}.pt')
            torch.save(ckpt_data, ckpt_file)

            logger.info(f"[Epoch {epoch}] Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            # -------------------
            # MOLECULE GENERATION (5 molecules)
            # -------------------
            model.eval()
            generated_smiles_raw = []
            with torch.no_grad():
                for data in train_data_loader:
                    data_batch = data[0].to(device)
                    output, _, _ = model(data_batch)

                    decoded_smiles = OneHotTokenizer().decode_one_hot(output.cpu().numpy())
                    for sm_tokens in decoded_smiles:
                        sm_str = "".join(sm_tokens).strip()
                        generated_smiles_raw.append(sm_str)
                        if len(generated_smiles_raw) >= 5:
                            break
                    if len(generated_smiles_raw) >= 5:
                        break

            # Build CSV of generated SMILES
            results_list = []
            real_data = np.load(os.path.join(finetune_base_dir, 'real_data', f'{target_name}.npy'), allow_pickle=True).tolist()

            for smiles in generated_smiles_raw:
                row_dict = {
                    'SMILES': smiles,
                    'Validity': False,
                    'Novelty': None,
                    'Uniqueness': None,
                    'logP': None,
                    'QED': None,
                    'SAS': None
                }
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        canonical = Chem.MolToSmiles(mol)
                        canonical = Chem.CanonSmiles(canonical)
                        row_dict['SMILES'] = canonical
                        row_dict['Validity'] = evaluation.check_valid(canonical)
                        if row_dict['Validity']:
                            row_dict['Novelty'] = evaluation.check_novel(canonical, real_data)
                            row_dict['Uniqueness'] = evaluation.check_unique(canonical, real_data)
                            props = evaluation.get_properties([canonical])
                            row_dict['logP'] = props['logP'][0]
                            row_dict['QED'] = props['QED'][0]
                            row_dict['SAS'] = props['SAS'][0]
                except Exception as ex:
                    logger.debug(f"SMILES parse error: {smiles} => {ex}")
                results_list.append(row_dict)

            df_generated = pd.DataFrame(results_list)
            csv_filename = os.path.join(target_gen_dir, f'generated_epoch_{epoch:03d}.csv')
            df_generated.to_csv(csv_filename, index=False)
            logger.info(f"Saved generated SMILES to {csv_filename}")

            # -------------------
            # TEXT EVALUATION LOG
            # -------------------
            txt_filename = os.path.join(target_eval_dir, f'evaluation_results_{target_name}_epoch_{epoch:03d}.txt')
            valid_df = df_generated[df_generated['Validity'] == True]
            invalid_df = df_generated[df_generated['Validity'] == False]

            with open(txt_filename, 'w') as f:
                f.write(f"Target: {target_name}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Train Loss: {train_loss:.4f}\n")
                f.write(f"Val Loss: {val_loss:.4f}\n\n")

                f.write("=== Generated SMILES (Valid) ===\n")
                for _, row in valid_df.iterrows():
                    f.write(f"{row['SMILES']}\n")
                f.write("\n")

                f.write("=== Generated SMILES (Invalid or Failed Parse) ===\n")
                for _, row in invalid_df.iterrows():
                    f.write(f"{row['SMILES']}\n")
                f.write("\n")

                f.write("=== Real SMILES Used for Fine-tuning ===\n")
                for s in real_data:
                    f.write(s + "\n")
                f.write("\n")

                f.write("=== Evaluation Metrics for Valid SMILES ===\n")
                if valid_df.empty:
                    f.write("No valid SMILES generated.\n")
                else:
                    f.write(f"Validity: {list(valid_df['Validity'])}\n")
                    f.write(f"Novelty: {list(valid_df['Novelty'])}\n")
                    f.write(f"Uniqueness: {list(valid_df['Uniqueness'])}\n\n")

                    f.write("=== Properties ===\n")
                    f.write(f"logP: {list(valid_df['logP'])}\n")
                    f.write(f"QED: {list(valid_df['QED'])}\n")
                    f.write(f"SAS: {list(valid_df['SAS'])}\n\n")

                f.write("=== NUV Filtered SMILES ===\n")
                if not valid_df.empty:
                    valid_smiles = valid_df['SMILES'].tolist()
                    nuv_filtered = evaluation.nuvfilter(valid_smiles, dataset=real_data, nuv=[True, True, True])
                    for s in nuv_filtered:
                        f.write(s + "\n")
                else:
                    f.write("No valid SMILES to filter.\n")

            logger.info(f"Saved evaluation metrics to {txt_filename}")

        # End epochs
        logger.info(f"Finished fine-tuning for target: {target_name}")

##############################################################################
# Main
##############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the model on a specific target and expression.")
    parser.add_argument('--target', type=str, required=True, help="Target name to process (e.g., AKT1)")
    parser.add_argument('--expression', type=str, required=True,
                        choices=['inhibit', 'overexpression'],
                        help="Expression type (inhibit or overexpression)")
    parser.add_argument('--resume', action='store_true',
                        help="If set, resume from the last checkpoint for the given target.")
    args = parser.parse_args()

    fine_tune_target(args.target, args.expression, path_project, resume=args.resume)
