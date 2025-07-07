import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from adopt import ADOPT
import warnings

# Please set the path
path_project = r''
smiles_gpt_path = r''
sys.path.append(smiles_gpt_path)
import smiles_gpt

log_dir = r''
path_checkpoint = r''
sys.path.append('')

######################################################################
#          the necessary function for tokenizing of SMILES sequence
######################################################################


class OneHotTokenizer():
    def __init__(self, charset=CHARSET, fixed_length=100):
        self.charset = charset
        self.fixed_length = fixed_length

    @staticmethod
    def get_one_hot_vector(idx, N):
        return list(map(int, [idx == i for i in range(N)]))

    @staticmethod
    def get_one_hot_index(chars, char):
        try:
            return chars.index(char)
        except ValueError:
            return None

    def pad_smiles(self, smiles):
        if len(smiles) <= self.fixed_length:
            return smiles.ljust(self.fixed_length)
        return smiles[:self.fixed_length]

    def encode_one_hot(self, smiles):
        one_hot_indexes = [self.get_one_hot_index(chars=self.charset, char=char) for char in self.pad_smiles(smiles)]
        one_hot_vectors = [self.get_one_hot_vector(idx=idx, N=len(self.charset)) for idx in one_hot_indexes]
        return np.array(one_hot_vectors)

    def tokenize(self, list_smiles):
        return np.array([self.encode_one_hot(smiles) for smiles in list_smiles])

    def decode_one_hot(self, list_encoded_smiles):
        list_smiles_get_back = []
        for smiles_idx in range(len(list_encoded_smiles)):
            smiles_string = ''
            for row_idx in range(len(list_encoded_smiles[smiles_idx])):
                if np.sum(list_encoded_smiles[smiles_idx][row_idx]) == 0:  # ✅ Handles empty rows
                    smiles_string += '?'  # ❗ Placeholder for empty one-hot vectors
                    continue
                one_hot = np.argmax(list_encoded_smiles[smiles_idx][row_idx])
                if one_hot < len(self.charset):  # ✅ Prevent index errors
                    smiles_string += self.charset[one_hot]
                else:
                    smiles_string += '?'  # ❗ Placeholder for unknown characters
            list_smiles_get_back.append([smiles_string.strip()])
        return list_smiles_get_back

def decode_smiles_from_indexes(vec, charset=CHARSET, decode=True):
    if decode:
        try:
            charset = np.array([v.decode('utf-8') for v in charset])
        except:
            pass
    return ''.join(map(lambda x: charset[x], vec)).strip()

def test():
    one_hot_tokenizer = OneHotTokenizer(charset=CHARSET, fixed_length=100)
    one_hot_vector = one_hot_tokenizer.get_one_hot_vector(idx=2, N=6)
    print(f'one_hot_vector = {one_hot_vector}')

    one_hot_index = one_hot_tokenizer.get_one_hot_index(chars='Testing', char='s')
    print(f'one_hot_index = {one_hot_index}')

    smiles = one_hot_tokenizer.pad_smiles(smiles='COc(c1)cccc1C#N')
    print(f'smiles = {smiles} with length = {len(smiles)}')

    smiles_encoded = one_hot_tokenizer.encode_one_hot(smiles='COc(c1)cccc1C#N')
    np.set_printoptions(threshold=np.inf)
    print(f'smiles_encoded = {smiles_encoded}')
    print(f'smiles_encoded.shape = {smiles_encoded.shape}')

    list_encoded_smiles = one_hot_tokenizer.tokenize(list_smiles=['COc(c1)cccc1C#N'])
    print(f'\ntokenizer for a list of Smiles = {list_encoded_smiles}')
    print(f'list_encoded_smiles.shape = {list_encoded_smiles.shape}')

    list_smiles_get_back = one_hot_tokenizer.decode_one_hot(list_encoded_smiles)
    print(f'list_smiles_get_back = {list_smiles_get_back}')

    print('\ndecode smiles from indexes - testing with fake smiles')
    fake_smiles = decode_smiles_from_indexes(vec=np.array([0, 3, 1, 2, 4, 5]),
                                             charset='abcdef')
    print(f'fake_smiles = {fake_smiles}')


##################################################################
#               Define the architecture of the model
#################################################################
import torch
import torch.utils.data
from torch import nn
import numpy as np

from torch import optim
import torch.nn.functional as F
import os
from datetime import datetime
from transformers import GPT2Tokenizer

import sys

try:
    import smiles_gpt
except ImportError as e:
    print(f"Error importing smiles_gpt: {e}")
    sys.exit(1)

from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

# Set the fraction of GPU memory to use
fraction = 0.8
torch.cuda.set_per_process_memory_fraction(fraction, device=0)

def vae_loss(x_reconstructed, x, z_mean, z_logvar, kl_weight=1.0):  # Added kl_weight
    bce_loss = F.binary_cross_entropy(input=x_reconstructed, target=x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return bce_loss + kl_weight * kl_loss  # Weighted KL loss

import sys
from biformer import BiFormer, Block

# BiFormer Parameters (Consider tuning these)
depth = [3, 4, 8, 3]
in_chans = 3
num_classes = 1000
embed_dim = [64, 128, 320, 512]
head_dim = 64
qk_scale = None
representation_size = None
drop_path_rate = 0.
drop_rate = 0.
use_checkpoint_stages = []
n_win = 7
kv_downsample_mode = 'ada_avgpool'
kv_per_wins = [2, 2, -1, -1]
topks = [8, 8, -1, -1]
side_dwconv = 5
layer_scale_init_value = -1
qk_dims = [None, None, None, None]
param_routing = False
diff_routing = False
soft_routing = False
pre_norm = True
pe = None
pe_stages = [0]
before_attn_dwconv = 3
auto_pad = True
kv_downsample_kernels = [4, 2, 1, 1]
kv_downsample_ratios = [4, 2, 1, 1]  # -> kv_per_win = [2, 2, 2, 1]
mlp_ratios = [4, 4, 4, 4]
param_attention = 'qkvo'
mlp_dwconv = False

# Drop path rates
dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]

class VAE(nn.Module):
    '''
    Input data:
        Shape = (batch, 100, 48)
    '''

    def __init__(self, checkpoint_path=os.path.join(smiles_gpt_path, "checkpoints/benchmark-10m")):
        super(VAE, self).__init__()

        self.conv_1 = nn.Conv1d(in_channels=100, out_channels=9, kernel_size=9, stride=1)
        self.conv_2 = nn.Conv1d(in_channels=9, out_channels=9, kernel_size=9, stride=1)
        self.conv_3 = nn.Conv1d(in_channels=9, out_channels=10, kernel_size=11, stride=1)

        self.biformer_module = Block(dim=1,
                                     drop_path=dp_rates[0],
                                     layer_scale_init_value=layer_scale_init_value,
                                     topk=topks[0],
                                     num_heads=1,
                                     n_win=n_win,
                                     qk_dim=qk_dims[0],
                                     qk_scale=qk_scale,
                                     kv_per_win=kv_per_wins[0],
                                     kv_downsample_ratio=kv_downsample_ratios[0],
                                     kv_downsample_kernel=kv_downsample_kernels[0],
                                     kv_downsample_mode=kv_downsample_mode,
                                     param_attention=param_attention,
                                     param_routing=param_routing,
                                     diff_routing=diff_routing,
                                     soft_routing=soft_routing,
                                     mlp_ratio=mlp_ratios[0],
                                     mlp_dwconv=mlp_dwconv,
                                     side_dwconv=side_dwconv,
                                     before_attn_dwconv=before_attn_dwconv,
                                     pre_norm=pre_norm,
                                     auto_pad=auto_pad)

        self.fc_0 = nn.Linear(in_features=100, out_features=435)
        self.fc_1 = nn.Linear(in_features=435, out_features=292)
        self.fc_2 = nn.Linear(in_features=435, out_features=292)
        self.fc_3 = nn.Linear(in_features=292, out_features=576 * 100)
        self.fc_4 = nn.Linear(in_features=576, out_features=48)  # Added fc_4 layer

        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        # Load pre-trained smiles-gpt model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
        self.smiles_gpt = GPT2LMHeadModel.from_pretrained(checkpoint_path)

        # Freeze all parameters in pre-trained model
        for param in self.smiles_gpt.parameters():
            param.requires_grad = False

        # Unfreeze the last two blocks (Experiment with more or fewer)
        for param in self.smiles_gpt.transformer.h[-2:].parameters():
            param.requires_grad = True

    def get_parameter_groups(self):
        # Separate GPT parameters from the rest
        gpt_params = list(self.smiles_gpt.parameters())
        non_gpt_params = [p for n, p in self.named_parameters() if not any(nd in n for nd in ['smiles_gpt'])]
        return gpt_params, non_gpt_params

    def encode(self, x):
        # Convolutional layer
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x1 = torch.permute(x, (0, 2, 1))

        x = torch.matmul(x, x1)[:, None, :, :]

        # BiFormer Module
        x = self.biformer_module(x)

        # Fatten 2 last dimension but keep the batch_size
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = F.selu(self.fc_0(x))

        # return z_mean and z_logvar
        return self.fc_1(x), self.fc_2(x)

    def sampling(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        epsilon = 1e-2 * torch.randn_like(input=std)
        return z_mean + std * epsilon

    def decode(self, z, temperature=1.0, top_k=10):
        batch_size = z.size(0)
        sequence_length = 100  # Fixed sequence length for SMILES
        embedding_size = 576  # Calculate embedding size dynamically

        # Ensure the reshaping is valid
        if z.size(1) != sequence_length * embedding_size:
            raise ValueError(
                f"Invalid tensor shape for reshaping. Expected {sequence_length * embedding_size} elements, but got {z.size(1)}.")

        # Reshape z to (batch_size, sequence_length, embedding_size)
        z = z.view(batch_size, sequence_length, embedding_size)

        # Generate outputs using the pre-trained GPT model
        outputs = self.smiles_gpt(inputs_embeds=z, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Use the last hidden state

        # Apply the fc_4 layer to the hidden states
        logits_processed = torch.softmax(self.fc_4(hidden_states), dim=-1)

        if top_k > 0:
            top_k_values, top_k_indices = torch.topk(logits_processed, top_k, dim=-1)
            logits_processed = torch.zeros_like(logits_processed).scatter_(-1, top_k_indices, top_k_values)

        return logits_processed

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        z = F.selu(self.fc_3(z))
        x_reconstructed = self.decode(z)
        return x_reconstructed, z_mean, z_logvar
    

def test_class_VAE():
    batch = 64
    inputs = torch.rand(batch, 100, 48)
    y, z_mean, z_logvar = VAE().forward(x=inputs)
    print(f'output: y.shape = {y.shape}')
    print(f'latent space: z_mean.shape = {z_mean.shape}')
    print(f'latent space: z_logvar.shape = {z_logvar.shape}')


if __name__ == '__main__':
    print('Run a test for forward VAE')
    test_class_VAE()

#######################################
#  defining the training function which will be used to train and fine-tune the model
######################################
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data

def train(train_data=os.path.join(path_project),
          val_data=os.path.join(path_project),
          path_checkpoint=r'',
          resume_from_checkpoint=r'',
          batch_size=512,
          epochs=1000,  # Remaining epochs to train
          log_dir=log_dir,
          log_file_name='training_improved.txt',
          learning_rate_gpt=1e-6,
          learning_rate_vae=1e-3,
          weight_decay=1e-5,
          kl_weight=1.0,  # Weight for the KL divergence term
          scheduler_type='ReduceLROnPlateau',  # 'ReduceLROnPlateau' or 'CosineAnnealingLR'
          gradient_clip=1.0
          ):

    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Construct the full log file path
    log_file_path = os.path.join(log_dir, log_file_name)

    # Open the log file for writing
    with open(log_file_path, 'a') as log:
        try:
            # Load training data
            train_data = np.load(train_data)['arr'].astype(np.float32)  # Use the train_data parameter
            train_data = torch.utils.data.TensorDataset(torch.from_numpy(train_data))
            train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

            # Load validation data
            val_data = np.load(val_data)['arr'].astype(np.float32)  # Use the val_data parameter
            val_data = torch.utils.data.TensorDataset(torch.from_numpy(val_data))
            val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

            # Model, optimizer, and device setup
            torch.manual_seed(42)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f'device = {device}')
            model = VAE()
            model.to(device)

            # Get parameter groups
            gpt_params, non_gpt_params = model.get_parameter_groups()

            # Create optimizer with different learning rates and weight decay
            optimizer = ADOPT([
                {'params': gpt_params, 'lr': learning_rate_gpt, 'weight_decay': weight_decay},
                {'params': non_gpt_params, 'lr': learning_rate_vae, 'weight_decay': weight_decay}
            ])

            # Create a learning rate scheduler
            if scheduler_type == 'ReduceLROnPlateau':
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10) # Increased patience
            elif scheduler_type == 'CosineAnnealingLR':
                scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

            # Load checkpoint if provided
            start_epoch, stop_epoch = 0, epochs
            train_loss = 0.0
            val_loss = 0.0
            if resume_from_checkpoint:
                checkpoint_files = [f for f in os.listdir(resume_from_checkpoint) if
                                    f.startswith('checkpoint_') and f.endswith('.pt')]
                checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
                if checkpoint_files:
                    last_checkpoint_file = checkpoint_files[-1]
                    last_checkpoint_path = os.path.join(resume_from_checkpoint, last_checkpoint_file)
                    print(f'Loading checkpoint from: {last_checkpoint_path}')
                    checkpoint = torch.load(last_checkpoint_path, map_location=device)# weights_only=True
                    print(f'Checkpoint keys: {checkpoint.keys()}')
                    model.load_state_dict(checkpoint['model'], strict=False)
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    start_epoch = checkpoint['epoch'] + 1
                    stop_epoch = start_epoch + epochs
                    train_loss = checkpoint.get('train_loss', 0.0)
                    val_loss = checkpoint.get('val_loss', 0.0)

                    # Load scheduler state if it exists
                    if 'scheduler' in checkpoint:
                        scheduler.load_state_dict(checkpoint['scheduler'])

            # Update learning rate scheduler if using ReduceLROnPlateau and resuming
            if scheduler_type == 'ReduceLROnPlateau' and resume_from_checkpoint:
                scheduler.step(val_loss)

            # Early stopping variables
            best_val_loss = float('inf')
            best_val_epoch = -1  
            patience = 30  
            patience_counter = 0

            for epoch in range(start_epoch, stop_epoch):
                model.train()
                train_loss = 0
                uniqueness_calc = []
                metric_calc_dr = []
                smiles_list = []
                optimizer.zero_grad()  # Reset gradients at the start of each epoch

                for batch_idx, data in enumerate(train_data_loader):
                    data = data[0].to(device)
                    output, z_mean, z_logvar = model(data)
                    loss = vae_loss(output, data, z_mean, z_logvar, kl_weight=kl_weight) # Use weighted KL loss
                    loss.backward()

                    # Gradient clipping
                    if gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                    optimizer.step()  # Update parameters
                    optimizer.zero_grad()  # Reset gradients

                    train_loss += loss.item()
                    if batch_idx % 100 == 0:
                        log.write(f'\nepoch/batch_idx: {epoch}/{batch_idx}\t loss = {loss: .4f}\n')
                        log.flush()
                        input_data = data.cpu().numpy()
                        input_smiles = OneHotTokenizer().decode_one_hot(list_encoded_smiles=[input_data[0]])
                        log.write(f'\tFor testing: The first input smiles of batch={batch_size} Smiles\n')
                        log.flush()
                        log.write(f'\t{input_smiles}\n')
                        log.flush()

                        output_data = output.cpu().detach().numpy()
                        output_smiles = OneHotTokenizer().decode_one_hot(list_encoded_smiles=[output_data[0]])
                        log.write(f'\tFor testing: The first output smiles of {len(output_data)} generated Smiles\n')
                        log.flush()
                        log.write(f'\t{output_smiles}\n')
                        log.flush()

                        smiles_list.append(input_smiles[0][0])
                        smiles_list.append(output_smiles[0][0])
                        log.write(f'\tLogged SMILES: input: {input_smiles[0][0]}, output: {output_smiles[0][0]}\n')
                        log.flush()

                train_loss /= len(train_data_loader.dataset)
                log.write(f'Average train loss of this epoch = {train_loss}\n')
                log.flush()

                # Validation loop
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for data in val_data_loader:
                        data = data[0].to(device)
                        output, z_mean, z_logvar = model(data)
                        loss = vae_loss(output, data, z_mean, z_logvar, kl_weight=kl_weight) # Use weighted KL loss
                        val_loss += loss.item()

                val_loss /= len(val_data_loader.dataset)
                log.write(f'Epoch {epoch} Validation Loss: {val_loss}\n')
                log.flush()

                # Step the scheduler based on validation loss
                if scheduler_type == 'ReduceLROnPlateau':
                    scheduler.step(val_loss)
                elif scheduler_type == 'CosineAnnealingLR':
                    scheduler.step()

                # Log the learning rate
                current_lr_gpt = optimizer.param_groups[0]["lr"]
                current_lr_vae = optimizer.param_groups[1]["lr"]
                print(
                    f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, LR_GPT: {current_lr_gpt}, LR_VAE: {current_lr_vae}')
                log.write(
                    f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, LR_GPT: {current_lr_gpt}, LR_VAE: {current_lr_vae}\n')
                log.flush()

                # Check for early stopping and best validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch  # Store the epoch number
                    patience_counter = 0
                    log.write(f'New best validation loss: {best_val_loss} at epoch {best_val_epoch}\n')
                    log.flush()

                    # Save the best model immediately
                    if path_checkpoint:
                        best_checkpoint_file = f'checkpoint_{best_val_epoch:03d}.pt'
                        best_checkpoint_save_path = os.path.join(path_checkpoint, f'best_checkpoint.pt')
                        torch.save({
                            'epoch': best_val_epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'scheduler': scheduler.state_dict()
                        }, best_checkpoint_save_path)
                        log.write(f'Best checkpoint saved to {best_checkpoint_save_path} \n')
                        log.flush()

                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        log.write(
                            f'Early stopping at epoch {epoch} as validation loss did not improve for {patience} epochs.\n')
                        log.flush()
                        break

                # Save SMILES strings to Excel file
                df = pd.DataFrame(smiles_list, columns=['SMILES'])
                excel_file_path = os.path.join(log_dir, f'epoch_{epoch}.xlsx')
                df.to_excel(excel_file_path, index=False)


        except Exception as e:
            log.write(f'Error occurred: {e}\n')
            log.flush()
            raise e  

    return train_loss, val_loss, uniqueness_calc, metric_calc_dr

# Example usage
if __name__ == '__main__':
    print('Starting training...')
    try:
        train_loss, val_loss, uniqueness_calc, metric_calc_dr = train(
            epochs=1000, # Train for more epochs
            learning_rate_gpt=5e-7,  
            learning_rate_vae=5e-4, 
            weight_decay=1e-6,      
            kl_weight=0.5,       
            scheduler_type='CosineAnnealingLR', 
            gradient_clip=1.0       
        )
        print(f'Final Train Loss: {train_loss}')
        print(f'Final Validation Loss: {val_loss}')
    except Exception as e:
        print(f'Error occurred: {e}')