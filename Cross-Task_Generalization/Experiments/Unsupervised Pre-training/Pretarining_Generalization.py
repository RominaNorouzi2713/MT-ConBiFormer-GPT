import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import warnings

# Please set the paths
path_project = r''
smiles_gpt_path = r''
sys.path.append(smiles_gpt_path)
import smiles_gpt

log_dir = r''
path_checkpoint = r''
sys.path.append('')

######################################################################
#          Utilities for tokenizing SMILES sequences
######################################################################

CHARSET = [' ', '#', '(', ')', '+', '-', '/', '1', '2', '3', '4', '5', '6', '7', '8', '=', '@', 'A',
           'B', 'C', 'F', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'S', '[', '\\', ']', 'c', 'd', 'e', 'g',
           'i', 'l', 'n', 'o', 'r', 's', 't', 'u']

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
                one_hot = np.argmax(list_encoded_smiles[smiles_idx][row_idx])
                smiles_string += self.charset[one_hot]
            list_smiles_get_back.append([smiles_string.strip()])
        return list_smiles_get_back

def decode_smiles_from_indexes(vec, charset=CHARSET, decode=True):
    if decode:
        try:
            charset = np.array([v.decode('utf-8') for v in charset])
        except:
            pass
    return ''.join(map(lambda x: charset[x], vec)).strip()

##################################################################
#          Define the architecture of the model
#################################################################
import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class VAE(nn.Module):
    '''
    Variational Autoencoder with a GPT-based decoder.
    Input Shape = (batch_size, 100, 44)
    '''
    def __init__(self, checkpoint_path=os.path.join(smiles_gpt_path, "checkpoints/benchmark-5m")):
        super(VAE, self).__init__()

        self.conv_1 = nn.Conv1d(in_channels=100, out_channels=9, kernel_size=9)
        self.conv_2 = nn.Conv1d(in_channels=9, out_channels=9, kernel_size=9)
        self.conv_3 = nn.Conv1d(in_channels=9, out_channels=10, kernel_size=11)

        self.fc_0 = nn.Linear(in_features=100, out_features=435)
        self.fc_mean = nn.Linear(in_features=435, out_features=292)
        self.fc_logvar = nn.Linear(in_features=435, out_features=292)

        self.relu = nn.ReLU()

    def encode(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x1 = torch.permute(x, (0, 2, 1))
        
        x = torch.matmul(x, x1)[:, None, :, :]

        x = x.view(x.size(0), -1)
        x = F.selu(self.fc_0(x))

        return self.fc_mean(x), self.fc_logvar(x)

    def sampling(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        epsilon = 1e-2 * torch.randn_like(input=std)
        return z_mean + std * epsilon

    def decode(self, z):
        pass

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        
        x_reconstructed = self.decode(z)
        
        return x_reconstructed, z_mean, z_logvar

def vae_loss(x_reconstructed, x, z_mean, z_logvar, kl_weight=1.0):
    bce_loss = F.binary_cross_entropy(input=x_reconstructed, target=x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return bce_loss + kl_weight * kl_loss


#######################################
#   Defining the training function
######################################
def train(train_data_path=os.path.join(),
          val_data_path=os.path.join(''),
          path_checkpoint=r'',
          batch_size=512,
          epochs=100,
          log_dir=log_dir):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, 'training_log.txt')

    with open(log_file_path, 'a') as log:
        try:
            train_data = np.load(train_data_path)['arr'].astype(np.float32)
            train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = VAE()
            
            optimizer = None
            
            best_val_loss = float('inf')
            
            for epoch in range(epochs):
                model.train()
                total_train_loss = 0

                log.write(f'Epoch {epoch} completed...\n')
                log.flush()


        except Exception as e:
            log.write(f'An error occurred: {e}\n')
            raise e