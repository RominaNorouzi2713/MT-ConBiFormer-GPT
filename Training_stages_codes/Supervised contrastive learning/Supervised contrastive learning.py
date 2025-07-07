
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
sys.path.append("")
from C10_vae_gt_r_b import OneHotTokenizer, VAE

# Setup logging
log_path = ""
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')
print(f"✅ Logs will be written to {log_path}")

# Dataset class with tokenization
tokenizer = OneHotTokenizer(fixed_length=100)

class SMILESDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.smiles = self.df['Canonical_SMILES'].values
        self.labels = self.df['Label'].values
        valid_smiles = [s for s in self.smiles if Chem.MolFromSmiles(s)]
        self.data = [tokenizer.encode_one_hot(s) for s in valid_smiles]
        self.labels = [l for s, l in zip(self.smiles, self.labels) if Chem.MolFromSmiles(s)]

        # Save valid tokenized dataset
        save_df = pd.DataFrame({
            'SMILES': valid_smiles,
            'label': self.labels
        })
        save_path = ''
        save_df.to_csv(save_path, index=False)
        print(f"✅ Tokenized SMILES saved to {save_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]).float(), torch.tensor(self.labels[idx]).long()

# SupCon loss
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(contrast, dim=1, keepdim=True)
        logits = contrast - logits_max.detach()
        exp_logits = torch.exp(logits) * (1 - torch.eye(batch_size).to(device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        return loss

# Encoder and decoder from VAE model
class Encoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        z_mean, _ = self.vae.encode(x)
        return z_mean

class Decoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        z_proj = F.selu(self.vae.fc_3(z))
        return self.vae.decode(z_proj)

# Projection head
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=292, proj_dim=128):
        super(ProjectionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, proj_dim)
        )

    def forward(self, x):
        return F.normalize(self.mlp(x), dim=1)

# Linear classifier
class LinearClassifier(nn.Module):
    def __init__(self, input_dim=292, num_classes=2):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Training loop
def train_supcon_model():
    dataset = SMILESDataset('')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    vae_model = VAE().cuda()

    # Load pretrained checkpoint
    checkpoint_path = ''
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    vae_model.load_state_dict(checkpoint['model'], strict=False)
    print("✅ Loaded VAE weights from best pretraining checkpoint.")
    logging.info("Loaded VAE weights from checkpoint.")

    encoder = Encoder(vae_model).cuda()
    decoder = Decoder(vae_model).cuda()
    projector = ProjectionHead().cuda()

    criterion_supcon = SupConLoss()
    criterion_recon = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()) + list(decoder.parameters()),
        lr=3e-4,
        weight_decay=1e-5
    )

    for epoch in range(1, 201):
        encoder.train(), projector.train(), decoder.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            z = encoder(x)
            z_proj = projector(z)
            x_recon = decoder(z)
            loss_supcon = criterion_supcon(z_proj, y)
            loss_recon = criterion_recon(x_recon, x)
            loss = loss_supcon + loss_recon
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")
        logging.info(f"Epoch {epoch} - Loss: {avg_loss:.4f}")

    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict()
    }, '/encoder_decoder_supcon.pt')

    encoder.eval()
    reps, targets = [], []
    for x, y in dataloader:
        with torch.no_grad():
            x = x.cuda()
            reps.append(encoder(x).cpu().numpy())
            targets.extend(y.numpy())
    X = np.vstack(reps)
    y = np.array(targets)
    clf = LogisticRegression(max_iter=2000).fit(X, y)
    y_pred = clf.predict(X)
    report = classification_report(y, y_pred, digits=4)
    print(report)
    with open('/classification_report.txt', 'w') as f:
        f.write(report)
    logging.info("Classifier evaluation complete.")

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('/confusion_matrix.png')
    logging.info("Saved confusion matrix plot.")

    # TSNE plot
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='viridis', s=10)
    plt.title('TSNE of Latent Representations')
    plt.savefig('/tsne_latent_plot.png')
    logging.info("Saved TSNE plot.")

if __name__ == '__main__':
    train_supcon_model()
