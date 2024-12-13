import torch
from torch import nn, optim
import torch.nn.functional as F
from dataset import WallDataset
import numpy as np
from tqdm import tqdm
from models import JEPA, Encoder, Predictor, Prober
from torch.utils.data import random_split
import sys

def create_dataloader(data_path, device="cuda", batch_size=64):
    ds = WallDataset(data_path=data_path, device=device, probing=False)
    train_ds, val_ds = random_split(ds, [0.8, 0.2])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True, drop_last=True, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size, shuffle=True, drop_last=True, pin_memory=False)
    return train_loader, val_loader

def vicreg_loss(x, y):
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1, y.shape[-1])

    invariance_loss = F.mse_loss(x, y)
    
    x_std = torch.sqrt(x.var(dim=0) + 1e-4)
    y_std = torch.sqrt(y.var(dim=0) + 1e-4)
    
    std_loss = torch.mean(F.relu(1 - x_std)) + torch.mean(F.relu(1 - y_std))
    
    def compute_covariance_loss(t):
        t = t - t.mean(dim=0)
        cov_matrix = (t.T @ t) / (t.shape[0] - 1)
        
        cov_matrix = cov_matrix - torch.eye(cov_matrix.shape[0], device=t.device)
        
        cov_loss = torch.sum(cov_matrix ** 2) / cov_matrix.shape[0]
        return cov_loss
    
    cov_x = compute_covariance_loss(x)
    cov_y = compute_covariance_loss(y)
    
    return invariance_loss, std_loss, (cov_x + cov_y)

def train_jepa_model(train_loader, val_loader, epochs=10, learning_rate=1e-4, device="cuda"):
    grad_clip = 1.0
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    inv_lam = int(sys.argv[1])
    var_gam = int(sys.argv[2])
    cov_mu = int(sys.argv[3])
    lr = float(sys.argv[4])
    
    jepa_model = JEPA(embedding_dim=256, action_dim=2, momentum=0.99).to(device)

    optimizer = optim.AdamW(jepa_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    print(f"using coefficient values: inv_lam: {inv_lam}, var_gam: {var_gam}, cov_mu: {cov_mu}, lr: {lr}")

    jepa_model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        val_epoch_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}")):
            observations = batch.states.to(device)
            actions = batch.actions.to(device)

            optimizer.zero_grad()
            predicted_states = jepa_model(observations, actions)
        
            with torch.no_grad():
                b, t, c, h, w = observations.size()
                targets = observations.view(-1, c, h, w)
                target_states = jepa_model.target_encoder(targets)
                target_states = target_states.view(b, t, -1)


            invariance_loss, std_loss, cov_loss = vicreg_loss(predicted_states, target_states)
            loss = inv_lam * invariance_loss + var_gam * std_loss + cov_mu * cov_loss
            loss.backward()
            nn.utils.clip_grad_norm_(jepa_model.parameters(), grad_clip)

            optimizer.step()
            jepa_model.update_target_encoder()

            epoch_loss += loss.item()

            if batch_idx % 500 == 0:
                print(
                    f"Training Epoch [{epoch+1}/{epochs}], Batch [{batch_idx} / {len(train_loader)}], Loss: {loss:.4f}, Inv Loss: {invariance_loss:.4f}, Std Loss: {std_loss:.4f}, Cov Loss: {cov_loss:.4f}"
                )

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}")):
                observations = batch.states.to(device)
                actions = batch.actions.to(device)
                predicted_states = jepa_model(observations, actions)

                b, t, c, h, w = observations.size()
                targets = observations.view(-1, c, h, w)
                target_states = jepa_model.target_encoder(targets)
                target_states = target_states.view(b, t, -1)

                invariance_loss, std_loss, cov_loss = vicreg_loss(predicted_states, target_states)
                loss = inv_lam * invariance_loss + var_gam * std_loss + cov_mu * cov_loss

                val_epoch_loss += loss.item()

                if batch_idx % 500 == 0:
                    print(
                        f"Validation Epoch [{epoch+1}/{epochs}], Batch [{batch_idx} / {len(train_loader)}], Loss: {loss:.4f}, Inv Loss: {invariance_loss:.4f}, Std Loss: {std_loss:.4f}, Cov Loss: {cov_loss:.4f}"
                    )
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_val_epoch_loss = val_epoch_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} - JEPA Training Loss: {avg_epoch_loss}")
        print(f"Epoch {epoch+1}/{epochs} - JEPA Validation Loss: {avg_val_epoch_loss}")
        scheduler.step()
        
        if best_train_loss > epoch_loss and best_val_loss > val_epoch_loss:
            best_train_loss = epoch_loss
            val_epoch_loss = val_epoch_loss
            torch.save(jepa_model.state_dict(), f"jepa_model_train_val_inv_lam_{inv_lam}_var_gam_{var_gam}_cov_mu_{cov_mu}_lr_{lr}.pth")
            print("JEPA model trained and saved.")

    return jepa_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "/scratch/DL24FA"
    train_loader, val_loader = create_dataloader(data_path=f"{data_path}/train", device=device, batch_size=64)
    train_jepa_model(train_loader, val_loader, epochs=30, device=device)

if __name__ == "__main__":
    main()

