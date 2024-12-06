import torch
from torch import nn, optim
from models import JEPA, Encoder, Predictor, Prober
from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
from tqdm import tqdm
from torch.utils.data import DataLoader
import os

def train_jepa_model(train_loader, epochs=10, learning_rate=1e-3, device="cuda"):
    encoder = Encoder(input_channels=2, embedding_dim=256).to(device)
    predictor = Predictor(embedding_dim=256, action_dim=2).to(device)

    jepa_model = JEPA(encoder, predictor).to(device)

    def jepa_loss(predicted, target):
        return torch.mean((predicted - target) ** 2)

    optimizer = optim.Adam(jepa_model.parameters(), lr=learning_rate)
    best_val_loss = 10000000

    jepa_model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            observations = batch.states.to(device)
            actions = batch.actions.to(device)

            optimizer.zero_grad()
        
            target_states = []
            _, timesteps, _, _, _ = observations.size()
            for t in range(timesteps):
                s_target = encoder(observations[:, t])
                target_states.append(s_target)
            target_states = torch.stack(target_states, dim=1)

            predicted_states = jepa_model(observations, actions)

            loss = jepa_loss(predicted_states, target_states)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - JEPA Training Loss: {epoch_loss / len(train_loader)}")
        
        if best_val_loss > epoch_loss:
            best_val_loss = epoch_loss
            torch.save(jepa_model.state_dict(), "trained_jepa_model.pth")
            print("JEPA model trained and saved.")

    return jepa_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = "/scratch/DL24FA"
    train_loader = create_wall_dataloader(data_path=f"{data_path}/train", probing=False, device=device, batch_size=64, train=True)
    # val_loader = create_wall_dataloader(data_path=f"{data_path}/val", probing=False, device=device, batch_size=64, train=False)

    train_jepa_model(train_loader, epochs=50, device=device)

if __name__ == "__main__":
    main()
