import torch
from torch import optim
import torch.nn as nn
import timm
from dataset import create_wall_dataloader
from tqdm import tqdm

class ViTEncoder(nn.Module):
    def __init__(self, img_size=64, in_chans=2, embed_dim=256, patch_size=8):
        super().__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            in_chans=in_chans,
            img_size=img_size,
            num_classes=0
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        features = self.vit(x)
        features = features.view(b, t, *features.shape[1:])
        return features

class Predictor(nn.Module):
    def __init__(self, embed_dim=768, action_dim=2, hidden_dim=512, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embed_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, embed_dim)

    def forward(self, latent_embeddings, actions):
        combined = torch.cat([latent_embeddings, actions], dim=-1)
        outputs, _ = self.lstm(combined)
        predictions = self.fc(outputs)
        return predictions
    
class JEPA(nn.Module):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, observations, actions, mode="train"):
        if mode == "train":
            encoded_observations = self.encoder(observations)
            first_observation = encoded_observations[:, 0]
            
            predictions = [first_observation]
            for n in range(actions.size(1)):
                pred = self.predictor(encoded_observations[:, n], actions[:, n])
                predictions.append(pred)
            
            predictions = torch.stack(predictions, dim=1)
            return predictions

        elif mode == "inference":
            s_prev = self.encoder(observations)[:, 0]
            
            predictions = [s_prev]
            for n in range(actions.size(1)):
                s_prev = self.predictor(s_prev, actions[:, n])
                predictions.append(s_prev)
            
            predictions = torch.stack(predictions, dim=1)
            return predictions


def train_jepa_model(train_loader, epochs=10, learning_rate=1e-3, device="cuda"):
    action_dim = 2
    embed_dim = 768

    encoder = ViTEncoder(img_size=65, in_chans=2, embed_dim=embed_dim).to(device)
    predictor = Predictor(embed_dim=embed_dim, action_dim=action_dim).to(device)
    jepa_model = JEPA(encoder, predictor).to(device)

    def jepa_loss(predicted, target):
        return torch.mean((predicted - target) ** 2)

    optimizer = optim.Adam(jepa_model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')

    jepa_model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            observations = batch.states.to(device)
            actions = batch.actions.to(device)

            optimizer.zero_grad()
        
            target_states = encoder(observations)
            predicted_states = jepa_model(observations, actions)

            loss = jepa_loss(predicted_states, target_states)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - JEPA Training Loss: {epoch_loss / len(train_loader)}")
        
        if best_val_loss > epoch_loss:
            best_val_loss = epoch_loss
            torch.save(jepa_model.state_dict(), "trained_jepa_vit_model.pth")
            print("JEPA model trained and saved.")

    return jepa_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = "/Users/archit/Desktop/DL/final-project/DL24FA"
    train_loader = create_wall_dataloader(data_path=f"{data_path}/train", probing=False, device=device, batch_size=64, train=True)
    # val_loader = create_wall_dataloader(data_path=f"{data_path}/val", probing=False, device=device, batch_size=64, train=False)

    train_jepa_model(train_loader, epochs=50, device=device)

if __name__ == "__main__":
    main()