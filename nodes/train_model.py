#!/usr/bin/env python3

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import os
import numpy as np

# --- 1. Custom Dataset for our CSV data (No changes here) ---
class DrivingDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # We add the position data as features now
        self.features = self.data[['tau_el', 'tau_er', 'tau_l', 'tau_r', 'tau_c', 'default_linear_x', 'default_angular_z',
                                   'angular_z_state']].values
        self.labels = self.data[['expert_linear_x', 'expert_angular_z']].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, labels

# --- 2. Neural Network Architecture ---
# We update the input size from 6 to 9 to include position data
class BehavioralCloningModel(nn.Module):
    def __init__(self):
        super(BehavioralCloningModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)

# --- 3. Improved Training Function ---
def train_model():
    # --- Data Loading and Splitting ---
    csv_file_path = os.path.expanduser('~/recorded_data_full.csv')
    dataset = DrivingDataset(csv_file_path)
    
    # Split dataset into training (80%) and validation (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders for both sets
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # --- Model, Loss, and Optimizer Setup ---
    model = BehavioralCloningModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 1000
    best_val_loss = float('inf') # Initialize with a very high value
    model_path = os.path.expanduser('~/il_model_full.pth')

    print("Starting model training...")
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        for features, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_dataloader)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad(): # No need to calculate gradients
            for features, labels in val_dataloader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(val_dataloader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # --- Save the Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f'New best model saved with validation loss: {best_val_loss:.4f}')

    print(f"Finished training. Best model saved to {model_path}")

if __name__ == '__main__':
    train_model()