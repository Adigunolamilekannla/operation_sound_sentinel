from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from src.utils.exception import CustomException
from src.utils.logger import logging
import sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def classification_report(y_true, y_pred):
    """
    Generate performance metrics for classification models.
    """
    try:
        logging.info("Calculating classification metrics...")

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

        logging.info("Classification metrics calculated successfully.")

        return {
            "accuracy_score": acc,
            "f1_score": f1,
            "precision_score": precision,
            "recall_score": recall
        }

    except Exception as e:
        raise CustomException(e, sys)
    

# ===================== CNN MODEL ==========================
class AudioCNN(nn.Module):
    def __init__(self, n_classes, in_channels=1, dropout=0.4):
        super().__init__()

        self.net = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.4),
            nn.MaxPool2d(2)
        )

        # Adaptive pooling for fixed-size output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.net(x)
        x = self.adaptive_pool(x)
        x = self.fc(x)
        return x


# ===================== OPTIMIZER AND LOSS ==========================
def get_optimizer(weight=None, n_classes=1):
    """
    Returns initialized model, loss function, and optimizer.
    """
    net = AudioCNN(n_classes=n_classes)
    lossFun = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=1e-5)
    return net, lossFun, optimizer


# ===================== TRAINING FUNCTION ==========================
def train_model(x_y_train_loader, device, num_epochs=20):
    torch.cuda.empty_cache()
    
    net, lossFun, optimizer = get_optimizer()
    net.to(device)

    accumulation_steps = 16

    for epoch in tqdm(range(num_epochs)):
        net.train()
        optimizer.zero_grad()

        batch_losses, batch_accs = [], []

        for i, (X, y) in enumerate(x_y_train_loader):
            X = X.to(device)
            y = y.to(device).float().view(-1, 1)

            # Forward pass
            y_pred = net(X.permute(0, 3, 1, 2))

            # Compute loss
            loss = lossFun(y_pred, y)
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(x_y_train_loader):
                optimizer.step()
                optimizer.zero_grad()

            preds = (torch.sigmoid(y_pred) >= 0.5).float()
            acc = (preds == y).float().mean().item()

            batch_losses.append(loss.item() * accumulation_steps)
            batch_accs.append(acc)

        logging.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Acc: {np.mean(batch_accs):.4f} | "
            f"Train Loss: {np.mean(batch_losses):.4f}"
        )

        torch.cuda.empty_cache()

    return net


# ===================== PREDICTION FUNCTION ==========================
def predict_audio(net, device, data_loader):
    """
    Predicts outputs on a given dataset.
    """
   

    preds, labels = [], []
    net.eval()

    with torch.no_grad():
        for X_val, y_val in data_loader:
            X_val = X_val.to(device)
            y_val = y_val.to(device).float().view(-1, 1)

            y_pred = net(X_val.permute(0, 3, 1, 2))
        

            preds_batch = (torch.sigmoid(y_pred) >= 0.5).float().squeeze()
            preds.extend(preds_batch.cpu().numpy())
            labels.extend(y_val.cpu().numpy())

    return preds, labels
