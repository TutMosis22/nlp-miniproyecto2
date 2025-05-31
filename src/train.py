import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

def train_model(model, X_train, y_train, X_val, y_val, epochs=5, lr=0.001):
    """
    Entrena el modelo y evalúa la precisión.
    """
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Convertimos a tensores de PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
        # Evaluación simple
        model.eval()
        with torch.no_grad():
            preds_val = model(X_val_tensor).round()
            acc = accuracy_score(y_val_tensor, preds_val)
            
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")