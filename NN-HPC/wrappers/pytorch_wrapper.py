import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from models.ft_transformer import FTTransformer

class SklearnFTTransformer(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        # Model params
        d_model=192,
        n_layers=3,
        n_heads=8,
        dropout=0.1,
        # Training params
        batch_size=256,
        epochs=100,
        learning_rate=1e-3,
        weight_decay=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        random_state=42,
        # Data params
        cat_indices=None, # List of indices for categorical columns
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.random_state = random_state
        self.cat_indices = cat_indices

    def fit(self, X, y):
        # Check inputs
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]
        
        # Determine categorical and numerical splits
        if self.cat_indices is None:
            # Assume no categoricals if not specified
            self.cat_indices_ = []
        else:
            # Normalize negative indices to positive
            n_cols = X.shape[1]
            self.cat_indices_ = [c if c >= 0 else n_cols + c for c in self.cat_indices]

        self.num_indices_ = [i for i in range(X.shape[1]) if i not in self.cat_indices_]
        
        # Calculate cardinalities for categorical columns
        # We assume input X has ordinal encoded categoricals (integers 0..N)
        # We reserve index 0 for "Unknown" (mapped from -1)
        self.cat_cardinalities_ = []
        if self.cat_indices_:
            for idx in self.cat_indices_:
                # Max value in train set (0..N-1)
                # We shift by +1, so max becomes N. Cardinality is N+1 (0..N)
                # Plus maybe a buffer? Safe to use max + 2
                max_val = int(np.max(X[:, idx]))
                card = max_val + 2 
                self.cat_cardinalities_.append(card)
        
        n_num_features = len(self.num_indices_)
        
        # Initialize Model
        torch.manual_seed(self.random_state)
        self.model_ = FTTransformer(
            n_num_features=n_num_features,
            cat_cardinalities=self.cat_cardinalities_,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=self.dropout
        ).to(self.device)
        
        # Prepare Data
        dataset = self._make_dataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer & Loss
        optimizer = optim.AdamW(
            self.model_.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()
        
        # Scheduler (OneCycleLR is very effective for Transformers)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.learning_rate, 
            steps_per_epoch=len(loader), 
            epochs=self.epochs,
            pct_start=0.3
        )
        
        # Training Loop
        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x_num, batch_x_cat, batch_y in loader:
                optimizer.zero_grad()
                
                preds = self.model_(batch_x_num, batch_x_cat)
                loss = criterion(preds.squeeze(), batch_y)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            # Optional: Print progress
            # if (epoch + 1) % 10 == 0:
            #     print(f"Epoch {epoch+1}/{self.epochs} Loss: {total_loss / len(loader):.4f}")
                
        return self

    def predict(self, X):
        check_is_fitted(self, ["model_", "num_indices_", "cat_indices_"])
        X = check_array(X, accept_sparse=False)
        
        dataset = self._make_dataset(X, y=None)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model_.eval()
        preds = []
        with torch.no_grad():
            for batch_x_num, batch_x_cat in loader:
                out = self.model_(batch_x_num, batch_x_cat)
                preds.append(out.cpu().numpy())
                
        return np.concatenate(preds).flatten()

    def _make_dataset(self, X, y=None):
        # Split Num/Cat
        if self.num_indices_:
            x_num = torch.tensor(X[:, self.num_indices_], dtype=torch.float32).to(self.device)
        else:
            x_num = None
            
        if self.cat_indices_:
            # Handle Ordinal Encoding with potential -1 for unknowns
            x_cat_np = X[:, self.cat_indices_].astype(np.int64)
            # Map -1 (unknown) to 0, and shift others by +1
            # If value is -1: -1 + 1 = 0.
            # If value is 0: 0 + 1 = 1.
            x_cat_np = x_cat_np + 1
            # Clip negative values just in case (though -1+1=0 is fine)
            x_cat_np = np.maximum(x_cat_np, 0)
            
            x_cat = torch.tensor(x_cat_np, dtype=torch.long).to(self.device)
        else:
            x_cat = None
            
        if y is not None:
            y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
            return TensorDataset(x_num, x_cat, y_tensor)
        else:
            # Hack for TensorDataset with optional tensors? 
            # TensorDataset requires all args to be tensors of same size dim 0.
            # If one is None, we can't use standard TensorDataset easily if we want to unpack cleanly.
            # Custom dataset is better.
            return _FTDataset(x_num, x_cat)

class _FTDataset(torch.utils.data.Dataset):
    def __init__(self, x_num, x_cat):
        self.x_num = x_num
        self.x_cat = x_cat
        self.n_samples = x_num.shape[0] if x_num is not None else x_cat.shape[0]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        xn = self.x_num[idx] if self.x_num is not None else torch.empty(0)
        xc = self.x_cat[idx] if self.x_cat is not None else torch.empty(0)
        return xn, xc
