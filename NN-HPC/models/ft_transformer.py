import torch
import torch.nn as nn
import torch.nn.functional as F

class ReGLU(nn.Module):
    """
    Rectified Gated Linear Unit (ReGLU).
    Computes: ReGLU(x) = a * ReLU(b), where x is split into a and b.
    """
    def forward(self, x):
        # x shape: (..., 2 * d_ff)
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)

class FeatureTokenizer(nn.Module):
    def __init__(self, n_num_features, cat_cardinalities, d_model):
        super().__init__()
        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities
        
        # Numerical Embeddings: x * W + b
        # Optimized implementation using broadcasting instead of loop over nn.Linear
        if n_num_features > 0:
            self.num_W = nn.Parameter(torch.randn(n_num_features, d_model) * 0.01)
            self.num_b = nn.Parameter(torch.zeros(n_num_features, d_model))
        
        # Categorical Embeddings
        if cat_cardinalities:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(card, d_model) for card in cat_cardinalities
            ])
        
        # [CLS] Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x_num, x_cat):
        """
        x_num: (batch_size, n_num_features) or None
        x_cat: (batch_size, n_cat_features) or None
        """
        batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]
        tokens = []
        
        # 1. [CLS] Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens.append(cls_tokens)
        
        # 2. Numerical Features
        if self.n_num_features > 0 and x_num is not None:
            # (batch, n_num, 1) * (1, n_num, d_model) + (1, n_num, d_model)
            # Broadcasting: x_num.unsqueeze(-1) is (batch, n_num, 1)
            # self.num_W is (n_num, d_model)
            num_embeds = x_num.unsqueeze(-1) * self.num_W.unsqueeze(0) + self.num_b.unsqueeze(0)
            tokens.append(num_embeds)
            
        # 3. Categorical Features
        if self.cat_cardinalities and x_cat is not None:
            cat_tokens = []
            for i, emb_layer in enumerate(self.cat_embeddings):
                cat_tokens.append(emb_layer(x_cat[:, i]).unsqueeze(1))
            tokens.append(torch.cat(cat_tokens, dim=1))
            
                # Concat all: (batch, 1 + n_num + n_cat, d_model)
        x = torch.cat(tokens, dim=1)
        return x

class FTTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        # Attention Block
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        # FeedForward Block (PreNorm + ReGLU)
        self.norm2 = nn.LayerNorm(d_model)
        
        # ReGLU requires input projection to 2 * d_ff
        # Standard Transformer d_ff is usually 4 * d_model.
        d_ff = int(4 * d_model * 2 / 3) # GEGLU/ReGLU paper suggestion: reduce d_ff slightly
        self.linear1 = nn.Linear(d_model, 2 * d_ff) 
        self.reglu = ReGLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # PreNorm Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)
        
        # PreNorm FeedForward with ReGLU
        x_norm = self.norm2(x)
        projected = self.linear1(x_norm)
        activated = self.reglu(projected)
        out = self.linear2(activated)
        x = x + self.dropout2(out)
        
        return x

class FTTransformer(nn.Module):
    def __init__(
        self, 
        n_num_features, 
        cat_cardinalities, 
        d_model=192, 
        n_layers=3, 
        n_heads=8, 
        dropout=0.1
    ):
        super().__init__()
        self.feature_tokenizer = FeatureTokenizer(n_num_features, cat_cardinalities, d_model)
        
        self.layers = nn.ModuleList([
            FTTransformerEncoderLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x_num, x_cat):
        # 1. Tokenize
        x = self.feature_tokenizer(x_num, x_cat)
        
        # 2. Transformer Layers
        for layer in self.layers:
            x = layer(x)
            
        # 3. Prediction Head (Use [CLS] token only)
        x = self.norm(x)
        cls_token = x[:, 0, :] # [CLS] is at index 0
        out = self.head(cls_token)
        
        return out

class FTTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        # Attention Block
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        # FeedForward Block (PreNorm + ReGLU)
        self.norm2 = nn.LayerNorm(d_model)
        
        # ReGLU requires input projection to 2 * d_ff
        # Standard Transformer d_ff is usually 4 * d_model.
        d_ff = 4 * d_model
        self.linear1 = nn.Linear(d_model, 2 * d_ff) 
        self.reglu = ReGLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # PreNorm Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)
        
        # PreNorm FeedForward with ReGLU
        x_norm = self.norm2(x)
        projected = self.linear1(x_norm)
        activated = self.reglu(projected)
        out = self.linear2(activated)
        x = x + self.dropout2(out)
        
        return x

class FTTransformer(nn.Module):
    """
    FT-Transformer for Regression on Tabular Data.
    Optimized for A100 (using standard PyTorch layers).
    """
    def __init__(self, 
                 n_num_features, 
                 cat_cardinalities, 
                 d_model=192, 
                 n_layers=3, 
                 n_heads=8, 
                 dropout=0.1):
        super().__init__()
        
        self.tokenizer = FeatureTokenizer(n_num_features, cat_cardinalities, d_model)
        
        self.layers = nn.ModuleList([
            FTTransformerEncoderLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Prediction Head (CLS token only)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_num, x_cat):
        """
        x_num: (batch, n_num_features)
        x_cat: (batch, n_cat_features) - LongTensor
        """
        # 1. Tokenization
        x = self.tokenizer(x_num, x_cat)
        
        # 2. Transformer Backbone
        for layer in self.layers:
            x = layer(x)
            
        # 3. Prediction Head (use [CLS] token at index 0)
        cls_output = x[:, 0, :]
        output = self.head(cls_output)
        
        return output
