import numpy as np
import pandas as pd
import torch
from torch import nn
from utils import *

import time
import matplotlib.pyplot as plt

class CubeTransformer(nn.Module):
    def __init__(self, d_model = 256, n_heads = 16, d_ffwd = 512, n_layers = 8, activation=nn.functional.gelu):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn((1, 1, d_model)))
        
        self.piece_embed_table = nn.Embedding(20, d_model)
        self.placement_embed_table = nn.Embedding(48, d_model)
        
        self.layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ffwd, 
                                                batch_first=True, 
                                                dropout=0, 
                                                activation=activation)
        self.stack = nn.TransformerEncoder(self.layer, n_layers)

        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        B = x.shape[0]
        
        # x : (B, 20, 2)
        piece_embeddings = self.piece_embed_table(x[:, :, 0]) # (B, 20, d_model)
        placement_embeddings = self.placement_embed_table(x[:, :, 1])
        embeddings = piece_embeddings + placement_embeddings # (B, 20, d_model)
        
        # concatenate cls token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, d_model)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1) # (B, 21, d_model)

        # feed through transformer
        embeddings = self.stack(embeddings) # (B, 21, d_model)
        
        # isolate cls and run through linear for final prediction
        cls_output = embeddings[:, 0]  # (B, d_model)
        return self.regressor(cls_output).squeeze(-1)  # (B,)
