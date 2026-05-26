# Import libraries 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 



# Multi layer perceptron ! 
class MLP(nn.Module): 
    def __init__(self, input, h1_dim, h2_dim, h3_dim, out_1, out_2): 
        super().__init__(); 
        # Setup our MLP structure 
        self.mlp_layers = nn.Sequential(
            nn.Linear(input, h1_dim), 
            nn.ReLU(), 
            nn.Linear(h1_dim, h2_dim), 
            nn.ReLU(), 
            nn.Linear(h2_dim, h3_dim) 
        ); 

        # Multi head output for our two transmitter arrays  
        self.classify_idx_1 = nn.Linear(h3_dim, out_1); 
        self.classify_book_1 = nn.Linear(out_1 + h3_dim, out_2); 
        self.classify_idx_2 = nn.Linear(h3_dim, out_1); 
        self.classify_book_2 = nn.Linear(out_1 + h3_dim, out_2); 


    # Forward pass 
    def forward(self, x): 
        x = self.mlp_layers(x); 

        # Predict beam index 
        logits_1a = self.classify_idx_1(x); 
        logits_1b = self.classify_idx_2(x); 

        cat_1a = torch.cat([x, logits_1a], dim = -1); 
        cat_1b = torch.cat([x, logits_1b], dim = -1); 

        # Predict code book 
        logits_2a = self.classify_book_1(cat_1a); 
        logits_2b = self.classify_idx_2(cat_1b); 

        return logits_1a, logits_1b, logits_2a, logits_2b; 