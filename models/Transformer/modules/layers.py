import torch.nn as nn
import torch
import math

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim//num_heads
        
        # Linear layer Wq, Wk, Wv (N*N)
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        
        # Output layer
        self.out_linear = nn.Linear(dim, dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(attn_drop)
        
        
        
        

    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        batch_size, num_image_tokens, _ = x.shape
        
        # Linear, get (batch_size, # tokens, dim)
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        # Reshape for multi-head, get (batch_size, # tokens, # heads, head_dim)
        Q = Q.view(batch_size, num_image_tokens, self.num_heads, self.head_dim)
        K = K.view(batch_size, num_image_tokens, self.num_heads, self.head_dim)
        V = V.view(batch_size, num_image_tokens, self.num_heads, self.head_dim)
           
        # Reorder, so that compute independently for each head, get (batch_size, # heads, # tokens, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Attention = dropout(softmax(QK^T/(d_k)**0.5))V
        A = torch.matmul(Q, K.transpose(-2,-1))/(self.head_dim**0.5)
        A_hat = torch.nn.functional.softmax(A, dim =-1)
        A_hat = self.dropout(A_hat)        
        B = torch.matmul(A_hat, V)
        
        # concat the output
        B = B.transpose(1, 2).reshape(batch_size, num_image_tokens, self.dim)
        output = self.out_linear(B)
        
        return output            
        

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    