import torch


class Encoder(torch.nn.Module):
    def __init__(self, emb_dim, num_heads, hidden_dim_ff):
        super().__init__()
        # emb_dim = 756
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads  # Dimension per head
        # print(self.head_dim)
        assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        
        self.linear_q = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_k = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_v = torch.nn.Linear(emb_dim, emb_dim)
        
        # Learnable bias for attention
        self.attn_embedding_bias = torch.nn.Parameter(torch.zeros(emb_dim))

        self.linear_concat = torch.nn.Linear(emb_dim, emb_dim)

        self.norm = torch.nn.LayerNorm(emb_dim)
        

    def forward(self, emb):
        batch_size = emb.size(0)
        num_patches = emb.size(1)
        
        # Transform embeddings for query, key, and value
        query = self.linear_q(emb).view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.linear_k(emb).view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.linear_v(emb).view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores and apply softmax
        scaling_factor = self.head_dim ** 0.5
        similarity_matrix = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor

        # putting in zero for mask as pad token is 0 (dont need to write mask as pad token = 0)
        similarity_matrix = similarity_matrix.masked_fill(0, -1e10)

        # Apply softmax to get attention weights
        soft_matrix = torch.softmax(similarity_matrix, dim=-1)
    
        
        # Apply attention weights to values and reshape back
        attention = torch.matmul(soft_matrix, value)
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch_size, num_patches, -1)  # recombine heads

        attention = self.linear_concat(attention)

        attention = self.norm(attention + emb)

        # Apply feedforward layer
        output = self.feedforward(attention)
        
        return output