import torch

# class goes over for multihead 

class Encoder(torch.nn.Module):
    def __init__(self, emb_dim, num_heads,vocab_size,hidden_dim_ff):
        super().__init__()
        # emb_dim = 756
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads  # Dimension per head
        # print(self.head_dim)
        assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.embedding  = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim= emb_dim)
        
        self.linear_q = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_k = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_v = torch.nn.Linear(emb_dim, emb_dim)
        
        # Learnable bias for attention
        self.attn_embedding_bias = torch.nn.Parameter(torch.zeros(emb_dim))

        self.linear_concat = torch.nn.Linear(emb_dim, emb_dim)

        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, hidden_dim_ff),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim_ff, emb_dim),
        )

        self.projection_labels = torch.nn.Linear(emb_dim,3)

        self.norm = torch.nn.LayerNorm(emb_dim)

        

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        input_ids = self.embedding(input_ids)
        
        # Transform embeddings for query, key, and value
        query = self.linear_q(input_ids).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.linear_k(input_ids).view(batch_size,seq_len , self.num_heads, self.head_dim).transpose(1, 2)
        value = self.linear_v(input_ids).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # # Transform embeddings for query, key, and value
        # query = self.linear_q(emb).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        # key = self.linear_k(emb).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        # value = self.linear_v(emb).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # Calculate attention scores and apply softmax
        scaling_factor = self.head_dim ** 0.5
        similarity_matrix = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor

        # putting in zero for mask as pad token is 0 (dont need to write mask as pad token = 0)
        mask = attention_mask.unsqueeze(1).unsqueeze(2)
        mask = mask.expand(-1, self.num_heads, seq_len, -1)
        similarity_matrix = similarity_matrix.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        soft_matrix = torch.softmax(similarity_matrix, dim=-1)
    
        
        # Apply attention weights to values and reshape back
        attention = torch.matmul(soft_matrix, value)
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch_size,seq_len,-1)  # recombine heads

        attention = self.linear_concat(attention)

        attention = self.norm(attention + input_ids)

        # Apply feedforward layer
        output = self.feedforward(attention)

        output = self.projection_labels(output)

        output = output.mean(dim=1)
        
        return output