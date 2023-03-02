class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(input_dim, 3 * d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, input_dim = x.size()
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        vals, attn = scaled_dot_product(q, k, v, mask)
        vals = vals.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        out = self.linear(vals)
        return out


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attn = F.softmax(scaled, dim=-1)
    vals = torch.matmul(attn, v)
    return vals, attn
