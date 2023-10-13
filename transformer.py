import numpy as np 
import math


L, d_k, d_v, = 0, 0, 0

class Linear:

    in_features: int 
    out_features: int
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, in_features:int, out_features:int):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.randn(out_features,in_features) * 0.01
        self.bias = np.random.randn(out_features,1) * 0.01

    def __call__(self, inp:np.ndarray) -> np.ndarray:
        return self.forward(inp)

    def forward(self, x:np.ndarray) -> np.ndarray:
        m = []
        for i in x:
            Z = np.dot(i, self.weights.T) + self.bias.T
            # Definition of ReLU
            A = Z * (Z > 0)
            m.append(A)
        return np.array(m)

class MultiheadAttention:

    input_dim: int
    d_model: int
    num_heads: int
    head_dim: int
    qkv_layer: Linear
    linear_layer: Linear

    def __init__(self, input_dim:int, d_model:int, num_heads:int):
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = Linear(input_dim , 3*d_model)
        self.linear_layer = Linear(d_model, d_model)
    
    def forward(self, x:np.ndarray, mask=None) -> np.ndarray:
        batch_size, sequence_length, input_dim = x.shape
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.transpose((0, 2, 1, 3))
        q, k, v = np.array_split(qkv, 3, axis=-1)
        attention, values = scaled_dot_product_attention(q, k, v, mask)
        values = values.reshape(batch_size, sequence_length, self.num_heads*self.head_dim)
        return self.linear_layer(values)

# define softmax
def softmax(x):
    exp_max = np.exp(x - np.max(x,axis=-1,keepdims=True))
    return exp_max/np.sum(exp_max,axis=-1,keepdims=True)

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Q: what user is looking for
    K: what I can offer
    V: what I can actually offer

    Each have same dimension N x 1
    """
    d_k = q.shape[-1]
    dim = list(range(len(k.shape)))
    dim[-1], dim[-2] = dim[-2], dim[-1]
    scaled = np.matmul(q, k.transpose(dim)) / math.sqrt(d_k)
    if mask:
        scaled = scaled + mask
    attention = softmax(scaled)
    new_V = np.matmul(attention, v)
    return attention, new_V

if __name__ == "__main__":
    input_dim = 1024
    d_model = 512
    num_heads = 8

    batch_size = 30
    sequence_length = 5
    x = np.random.randn(batch_size, sequence_length, input_dim)

    model = MultiheadAttention(input_dim, d_model, num_heads)
    out = model.forward(x)
    print(out)