import numpy as np 
import math


L, d_k, d_v, = 0, 0, 0

class Linear:

    in_features: int 
    out_features: int
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, in_features:int, out_features:int, bias:bool=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.empty([out_features, in_features])

        if bias:
            self.bias = np.empty([out_features, 1])

    def __call__(self, inp:np.ndarray) -> np.ndarray:
        return self.forward(inp)

    def forward(self, inp:np.ndarray, bias:bool=True) -> np.ndarray:
        return np.matmul(self.weights, inp.T)+self.bias if bias else np.matmul(self.weights, inp.T)

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
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.reshape(batch_size, sequence_length, self.num_heads*self.head_dim)
        return self.linear_layer(values)

# define softmax
def softmax(x):
    return (np.exp(x).T / np.sum(np.exp(x), axis=-1)).T

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: what user is looking for
    K: what I can offer
    V: what I can actually offer

    Each have same dimension N x 1
    """
    d_k = Q.shape[-1]
    scaled = np.matmul(Q, K.transpose(-1,-2)) / math.sqrt(d_k)
    if mask:
        scaled = scaled + mask
    attention = softmax(scaled)
    new_V = np.matmul(attention, V)
    return attention, new_V


input_dim = 1024
d_model = 512
num_heads = 8

batch_size = 30
sequence_length = 5
x = np.random.randn(batch_size, sequence_length, input_dim)

model = MultiheadAttention(input_dim, d_model, num_heads)
out = model.forward(x)