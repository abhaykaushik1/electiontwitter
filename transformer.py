import numpy as np 
import math


L, d_k, d_v, = 0, 0, 0

class Linear:

    in_features: int 
    out_features: int
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, in_features:int, out_features:int, bias:bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.empty([out_features, in_features])

        if bias:
            self.bias = np.empty([out_features, 1])
    
    def __call__(self, inp:np.ndarray) -> np.ndarray:
        return self.forward(inp)

    def forward(self, inp:np.ndarray) -> np.ndarray:
        pass


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
    scaled = np.matmul(Q, K.T)/math.sqrt(d_k)

    if mask:
        scaled = scaled + mask
    
    attention = softmax(scaled)
    new_V = np.matmul(attention, V)

    return attention, new_V
