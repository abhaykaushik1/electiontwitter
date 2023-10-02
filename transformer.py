import numpy as np 
import math


L, d_k, d_v, = 0, 0, 0

# define softmax
def softmax(x):
    return (np.exp(x).T / np.sum(np.exp(x), axis=-1)).T

def scaled_dot_product_attention(Q, K , V, mask=None):
    d_k = Q.shape[-1]
    scaled = np.matmul(Q, K.T)/math.sqrt(d_k)

    if mask:
        scaled = scaled + mask
    
    attention = softmax(scaled)
    new_V = np.matmul(attention, V)

    return attention, new_V
