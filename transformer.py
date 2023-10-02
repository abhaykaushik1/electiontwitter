import numpy as np 
import math

# self attention
L, d_k, d_v, = 0, 0, 0
Q = np.random.randn(L, d_k)
K = np.random.randn(L, d_k)
V = np.random.randn(L, d_v)

# need scaled values in the input in softmax function to get the equation for attention since it helps keep values in range
scaled = np.matmul(Q, K.T) / math.sqrt(d_k)

# Need mask to ensure words don't get context from words in the future
M = np.tril(np.ones((L, L)))
M[M == 0] = -np.infty
M[M == 1] = 0

# Now to get the matrix we input into softmax function
s_in = scaled + M

# define softmax
def softmax(x):
    return (np.exp(x).T / np.sum(np.exp(x), axis=-1)).T

attention = softmax(s_in)
