import numpy as np

# X = np.array([[1,2,3], [3,2,1]])
# v = np.array([2,6,2])

# # X.shape == (N, M), v.shape == (M,)
# def mat_prod(X, v):
#     u = np.zeros(X.shape[0])
#     for i in range(X.shape[0]):
#         for j in range(X.shape[1]):
#             u[i] += X[i, j] * v[j]
#     return u

# print(mat_prod(X, v))
# print(X@v)
# print(np.matmul(X,v))
# print(np.dot(X,v))

# print(np.dot(v,X))

X = np.array([[9,54,36], [-3, 69, 69]])
v = np.array([2,6])
print(X - np.mean(X, axis=1, keepdims=True))
print(X - np.expand_dims(np.mean(X, axis=1), 1))