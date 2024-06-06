import numpy as np
import math
import time


def dpp(kernel, top_k, epsilon=1E-10):
    """
    fast implementation of the greedy algorithm
    :param kernel: 2-d array
    :param top_k: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel.shape[0]
    cis = np.zeros((top_k, item_size))
    di2s = np.copy(np.diag(kernel))  # shape为(item_size,)
    S = list()  # 候选集合
    j = np.argmax(di2s)  # 第一个取精排分最高的
    S.append(j)
    while len(S) < top_k:
        k = len(S) - 1
        ci_optimal = cis[:k, j]
        di_optimal = math.sqrt(di2s[j])
        elements = kernel[j, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        j = np.argmax(di2s)
        if di2s[j] < epsilon:
            break
        S.append(j)
    return S


if __name__ == '__main__':
    item_size = 5000  # 出精排5000
    feature_dimension = 128  # 每个向量维度为128维
    max_length = 1000  # 5000里面挑选top1000
    np.random.seed(200)
    scores = np.exp(0.01 * np.random.randn(item_size) + 0.2)  # shape (item_size,) 该用户与候选商品的相关性，可用预估ctr作相关性
    feature_vectors = np.random.randn(item_size, feature_dimension)  # shape (item_size,dim)#商品embedding

    feature_vectors /= np.linalg.norm(feature_vectors, axis=1, keepdims=True)  # feature除以每个行向量的2范数（也就是行向量的模）
    similarities = np.dot(feature_vectors, feature_vectors.T)
    kernel_matrix = scores.reshape((item_size, 1)) * similarities * scores.reshape((1, item_size))  # k核矩阵
    # b = np.linalg.cholesky(kernel_matrix)
    print(kernel_matrix.shape)
    print('kernel dpp generated!')
    t = time.time()
    result = dpp(kernel_matrix, max_length)
    print(result)
    a = time.time() - t
    print(a)
    print('algorithm running time: ' + '\t' + "{0:.4e}".format(a))
