import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def get_position_encoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)  # k 指单词在句子中的位置， i 指向量中每一个分量的位置， d 指向量维度
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return P


P = get_position_encoding(seq_len=4, d=4, n=100)
print(P)

P1 = get_position_encoding(seq_len=3, d=4, n=100)
print(P1)

P2 = get_position_encoding(seq_len=4, d=6, n=100)
print(P2)

