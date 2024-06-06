import time

import numpy as np


class DPPModel(object):
    def __init__(self, **kwargs):
        self.item_count = kwargs['item_count']
        self.item_embed_size = kwargs['item_embed_size']
        self.k = kwargs['k']
        self.epsilon = kwargs['epsilon']

    def build_kernel_matrix(self):
        np.random.seed(200)
        rank_score = np.exp(0.01 * np.random.randn(self.item_count) + 0.2)  # 用户和每个item的相关性 类比精排模型分
        # rank_score = np.random.random(size=(self.item_count))
        print(rank_score)
        item_embedding = np.random.randn(self.item_count, self.item_embed_size)  # item的embedding
        item_embedding = item_embedding / np.linalg.norm(item_embedding, axis=1, keepdims=True)
        sim_matrix = np.dot(item_embedding, item_embedding.T)  # item之间的相似度矩阵
        kernel_matrix = rank_score.reshape((self.item_count, 1)) * sim_matrix * rank_score.reshape((1, self.item_count))
        self.kernel_matrix = kernel_matrix

    def dpp(self):
        c = np.zeros((self.k, self.item_count))
        d = np.copy(np.diag(self.kernel_matrix))
        j = np.argmax(d)
        Yg = [j]
        iter = 0
        Z = list(range(self.item_count))
        while len(Yg) < self.k:
            Z_Y = set(Z).difference(set(Yg))  # Z_Y就是原集合Z里面 去除了已经挑选过得集合 Yg
            for i in Z_Y:  # 在剩余的集合里面继续挑选
                if iter == 0:
                    ei = self.kernel_matrix[j, i] / np.sqrt(d[j])
                else:
                    ei = (self.kernel_matrix[j, i] - np.dot(c[:iter, j], c[:iter, i])) / np.sqrt(d[j])
                c[iter, i] = ei
                d[i] = d[i] - ei * ei
            d[j] = 0
            j = np.argmax(d)
            if d[j] < self.epsilon:
                break
            Yg.append(j)
            iter += 1
        return Yg


if __name__ == "__main__":
    kwargs = {
        'item_count': 5000,
        'item_embed_size': 128,
        'k': 1000,  # 最多从 item_count 中 挑选多少个
        'epsilon': 1e-10
    }
    dpp_model = DPPModel(**kwargs)
    dpp_model.build_kernel_matrix()
    t = time.time()
    print(dpp_model.dpp())
    a = time.time() - t
    print(a)
    print('algorithm running time: ' + '\t' + "{0:.4e}".format(a))


#   1. 输入:n 个物品的向量表征 v1,··· ,vn ∈ Rd 和分数 reward1,··· ,rewardn。
#   2. 计算n×n的相似度矩阵A，它的第(i,j)个元素等于aij =vTi vj。时间复杂度为 O(n2d)。
#   3. 选中 reward 分数最高的物品，记作 i。初始化集合 S = {i} 和 1×1 的矩阵 L = [1] 。(由于 aii = v⊤i vi = 1，此时 AS = [aii] = LL⊤。)
#   4. 做循环，从t=1到k−1:
#       (a). 对于每一个 i ∈ R:
#           I. 行向量 [aTi , 1] 是矩阵 AS ∪{i} 的最后一行。
#           II. 求解线性方程组 ai = Lci，得到 ci。时间复杂度为 O(|S|2)。
#           III. 计算d2i =1−cTi ci。
#       (b). 求解 (1.16):i⋆ = argmaxi∈R θ · rewardi + (1 − θ) · log d2i .
#       (c). 更新集合S←S∪{i⋆}。
#       (d). 更新下三角矩阵
#   5. 返回集合 S，其中包含 k 个物品。
