import numpy as np 
from torch import FloatTensor
from torch.autograd import Variable
from itertools import combinations

class Data:

    def __init__(self, N=100, D=2, **kw):
        self.N = N
        self.D = D
        self.points = np.random.uniform(-1, 1, size=(N, D))
        self.x_star = np.random.uniform(-1, 1, size=(1, D))
        self.rank_mat = self.build_rank_mat()

    def iterator(self, training_iterations=1):
        combos = [(i,j) for i,j in combinations(range(self.N), 2)]
        for b in range(training_iterations):
            for i, j in combos:
                yield self.points[i,:], self.points[j,:], self.rank_mat[i,j]
            np.random.shuffle(combos)

    def build_rank_mat(self):
        mat = np.zeros((self.N, self.N))
        for i, j in combinations(range(self.N), 2):
            d = self.distance(self.points[i,:], self.points[j,:], self.x_star)
            mat[i,j] = d
        return mat

    ''' 1 if i --> x smaller than j --> x, else -1 '''
    def distance(self, i, j, x):
        dist_i, dist_j = 0, 0
        if isinstance(i, np.ndarray):
            dist_i = np.linalg.norm(i - x)
            dist_j = np.linalg.norm(j - x)
        rank_ij = np.sign(dist_j - dist_i)
        return rank_ij

if __name__ == "__main__":
    d = data(3, 1)

    print(d.points)
    for foo in d.iterator():
        print(foo)
