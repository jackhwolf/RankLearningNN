import numpy as np 
from torch import FloatTensor
from torch.autograd import Variable
from itertools import combinations

class Data:

    def __init__(self, beer=False, N=100, D=2, training_iterations=1, batch_size=4, **kw):
        self.beer = beer
        self.N = N
        if beer:
            self.points = np.load('beer-data/beers_processed.npy')
            self.D = self.points.shape[1]
            mask = np.array([False] * self.points.shape[0])
            mask[:N] = True
            np.random.shuffle(mask)
            self.points = self.points[mask,:]
        else:
            self.points = np.random.uniform(-1, 1, size=(N, D))
            self.D = D
        self.training_iterations = training_iterations
        self.batch_size = batch_size
        self.x_star = np.random.uniform(-1, 1, size=(1, self.D))
        self.rank_mat = self.build_rank_mat()

    def training_iterator(self):
        combos = [(i,j) for i,j in combinations(range(self.N), 2)]
        combos = np.array(combos)
        K = combos.shape[0]
        for ti in range(self.training_iterations):
            for step in range(int(K / self.batch_size) + 1):
                combos_slice = combos[step*self.batch_size:(step+1)*self.batch_size,:]
                i, j = combos_slice[:,0], combos_slice[:,1]
                i, j = self.points[i,:], self.points[j,:]
                ranks = np.array([self.rank_mat[ik,jk] for ik,jk in combos_slice])
                yield i, j, ranks
            np.random.shuffle(combos)
            
    def prediction_iterator(self):
        combos = [(i,j) for i,j in combinations(range(self.N), 2)]
        np.random.shuffle(combos)
        for i, j in combos:
            yield self.points[i,:], self.points[j,:], self.rank_mat[[i],[j]]

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
    d = Data(5, 2)
    
    for (i, j, ranks) in d.training_iterator():
        print(i)
        print(j)
        print(ranks)
        print("=================")
