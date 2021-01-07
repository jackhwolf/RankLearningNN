import torch
from numpy.random import uniform as np_uni
from torch.autograd import Variable
from hingeloss import hingeloss

class RankLearner:

    def __init__(self, D=2, criterion='MSELoss', lr=0.01, \
                                weight_decay=1e-5, optimizer='SGD', epochs=1000, **kw):
        self.x_hat_fc = Variable(torch.Tensor(np_uni(-1, 1, (1,D))).type(torch.FloatTensor), requires_grad=True)
        self.D = D
        try:
            self.criterion = getattr(torch.nn, criterion)()
        except:
            cmap = {'Hinge': hingeloss()}
            self.criterion = cmap[criterion]
        self.lr = lr 
        self.weight_decay = weight_decay
        opt = getattr(torch.optim, optimizer)
        self.optimizer = opt([self.x_hat_fc], lr=self.lr, weight_decay=self.weight_decay)
        self.epochs = epochs
        self.const_inp = torch.FloatTensor([1])

    def learn_pairwise_rank(self, point_i, point_j, true_rank_ij):
        true_rank_ij = self.to_var(true_rank_ij)
        for i in range(self.epochs):
            pred_rank_ij = self.forward(point_i, point_j)
            loss = self.criterion(pred_rank_ij, true_rank_ij).sum()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()

    def forward(self, point_i, point_j):
        point_i, point_j = self.to_var(point_i), self.to_var(point_j)
        dist_i = self.forward_one(point_i)
        dist_j = self.forward_one(point_j)
        pred_rank_ij = dist_j - dist_i
        # pred_rank_ij = (dist_j - dist_i).sign()  # grad(sign)=0, so x_hat will never update
        return pred_rank_ij

    def forward_one(self, point):
        return (point-(self.const_inp.matmul(self.x_hat_fc))).pow(2).sum() 

    def predict_pairwise_rank(self, point_i, point_j):
        with torch.no_grad():
            return self.forward(point_i, point_j).sign().detach().numpy()

    @property
    def current_x_hat(self):
        return self.x_hat_fc.detach().numpy()

    def to_var(self, foo):
        return Variable(torch.FloatTensor(foo))
