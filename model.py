import torch
from torch.autograd import Variable

class RankLearner:

    def __init__(self, D=2, criterion='MSELoss', optimizer='SGD', lr=0.01, weight_decay=1e-5, epochs=1000, **kw):
        self.D = D
        self.x_hat_fc = Variable((torch.randn(1, D).type(torch.FloatTensor)*2)-1, requires_grad=True)
        self.const_inp = torch.FloatTensor([1])
        self.criterion = getattr(torch.nn, criterion)()
        self.lr = lr 
        self.weight_decay = weight_decay
        self.optimizer = getattr(torch.optim, optimizer)([self.x_hat_fc], lr=self.lr, weight_decay=self.weight_decay)
        self.epochs = 100

    def learn_pairwise_rank(self, point_i, point_j, true_rank_ij):
        true_rank_ij = self.to_var(true_rank_ij)
        for i in range(self.epochs):
            pred_rank_ij = self.forward(point_i, point_j)
            loss = self.criterion(pred_rank_ij, true_rank_ij)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()

    def forward(self, point_i, point_j):
        point_i, point_j = self.to_var(point_i), self.to_var(point_j)
        dist_i = (point_i-(self.const_inp.matmul(self.x_hat_fc))).pow(2).sum() 
        dist_j = (point_j-(self.const_inp.matmul(self.x_hat_fc))).pow(2).sum() 
        pred_rank_ij = dist_j - dist_i
        return pred_rank_ij

    def predict_pairwise_rank(self, point_i, point_j):
        with torch.no_grad():
            return self.forward(point_i, point_j).sign().detach().numpy()

    @property
    def current_x_hat(self):
        return self.x_hat_fc.detach().numpy()

    def to_var(self, foo):
        return Variable(torch.FloatTensor([foo]))
