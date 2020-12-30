import warnings
warnings.filterwarnings('ignore')
import yaml
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score
from data import Data
from model import RankLearner

class Experiment:

    def __init__(self, input_filename):
        with open(input_filename) as f:
            self.parameters = yaml.load(f, Loader=yaml.FullLoader)
        os.makedirs('Results/', exist_ok=True)
        self.data = Data(**self.parameters)
        self.model = RankLearner(**self.parameters)

    def learn_pairwise_ranks(self):
        training_iterations = self.parameters['training_iterations']
        for point_i, point_j, rank_ij in self.data.iterator(training_iterations):
            self.model.learn_pairwise_rank(point_i, point_j, rank_ij)

    def report(self):
        pred = np.array(list(self.predict_pairwise_ranks()))
        true, pred = pred[:,0], pred[:,1]
        out = {}
        out = self.parameters.copy()
        out['x_star'] = self.data.x_star.flatten().tolist()
        out['x_hat'] = self.model.current_x_hat.flatten().tolist()
        out['accuracy'] = accuracy_score(true, pred)
        if os.path.exists('Results/main.json'):
            res = json.loads(open('Results/main.json', 'r').read())
        else:
            res = []
        res.append(out)
        open('Results/main.json', 'w').write(json.dumps(res))
        return out

    def predict_pairwise_ranks(self):
        for point_i, point_j, true_rank_ij in self.data.iterator(1):
            pred_rank_ij = self.model.predict_pairwise_rank(point_i, point_j)
            yield true_rank_ij, pred_rank_ij

if __name__ == "__main__":
    import sys
    e = Experiment(sys.argv[1])
    e.learn_pairwise_ranks()
    print(e.report())