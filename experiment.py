import warnings
warnings.filterwarnings('ignore')
import yaml
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from distributed import worker_client
from data import Data
from model import RankLearner
from grapher import Graph
from time import time

class Experiment:
    
    def __init__(self, input_filename):
        with open(input_filename) as f:
            p = yaml.load(f, Loader=yaml.FullLoader)
            p = {k: [v] if not isinstance(v, list) else v for k, v in p.items()}
            p = list(ParameterGrid(p))
        self.parameters = p
        
    def run(self):
        reports = []
        with worker_client() as wc:
            for p in self.parameters:
                reports.append(wc.submit(SingleExperiment(p).run))
            reports = wc.gather(reports)
        if os.path.exists('Results/main.json'):
            res = json.loads(open('Results/main.json', 'r').read())
        else:
            res = []
        res.extend(reports)
        open('Results/main.json', 'w').write(json.dumps(res))
        return reports
        
class SingleExperiment:

    def __init__(self, params):
        self.parameters = params
        os.makedirs('Results/', exist_ok=True)
        self.data = Data(**self.parameters)
        self.model = RankLearner(**self.parameters)
        
    def run(self):
        print(self.parameters)
        self.learn_pairwise_ranks()
        return self.report()

    def learn_pairwise_ranks(self):
        print("X*        : ", self.data.x_star)
        print("Initial X^: ", self.model.current_x_hat)
        for point_i, point_j, rank_ij in self.data.training_iterator():
            self.model.learn_pairwise_rank(point_i, point_j, rank_ij)

    def report(self):
        os.makedirs('Results', exist_ok=True)
        out = self.parameters.copy()
        out['x_star'] = self.data.x_star.flatten().tolist()
        out['x_hat'] = self.model.current_x_hat.flatten().tolist()
        out['points'] = self.data.points.tolist()
        out['ranks'] = self.data.rank_mat.tolist()
        out['accuracy'] = self.predict_pairwise_ranks()
        out['timestamp'] = int(time())
        out.update(Graph(out))
        return out

    def predict_pairwise_ranks(self):
        pred = []
        for point_i, point_j, true_rank_ij in self.data.prediction_iterator():
            pred_rank_ij = self.model.predict_pairwise_rank(point_i, point_j)
            pred.append([true_rank_ij[0], pred_rank_ij])
        pred = np.array(pred)
        true, pred = pred[:,0], pred[:,1]
        return np.round(accuracy_score(true, pred), 3)

