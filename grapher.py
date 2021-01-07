from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from uuid import uuid4

def Graph(report):
    fname = None
    if report['D'] == 2:
        fname = report.get('graph_filename', uuid4().hex + '.png')
        fig, (ax, table_ax) = plt.subplots(ncols=2, figsize=(10,5))
        points, x_star, x_hat = np.array(report['points']), \
            np.array(report['x_star']), np.array(report['x_hat'])
        ax.set_xlim((min(-1, x_star[0], x_hat[0])-0.05, max(1, x_star[0], x_hat[0])+0.05))
        ax.set_ylim((min(-1, x_star[1], x_hat[1])-0.05, max(1, x_star[1], x_hat[1])+0.05))
        ticks = np.linspace(-1, 1, 9)
        x_h_str = np.round(x_hat,3).flatten().tolist()
        x_s_str = np.round(x_star,3).flatten().tolist()
        fig.suptitle(f"True Optimal X (X*) compared to Learned Optimal X (X^)\n{x_s_str}, {x_h_str}")
        ax.scatter(points[:,0], points[:,1], c='b', label='Points')
        ax.scatter(x_star[0], x_star[1], c='g', label='True optimal X (X*)')
        ax.scatter(x_hat[0], x_hat[1], c='tab:orange', label='Learned optimal X (X^)')
        ax.legend(loc=(1.025, 0.825))
        table = []
        table.append(['Prediction Accuracy', np.round(report['accuracy'],3)])
        table.append(['True Optimal X (X*)', x_s_str])
        table.append(['Learned Optimal X (X^)', x_h_str])
        table.append(['Loss Function', report['criterion']])
        table.append(['Optimizer', report['optimizer']])
        table.append(['Learning Rate', '{:.2e}'.format(report['lr'])])
        table.append(['Weight Decay', '{:.2e}'.format(report['weight_decay'])])
        table.append(['Dataset Iterations', report['training_iterations']])
        table.append(['Training Epochs', report['epochs']])
        table_ax.axis('tight')
        table_ax.axis('off')
        table_ax.table(table, loc='center', fontsize=25)
        fig.savefig('Results/' + fname, facecolor='white', dpi=100, bbox_inches='tight')
        plt.close()
        del fig, ax
    out = {'graph_filename': fname}
    return out

def regraph():
    import pandas as pd
    res = pd.read_json('Results/main.json')
    k = res.shape[0]
    for i, row in res.iterrows():
        print(i, '/', k)
        Graph(row.to_dict())
        
if __name__ == '__main__':
    regraph()
