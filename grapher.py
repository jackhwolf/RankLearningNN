import matplotlib.pyplot as plt
import numpy as np
from uuid import uuid4

def Graph(report):
    fname = ''
    if report['D'] == 2:
        fname = uuid().hex + '.png'
        fig, ax = plt.subplots(figsize=(5,5))
        points, x_star, x_hat = np.array(report['points']), \
            np.array(report['x_star']), np.array(report['x_hat'])
        ax.set_title("Learned X^ compared to X*")
        ax.scatter(points[:,0], points[:,1], c='b', label='Points')
        ax.scatter(x_star[0], x_star[1], c='g', label='True optimal metric')
        ax.scatter(x_hat[0], x_hat[1], c='tab:orange', label='Learned optimal metric')
        fig.savefig('Results/' + fname, facecolor='white', dpi=100, bbox_inches='tight')
    out = {'graph_filename': fname}
    return out
