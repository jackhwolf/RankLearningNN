from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from collections import OrderedDict
import json
import jinja2
import numpy as np

def load_results(sortkey='timestamp', ascending=True):
    results = pd.read_json('Results/main.json')
    del results['points']
    del results['ranks']
    results = results.sort_values(by=[sortkey], ascending=ascending)
    results['timestamp'] = results['timestamp'].astype(str)
    out = []
    for _, row in results.iterrows():
        row = OrderedDict(row)
        row['x_hat'] = ["{:0.4f}".format(v) for v in row['x_hat']]
        row['x_star'] = ["{:0.4f}".format(v) for v in row['x_star']]
        row['accuracy'] = "{:0.4f}".format(row['accuracy'])
        out.append(row)
    return out, row.keys()

app = FastAPI()
app.mount("/images", StaticFiles(directory="Results"), name="static")

html = """
<!DOCTYPE html>
<html>
    <head>
    <title> RankLearningNN Results</title>
        <style>
            table, th, td {
              border: 1px solid black;
            }
        </style>
    </head>
<body>
    <h1>RankLearningNN Results</h1>
    <table id="results_table">
        <tr>
        {% for key in keys %}
            <th><a href="?sortkey={{key}}">{{key}}</a></th>
        {% endfor %}
        </tr>
        {% for item in items %}
        <tr>
            {% for key in keys %}
            {% if key != "graph_filename" %}
                <td>{{item[key]}}</td>
            {% else %}
                <td><img src={{"images/" + item[key]}} width="500"></td>
            {% endif %}
            {% endfor %}
        </tr>
        {% endfor %}
    </table>
    
    <script>
    </script>
</body>
</html>
"""

@app.get('/ranklearning_results')
def ranklearning_results(sortkey: str='timestamp', ascending: bool=False):
    items, keys = load_results(sortkey, ascending)
    return HTMLResponse(jinja2.Template(html).render(items=items, keys=keys))

if __name__ == '__main__':
    '''
    usage: 
    - on server, run python3 api.py to host on server's localhost:5003
    - run local ssh tunnel like ssh opt@wisc.edu -L 5083:localhost:5003
    - on local browser go to http://localhost:5083/ranklearning_results
    '''
    import uvicorn
    uvicorn.run('api:app', port=5003)