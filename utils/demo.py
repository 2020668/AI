import graphviz
import os
from data.common_data import DATA_DIR

with open(os.path.join(DATA_DIR, 'tree.dot')) as f:
    dot_graph = f.read()
dot = graphviz.Source(dot_graph)
dot.view()
