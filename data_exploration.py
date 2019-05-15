#%%
import pandas as pd 
import pickle 
import networkx as nx
#%%
hash_graphe=pickle.load( open( "hash_graph.pkl", "rb" ) )

#%%
g = nx.DiGraph()
#%%
g.add_nodes_from(hash_graphe.keys())
#%%
for k, v in hash_graphe.items():
    g.add_edges_from(([(k, t) for t in v]))
pickle.dump( favorite_color, open( "graph.pkl", "wb" ) )