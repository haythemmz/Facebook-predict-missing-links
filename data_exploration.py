#%%
import pandas as pd 
import pickle 
import networkx as nx
#%%
hash_graphe=pickle.load( open( "hash_graph.pkl", "rb" ) )

#%%
l=list(hash_graphe.keys())[:1000]
sub_hash={}
for j in l:
    sub_hash[j]=hash_graphe[j]
#%%
print(sub_hash)




#%%
g = nx.DiGraph()
#%%
g.add_nodes_from(hash_graphe.keys())
#%%
for k, v in hash_graphe.items():
    g.add_edges_from(([(k, t) for t in v]))
pickle.dump( g, open( "graph.pkl", "wb" ) )