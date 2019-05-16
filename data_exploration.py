#%%
import pandas as pd 
import pickle 
import networkx as nx
import matplotlib.pyplot as plt 
#%%
hash_graphe=pickle.load( open( "hash_graph.pkl", "rb" ) )

# dont forget to remove duplicated lines 

#%%
l=list(hash_graphe.keys())[:100000]
sub_hash={}
for j in l:
    sub_hash[j]=hash_graphe[j]
#%%
print(len(list(sub_hash.keys())))
print(len(list(hash_graphe.keys())))




#%%
g_small = nx.DiGraph()
#%%
g_small.add_nodes_from(sub_hash.keys())
#%%
for k, v in sub_hash.items():
    g_small.add_edges_from(([(k, t) for t in v]))
pickle.dump( g_small, open( "graph_small.pkl", "wb" ) )

#%%
nx.info(g_small)

#%%
sub_grah=g_small.edge_subgraph(list(g_small.edges)[0:10])
pos = nx.circular_layout(sub_grah)
nx.draw(sub_grah,pos=pos,edges=sub_grah.edges, node_color = 'r',edge_color='b', wight=10)

#%%
print("number of persones {}".format(len(g_small.nodes)))
# number of followers for each person 
###################################################################################
##                                                                               ##
##                                 EDA                                           ##
##                                                                               ##
###################################################################################
#%%

followers_number=list(dict(g_small.in_degree()).values())
followers_number.sort()
#%%
plt.figure(figsize=(10,6))
plt.tight_layout()

plt.plot(followers_number)
plt.xlabel("network users")
plt.ylabel("number of followers")
plt.show()

#%%

