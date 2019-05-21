
#%%
import pandas as pd 
import pickle 
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np  
#%%
hash_graphe=pickle.load( open( "hash_graph.pkl", "rb" ) )

# dont forget to remove duplicated lines 

#%%
l=list(hash_graphe.keys())[:35000]
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
print(nx.info(g_small))

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
g=g_small
#%%
nodes_list=list(g.nodes)
edges_list=list(g.edges)
missing_edges=set([])
a=len(nodes_list) # change to edges_list
#%%

while (len(missing_edges)<a):
    print(len(missing_edges))
    first, second=tuple(np.random.choice(nodes_list,2))
    first=int(first)
    second=int(second)
    if first != second and (first,second) not in  edges_list and (first,second) not in  missing_edges:
        try :
            if nx.shortest_path_length(g,source=first,target=second) > 2 :
                missing_edges.add((frist,second))
            else :
                continue
        except:
            missing_edges.add((first,second))
    else:
        continue
pickle.dump( missing_edges, open( "missing_edges-35000.p", "wb" ) )
#%%
x,y=edges_list[0]
print(type(x))
nx.shortest_path_length(g,source=x,target=y)

#%%
missing_edges=pickle.load( open( "missing_edges-35000.p", "rb" ) )

#%%
edges_dict={}
for j in edges_list:
    edges_dict[j]=1
for i in missing_edges:
    edges_dict[i]=0
#%%
pickle.dump( edges_dict, open( "edges_dict.pkl", "wb" ) )
