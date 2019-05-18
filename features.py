#%%
import pickle
import math 
import networkx as nx
import numpy as np 
#%%
small_graph=pickle.load(open("graph_small.pkl","rb"))
sub_hash=pickle.load(open("sub_hash.pkl", "rb"))

#%%
def jobcob_followee(a,b,graph,f):
    x=set(graph.successors(a))
    y=set(graph.successors(b))
    
    try:
        if len(x)==0 or len(y)==0:
            return 0
        else:
            jacob= len(x.intersection(y))/len(x.intersection(y))
            return jacob
    except:
        return 0

#%%

def jobcob_follower(a,b,graph):
    x=set(graph.predecessors(a))
    y=set(graph.predecessors(b))
    
    try:
        if len(x)==0 or len(y)==0:
            return 0
        else:
            jacob= len(x.intersection(y))/len(x.intersection(y))
            return jacob
    except:
        return 0

#%%
def cosine_similarity_followee(a,b):
    x=set(graph.successors(a))
    y=set(graph.successors(b))
    try:
        if len(x)==0 or len(y)==0:
            return 0
        else:
            cosin= len(x.intersection(y))/math.sqrt(len(x)*len(y))
            return cosin
    except:
        return 0

#%%

def cosine_similarity_follower(a,b):
    x=set(graph.predecessors(a))
    y=set(graph.predecessors(b))
    try:
        if len(x)==0 or len(y)==0:
            return 0
        else:
            cosin= len(x.intersection(y))/math.sqrt(len(x)*len(y))
            return cosin
    except:
        return 0
#%%
page_rank=nx.pagerank(small_graph)
pickle.dump( page_rank, open( "pagerank_smallgraph.pkl", "wb" ) )

#%%
print(page_rank[max(page_rank,key=page_rank.get() )])
#%%
print(min(page_rank.values()))


#%%
def shortest_path(a,b,graph):
    path=-1
    try :
        if graph.has_edge(a,b):
            graph.remove_edge(a,b)
            path=nx.shortest_path_length(graph,source=a,target=b)
            graph.add_edge(a,b)
        else: 
            path=nx.shortest_path_length(graph,source=a,target=b)
        return path
    except:
        return path 


#%%
print(shortest_path(1,2,small_graph))

#%%
def adar_index(a,b,graph):
    x=set(graph.successors(a))
    y=set(graph.successors(b))
    s=0
    try:
        n=list(x.intersection(y))
        if len(n) != 0 :
            for i in n :
                s=s+(1/math.log10(len(graph.predecessors(i))))
            return s
        else:
            return 0

    except:
        return 0

#%%
def follow_back(a,b,graph):
    if graph.has_edge(b,a):
        return 1
    else:
        0

#%%
centrality = nx.katz_centrality(small_graph)
pickle.dump( centrality, open( "centrality_smallgraph.pkl", "wb" ) )

#%%


def belong_to_same_weakly_connected_components(a,b,graph):
    wcc=list(nx.weakly_connected_components(graph))
    if graph.has_edge(a,b):
        return 1 
    elif graph.has_edge(b,a):
        return 1 
    else:
        for j in wcc :
            if a in j and b in j :
                return 1
        return 0

#%%
def HITS_score(a):
    pass
h,a=nx.hits(small_graph,max_iter=1000)
print(h,a)