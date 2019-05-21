#%%
import pickle
import math 
import networkx as nx
import numpy as np 
#%%
small_graph=pickle.load(open("graph_small.pkl","rb"))
sub_hash=pickle.load(open("sub_hash.pkl", "rb"))
edges_dict=pickle.load(open("edges_dict.pkl", "rb"))

#%%
def jobcob_followee(a,b,graph):
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
def cosine_similarity_followee(a,b,graph):
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

def cosine_similarity_follower(a,b,graph):
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

h,a=nx.hits(small_graph,max_iter=1000)


#%%
df=pd.DataFrame.from_dict(edges_dict, orient='index',columns=['has_edges'])
#%%
df['jacob_followee']=[jobcob_followee(x[0],x[1],graph=small_graph) for x in df.index]


#%%
df['jacob_follower']=[jobcob_follower(x[0],x[1],graph=small_graph) for x in df.index]

#%%
df['cosine_similarity_followee']=[cosine_similarity_followee(x[0],x[1],graph=small_graph) for x in df.index]
#%%

df['cosine_similarity_follower']=[cosine_similarity_follower(x[0],x[1],graph=small_graph) for x in df.index]

#%%
page_rank=pickle.load(open("pagerank_smallgraph.pkl", "rb"))

#%%
df['small_graph_diff']=[page_rank[x[0]]-page_rank[x[1]] for x in df.index]

#%%
df['shortest_path']=[shortest_path(x[0],x[1],graph=small_graph) for x in df.index]
#%%

df['adar_index']=[adar_index(x[0],x[1],graph=small_graph) for x in df.index]

#%%

df['follow_back']=[follow_back(x[0],x[1],graph=small_graph) for x in df.index]

#%%
centrality=pickle.load(open("centrality_smallgraph.pkl", "rb"))
#%%
df['centrality_diff']=[centrality[x[0]]-centrality[x[1]] for x in df.index]

#%%
df['follow_back']=[follow_back(x[0],x[1],graph=small_graph) for x in df.index]

belong_to_same_weakly_connected_components(a,b,graph)
#%%

df['belong_to_same_wcc']=[belong_to_same_weakly_connected_components(x[0],x[1],graph=small_graph) for x in df.index]
#%%
df.head()
