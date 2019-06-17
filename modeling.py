#%%
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn import tree
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.metrics import confusion_matrix

import numpy as np 
#%%
data=pd.read_pickle("pickles/data.pkl")

#%%
data.head()

#%%
data.dtypes

#%%
data.shape
#%%
data['has_edges'].value_counts().plot(kind='bar')


#balanced data
#%%
plt.figure(figsize=(9, 8))
sns.distplot(data['jacob_followee'], color='g', bins=50)

#%%
data['jacob_followee'].hist(bins=10)

#%%
data['jacob_followee'].value_counts().hist(bins=10)

#%%
data['cosine_similarity_followee'].hist(bins=10)
#%%
data[data['cosine_similarity_followee']!=0]['cosine_similarity_followee'].hist(bins=50)

#%%
data['adar_index'].value_counts()

#%%
'centrality_diff'
'small_graph_diff'

#%%
data['centrality_diff'].hist(bins=10)

#%%
data[data['centrality_diff']!=0]['centrality_diff'].hist(bins=50)

#%%
data['small_graph_diff'].hist(bins=10)


#%%
data['follow_back'].value_counts()


#%%
pd.crosstab(data.follow_back, df.has_edges, margins=True)

#%%
data=data.drop(columns=['adar_index'])

#%%
clf = tree.DecisionTreeClassifier()

#%%
X=data.drop(columns=['has_edges']).values
y=data['has_edges'].values
#%%
clf=clf.fit(X, y)

#%%
tree.plot_tree(clf.fit(X, y))

#%%
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'
#%%
dot_data = tree.export_graphviz(clf, out_file=None,feature_names=data.drop(columns=['has_edges']).columns,  
                    class_names='has_edges',  
                    filled=True, rounded=True,  
                    special_characters=True)  

#%%
graph = graphviz.Source(dot_data)

#%%
graph

#%%
import pydotplus
gr = pydotplus.graph_from_dot_data(dot_data)  



#%%
from IPython.display import Image  

Image(gr.create_png())


#%%
gr.write_png("DT.png")

#%%
clf.feature_importances_

#%%
print(dict(zip(data.drop(columns=['has_edges']).columns, clf.feature_importances_)))

#%%
feature_importance= pd.DataFrame(clf.feature_importances_,
                                index =data.drop(columns=['has_edges']).columns,
                                columns=['importance']).sort_values('importance',ascending=False)

#%%
feature_importance.plot(y='importance', use_index=True,kind='bar', grid=True,figsize=(5,2) )

#%%
X_embedded = TSNE(n_components=2).fit_transform(X)
pd.DataFrame(data=X_embedded).to_pickle("tsne.pkl")
#%%
print("aa")

#%%
cv = StratifiedKFold(n_splits=5,shuffle=False,random_state=0)
xgb_para={'eta': 0.02, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.9, 'alpha' : '0.1','lambda':0.01,
           'n_estimators': 600,'gamma':1.5,'objective': 'binary:logistic','eval_metric': 'auc', 'seed': 0, 'silent': False}
oof_xgb_ = np.zeros(len(X))
predictions_xg = np.zeros(len(X))


for train_index,test_index in cv.split(X,y):
    train_X, valid_X = X[train_index], X[test_index]
    train_y, valid_y = y[train_index], y[test_index]
    d_train = xgb.DMatrix(train_X, train_y ) 
    d_valid = xgb.DMatrix(valid_X, valid_y)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(xgb_para, d_train, 1000,evals=watchlist,maximize=True, verbose_eval=50,early_stopping_rounds=100 ) 
    predictions_xg[test_index] = model.predict(xgb.DMatrix(valid_X ), ntree_limit=model.best_ntree_limit)
    

#%%
predictions_xg.shape

#%%
predicted = (predictions_xg [:] > 0.5 ).astype('int')


#%%
pd.Series(predicted).value_counts()

#%%
cm = confusion_matrix(y,predicted)
cm

#%%
thresholds=np.linspace(0.1,0.9,9)

#%%
thresholds

#%%
tn, fp, fn, tp = confusion_matrix(y,predicted).ravel()    


#%%
print("tn ={} , fp={} , fn={} ,tp={}".format(tn,fp,fn,tp))

#%%
negative=tn+fn
postive=tp+fp

#%%
print(negative)

#%%
print(postive)

#%%
data['has_edges'].value_counts()

#%%
false_positives = np.logical_and(y != predicted, predicted == 1)

#%%
X[false_positives].shape

#%%
X.shape

#%%
b=false_positives.astype(int) 

#%%
a=np.ones_like(y)

#%%
c=np.multiply(a, b)

#%%
np.array_equal(c,b)

#%%
thresholds=np.linspace(0.1,0.9,9)
a=np.ones_like(y)
for j in thresholds:
    predicted = (predictions_xg [:] > j ).astype('int')
    false_positives = np.logical_and(y != predicted, predicted == 1).astype("int")
    a=np.multiply(a, false_positives)







#%%
X[a].shape


#%%
pd.Series(a).value_counts()

#%%
