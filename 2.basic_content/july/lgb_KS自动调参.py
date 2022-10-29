#!/usr/bin/env python
# coding: utf-8

# 调参方法
# * offks + 0.8(offks - devks)最大化  
# * (devks+offks)/2 最大化 
# * offks最大化  
# 
# 这里只举例offks + 0.8(offks - devks)最大化

# In[1]:


import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import numpy as np
import random
import math
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# In[2]:


data = pd.read_csv('Acard.txt')
data.head()

# In[3]:


train = data[data.obs_mth != '2018-11-30'].reset_index().copy()
val = data[data.obs_mth == '2018-11-30'].reset_index().copy()
feature_lst = ['person_info','finance_info','credit_info','act_info']
x = train[feature_lst]
y = train['bad_ind']

val_x =  val[feature_lst]
val_y = val['bad_ind']


train_x,test_x,train_y,test_y = train_test_split(x,y,random_state=0,test_size=0.2)

# In[20]:


#改变我们想去调整的参数为value，设置调参区间
min_value = 40
max_value = 60
for value in  range(min_value,max_value+1):
    best_omd = -1
    best_value = -1
    best_ks=[]
    def  lgb_test(train_x,train_y,test_x,test_y):
        clf =lgb.LGBMClassifier(boosting_type = 'gbdt',
                               objective = 'binary',
                               metric = 'auc',
                               learning_rate = 0.1,
                               n_estimators = value,
                               max_depth = 5,
                               num_leaves = 20,
                               max_bin = 45,
                               min_data_in_leaf = 6,
                               bagging_fraction = 0.6,
                               bagging_freq = 0,
                               feature_fraction = 0.8,
                               silent=True
                               )
        clf.fit(train_x,train_y,eval_set = [(train_x,train_y),(test_x,test_y)],eval_metric = 'auc')
        return clf,clf.best_score_['valid_1']['auc'],
    lgb_model , lgb_auc  = lgb_test(train_x,train_y,test_x,test_y)

    y_pred = lgb_model.predict_proba(x)[:,1]
    fpr_lgb_train,tpr_lgb_train,_ = roc_curve(y,y_pred)
    train_ks = abs(fpr_lgb_train - tpr_lgb_train).max()

    y_pred = lgb_model.predict_proba(val_x)[:,1]
    fpr_lgb,tpr_lgb,_ = roc_curve(val_y,y_pred)
    val_ks = abs(fpr_lgb - tpr_lgb).max()
    
    Omd= val_ks + 0.8*(val_ks - train_ks)
    if Omd>best_omd:
        best_omd = Omd
        best_value = value
        best_ks = [train_ks,val_ks]
print('best_value:',best_value)
print('best_ks:',best_ks)

# In[ ]:



