#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
import collections
import numpy as np
import numbers
import random
import sys
import pickle
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from importlib import reload
from matplotlib import pyplot as plt
import importlib,sys
importlib.reload(sys)

# from scorecard_functions import *
from sklearn.linear_model import LogisticRegressionCV
# -*- coding: utf-8 -*-

# In[2]:


# 显示所有列
pd.set_option('display.max_columns', 500)  #最多显示五列
 # 显示所有行
pd.set_option('display.max_rows', None)

# In[3]:


import warnings
warnings.filterwarnings("ignore")


# In[4]:


folderOfData="./data/"
categorical_var = ['UserInfo_1', 'UserInfo_3', 'UserInfo_5', 'UserInfo_6', 'UserInfo_7', 'UserInfo_9', 'UserInfo_10', 'UserInfo_14', 'UserInfo_15', 'UserInfo_16', 'UserInfo_19', 'Education_Info1', 'Education_Info2', 'Education_Info3', 'Education_Info4', 'Education_Info5', 'Education_Info6', 'Education_Info7', 'Education_Info8', 'WeblogInfo_19', 'WeblogInfo_20', 'WeblogInfo_21', 'WeblogInfo_55', 'SocialNetwork_11', 'SocialNetwork_12', 'SocialNetwork_13', 'SocialNetwork_17']


# In[5]:


###################################
# Step 3: 基于卡方分箱法对变量进行分箱#
###################################
'''
对不同类型的变量，分箱的处理是不同的：
（1）数值型变量可直接分箱
（2）取值个数较多的类别型变量，需要用bad rate做编码转换成数值型变量，再分箱
（3）取值个数较少的类别型变量不需要分箱，但是要检查是否每个类别都有好坏样本。如果有类别只有好或坏，需要合并
'''

#for each categorical variable, if it has distinct values more than 5, we use the ChiMerge to merge it

trainData = pd.read_csv(folderOfData+'allData_1.csv',header = 0, encoding='gbk')
trainData.drop_duplicates(inplace=True)#inplace=True表示替换原数据
trainData.shape

# In[6]:


allFeatures = list(trainData.columns)
allFeatures.remove('ListingInfo')
allFeatures.remove('target')
trainData.shape

# In[7]:


trainData.head()

# In[8]:


trainData.describe()

# In[9]:


# trainData.describe([0.01,0.1,0.25,.5,.75,.9,.99]).T

# In[10]:


trainData[['UserInfo_7','UserInfo_9','UserInfo_19']].head()

# In[11]:


numerical_var = [i for i in allFeatures if i not in categorical_var]
len(numerical_var)

# In[12]:


# #将特征区分为数值型和类别型
# numerical_var = []
# for var in allFeatures:
#     uniq_vals = list(set(trainData[var]))
#     if np.nan in uniq_vals:
#         uniq_vals.remove(np.nan)
#     if len(uniq_vals) >= 10 and isinstance(uniq_vals[0],numbers.Real):
#         numerical_var.append(var)

# categorical_var = [i for i in allFeatures if i not in numerical_var]

# for col in categorical_var:
#     #for Chinese character, upper() is not valid
#     if col not in ['UserInfo_7','UserInfo_9','UserInfo_19']:
#         trainData[col] = trainData[col].map(lambda x: str(x).upper())

# In[13]:


trainData[numerical_var] = trainData[numerical_var].astype(float)
trainData[categorical_var] = trainData[categorical_var].astype(str)

# In[14]:


trainData[categorical_var].head()

# In[15]:


trainData['Education_Info1'].unique()

# In[16]:


from scorecard_functions_V3 import *

# In[17]:


trainData.head()

# In[18]:


deleted_features = []   #将处理过的变量删除，防止对后面建模的干扰
encoded_features = {}   #将bad rate编码方式保存下来，在以后的测试和生产环境中需要使用
merged_features = {}    #将类别型变量合并方案保留下来
var_IV = {}  #save the IV values for binned features       #将IV值保留和WOE值
var_WOE = {}

col="UserInfo_1"

# In[19]:


len(set(trainData[col]))

# In[20]:


trainData[col].head()

# In[21]:


trainData.info()

# In[22]:


trainData.head()

# In[23]:


display_flag =True

# In[24]:


def BinBadRate(df, col, target, grantRateIndicator=0):
    '''
    :param df: 需要计算好坏比率的数据集
    :param col: 需要计算好坏比率的特征
    :param target: 好坏标签
    :param grantRateIndicator: 1返回总体的坏样本率，0不返回
    :return: 每箱的坏样本率，以及总体的坏样本率（当grantRateIndicator＝＝1时）
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad * 1.0 / x.total, axis=1)
    dicts = dict(zip(regroup[col],regroup['bad_rate']))
    if grantRateIndicator==0:
        return (dicts, regroup)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    overallRate = B * 1.0 / N
    return (dicts, regroup, overallRate)

def BadRateEncoding(df, col, target):
    '''
    :return: 在数据集df中，用坏样本率给col进行编码。target表示坏样本标签
    '''
    regroup = BinBadRate(df, col, target, grantRateIndicator=0)[1]
    if display_flag:
        print("regroup:",regroup)
    
    br_dict = regroup[[col,'bad_rate']].set_index([col]).to_dict(orient='index')
    for k, v in br_dict.items():
        br_dict[k] = v['bad_rate']
    badRateEnconding = df[col].map(lambda x: br_dict[x])
    return {'encoding':badRateEnconding, 'bad_rate':br_dict}

# In[25]:


def MergeBad0(df,col,target, direction='bad'):
    '''
     :param df: 包含检验0％或者100%坏样本率
     :param col: 分箱后的变量或者类别型变量。检验其中是否有一组或者多组没有坏样本或者没有好样本。如果是，则需要进行合并
     :param target: 目标变量，0、1表示好、坏
     :return: 合并方案，使得每个组里同时包含好坏样本
     '''
    regroup = BinBadRate(df, col, target)[1]
    if direction == 'bad':
        # 如果是合并0坏样本率的组，则跟最小的非0坏样本率的组进行合并
        regroup = regroup.sort_values(by  = 'bad_rate')
    else:
        # 如果是合并0好样本率的组，则跟最小的非0好样本率的组进行合并
        regroup = regroup.sort_values(by='bad_rate',ascending=False)
    if display_flag:
        display(regroup.head())
    regroup.index = range(regroup.shape[0])
    col_regroup = [[i] for i in regroup[col]]
    del_index = []
    for i in range(regroup.shape[0]-1):
        col_regroup[i+1] = col_regroup[i] + col_regroup[i+1]
        del_index.append(i)
        if direction == 'bad':
            if regroup['bad_rate'][i+1] > 0:
                break
        else:
            if regroup['bad_rate'][i+1] < 1:
                break
    col_regroup2 = [col_regroup[i] for i in range(len(col_regroup)) if i not in del_index]
    if display_flag:
        display(print(del_index,col_regroup2))
    
    newGroup = {}
    for i in range(len(col_regroup2)):
        for g2 in col_regroup2[i]:
            newGroup[g2] = 'Bin '+str(i)
    return newGroup

# ### 列子

# In[26]:


col="UserInfo_1"
trainData[col].unique()

# In[27]:


encoding_result = BadRateEncoding(trainData, col, 'target')
# encoding_result

# In[28]:


mergeBin = MergeBad0(trainData, col, 'target')
print(mergeBin)

# In[29]:


regroup = BinBadRate(trainData, col, 'target', grantRateIndicator=0)
regroup[1]

# ## 正流程

# In[30]:


display_flag = False

# In[31]:


categorical_var

# In[32]:


trainData.head(2)

# In[33]:


'''
对于类别型变量，按照以下方式处理
1，如果变量的取值个数超过5，计算bad rate进行编码，保留。
2，除此之外，其他任何类别型变量如果有某个取值中，对应的样本全部是坏样本或者是好样本，进行合并, 对bad_rate从小到大排列，从前遍历，对为0数据合并，如果连着三个都是负样本会合并到一起，最大箱占比>0.9也会剔除。。
'''
deleted_features = []   #将处理过的变量删除，防止对后面建模的干扰
encoded_features = {}   #将bad rate编码方式保存下来，在以后的测试和生产环境中需要使用
merged_features = {}    #将类别型变量合并方案保留下来
var_IV = {}  #save the IV values for binned features       #将IV值保留和WOE值
var_WOE = {}
for col in categorical_var:
    print('we are processing {}'.format(col))
    if len(set(trainData[col]))>5:
        print('{} is encoded with bad rate'.format(col))
        col0 = str(col)+'_encoding'

        #(1), 计算坏样本率并进行编码
        encoding_result = BadRateEncoding(trainData, col, 'target')
        trainData[col0], br_encoding = encoding_result['encoding'],encoding_result['bad_rate']

        #(2), 将（1）中的编码后的变量也加入数值型变量列表中，为后面的卡方分箱做准备
        numerical_var.append(col0)

        #(3), 保存编码结果
        encoded_features[col] = [col0, br_encoding]

        #(4), 删除原始值

        deleted_features.append(col)
    else:
        bad_bin = trainData.groupby([col])['target'].sum()
        #对于类别数少于5个，但是出现0坏样本的特征需要做处理
        if min(bad_bin) == 0:
            print('{} has 0 bad sample!'.format(col))
            col1 = str(col) + '_mergeByBadRate'
            #(1), 找出最优合并方式，使得每一箱同时包含好坏样本
            mergeBin = MergeBad0(trainData, col, 'target')
            #(2), 依照（1）的结果对值进行合并
            trainData[col1] = trainData[col].map(mergeBin)
            maxPcnt = MaximumBinPcnt(trainData, col1)
            #如果合并后导致有箱占比超过90%，就删除。
            if maxPcnt > 0.9:
                print('{} is deleted because of large percentage of single bin'.format(col))
                deleted_features.append(col)
                categorical_var.remove(col)
                del trainData[col]
                continue
            #(3) 如果合并后的新的变量满足要求，就保留下来
            merged_features[col] = [col1, mergeBin]
            WOE_IV = CalcWOE(trainData, col1, 'target')
            var_WOE[col1] = WOE_IV['WOE']
            var_IV[col1] = WOE_IV['IV']
            # del trainData[col]
            deleted_features.append(col)
            
        else:
            WOE_IV = CalcWOE(trainData, col, 'target')
            var_WOE[col] = WOE_IV['WOE']
            var_IV[col] = WOE_IV['IV']

# In[34]:


trainData['Education_Info1'].unique()

# In[35]:


var_WOE['Education_Info5'],var_IV['Education_Info5']

# In[36]:


trainData['Education_Info1'].unique()

# In[37]:


'''
对于连续型变量，处理方式如下：
1，利用卡方分箱法将变量分成5个箱
2，检查坏样本率的单带性，如果发现单调性不满足，就进行合并，直到满足单调性
'''

'''
ChiMerge函数的 技术细节：
1. 可以设置缺失值或者异常值不进行分箱。
2. 等分数据集时，根据比例对应的数据。
3. 多箱合并时，chi2考虑最小合并，从0到N-1遍历，（0,1），（1,2）组队，对最小值替换，对合并值剔除。
4. 多项合并，检查是否有分箱没有分到坏样本，根据分箱的个数，确定索引，然后找分箱号前后，计算chi2进行合并。
5. 可选，对分箱的均匀性进行合并。
6. 最后返回所有分箱的分位点。

Monotone_Merge函数的 技术细节：
1. 获取当前特征的非单调的个数和索引。
2. 遍历所有的非单调的个数，
        其中每个判断与前合并，与后合并，非单调点个数的减少，均匀性进行判断。得到单个点的情况。
   综合考虑每个非单调点的个数或者均匀情况。
3. 满足单调性，或者总箱数等于2，结束。

'''

var_cutoff = {}
for col in numerical_var:
    print("{} is in processing".format(col))
    col1 = str(col) + '_Bin'

    #(1),用卡方分箱法进行分箱，并且保存每一个分割的端点。例如端点=[10,20,30]表示将变量分为x<10,10<x<20,20<x<30和x>30.
    #特别地，缺失值-1不参与分箱
    if -1 in set(trainData[col]):
        special_attribute = [-1]
    else:
        special_attribute = []
    cutOffPoints = ChiMerge(trainData, col, 'target',special_attribute=special_attribute)
    print("cutOffPoints:",cutOffPoints)
    var_cutoff[col] = cutOffPoints
    trainData[col1] = trainData[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))
    
    #(2), check whether the bad rate is monotone
    BRM = BadRateMonotone(trainData, col1, 'target',special_attribute=special_attribute)
    if not BRM:
        if special_attribute == []:
            bin_merged = Monotone_Merge(trainData, 'target', col1)
            removed_index = []
            for bin in bin_merged:
                if len(bin)>1:
                    indices = [int(b.replace('Bin ','')) for b in bin]
                    removed_index = removed_index+indices[0:-1]
            removed_point = [cutOffPoints[k] for k in removed_index]
            for p in removed_point:
                cutOffPoints.remove(p)
            var_cutoff[col] = cutOffPoints
            trainData[col1] = trainData[col].map(lambda x: AssignBin(x, cutOffPoints, special_attribute=special_attribute))
        else:
            cutOffPoints2 = [i for i in cutOffPoints if i not in special_attribute]
            temp = trainData.loc[~trainData[col].isin(special_attribute)]
            bin_merged = Monotone_Merge(temp, 'target', col1)
            removed_index = []
            for bin in bin_merged:
                if len(bin) > 1:
                    indices = [int(b.replace('Bin ', '')) for b in bin]
                    removed_index = removed_index + indices[0:-1]
            removed_point = [cutOffPoints2[k] for k in removed_index]
            for p in removed_point:
                cutOffPoints2.remove(p)
            cutOffPoints2 = cutOffPoints2 + special_attribute
            var_cutoff[col] = cutOffPoints2
            trainData[col1] = trainData[col].map(lambda x: AssignBin(x, cutOffPoints2, special_attribute=special_attribute))

    #(3), 分箱后再次检查是否有单一的值占比超过90%。如果有，删除该变量
    maxPcnt = MaximumBinPcnt(trainData, col1)
    if maxPcnt > 0.9:
        # del trainData[col1]
        deleted_features.append(col)
        numerical_var.remove(col)
        print('we delete {} because the maximum bin occupies more than 90%'.format(col))
        continue
    
    WOE_IV = CalcWOE(trainData, col1, 'target')
    var_IV[col] = WOE_IV['IV']
    var_WOE[col] = WOE_IV['WOE']
    #del trainData[col]

# In[38]:


numerical_var

# In[39]:


categorical_var

# In[40]:


trainData.head(2)

# In[43]:


trainData.columns

# In[41]:


len(numerical_var) + len(categorical_var), trainData.shape

# In[45]:


for i in numerical_var:
    if i not in list(trainData.columns):
        print(i)

# In[46]:


for i in categorical_var:
    if i not in list(trainData.columns):
        print(i)

# In[47]:


trainData.to_csv(folderOfData+'allData_2.csv', header=True,encoding='gbk', columns = trainData.columns, index=False)

with open(folderOfData+'var_WOE.pkl',"wb") as f:
    f.write(pickle.dumps(var_WOE))

with open(folderOfData+'var_IV.pkl',"wb") as f:
    f.write(pickle.dumps(var_IV))


with open(folderOfData+'var_cutoff.pkl',"wb") as f:
    f.write(pickle.dumps(var_cutoff))


with open(folderOfData+'merged_features.pkl',"wb") as f:
    f.write(pickle.dumps(merged_features))
    

# In[48]:


var_WOE.keys()

# In[49]:




# In[60]:


#########################################################
# Step 4: Select variables with IV > 0.02 and assign WOE#
#########################################################
trainData = pd.read_csv(folderOfData+'allData_2.csv', header=0, encoding='gbk')

trainData[numerical_var] = trainData[numerical_var].astype(float)
trainData[categorical_var] = trainData[categorical_var].astype(str)

num2str = ['SocialNetwork_13','SocialNetwork_12','UserInfo_6','UserInfo_5','UserInfo_10']
for col in num2str:
    trainData[col] = trainData[col].map(lambda x: str(x))


for col in var_WOE.keys():
    print(col)
    col2 = str(col)+"_WOE"
    if col in var_cutoff.keys():
        cutOffPoints = var_cutoff[col]
        special_attribute = []
        if - 1 in cutOffPoints:
            special_attribute = [-1]
        binValue = trainData[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))
        trainData[col2] = binValue.map(lambda x: var_WOE[col][x])
    else:
        print(col,col2)
        trainData[col2] = trainData[col].map(lambda x: var_WOE[col][x])

trainData.to_csv(folderOfData+'allData_3.csv', header=True,encoding='gbk', columns = trainData.columns, index=False)

# In[61]:


### (i) select the features with IV above the thresould
trainData = pd.read_csv(folderOfData+'allData_3.csv', header=0,encoding='gbk')
all_IV = list(var_IV.values())
all_IV = sorted(all_IV, reverse=True)
plt.bar(x=range(len(all_IV)), height = all_IV)
iv_threshould = 0.01
varByIV = [k for k, v in var_IV.items() if v > iv_threshould]

# In[69]:


### (ii) check the collinearity of any pair of the features with WOE after (i)

var_IV_selected = {k:var_IV[k] for k in varByIV}
var_IV_sorted = sorted(var_IV_selected.items(), key=lambda d:d[1], reverse = True)
var_IV_sorted = [i[0] for i in var_IV_sorted]

removed_var  = []
roh_thresould = 0.4

for i in range(len(var_IV_sorted)-1):
    if var_IV_sorted[i] not in removed_var:
        x1 = var_IV_sorted[i]+"_WOE"
        for j in range(i+1,len(var_IV_sorted)):
            if var_IV_sorted[j] not in removed_var:
                x2 = var_IV_sorted[j] + "_WOE"
                roh = np.corrcoef([trainData[x1], trainData[x2]])[0, 1]
                if abs(roh) >= roh_thresould:
                    print('the correlation coeffient between {0} and {1} is {2}'.format(x1, x2, str(roh)))
                    if var_IV[var_IV_sorted[i]] > var_IV[var_IV_sorted[j]]:
                        removed_var.append(var_IV_sorted[j])
                    else:
                        removed_var.append(var_IV_sorted[i])

var_IV_sortet_2 = [i for i in var_IV_sorted if i not in removed_var]

# In[70]:


### (iii) check the multi-colinearity according to VIF > 10
for i in range(len(var_IV_sortet_2)):
    x0 = trainData[var_IV_sortet_2[i]+'_WOE']
    x0 = np.array(x0)
    X_Col = [k+'_WOE' for k in var_IV_sortet_2 if k != var_IV_sortet_2[i]]
    X = trainData[X_Col]
    X = np.matrix(X)
    regr = LinearRegression()
    clr= regr.fit(X, x0)
    x_pred = clr.predict(X)
    R2 = 1 - ((x_pred - x0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum()
    vif = 1/(1-R2)
    if vif > 10:
        print("Warning: the vif for {0} is {1}".format(var_IV_sortet_2[i], vif))

# In[71]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# In[72]:


#########################
# Step 5: 应用逻辑回归模型#
#########################
multi_analysis = [i+'_WOE' for i in var_IV_sortet_2]
y = trainData['target']
X = trainData[multi_analysis].copy()
X['intercept'] = [1]*X.shape[0]


LR = sm.Logit(y, X).fit()
summary = LR.summary2()
pvals = LR.pvalues.to_dict()
params = LR.params.to_dict()

#发现有变量不显著，因此需要单独检验显著性
varLargeP = {k: v for k,v in pvals.items() if v >= 0.1}
varLargeP = sorted(varLargeP.items(), key=lambda d:d[1], reverse = True)
varLargeP = [i[0] for i in varLargeP]
p_value_list = {}
for var in varLargeP:
    X_temp = trainData[var].copy().to_frame()
    X_temp['intercept'] = [1] * X_temp.shape[0]
    LR = sm.Logit(y, X_temp).fit()
    p_value_list[var] = LR.pvalues[var]
for k,v in p_value_list.items():
    print("{0} has p-value of {1} in univariate regression".format(k,v))


#发现有变量的系数为正，因此需要单独检验正确性
varPositive = [k for k,v in params.items() if v >= 0]
coef_list = {}
for var in varPositive:
    X_temp = trainData[var].copy().to_frame()
    X_temp['intercept'] = [1] * X_temp.shape[0]
    LR = sm.Logit(y, X_temp).fit()
    coef_list[var] = LR.params[var]
for k,v in coef_list.items():
    print("{0} has coefficient of {1} in univariate regression".format(k,v))


selected_var = [multi_analysis[0]]
for var in multi_analysis[1:]:
    try_vars = selected_var+[var]
    X_temp = trainData[try_vars].copy()
    X_temp['intercept'] = [1] * X_temp.shape[0]
    LR = sm.Logit(y, X_temp).fit()
    #summary = LR.summary2()
    pvals, params = LR.pvalues, LR.params
    del params['intercept']
    if max(pvals)<0.1 and max(params)<0:
        selected_var.append(var)

LR.summary2()

y_pred = LR.predict(X_temp)
roc_auc_score(trainData['target'], y_pred)

# In[73]:


def Prob2Score(prob, basePoint, PDO):
    #将概率转化成分数且为正整数
    y = np.log(prob/(1-prob))
    y2 = basePoint+PDO/np.log(2)*(-y)
    score = y2.astype("int")
    return score

### 计算KS值
def KS(df, score, target, plot = True):
    '''
    :param df: 包含目标变量与预测值的数据集
    :param score: 得分或者概率
    :param target: 目标变量
    :return: KS值
    :return: KS值
    '''
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total':total, 'bad':bad})
    all['good'] = all['total'] - all['bad']
    all[score] = all.index
    all = all.reset_index(drop=True)
    display(all.head())
    all = all.sort_values(by=score)
    all.index = range(len(all))
    all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
    all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
    KS_list = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    KS = max(KS_list)
    plot=True
    if plot:
        plt.plot(all[score], all['badCumRate'])
        plt.plot(all[score], all['goodCumRate'])
        plt.title('KS ={}%'.format(int(KS*100)))
        plt.show()
    return KS

# In[74]:


def ROC_AUC(df, score, target, plot = True):
    df2 = df.copy()
    s = list(set(df2[score]))
    s.sort()
    tpr_list =[]
    fpr_list = []
    for k in s:
        df2['label_temp'] = df[score].map(lambda x : int(x<=k))
        temp = df2.groupby([target,'label_temp']).size()
        if temp.shape[0]<4:
            continue
        TP,FN,FP,TN = temp[1][1],temp[1][0],temp[0][1],temp[0][0]
        TPR, FPR = TP/(TP+FN), FP/(FP+TN)
        tpr_list.append(TPR)
        fpr_list.append(FPR)

    ROC_df = pd.DataFrame({'tpr':tpr_list, 'fpr':fpr_list})
    ROC_df = ROC_df.sort_values(by = 'tpr')
    auc = 0
    ROC_mat = np.mat(ROC_df)
    for i in range(1,ROC_mat.shape[0]):
        auc = auc + (ROC_mat[i,1] + ROC_mat[i-1,1])*(ROC_mat[i,0] - ROC_mat[i-1,0])*0.5
    if plot:
        plt.plot(ROC_df['fpr'],ROC_df['tpr'])
        plt.plot([0,1],[0,1])
        plt.title("AUC={}%".format(int(auc*100)))
    return auc

# In[75]:


#########################
# Step 6: 尺度化与性能检验#
#########################
plot = True
scores = Prob2Score(y_pred, 200, 100)
plt.hist(scores,bins=100)
y_preds = [ 1 if i > 0.5 else 0 for i in y_pred]
scorecard = pd.DataFrame({'y_pred':y_preds, 'y_real':list(trainData['target']),'score':y_pred})
display(scorecard.head())
print(KS(scorecard,'score','y_real'))

# 也可用sklearn带的函数
roc_auc_score(trainData['target'], y_pred)

# In[ ]:



