#!/usr/bin/env python
# coding: utf-8

# In[50]:


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

# In[51]:


################################
######## UDF: 自定义函数 ########
################################
### 对时间窗口，计算累计产比 ###
def TimeWindowSelection(df, daysCol, time_windows):
    '''
    :param df: the dataset containg variabel of days
    :param daysCol: the column of days
    :param time_windows: the list of time window
    :return:
    '''
    freq_tw = {}
    for tw in time_windows:
        freq = sum(df[daysCol].apply(lambda x: int(x<=tw)))
        freq_tw[tw] = freq
    return freq_tw


def DeivdedByZero(nominator, denominator):
    '''
    当分母为0时，返回0；否则返回正常值
    '''
    if denominator == 0:
        return 0
    else:
        return nominator*1.0/denominator

def MakeupRandom(x, sampledList):
    if x==x:
        return x
    else:
        randIndex = random.randint(0, len(sampledList)-1)
        return sampledList[randIndex]

# In[52]:


# 显示所有列
pd.set_option('display.max_columns', 500)  #最多显示五列
 # 显示所有行
pd.set_option('display.max_rows', None)


# In[53]:


############################################################
#Step 0: 数据分析的初始工作, 包括读取数据文件、检查用户Id的一致性等#
############################################################
folderOfData = './data/'
data1 = pd.read_csv(folderOfData+'PPD_LogInfo_3_1_Training_Set.csv', header = 0)
data2 = pd.read_csv(folderOfData+'PPD_Training_Master_GBK_3_1_Training_Set.csv', header = 0,encoding = 'gbk')
data3 = pd.read_csv(folderOfData+'PPD_Userupdate_Info_3_1_Training_Set.csv', header = 0)

# In[54]:


display(data1.shape,data2.shape,data3.shape)
display(data1.head(),data2.head(),data3.head())

# In[55]:


%%time

# compare whether the four city variables match
def place_is_same(a,b,c,d):
    if a == b:
        if a == c:
            if a == d:
                return 1
    return 0

data2['city_match'] = data2.apply(lambda x: place_is_same(x['UserInfo_2'],x['UserInfo_4'],x['UserInfo_8'],x['UserInfo_20']),axis = 1)
del data2['UserInfo_2']
del data2['UserInfo_4']
del data2['UserInfo_8']
del data2['UserInfo_20']

# In[56]:


# %%time
# import swifter
# # compare whether the four city variables match
# def place_is_same(a,b,c,d):
#     if a == b:
#         if a == c:
#             if a == d:
#                 return 1
#     return 0

# data11['city_match'] = data11.swifter.apply(lambda x: place_is_same(x['UserInfo_2'],x['UserInfo_4'],x['UserInfo_8'],x['UserInfo_20']),axis = 1)
# del data11['UserInfo_2']
# del data11['UserInfo_4']
# del data11['UserInfo_8']
# del data11['UserInfo_20']

# In[57]:


### 提取申请日期，计算日期差，查看日期差的分布
data1['logInfo'] = data1['LogInfo3'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
data1['Listinginfo'] = data1['Listinginfo1'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
data1['ListingGap'] = data1[['logInfo','Listinginfo']].apply(lambda x: (x[1]-x[0]).days,axis = 1)
plt.hist(data1['ListingGap'],bins=200)
plt.xlim(xmin = 0, xmax = 400)
plt.title('Days between login date and listing date')

# In[58]:


ListingGap2 = data1['ListingGap'].map(lambda x: min(x,365))
plt.hist(ListingGap2,bins=200)
plt.title('Days between login date and listing date')

# In[59]:


timeWindows = TimeWindowSelection(data1, 'ListingGap', range(30,361,30))
timeWindows

# In[60]:


'''
使用180天作为最大的时间窗口计算新特征
所有可以使用的时间窗口可以有7 days, 30 days, 60 days, 90 days, 120 days, 150 days and 180 days.
在每个时间窗口内，计算总的登录次数，不同的登录方式，以及每种登录方式的平均次数
'''
time_window = [7, 30, 60, 90, 120, 150, 180]
var_list = ['LogInfo1','LogInfo2']
data1GroupbyIdx = pd.DataFrame({'Idx':data1['Idx'].drop_duplicates()})
data1GroupbyIdx.head()

# In[61]:


data1.head()

# In[62]:


for tw in time_window:
    data1['TruncatedLogInfo'] = data1['Listinginfo'].map(lambda x: x + datetime.timedelta(-tw))
    temp = data1.loc[data1['logInfo'] >= data1['TruncatedLogInfo']]
    for var in var_list:
        #count the frequences of LogInfo1 and LogInfo2
        count_stats = temp.groupby(['Idx'])[var].count().to_dict()
        data1GroupbyIdx[str(var)+'_'+str(tw)+'_count'] = data1GroupbyIdx['Idx'].map(lambda x: count_stats.get(x,0))

        # count the distinct value of LogInfo1 and LogInfo2
        Idx_UserupdateInfo1 = temp[['Idx', var]].drop_duplicates()
        print(temp[['Idx', var]].shape, Idx_UserupdateInfo1.shape)
        uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])[var].count().to_dict()
        data1GroupbyIdx[str(var) + '_' + str(tw) + '_unique'] = data1GroupbyIdx['Idx'].map(lambda x: uniq_stats.get(x,0))

        # calculate the average count of each value in LogInfo1 and LogInfo2
        data1GroupbyIdx[str(var) + '_' + str(tw) + '_avg_count'] = data1GroupbyIdx[[str(var)+'_'+str(tw)+'_count',str(var) + '_' + str(tw) + '_unique']].\
            apply(lambda x: DeivdedByZero(x[0],x[1]), axis=1)
        
data1GroupbyIdx.head()        

# In[63]:


data3.head()

# In[64]:


data3['ListingInfo'] = data3['ListingInfo1'].map(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d'))
data3['UserupdateInfo'] = data3['UserupdateInfo2'].map(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d'))
data3['ListingGap'] = data3[['UserupdateInfo','ListingInfo']].apply(lambda x: (x[1]-x[0]).days,axis = 1)
# collections.Counter(data3['ListingGap'])

# plt.hist(data3[['ListingGap']],bins=200)
# plt.title('Days between login date and listing date')

# In[65]:


hist_ListingGap = np.histogram(data3['ListingGap'])
hist_ListingGap

# In[66]:


hist_ListingGap = pd.DataFrame({'Freq':hist_ListingGap[0],'gap':hist_ListingGap[1][1:]})
display(hist_ListingGap.head())
hist_ListingGap['CumFreq'] = hist_ListingGap['Freq'].cumsum()
display(hist_ListingGap.head())
hist_ListingGap['CumPercent'] = hist_ListingGap['CumFreq'].map(lambda x: x*1.0/hist_ListingGap.iloc[-1]['CumFreq'])
hist_ListingGap.head()

# In[67]:


data3.head()

# In[68]:


'''
对 QQ和qQ, Idnumber和idNumber,MOBILEPHONE和PHONE 进行统一
在时间切片内，计算
 (1) 更新的频率
 (2) 每种更新对象的种类个数
 (3) 对重要信息如IDNUMBER,HASBUYCAR, MARRIAGESTATUSID, PHONE的更新
'''
#对某些统一的字段进行统一
def ChangeContent(x):
    y = x.upper()
    if y == '_MOBILEPHONE':
        y = '_PHONE'
    return y
data3['UserupdateInfo1'] = data3['UserupdateInfo1'].map(ChangeContent)
data3GroupbyIdx = pd.DataFrame({'Idx':data3['Idx'].drop_duplicates()})

time_window = [7, 30, 60, 90, 120, 150, 180]
for tw in time_window:
    print(tw)
    data3['TruncatedLogInfo'] = data3['ListingInfo'].map(lambda x: x + datetime.timedelta(-tw))
    temp = data3.loc[data3['UserupdateInfo'] >= data3['TruncatedLogInfo']]

    #frequency of updating
    freq_stats = temp.groupby(['Idx'])['UserupdateInfo1'].count().to_dict()
    data3GroupbyIdx['UserupdateInfo_'+str(tw)+'_freq'] = data3GroupbyIdx['Idx'].map(lambda x: freq_stats.get(x,0))

    # number of updated types
    Idx_UserupdateInfo1 = temp[['Idx','UserupdateInfo1']].drop_duplicates()
    uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])['UserupdateInfo1'].count().to_dict()
    data3GroupbyIdx['UserupdateInfo_' + str(tw) + '_unique'] = data3GroupbyIdx['Idx'].map(lambda x: uniq_stats.get(x, x))

    #average count of each type
    data3GroupbyIdx['UserupdateInfo_' + str(tw) + '_avg_count'] = data3GroupbyIdx[['UserupdateInfo_'+str(tw)+'_freq', 'UserupdateInfo_' + str(tw) + '_unique']]. \
        apply(lambda x: x[0] * 1.0 / x[1], axis=1)

    #whether the applicant changed items like IDNUMBER,HASBUYCAR, MARRIAGESTATUSID, PHONE
    Idx_UserupdateInfo1['UserupdateInfo1'] = Idx_UserupdateInfo1['UserupdateInfo1'].map(lambda x: [x])
    Idx_UserupdateInfo1_V2 = Idx_UserupdateInfo1.groupby(['Idx'])['UserupdateInfo1'].sum()
    for item in ['_IDNUMBER','_HASBUYCAR','_MARRIAGESTATUSID','_PHONE']:
        item_dict = Idx_UserupdateInfo1_V2.map(lambda x: int(item in x)).to_dict()
        data3GroupbyIdx['UserupdateInfo_' + str(tw) + str(item)] = data3GroupbyIdx['Idx'].map(lambda x: item_dict.get(x, x))
data3GroupbyIdx.head()

# In[69]:


# Combine the above features with raw features in PPD_Training_Master_GBK_3_1_Training_Set
allData = pd.concat([data2.set_index('Idx'), data3GroupbyIdx.set_index('Idx'), data1GroupbyIdx.set_index('Idx')],axis= 1)
allData.to_csv(folderOfData+'allData_0.csv',encoding = 'gbk')

# In[70]:


display(allData.head())
allData.shape

# # 2 进一步处理文件

# In[324]:


#######################################
# Step 2: 对类别型变量和数值型变量进行补缺#
######################################
allData = pd.read_csv(folderOfData+'allData_0.csv',header = 0,encoding = 'gbk')
allFeatures = list(allData.columns)
allFeatures.remove('target')
if 'Idx' in allFeatures:
    allFeatures.remove('Idx')
allFeatures.remove('ListingInfo')

# In[325]:


allData.head()

# In[326]:


# 1、判断数值是否为空，可以用pd.isna,pd.isnull,np.isnan；
# 2、判断字符串是否为空，可以用pd.isna,pd.isnull；
# 3、判断时间是否为空，可以用pd.isna,pd.isnull，np.isnat；
# 4、判断转换类型后的字符串，空值也转换成了字符串nan，所以不能用常规方法判断了

# In[327]:


a=pd.Series(pd.isnull(allData).mean())

# In[328]:


#检查是否有常数型变量，并且检查是类别型还是数值型变量
numerical_var = []
for col in allFeatures:
    if len(set(allData[col])) == 1:
        print('delete {} from the dataset because it is a constant'.format(col))
        del allData[col]
        allFeatures.remove(col)
    else:
        #uniq_vals = list(set(allData[col]))
        #if np.nan in uniq_vals:
            #uniq_vals.remove(np.nan)
        uniq_valid_vals = [i for i in allData[col] if i == i]
        uniq_valid_vals = list(set(uniq_valid_vals))
        if len(uniq_valid_vals) >= 10 and isinstance(uniq_valid_vals[0], numbers.Real):
            numerical_var.append(col)

categorical_var = [i for i in allFeatures if i not in numerical_var]
len(allFeatures),len(numerical_var), len(categorical_var)

# In[329]:


def values_counts_all(df, col):
    v_dict = df[col].value_counts()
    v_dict['nan'] = pd.isna(df[col]).sum()
    v_df = pd.Series(v_dict)
    return v_df
s = values_counts_all(allData,'WeblogInfo_1')
display(s)

# In[330]:


#检查变量的最多值的占比情况,以及每个变量中占比最大的值
records_count = allData.shape[0]
col_most_values,col_large_value = {},{}
for col in allFeatures:
    value_count = allData[col].groupby(allData[col]).count()
#     value_count = values_counts_all(allData,col)
    col_most_values[col] = max(value_count)/records_count
    large_value = value_count[value_count== max(value_count)].index[0]
    col_large_value[col] = large_value
    
col_most_values_df = pd.DataFrame.from_dict(col_most_values, orient = 'index')
col_most_values_df.columns = ['max percent']
display(col_most_values_df.sort_values('max percent',ascending = False).head(30))
col_most_values_df = col_most_values_df.sort_values(by = 'max percent', ascending = False)
pcnt = list(col_most_values_df[:500]['max percent'])
vars = list(col_most_values_df[:500].index)
plt.bar(range(len(pcnt)), height = pcnt)
plt.title('Largest Percentage of Single Value in Each Variable')

# In[331]:


#计算多数值产比超过90%的字段中，少数值的坏样本率是否会显著高于多数值
large_percent_cols = list(col_most_values_df[col_most_values_df['max percent']>=0.8].index)
large_percent_cols_del = list()
bad_rate_diff = {}
for col in large_percent_cols:
    large_value = col_large_value[col]
    temp = allData[[col,'target']]
    temp[col] = temp.apply(lambda x: int(x[col]==large_value),axis=1)
    bad_rate = temp.groupby(col).mean()
    if bad_rate.iloc[0]['target'] == 0:
        bad_rate_diff[col] = 0
        continue
    bad_rate_diff[col] = np.log(bad_rate.iloc[0]['target']/bad_rate.iloc[1]['target'])
    if abs(bad_rate_diff[col]) < 1:
        large_percent_cols_del.append(col)

bad_rate_diff_sorted = sorted(bad_rate_diff.items(),key=lambda x: x[1], reverse=True)
bad_rate_diff_sorted_values = [x[1] for x in bad_rate_diff_sorted]
plt.bar(x = range(len(bad_rate_diff_sorted_values)), height = bad_rate_diff_sorted_values)

# In[332]:


len(large_percent_cols_del), len(large_percent_cols)

# In[333]:


#由于所有的少数值的坏样本率并没有显著高于多数值，意味着这些变量可以直接剔除
for col in large_percent_cols_del:
    if col in numerical_var:
        numerical_var.remove(col)
    else:
        categorical_var.remove(col)
    allFeatures.remove(col)
    del allData[col]

# In[334]:


allData.shape, len(numerical_var), len(categorical_var)

# In[335]:


'''
对类别型变量，如果缺失超过80%, 就删除，否则当成特殊的状态
'''

def MissingCategorial(df,x):
    return pd.isna(df[x]).mean()

missing_pcnt_threshould_1 = 0.6
need_to_deal_col = []
for col in allFeatures:
    missingRate = MissingCategorial(allData,col)
    print('{0} has missing rate as {1}'.format(col,missingRate))
    if missingRate > missing_pcnt_threshould_1:
        if col in numerical_var:
            numerical_var.remove(col)
        else:
            categorical_var.remove(col)
        print('delete')
        del allData[col]
    if 0 < missingRate < missing_pcnt_threshould_1:
        # In this way we convert NaN to NAN, which is a string instead of np.nan
        # allData[col] = allData[col].map(lambda x: str(x).upper())
        need_to_deal_col.append(col)

# In[336]:


allData.shape, len(numerical_var), len(categorical_var)

# In[337]:


# 最大占比原因删除 
# 缺失值删除。    

# In[338]:


len(need_to_deal_col),need_to_deal_col[:2]

# In[339]:


allData[~pd.isna(allData['UserInfo_1'])].head()

# In[344]:


bad_rate_diff={}
large_percent_cols_del = []
record_num = allData.shape[0]
missing_df = allData[need_to_deal_col]
for col in need_to_deal_col:
    temp = allData[pd.isna(allData[col])]
    other_temp = allData[~pd.isna(allData[col])]
#     display(temp[col].index)
    miss_index = temp[col].index
    bad_rate = temp['target'].mean()
    if temp.shape[0] / record_num > 0.2:
        bad_rate_diff[col] = np.log(bad_rate/(1-bad_rate+1e-10))
        if abs(bad_rate_diff[col]) > 1:
            if col in categorical_var:
                allData.loc[miss_index,col] = 'NAN'
            else:
                allData.loc[miss_index,col] = -100
            continue
    random_list = random.sample(list(other_temp[col]), len(temp))                       
    allData.loc[miss_index,col] = random_list                      

# In[345]:


allData_bk = allData.copy()
allData_bk.shape

# In[346]:


import numpy as np

# In[347]:


# '''
# 检查数值型变量, 缺失数值随机采样。
# '''
# def MissingContinuous(df,x):
#     missing_vals = df[x].map(lambda x: int(np.isnan(x)))
#     return sum(missing_vals) * 1.0 / df.shape[0]

# missing_pcnt_threshould_2 = 0.8
# deleted_var = []
# for col in numerical_var:
#     missingRate = MissingContinuous(allData, col)
#     print('{0} has missing rate as {1}'.format(col, missingRate))
#     if missingRate > missing_pcnt_threshould_2:
#         deleted_var.append(col)
#         print('we delete variable {} because of its high missing rate'.format(col))
#     else:
#         if missingRate > 0:
#             not_missing = allData.loc[allData[col] == allData[col]][col]
#             #makeuped = allData[col].map(lambda x: MakeupRandom(x, list(not_missing)))
#             missing_position = allData.loc[allData[col] != allData[col]][col].index
#             not_missing_sample = random.sample(list(not_missing), len(missing_position))
# #             print(not_missing.shape)
# #             print(np.median(not_missing))
# #             print(len(not_missing_sample),not_missing_sample[:10])
# #             allData.loc[missing_position,col] = not_missing_sample
#             allData.loc[missing_position,col] = np.median(not_missing)
#             #del allData[col]
#             #allData[col] = makeuped
#             missingRate2 = MissingContinuous(allData, col)
#             print('missing rate after making up is:{}'.format(str(missingRate2)))
#             break
    
# if deleted_var != []:
#     for col in deleted_var:
#         numerical_var.remove(col)
#         del allData[col]

# In[348]:


allData.shape

# In[349]:


# allData.to_csv(folderOfData+'allData_1.csv',
#                header=True,encoding='gbk', columns = allData.columns,
#                index=False，float_format="%.5f")

# In[350]:


allData.describe()

# In[351]:


allData[categorical_var].head()

# In[356]:


allData.shape, len(numerical_var), len(categorical_var)

# In[353]:


allData[numerical_var] = allData[numerical_var].astype(float)
allData[categorical_var] = allData[categorical_var].astype(str)

# In[354]:


allData.head()

# In[358]:


allData.to_csv(folderOfData+'allData_1.csv',header=True,encoding='gbk', columns = allData.columns,
               index=False, index_label = True)

# In[360]:


print(categorical_var)

# In[ ]:



