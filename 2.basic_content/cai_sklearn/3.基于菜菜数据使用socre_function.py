#!/usr/bin/env python
# coding: utf-8

# In[1]:


%matplotlib inline
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
 
#其实日常在导库的时候，并不是一次性能够知道我们要用的所有库的。通常都是在建模过程中逐渐导入需要的库。

# In[2]:


import warnings
warnings.filterwarnings("ignore")

# In[3]:


data = pd.read_csv(r"./rankingcard.csv",index_col=0)

# In[4]:


data.shape

# In[5]:


data.head()

# In[6]:


#观察数据类型
data.head()#注意可以看到第一列为标签，剩下的10列为特征
 
#观察数据结构
data.shape#(150000, 11)
data.info()

# In[7]:


#去除重复值
data.drop_duplicates(inplace=True)#inplace=True表示替换原数据
 
data.info()
 
#删除之后千万不要忘记，恢复索引
data.index = range(data.shape[0])
 
data.info()

# In[8]:


#探索缺失值
data.info()
data.isnull().sum()/data.shape[0]#得到缺失值的比例
#data.isnull().mean()#上一行代码的另一种形式书写

# In[9]:


data["NumberOfDependents"].fillna(int(data["NumberOfDependents"].mean()),inplace=True)
#这里用均值填补家庭人数这一项 
#如果你选择的是删除那些缺失了2.5%的特征，千万记得恢复索引哟~
 
data.info()
data.isnull().sum()/data.shape[0]


# In[10]:


def fill_missing_rf(X,y,to_fill):

    """
    使用随机森林填补一个特征的缺失值的函数

    参数：
    X：要填补的特征矩阵
    y：完整的，没有缺失值的标签
    to_fill：字符串，要填补的那一列的名称
    """

    #构建我们的新特征矩阵和新标签
    df = X.copy()
    fill = df.loc[:,to_fill]
    df = pd.concat([df.loc[:,df.columns != to_fill],pd.DataFrame(y)],axis=1)

    # 找出我们的训练集和测试集
    Ytrain = fill[fill.notnull()]
    Ytest = fill[fill.isnull()]
    Xtrain = df.iloc[Ytrain.index,:]
    Xtest = df.iloc[Ytest.index,:]

    #用随机森林回归来填补缺失值
    from sklearn.ensemble import RandomForestRegressor as rfr
    rfr = rfr(n_estimators=100)
    rfr = rfr.fit(Xtrain, Ytrain)
    Ypredict = rfr.predict(Xtest)

    return Ypredict

# In[11]:


X = data.iloc[:,1:]
y = data["SeriousDlqin2yrs"]#y = data.iloc[:,0]
X.shape#(149391, 10)

#=====[TIME WARNING:1 min]=====#
y_pred = fill_missing_rf(X,y,"MonthlyIncome")

#注意可以通过以下代码检验数据是否数量相同
# y_pred.shape ==  data.loc[data.loc[:,"MonthlyIncome"].isnull(),"MonthlyIncome"].shape

#确认我们的结果合理之后，我们就可以将数据覆盖了
data.loc[data.loc[:,"MonthlyIncome"].isnull(),"MonthlyIncome"] = y_pred

data.info()

# In[12]:


#描述性统计
# data.describe()
data.describe([0.01,0.1,0.25,.5,.75,.9,.99]).T

# In[13]:


#异常值也被我们观察到，年龄的最小值居然有0，这不符合银行的业务需求，即便是儿童账户也要至少8岁，我们可以
# 查看一下年龄为0的人有多少
(data["age"] == 0).sum()
#发现只有一个人年龄为0，可以判断这肯定是录入失误造成的，可以当成是缺失值来处理，直接删除掉这个样本
data = data[data["age"] != 0]
 
"""
另外，有三个指标看起来很奇怪：
 
"NumberOfTime30-59DaysPastDueNotWorse"
"NumberOfTime60-89DaysPastDueNotWorse"
"NumberOfTimes90DaysLate"
 
这三个指标分别是“过去两年内出现35-59天逾期但是没有发展的更坏的次数”，“过去两年内出现60-89天逾期但是没
有发展的更坏的次数”,“过去两年内出现90天逾期的次数”。这三个指标，在99%的分布的时候依然是2，最大值却是
98，看起来非常奇怪。一个人在过去两年内逾期35~59天98次，一年6个60天，两年内逾期98次这是怎么算出来的？
 
我们可以去咨询业务人员，请教他们这个逾期次数是如何计算的。如果这个指标是正常的，那这些两年内逾期了98次的
客户，应该都是坏客户。在我们无法询问他们情况下，我们查看一下有多少个样本存在这种异常：
 
"""
data[data.loc[:,"NumberOfTimes90DaysLate"] > 90]
data[data.loc[:,"NumberOfTimes90DaysLate"] > 90].count()
data.loc[:,"NumberOfTimes90DaysLate"].value_counts()
 
#有225个样本存在这样的情况，并且这些样本，我们观察一下，标签并不都是1，他们并不都是坏客户。因此，我们基
# 本可以判断，这些样本是某种异常，应该把它们删除。
 
data = data[data.loc[:,"NumberOfTimes90DaysLate"] < 90]
#一定要恢复索引
data.index = range(data.shape[0])
data.info()

# In[14]:


#探索标签的分布
X = data.iloc[:,1:]
y = data.iloc[:,0]
 
y.value_counts()#查看每一类别值得数据量，查看样本是否均衡
 
n_sample = X.shape[0]
 
n_1_sample = y.value_counts()[1]
n_0_sample = y.value_counts()[0]
 
print('样本个数：{}; 1占{:.2%}; 0占{:.2%}'.format(n_sample,n_1_sample/n_sample,n_0_sample/n_sample))
#样本个数：149165; 1占6.62%; 0占93.38%

# In[15]:


from sklearn.model_selection import train_test_split
X = pd.DataFrame(X)
y = pd.DataFrame(y)
 
X_train, X_vali, Y_train, Y_vali = train_test_split(X,y,test_size=0.3,random_state=420)
model_data = pd.concat([Y_train, X_train], axis=1)#训练数据构建模型
model_data.index = range(model_data.shape[0])
model_data.columns = data.columns
 
vali_data = pd.concat([Y_vali, X_vali], axis=1)#验证集
vali_data.index = range(vali_data.shape[0])
vali_data.columns = data.columns
 
model_data.to_csv(r".\model_data.csv")#训练数据
vali_data.to_csv(r".\vali_data.csv")#验证数据


# In[16]:


from scorecard_functions_V3 import *

# In[17]:


deleted_features = []   #将处理过的变量删除，防止对后面建模的干扰
encoded_features = {}   #将bad rate编码方式保存下来，在以后的测试和生产环境中需要使用
merged_features = {}    #将类别型变量合并方案保留下来
var_IV = {}  #save the IV values for binned features       #将IV值保留和WOE值
var_WOE = {}


# In[18]:


model_data.head()

# In[19]:


numerical_var = ['RevolvingUtilizationOfUnsecuredLines','age',
                 'DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans']

categorical_var = ["NumberOfTime30-59DaysPastDueNotWorse"
            ,"NumberOfTimes90DaysLate"
            ,"NumberRealEstateLoansOrLines"
            ,"NumberOfTime60-89DaysPastDueNotWorse"
            ,"NumberOfDependents"]

# In[20]:


def BinBadRate(df, col, SeriousDlqin2yrs, grantRateIndicator=0):
    '''
    :param df: 需要计算好坏比率的数据集
    :param col: 需要计算好坏比率的特征
    :param SeriousDlqin2yrs: 好坏标签
    :param grantRateIndicator: 1返回总体的坏样本率，0不返回
    :return: 每箱的坏样本率，以及总体的坏样本率（当grantRateIndicator＝＝1时）
    '''
    total = df.groupby([col])[SeriousDlqin2yrs].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[SeriousDlqin2yrs].sum()
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

def BadRateEncoding(df, col, SeriousDlqin2yrs):
    '''
    :return: 在数据集df中，用坏样本率给col进行编码。SeriousDlqin2yrs表示坏样本标签
    '''
    regroup = BinBadRate(df, col, SeriousDlqin2yrs, grantRateIndicator=0)[1]
    if display_flag:
        print("regroup:",regroup)
    
    br_dict = regroup[[col,'bad_rate']].set_index([col]).to_dict(orient='index')
    for k, v in br_dict.items():
        br_dict[k] = v['bad_rate']
    badRateEnconding = df[col].map(lambda x: br_dict[x])
    return {'encoding':badRateEnconding, 'bad_rate':br_dict}

def MergeBad0(df,col,SeriousDlqin2yrs, direction='bad'):
    '''
     :param df: 包含检验0％或者100%坏样本率
     :param col: 分箱后的变量或者类别型变量。检验其中是否有一组或者多组没有坏样本或者没有好样本。如果是，则需要进行合并
     :param SeriousDlqin2yrs: 目标变量，0、1表示好、坏
     :return: 合并方案，使得每个组里同时包含好坏样本
     '''
    regroup = BinBadRate(df, col, SeriousDlqin2yrs)[1]
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


# In[21]:


model_data.head(1)

# In[22]:


display_flag =False

# In[23]:


col = 'NumberOfTimes90DaysLate'
encoding_result = BadRateEncoding(model_data, col, 'SeriousDlqin2yrs')

# In[24]:


mergeBin = MergeBad0(model_data, col, 'SeriousDlqin2yrs')

# In[25]:


regroup = BinBadRate(model_data, col, 'SeriousDlqin2yrs', grantRateIndicator=0)
regroup[1]

# In[26]:


deleted_features = []   #将处理过的变量删除，防止对后面建模的干扰
encoded_features = {}   #将bad rate编码方式保存下来，在以后的测试和生产环境中需要使用
merged_features = {}    #将类别型变量合并方案保留下来
var_IV = {}  #save the IV values for binned features       #将IV值保留和WOE值
var_WOE = {}
for col in categorical_var:
    print('we are processing {}'.format(col))
    if len(set(model_data[col]))>5:
        print('{} is encoded with bad rate'.format(col))
        col0 = str(col)+'_encoding'

        #(1), 计算坏样本率并进行编码
        encoding_result = BadRateEncoding(model_data, col, 'SeriousDlqin2yrs')
        model_data[col0], br_encoding = encoding_result['encoding'],encoding_result['bad_rate']

        #(2), 将（1）中的编码后的变量也加入数值型变量列表中，为后面的卡方分箱做准备
        numerical_var.append(col0)
        
        #(3), 保存编码结果
        encoded_features[col] = [col0, br_encoding]

        #(4), 删除原始值

        deleted_features.append(col)
    else:
        bad_bin = model_data.groupby([col])['SeriousDlqin2yrs'].sum()
        #对于类别数少于5个，但是出现0坏样本的特征需要做处理
        if min(bad_bin) == 0:
            print('{} has 0 bad sample!'.format(col))
            col1 = str(col) + '_mergeByBadRate'
            #(1), 找出最优合并方式，使得每一箱同时包含好坏样本
            mergeBin = MergeBad0(model_data, col, 'SeriousDlqin2yrs')
            #(2), 依照（1）的结果对值进行合并
            model_data[col1] = model_data[col].map(mergeBin)
            maxPcnt = MaximumBinPcnt(model_data, col1)
            #如果合并后导致有箱占比超过90%，就删除。
            if maxPcnt > 0.9:
                print('{} is deleted because of large percentage of single bin'.format(col))
                deleted_features.append(col)
                categorical_var.remove(col)
                del model_data[col]
                continue
            #(3) 如果合并后的新的变量满足要求，就保留下来
            merged_features[col] = [col1, mergeBin]
            WOE_IV = CalcWOE(model_data, col1, 'SeriousDlqin2yrs')
            var_WOE[col1] = WOE_IV['WOE']
            var_IV[col1] = WOE_IV['IV']
            # del model_data[col]
            deleted_features.append(col)
            
        else:
            WOE_IV = CalcWOE(model_data, col, 'SeriousDlqin2yrs')
            var_WOE[col] = WOE_IV['WOE']
            var_IV[col] = WOE_IV['IV']

# In[27]:


var_WOE.keys()

# In[28]:


numerical_var

# In[29]:


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
    if -1 in set(model_data[col]):
        special_attribute = [-1]
    else:
        special_attribute = []
    cutOffPoints = ChiMerge(model_data, col, 'SeriousDlqin2yrs',special_attribute=special_attribute)
    print("cutOffPoints:",cutOffPoints)
    var_cutoff[col] = cutOffPoints
    model_data[col1] = model_data[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))
    
    #(2), check whether the bad rate is monotone
    BRM = BadRateMonotone(model_data, col1, 'SeriousDlqin2yrs',special_attribute=special_attribute)
    if not BRM:
        if special_attribute == []:
            bin_merged = Monotone_Merge(model_data, 'SeriousDlqin2yrs', col1)
            removed_index = []
            for bin in bin_merged:
                if len(bin)>1:
                    indices = [int(b.replace('Bin ','')) for b in bin]
                    removed_index = removed_index+indices[0:-1]
            removed_point = [cutOffPoints[k] for k in removed_index]
            for p in removed_point:
                cutOffPoints.remove(p)
            var_cutoff[col] = cutOffPoints
            model_data[col1] = model_data[col].map(lambda x: AssignBin(x, cutOffPoints, special_attribute=special_attribute))
        else:
            cutOffPoints2 = [i for i in cutOffPoints if i not in special_attribute]
            temp = model_data.loc[~model_data[col].isin(special_attribute)]
            bin_merged = Monotone_Merge(temp, 'SeriousDlqin2yrs', col1)
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
            model_data[col1] = model_data[col].map(lambda x: AssignBin(x, cutOffPoints2, special_attribute=special_attribute))

    #(3), 分箱后再次检查是否有单一的值占比超过90%。如果有，删除该变量
    maxPcnt = MaximumBinPcnt(model_data, col1)
    if maxPcnt > 0.9:
        # del model_data[col1]
        deleted_features.append(col)
        numerical_var.remove(col)
        print('we delete {} because the maximum bin occupies more than 90%'.format(col))
        continue
    
    WOE_IV = CalcWOE(model_data, col1, 'SeriousDlqin2yrs')
    var_IV[col] = WOE_IV['IV']
    var_WOE[col] = WOE_IV['WOE']
    #del model_data[col]

# In[30]:


len(numerical_var) + len(categorical_var), model_data.shape

# In[31]:


var_WOE.keys()

# In[32]:


vali_data.columns

# In[33]:


vali_data['NumberOfTime30-59DaysPastDueNotWorse_encoding'] = vali_data['NumberOfTime30-59DaysPastDueNotWorse']

# In[34]:


for col in var_WOE.keys():
    print(col)
    col2 = str(col)+"_WOE"
    if col in var_cutoff.keys():
        cutOffPoints = var_cutoff[col]
        special_attribute = []
        if - 1 in cutOffPoints:
            special_attribute = [-1]
        binValue = model_data[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))
        model_data[col2] = binValue.map(lambda x: var_WOE[col][x])
        
        binValue = vali_data[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))
        vali_data[col2] = binValue.map(lambda x: var_WOE[col][x])        
        
        
    else:
        print(col,col2)
        model_data[col2] = model_data[col].map(lambda x: var_WOE[col][x])
        
        binValue = vali_data[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))
        vali_data[col2] = binValue.map(lambda x: var_WOE[col][x])     
        

# In[35]:


var_IV

# In[36]:


vali_data.head(1)

# In[37]:


from matplotlib import pyplot as plt


all_IV = list(var_IV.values())
all_IV = sorted(all_IV, reverse=True)
plt.bar(x=range(len(all_IV)), height = all_IV)
iv_threshould = 0.01
varByIV = [k for k, v in var_IV.items() if v > iv_threshould]

# In[38]:


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
                roh = np.corrcoef([model_data[x1], model_data[x2]])[0, 1]
                if abs(roh) >= roh_thresould:
                    print('the correlation coeffient between {0} and {1} is {2}'.format(x1, x2, str(roh)))
                    if var_IV[var_IV_sorted[i]] > var_IV[var_IV_sorted[j]]:
                        removed_var.append(var_IV_sorted[j])
                    else:
                        removed_var.append(var_IV_sorted[i])

var_IV_sortet_2 = [i for i in var_IV_sorted if i not in removed_var]

# In[39]:


var_IV_sortet_2

# In[40]:


model_data.head(1)

# In[41]:


### (iii) check the multi-colinearity according to VIF > 10
from sklearn.linear_model import LinearRegression

for i in range(len(var_IV_sortet_2)):
    x0 = model_data[var_IV_sortet_2[i]+'_WOE']
    x0 = np.array(x0)
    X_Col = [k+'_WOE' for k in var_IV_sortet_2 if k != var_IV_sortet_2[i]]
    X = model_data[X_Col]
    X = np.matrix(X)
    regr = LinearRegression()
    clr= regr.fit(X, x0)
    x_pred = clr.predict(X)
    R2 = 1 - ((x_pred - x0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum()
    vif = 1/(1-R2)
    if vif > 10:
        print("Warning: the vif for {0} is {1}".format(var_IV_sortet_2[i], vif))

# In[42]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# In[43]:


#########################
# Step 5: 应用逻辑回归模型#
#########################
multi_analysis = [i+'_WOE' for i in var_IV_sortet_2]
y = model_data['SeriousDlqin2yrs']
X = model_data[multi_analysis].copy()
X['intercept'] = [1]*X.shape[0]

# In[45]:


import statsmodels.api as sm

# In[46]:


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
    X_temp = model_data[var].copy().to_frame()
    X_temp['intercept'] = [1] * X_temp.shape[0]
    LR = sm.Logit(y, X_temp).fit()
    p_value_list[var] = LR.pvalues[var]
for k,v in p_value_list.items():
    print("{0} has p-value of {1} in univariate regression".format(k,v))


#发现有变量的系数为正，因此需要单独检验正确性
varPositive = [k for k,v in params.items() if v >= 0]
coef_list = {}
for var in varPositive:
    X_temp = model_data[var].copy().to_frame()
    X_temp['intercept'] = [1] * X_temp.shape[0]
    LR = sm.Logit(y, X_temp).fit()
    coef_list[var] = LR.params[var]
for k,v in coef_list.items():
    print("{0} has coefficient of {1} in univariate regression".format(k,v))


selected_var = [multi_analysis[0]]
for var in multi_analysis[1:]:
    try_vars = selected_var+[var]
    X_temp = model_data[try_vars].copy()
    X_temp['intercept'] = [1] * X_temp.shape[0]
    LR = sm.Logit(y, X_temp).fit()
    #summary = LR.summary2()
    pvals, params = LR.pvalues, LR.params
    del params['intercept']
    if max(pvals)<0.1 and max(params)<0:
        selected_var.append(var)

LR.summary2()

y_pred = LR.predict(X_temp)
roc_auc_score(model_data['SeriousDlqin2yrs'], y_pred)

# In[47]:


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

# In[48]:


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

# In[49]:


#########################
# Step 6: 尺度化与性能检验#
#########################
plot = True
scores = Prob2Score(y_pred, 200, 100)
plt.hist(scores,bins=100)
y_preds = [ 1 if i > 0.5 else 0 for i in y_pred]
scorecard = pd.DataFrame({'y_pred':y_preds, 'y_real':list(model_data['SeriousDlqin2yrs']),'score':y_pred})
display(scorecard.head())
print(KS(scorecard,'score','y_real'))

# 也可用sklearn带的函数
roc_auc_score(model_data['SeriousDlqin2yrs'], y_pred)

# In[50]:


from sklearn.metrics import confusion_matrix
confusion_matrix(model_data['SeriousDlqin2yrs'], y_preds)

# In[51]:


1-6.0/102

# In[52]:


Valid_X = vali_data.copy()
Valid_X['intercept'] = [1] * Valid_X.shape[0]

# In[53]:


need_columns = list(X_temp.columns)

# In[54]:


valid_y_pred = LR.predict(Valid_X[need_columns])

# In[55]:


valid_y_preds = [ 1 if i > 0.5 else 0 for i in valid_y_pred]

# In[56]:


valid_scorecard = pd.DataFrame({'y_pred':valid_y_preds, 'y_real':list(vali_data['SeriousDlqin2yrs']),'score':valid_y_pred})
display(valid_scorecard.head())
print(KS(valid_scorecard,'score','y_real'))

# 也可用sklearn带的函数
roc_auc_score(vali_data['SeriousDlqin2yrs'], valid_y_pred)

# In[57]:


from sklearn.metrics import confusion_matrix
confusion_matrix(vali_data['SeriousDlqin2yrs'], valid_y_preds)

# In[58]:


40/42

# In[ ]:


safdagag

# In[ ]:



