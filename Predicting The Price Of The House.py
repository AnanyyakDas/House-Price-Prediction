#!/usr/bin/env python
# coding: utf-8

# # Data Understanding and Importing Libraries

# In[219]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[220]:


import warnings
warnings.filterwarnings('ignore')


# In[221]:


# Reading the housing data
housing=pd.read_excel(r'C:\Users\Dell\Desktop\DS-assignment\DS - Assignment Part 1 data set.xlsx')
housing.head()


# In[222]:


housing.info()


# -- Transaction date is in incorrect format we need to change that

# In[223]:


# we will extract only year from it 
housing['Transaction date'] = pd.to_datetime(housing['Transaction date'], format='%y%m%d') 


# In[224]:


housing.isnull().sum()


# 1. No need for data cleaning as there is no null values in the housing dataset

# In[225]:


housing1=housing


# In[226]:


housing1.drop(['longitude','latitude','Transaction date'],axis=1,inplace=True)


# In[227]:


housing1.head()


# 2. Making a Total House price column 

# In[228]:


housing1['House Price']=round(housing1['House size (sqft)']*housing1['House price of unit area'],0)


# In[229]:


housing1.head()


# In[230]:


housing1.shape


# # Univariate and Bivariate Analysis

# In[231]:


housing1.groupby(['Number of bedrooms'])['House Price'].median().plot.barh()
plt.legend()
plt.show()


# In[232]:


housing1.groupby(['Number of bedrooms'])['House size (sqft)'].median().plot.barh()
plt.legend()
plt.show()


# In[233]:


plt.figure(figsize=[10,5])
sns.heatmap(data=housing1.corr(),annot=True,cmap='RdYlGn')
plt.title('Correlation Map Of Housing Factors ')
plt.show()


# In[234]:


housing1.groupby(['House Age'])['House Price'].median().plot()
plt.legend()
plt.show()


# In[235]:


housing1.head()


# In[236]:


housing1['Number of convenience stores'].value_counts()


# # Splitting the Data into Train and Test DataSet

# In[237]:


from sklearn.model_selection import train_test_split
df_train,df_test=train_test_split(housing1,train_size=0.7,test_size=0.3,random_state=100)


# In[238]:


print(df_train.shape)
print(df_test.shape)


# ### Rescaling the features

# In[239]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()


# In[240]:


housing1.head()


# In[241]:


numerical_vars=['House Age','Distance from nearest Metro station (km)','Number of convenience stores','Number of bedrooms','House size (sqft)','House price of unit area','House Price']

df_train[numerical_vars]=scaler.fit_transform(df_train[numerical_vars])

df_train.head()


# ### Dividing X and y sets for model building

# In[242]:


y_train=df_train.pop('House Price')
X_train=df_train


# # Model Building  

# ### Model Building Using RFE ( Recursive Feature Elimination )

# In[243]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[244]:


lm=LinearRegression()
lm.fit(X_train,y_train)

rfe=RFE(lm,10)
rfe=rfe.fit(X_train,y_train)


# In[245]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[246]:


col = X_train.columns[rfe.support_]
col


# In[247]:


X_train.columns[~rfe.support_]


# # Building model using Statsmodel for detailed statistics

# ### Model 1

# In[248]:


import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[249]:


X_train_rfe = X_train[col]


# In[250]:


X_train_rfe = sm.add_constant(X_train_rfe)


# In[251]:


lm = sm.OLS(y_train,X_train_rfe).fit()
print(lm.summary())


# In[252]:


vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# -- After checking the p value and VIFs 'House Age' is insignificant to other characteristics for house price

# In[253]:


X_train_rfe=X_train_rfe.drop(['House Age'],axis=1)


# ### Model 2

# In[254]:


X_train_lm = sm.add_constant(X_train_rfe)


# In[255]:


lm = sm.OLS(y_train,X_train_rfe).fit()
print(lm.summary())


# In[256]:


vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# -- p value & vif is higher of 'Number of bedrooms' 

# In[257]:


X_train_rfe=X_train_rfe.drop(['Number of bedrooms'],axis=1)


# ### Model 3

# In[258]:


X_train_lm = sm.add_constant(X_train_rfe)


# In[259]:


lm = sm.OLS(y_train,X_train_rfe).fit()
print(lm.summary())


# In[260]:


vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

-- 'Distance from nearest Metro station (km)' is now insignificant related to the price prediction according to P-Values and       vif
# In[261]:


X_train_rfe=X_train_rfe.drop(['Distance from nearest Metro station (km)'],axis=1)


# ### Model 4 

# In[262]:


X_train_lm = sm.add_constant(X_train_rfe)


# In[263]:


lm = sm.OLS(y_train,X_train_rfe).fit()
print(lm.summary())


# -- Model 4 is the final model as the vif's and the p-values are quite low for their respected variable for price prediction

# # Residual Analysis of the Train Data

# In[265]:


y_train_price = lm.predict(X_train_lm)


# In[267]:


fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                   
plt.xlabel('Errors', fontsize = 18)
plt.show()


# # Making Predictions

# In[268]:


numerical_vars=['House Age','Distance from nearest Metro station (km)','Number of convenience stores','Number of bedrooms','House size (sqft)','House price of unit area','House Price']

df_test[numerical_vars]=scaler.fit_transform(df_test[numerical_vars])

df_test.head()


# In[270]:


y_test = df_test.pop('House Price')
X_test = df_test


# In[272]:


X_test_rfe = X_test[X_train_rfe.columns[1:]] 
X_test_rfe = sm.add_constant(X_test_rfe)


# In[273]:


y_pred = lm.predict(X_test_rfe)


# # Model Evaluation

# In[274]:


fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)               
plt.xlabel('y_test', fontsize=18)                          
plt.ylabel('y_pred', fontsize=16)
plt.show()


# In[277]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)*100


# In[ ]:




